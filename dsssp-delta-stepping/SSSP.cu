#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <climits>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/transform.h>

#define INF 2147483647
#define BLOCK_SIZE 256

#define STATE_NONE 0
#define STATE_NEAR 1
#define STATE_FAR  2
#define STATE_PROC 3

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)    \
                  << " at " << __FILE__ << ":" << __LINE__       \
                  << std::endl;                                   \
        std::exit(EXIT_FAILURE);                                  \
    }                                                               \
} while (0)

// KERNELS

__global__ void init_kernel(int N, int source, int *tent, int *state)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v < N) {
        tent[v] = INF;
        state[v] = STATE_NONE;
    }

    if (v == source) {
        tent[v] = 0;
        state[v] = STATE_NEAR;
    }
}

// Initial adaptive step: the source has no delta yet, so relax all of its
// outgoing edges into Far. This creates the first candidate set from which
// adaptive delta is chosen.
__global__ void seed_source_kernel(
    const int *offsets,
    const int *neighs,
    const int *weights,
    int *tent,
    int *state,
    int source,
    int *far_queue,
    int *far_size,
    int *bucket_id)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = offsets[source];
    int end   = offsets[source + 1];

    if (begin + e >= end) return;

    int pos = begin + e;
    int v = neighs[pos];
    int nd = weights[pos];

    if (nd >= INF) return;

    int old = atomicMin(&tent[v], nd);
    if (nd >= old) return;

    bucket_id[v] = 0;

    if (atomicCAS(&state[v], STATE_NONE, STATE_FAR) == STATE_NONE) {
        int p = atomicAdd(far_size, 1);
        far_queue[p] = v;
    }
}

// Light-edge relaxation. current_near is read-only during this kernel;
// next_near receives vertices that need another light-edge iteration.
//
// state transitions:
// NONE -> NEAR
// FAR  -> NEAR
// PROC -> NEAR   (vertex was already processed but got improved)
//
// A vertex is inserted into Far only when it has no existing queue entry.
__global__ void relax_light_kernel(
    const int *offsets,
    const int *neighs,
    const int *weights,
    int *tent,
    int *state,
    const int *current_near,
    int current_size,
    int *next_near,
    int *next_size,
    int *far_queue,
    int *far_size,
    int current_dist,
    int delta,
    int *bucket_id,
    int phase,
    int *settled_tag,
    int *settled_queue,
    int *settled_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= current_size) return;

    int u = current_near[i];
    int du = tent[u];
    if (du == INF) return;

    atomicCAS(&state[u], STATE_NEAR, STATE_PROC);

    // Add u to the current bucket's settled list exactly once.
    int old_tag = atomicExch(&settled_tag[u], phase);
    if (old_tag != phase) {
        int p = atomicAdd(settled_size, 1);
        settled_queue[p] = u;
    }

    int begin = offsets[u];
    int end   = offsets[u + 1];

    for (int e = begin; e < end; ++e) {
        int v = neighs[e];
        int w = weights[e];

        // Only light edges belong to the repeated inner phase.
        if (w > delta) continue;

        int nd = du + w;
        if (nd >= INF) continue;

        int old = atomicMin(&tent[v], nd);
        if (nd >= old) continue;

        if (delta > 0)
            bucket_id[v] = nd / delta;

        // convention used is right-open:
        // [current_dist, current_dist + delta).
        bool near;
        if (delta == 0)
            near = (nd == current_dist);
        else
            near = (nd < current_dist + delta);

        if (near) {
            while (true) {
                int s = atomicAdd(&state[v], 0);

                if (s == STATE_NONE) {
                    if (atomicCAS(&state[v], STATE_NONE, STATE_NEAR)
                        == STATE_NONE) {
                        int p = atomicAdd(next_size, 1);
                        next_near[p] = v;
                        break;
                    }
                }
                else if (s == STATE_FAR) {
                    if (atomicCAS(&state[v], STATE_FAR, STATE_NEAR)
                        == STATE_FAR) {
                        int p = atomicAdd(next_size, 1);
                        next_near[p] = v;
                        break;
                    }
                }
                else if (s == STATE_PROC) {
                    if (atomicCAS(&state[v], STATE_PROC, STATE_NEAR)
                        == STATE_PROC) {
                        int p = atomicAdd(next_size, 1);
                        next_near[p] = v;
                        break;
                    }
                }
                else {
                    // Already NEAR. Its existing queue occurrence is enough.
                    break;
                }
            }
        }
        else {
            while (true) {
                int s = atomicAdd(&state[v], 0);

                if (s == STATE_NONE) {
                    if (atomicCAS(&state[v], STATE_NONE, STATE_FAR)
                        == STATE_NONE) {
                        int p = atomicAdd(far_size, 1);
                        far_queue[p] = v;
                        break;
                    }
                }
                else if (s == STATE_PROC) {
                    if (atomicCAS(&state[v], STATE_PROC, STATE_FAR)
                        == STATE_PROC) {
                        int p = atomicAdd(far_size, 1);
                        far_queue[p] = v;
                        break;
                    }
                }
                else {
                    // Already represented by Near or Far.
                    break;
                }
            }
        }
    }
}

// Heavy edges are relaxed exactly once per bucket transition, after the
// light-edge closure. Since w > delta, an improved target is necessarily in
// a future bucket and therefore belongs in Far.
__global__ void relax_heavy_kernel(
    const int *offsets,
    const int *neighs,
    const int *weights,
    int *tent,
    int *state,
    const int *settled_queue,
    int settled_size,
    int *far_queue,
    int *far_size,
    int delta,
    int *bucket_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= settled_size) return;

    int u = settled_queue[i];
    int du = tent[u];
    if (du == INF) return;

    for (int e = offsets[u]; e < offsets[u + 1]; ++e) {
        int v = neighs[e];
        int w = weights[e];

        if (w <= delta) continue;

        int nd = du + w;
        if (nd >= INF) continue;

        int old = atomicMin(&tent[v], nd);
        if (nd >= old) continue;

        if (delta > 0)
            bucket_id[v] = nd / delta;

        // Heavy relaxation always produces a future distance.
        while (true) {
            int s = atomicAdd(&state[v], 0);

            if (s == STATE_NONE) {
                if (atomicCAS(&state[v], STATE_NONE, STATE_FAR)
                    == STATE_NONE) {
                    int p = atomicAdd(far_size, 1);
                    far_queue[p] = v;
                    break;
                }
            }
            else if (s == STATE_NEAR) {
                // An existing smaller Near distance wins, so this normally
                // cannot happen after atomicMin succeeds. Keep it represented
                // by the existing Near entry if it does.
                break;
            }
            else if (s == STATE_PROC) {
                if (atomicCAS(&state[v], STATE_PROC, STATE_FAR)
                    == STATE_PROC) {
                    int p = atomicAdd(far_size, 1);
                    far_queue[p] = v;
                    break;
                }
            }
            else {
                // Already in Far.
                break;
            }
        }
    }
}

__global__ void finish_near_kernel(
    const int *current_near,
    int n,
    int *state)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicCAS(&state[current_near[i]], STATE_PROC, STATE_NONE);
}

// Promote a prefix of Far into Near. Entries that are stale because a vertex
// was moved elsewhere are ignored by the state check.
__global__ void promote_kernel(
    const int *far_queue,
    int count,
    int *state,
    int *near_queue,
    int *near_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    int v = far_queue[i];

    if (atomicCAS(&state[v], STATE_FAR, STATE_NEAR) == STATE_FAR) {
        int p = atomicAdd(near_size, 1);
        near_queue[p] = v;
    }
}

// THRUST FUNCTORS

struct TentComparator {
    const int *tent;

    __host__ __device__
    bool operator()(int a, int b) const {
        return tent[a] < tent[b];
    }
};

struct LowerBoundComparator {
    const int *tent;

    __host__ __device__
    bool operator()(int vertex, int value) const {
        return tent[vertex] < value;
    }
};

struct UpperBoundComparator {
    const int *tent;

    __host__ __device__
    bool operator()(int value, int vertex) const {
        return value < tent[vertex];
    }
};

struct IsNotFar {
    const int *state;

    __host__ __device__
    bool operator()(int v) const {
        return state[v] != STATE_FAR;
    }
};

// Key = forward physical-slot distance from current slot in the modular ring.
struct RingSlotFunctor {
    const int *bucket_id;
    int M;
    int current_slot;

    __host__ __device__
    int operator()(int v) const {
        int slot = bucket_id[v] % M;
        int d = slot - current_slot;
        if (d < 0) d += M;
        return d;
    }
};

// HOST HELPERS

static void set_device_int(int *p, int x)
{
    CUDA_CHECK(cudaMemcpy(p, &x, sizeof(int), cudaMemcpyHostToDevice));
}

static int get_device_int(const int *p)
{
    int x;
    CUDA_CHECK(cudaMemcpy(&x, p, sizeof(int), cudaMemcpyDeviceToHost));
    return x;
}

static void compact_far(
    int *d_far,
    int &h_far_size,
    int *d_far_size,
    const int *d_state)
{
    if (h_far_size == 0) return;

    thrust::device_ptr<int> p(d_far);

    IsNotFar pred;
    pred.state = d_state;

    thrust::device_ptr<int> new_end =
        thrust::remove_if(p, p + h_far_size, pred);

    h_far_size = static_cast<int>(new_end - p);
    set_device_int(d_far_size, h_far_size);
}

// SINGLE SOURCE DRIVER

void run_delta_stepping_single_source(
    int N,
    int source,
    int delta_mode,
    int K,
    int M,
    const int *d_offsets,
    const int *d_neighs,
    const int *d_weights,
    int *d_tent,
    int *d_state,
    int *d_near_a,
    int *d_near_b,
    int *d_far,
    int *d_near_size,
    int *d_next_near_size,
    int *d_far_size,
    int *d_bucket_id,
    int *d_ring_key,
    int *d_settled_queue,
    int *d_settled_size,
    int *d_settled_tag,
    std::ofstream &outfile)
{
    int blocksN = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int zero = 0;
    int one = 1;

    // Per-source initialization. No cudaMalloc/cudaFree here.
    init_kernel<<<blocksN, BLOCK_SIZE>>>(
        N, source, d_tent, d_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    set_device_int(d_near_size, one);
    set_device_int(d_next_near_size, zero);
    set_device_int(d_far_size, zero);
    set_device_int(d_settled_size, zero);

    CUDA_CHECK(cudaMemcpy(
        d_near_a, &source, sizeof(int), cudaMemcpyHostToDevice));

    // phase is positive because settled_tag starts at zero.
    int phase = 1;

    int *current_near = d_near_a;
    int *next_near = d_near_b;
    int h_near_size = 1;
    int h_far_size = 0;

    int current_dist = 0;
    int delta = delta_mode;
    long long current_bucket = 0;

    thrust::device_ptr<int> far_ptr(d_far);
    thrust::device_ptr<int> key_ptr(d_ring_key);

    // Adaptive mode needs a first candidate set before delta exists.
    // So the source is therefore seeded into Far by relaxing all of its
    // outgoing edges once.
    if (delta_mode == -1) {
        int source_degree = 0;
        CUDA_CHECK(cudaMemcpy(
            &source_degree,
            d_offsets + source + 1,
            sizeof(int), cudaMemcpyDeviceToHost));

        int source_begin = 0;
        CUDA_CHECK(cudaMemcpy(
            &source_begin,
            d_offsets + source,
            sizeof(int), cudaMemcpyDeviceToHost));
        source_degree -= source_begin;

        if (source_degree > 0) {
            int grid = (source_degree + BLOCK_SIZE - 1) / BLOCK_SIZE;
            seed_source_kernel<<<grid, BLOCK_SIZE>>>(
                d_offsets, d_neighs, d_weights,
                d_tent, d_state,
                source, d_far, d_far_size, d_bucket_id);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Source has now been consumed.
        set_device_int(d_near_size, zero);
        h_near_size = 0;
        h_far_size = get_device_int(d_far_size);
    }

    // MAIN DELTA-STEPPING LOOP
    while (true) {
        // settled_queue belongs to the whole current bucket transition,
        // not to one individual light-edge iteration.
        set_device_int(d_settled_size, zero);

        // LIGHT PHASE: repeatedly drain Near.
        while (h_near_size > 0) {
            set_device_int(d_next_near_size, zero);

            int grid = (h_near_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            relax_light_kernel<<<grid, BLOCK_SIZE>>>(
                d_offsets, d_neighs, d_weights,
                d_tent, d_state,
                current_near, h_near_size,
                next_near, d_next_near_size,
                d_far, d_far_size,
                current_dist, delta,
                d_bucket_id,
                phase,
                d_settled_tag,
                d_settled_queue,
                d_settled_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            finish_near_kernel<<<grid, BLOCK_SIZE>>>(
                current_near, h_near_size, d_state);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_next_size = get_device_int(d_next_near_size);

            std::swap(current_near, next_near);
            std::swap(d_near_size, d_next_near_size);
            h_near_size = h_next_size;
        }

        // Heavy phase: process every vertex settled in this bucket once.
        int h_settled_size = get_device_int(d_settled_size);

        if (h_settled_size > 0 && delta >= 0) {
            int grid = (h_settled_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            relax_heavy_kernel<<<grid, BLOCK_SIZE>>>(
                d_offsets, d_neighs, d_weights,
                d_tent, d_state,
                d_settled_queue, h_settled_size,
                d_far, d_far_size,
                delta, d_bucket_id);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        ++phase;
        if (phase == INT_MAX) {
            // Extremely unlikely, but allows the tag array to be reused.
            CUDA_CHECK(cudaMemset(d_settled_tag, 0, N * sizeof(int)));
            phase = 1;
        }

        h_far_size = get_device_int(d_far_size);
        if (h_near_size == 0 && h_far_size == 0)
            break;

        // ADAPTIVE MODE
        if (delta_mode == -1) {
            if (h_far_size == 0) break;

            TentComparator cmp;
            cmp.tent = d_tent;

            thrust::sort(
                far_ptr,
                far_ptr + h_far_size,
                cmp);

            int first_v;
            CUDA_CHECK(cudaMemcpy(
                &first_v, d_far, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                &current_dist, d_tent + first_v,
                sizeof(int), cudaMemcpyDeviceToHost));

            int promote_count = 0;

            if (h_far_size <= K) {
                int last_v;
                CUDA_CHECK(cudaMemcpy(
                    &last_v,
                    d_far + h_far_size - 1,
                    sizeof(int), cudaMemcpyDeviceToHost));

                int last_d;
                CUDA_CHECK(cudaMemcpy(
                    &last_d,
                    d_tent + last_v,
                    sizeof(int), cudaMemcpyDeviceToHost));

                delta = last_d - current_dist;
                if (delta <= 0) delta = 1;

                LowerBoundComparator lb;
                lb.tent = d_tent;

                int cutoff = current_dist + delta;
                auto cut = thrust::lower_bound(
                    far_ptr,
                    far_ptr + h_far_size,
                    cutoff,
                    lb);

                promote_count = static_cast<int>(cut - far_ptr);
            }
            else {
                int kth_v;
                CUDA_CHECK(cudaMemcpy(
                    &kth_v,
                    d_far + K - 1,
                    sizeof(int), cudaMemcpyDeviceToHost));

                int V;
                CUDA_CHECK(cudaMemcpy(
                    &V,
                    d_tent + kth_v,
                    sizeof(int), cudaMemcpyDeviceToHost));

                LowerBoundComparator lb;
                lb.tent = d_tent;

                auto first_V = thrust::lower_bound(
                    far_ptr,
                    far_ptr + h_far_size,
                    V,
                    lb);

                int I_safe = static_cast<int>(first_V - far_ptr);

                if (I_safe == 0) {
                    // K duplicate minimum entries: use upper_bound to find
                    // the complete equal-minimum prefix.
                    UpperBoundComparator ub;
                    ub.tent = d_tent;

                    auto up = thrust::upper_bound(
                        far_ptr,
                        far_ptr + h_far_size,
                        current_dist,
                        ub);

                    promote_count = static_cast<int>(up - far_ptr);
                    delta = 1;
                }
                else {
                    delta = V - current_dist;
                    if (delta <= 0) delta = 1;
                    promote_count = I_safe;
                }
            }

            // Adaptive delta must appear before the corresponding batch.
            outfile << delta << '\n';

            set_device_int(d_near_size, zero);

            if (promote_count > 0) {
                int grid = (promote_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                promote_kernel<<<grid, BLOCK_SIZE>>>(
                    d_far, promote_count,
                    d_state,
                    current_near,
                    d_near_size);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            compact_far(
                d_far, h_far_size, d_far_size, d_state);

            h_near_size = get_device_int(d_near_size);
            continue;
        }

        // STATIC MODE: MODULAR CIRCULAR BUCKET RING
        else {
            if (h_far_size == 0) break;

            // Delta=0 has no useful modulo bucket width. Handle it as a
            // zero-weight closure followed by minimum-distance extraction.
            if (delta == 0) {
                TentComparator cmp;
                cmp.tent = d_tent;

                thrust::sort(
                    far_ptr,
                    far_ptr + h_far_size,
                    cmp);

                int first_v;
                CUDA_CHECK(cudaMemcpy(
                    &first_v, d_far, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(
                    &current_dist,
                    d_tent + first_v,
                    sizeof(int), cudaMemcpyDeviceToHost));

                UpperBoundComparator ub;
                ub.tent = d_tent;

                auto up = thrust::upper_bound(
                    far_ptr,
                    far_ptr + h_far_size,
                    current_dist,
                    ub);

                int promote_count = static_cast<int>(up - far_ptr);

                set_device_int(d_near_size, zero);

                if (promote_count > 0) {
                    int grid = (promote_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    promote_kernel<<<grid, BLOCK_SIZE>>>(
                        d_far, promote_count,
                        d_state,
                        current_near,
                        d_near_size);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                compact_far(
                    d_far, h_far_size, d_far_size, d_state);

                h_near_size = get_device_int(d_near_size);
                continue;
            }

            // Physical ring:
             // M = ceil(Wmax / delta) + 1, 
             // slot = logical_bucket mod M
            RingSlotFunctor ring_key;
            ring_key.bucket_id = d_bucket_id;
            ring_key.M = M;
            ring_key.current_slot =
                static_cast<int>(current_bucket % M);

            thrust::transform(
                far_ptr,
                far_ptr + h_far_size,
                key_ptr,
                ring_key);

            thrust::sort_by_key(
                key_ptr,
                key_ptr + h_far_size,
                far_ptr);

            int offset;
            CUDA_CHECK(cudaMemcpy(
                &offset,
                d_ring_key,
                sizeof(int), cudaMemcpyDeviceToHost));

            current_bucket += offset;
            current_dist = static_cast<int>(
                current_bucket * static_cast<long long>(delta));

            // All entries with this physical key correspond to the next
            // logical bucket under the bounded-horizon invariant.
            auto bucket_end = thrust::upper_bound(
                key_ptr,
                key_ptr + h_far_size,
                offset);

            int promote_count =
                static_cast<int>(bucket_end - key_ptr);

            set_device_int(d_near_size, zero);

            if (promote_count > 0) {
                int grid = (promote_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                promote_kernel<<<grid, BLOCK_SIZE>>>(
                    d_far, promote_count,
                    d_state,
                    current_near,
                    d_near_size);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            compact_far(
                d_far, h_far_size, d_far_size, d_state);

            h_near_size = get_device_int(d_near_size);
        }
    }
}

// MAIN

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open input file "
                  << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to open output file "
                  << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;

    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
        infile >> sources[i];

    std::vector<int> offsets(N + 1);
    std::vector<int> neighs(E);
    std::vector<int> weights(E);

    for (int i = 0; i <= N; ++i)
        infile >> offsets[i];

    for (int i = 0; i < E; ++i)
        infile >> neighs[i];

    for (int i = 0; i < E; ++i)
        infile >> weights[i];

    infile.close();

    // Wmax is known once at startup, so the static ring size can also be
    // computed once and reused for every source.
    int Wmax = 0;
    for (int w : weights)
        Wmax = std::max(Wmax, w);

    // ONE-TIME GPU ALLOCATIONS

    int *d_offsets = nullptr;
    int *d_neighs = nullptr;
    int *d_weights = nullptr;

    int *d_tent = nullptr;
    int *d_state = nullptr;

    int *d_near_a = nullptr;
    int *d_near_b = nullptr;
    int *d_far = nullptr;

    int *d_near_size = nullptr;
    int *d_next_near_size = nullptr;
    int *d_far_size = nullptr;

    int *d_bucket_id = nullptr;
    int *d_ring_key = nullptr;

    int *d_settled_queue = nullptr;
    int *d_settled_size = nullptr;
    int *d_settled_tag = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_offsets, (N + 1) * sizeof(int)));

    if (E > 0) {
        CUDA_CHECK(cudaMalloc(
            &d_neighs, E * sizeof(int)));
        CUDA_CHECK(cudaMalloc(
            &d_weights, E * sizeof(int)));
    }

    CUDA_CHECK(cudaMalloc(&d_tent, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_state, N * sizeof(int)));

    // Two Near buffers are the important queue-buffer correction.
    CUDA_CHECK(cudaMalloc(&d_near_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_near_b, N * sizeof(int)));

    // One global Far queue, O(N).
    CUDA_CHECK(cudaMalloc(&d_far, N * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_near_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_near_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_far_size, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_bucket_id, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ring_key, N * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_settled_queue, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_settled_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_settled_tag, N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_offsets,
        offsets.data(),
        (N + 1) * sizeof(int),
        cudaMemcpyHostToDevice));

    if (E > 0) {
        CUDA_CHECK(cudaMemcpy(
            d_neighs,
            neighs.data(),
            E * sizeof(int),
            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(
            d_weights,
            weights.data(),
            E * sizeof(int),
            cudaMemcpyHostToDevice));
    }

    // Static modular ring size.
    int M = 1;
    if (delta_mode > 0) {
        M = (Wmax + delta_mode - 1) / delta_mode + 1;
        if (M < 1) M = 1;
    }

    // settled_tag is initially zero; source driver starts phase at 1.
    CUDA_CHECK(cudaMemset(
        d_settled_tag, 0, N * sizeof(int)));

    std::vector<int> h_tent(N);

    // MULTI-SOURCE LOOP

    for (int source : sources) {
        outfile << source << '\n';

        run_delta_stepping_single_source(
            N,
            source,
            delta_mode,
            K,
            M,
            d_offsets,
            d_neighs,
            d_weights,
            d_tent,
            d_state,
            d_near_a,
            d_near_b,
            d_far,
            d_near_size,
            d_next_near_size,
            d_far_size,
            d_bucket_id,
            d_ring_key,
            d_settled_queue,
            d_settled_size,
            d_settled_tag,
            outfile);

        CUDA_CHECK(cudaMemcpy(
            h_tent.data(),
            d_tent,
            N * sizeof(int),
            cudaMemcpyDeviceToHost));

        for (int v = 0; v < N; ++v)
            outfile << h_tent[v] << '\n';
    }

    // CLEANUP
    CUDA_CHECK(cudaFree(d_offsets));
    if (E > 0) {
        CUDA_CHECK(cudaFree(d_neighs));
        CUDA_CHECK(cudaFree(d_weights));
    }

    CUDA_CHECK(cudaFree(d_tent));
    CUDA_CHECK(cudaFree(d_state));

    CUDA_CHECK(cudaFree(d_near_a));
    CUDA_CHECK(cudaFree(d_near_b));
    CUDA_CHECK(cudaFree(d_far));

    CUDA_CHECK(cudaFree(d_near_size));
    CUDA_CHECK(cudaFree(d_next_near_size));
    CUDA_CHECK(cudaFree(d_far_size));

    CUDA_CHECK(cudaFree(d_bucket_id));
    CUDA_CHECK(cudaFree(d_ring_key));

    CUDA_CHECK(cudaFree(d_settled_queue));
    CUDA_CHECK(cudaFree(d_settled_size));
    CUDA_CHECK(cudaFree(d_settled_tag));

    outfile.close();
    return 0;
}
