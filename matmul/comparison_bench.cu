#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<chrono>
#include<vector>
#include<algorithm>
#include<omp.h>
#include<cuda.h>
using namespace std;

#ifndef TILE
#define TILE 16
#endif

__global__ void fusedTiled(int p, int q, int r, int *A, int *B,
                           int *C, int *D, int *E){
    int tx = threadIdx.x, ty = threadIdx.y;
    int rowBase = blockIdx.y*TILE, colBase = blockIdx.x*TILE;
    int row = rowBase + ty, col = colBase + tx;

    __shared__ int As[TILE][TILE];
    __shared__ int Bs[TILE][TILE];
    __shared__ int Cs[TILE][TILE];
    __shared__ int Ds[TILE][TILE];
    int sum = 0;

    for(int t = 0; t < q; t += TILE){
        int ar = t + ty, ac = rowBase + tx;          // As[k][ty] == A[t+k][row]
        As[ty][tx] = (ar < q && ac < p) ? A[ar*p + ac] : 0;
        int bc = colBase + tx;                       // Bs[k][tx] == B[t+k][col]
        Bs[ty][tx] = (ar < q && bc < r) ? B[ar*r + bc] : 0;
        int cc = t + tx;                             // Cs[ty][k] == C[row][t+k]
        Cs[ty][tx] = (row < p && cc < q) ? C[row*q + cc] : 0;
        int dr = colBase + ty;                       // Ds[k][tx] == D[col][t+k]
        Ds[tx][ty] = (dr < r && cc < q) ? D[dr*q + cc] : 0;
        __syncthreads();

        for(int k = 0; k < TILE; k++)
            sum += As[k][ty]*Bs[k][tx] + Cs[ty][k]*Ds[k][tx];
        __syncthreads();
    }
    if(row < p && col < r) E[row*r + col] = sum;
}

__global__ void origTiled(int p, int q, int r, int *A, int *B,
                          int *C, int *D, int *E){
    unsigned tx = threadIdx.x;
    unsigned ty = threadIdx.y;
    unsigned row = blockIdx.y*TILE + ty;
    unsigned col = blockIdx.x*TILE + tx;
    __shared__ int As[TILE][TILE];
    __shared__ int Bs[TILE][TILE];
    __shared__ int Cs[TILE][TILE];
    __shared__ int Ds[TILE][TILE];
    int sum = 0;

    for(int tile = 0; tile < q; tile += TILE){
        int ar = tile + ty;
        int ac = blockIdx.y*TILE + tx;
        if(ar < q && ac < p) As[ty][tx] = A[ar*p + ac];
        else As[ty][tx] = 0;

        int B_row = tile + ty;
        int bc = blockIdx.x*TILE + tx;
        if(B_row < q && bc < r) Bs[ty][tx] = B[B_row*r + bc];
        else Bs[ty][tx] = 0;

        __syncthreads();
        for(int k = 0; k < TILE; k++) sum += As[k][ty]*Bs[k][tx];
        __syncthreads();
    }

    for(int tile = 0; tile < q; tile += TILE){
        int cr = blockIdx.y*TILE + ty;
        int dr = blockIdx.x*TILE + ty;
        int cc = tile + tx;
        int dc = tile + tx;
        if(cc < q && cr < p) Cs[ty][tx] = C[cr*q + cc];
        else Cs[ty][tx] = 0;
        if(dr < r && dc < q) Ds[tx][ty] = D[dr*q + dc];
        else Ds[tx][ty] = 0;

        __syncthreads();
        for(int k = 0; k < TILE; k++) sum += Cs[ty][k]*Ds[k][tx];
        __syncthreads();
    }
    if(row < p && col < r) E[row*r + col] = sum;
}

__global__ void fusedNaive(int p, int q, int r, int *A, int *B,
                           int *C, int *D, int *E){
    int row = blockIdx.y*TILE + threadIdx.y;
    int col = blockIdx.x*TILE + threadIdx.x;
    if(row >= p || col >= r) return;
    int sum = 0;
    for(int k = 0; k < q; k++)
        sum += A[k*p + row]*B[k*r + col] + C[row*q + k]*D[col*q + k];
    E[row*r + col] = sum;
}

void cpuFused(int p, int q, int r, const int *A, const int *B,
              const int *C, const int *D, int *E, bool parallel){
    vector<int> Dt((size_t)q*r);
    for(int j = 0; j < r; j++)
        for(int k = 0; k < q; k++) Dt[(size_t)k*r + j] = D[(size_t)j*q + k];
    memset(E, 0, (size_t)p*r*sizeof(int));

    #pragma omp parallel for schedule(static) if(parallel)
    for(int i = 0; i < p; i++){
        int *Ei = E + (size_t)i*r;
        for(int k = 0; k < q; k++){
            int a = A[(size_t)k*p + i], c = C[(size_t)i*q + k];
            const int *Bk = B + (size_t)k*r, *Dk = Dt.data() + (size_t)k*r;
            for(int j = 0; j < r; j++) Ei[j] += a*Bk[j] + c*Dk[j];
        }
    }
}

using Clock = chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){
    return chrono::duration<double, milli>(b - a).count();
}
static double median(vector<double> v){
    sort(v.begin(), v.end()); return v[v.size()/2];
}

struct GpuTimes { double h2d, kernel, d2h; };

enum Variant { NAIVE = 0, TILED = 1, TILED4 = 2 };

GpuTimes runGpu(int variant, int p, int q, int r, int *A, int *B,
                int *C, int *D, int *E){
    int *dA, *dB, *dC, *dD, *dE;
    cudaMalloc((void**)&dA, (size_t)q*p*sizeof(int));
    cudaMalloc((void**)&dB, (size_t)q*r*sizeof(int));
    cudaMalloc((void**)&dC, (size_t)p*q*sizeof(int));
    cudaMalloc((void**)&dD, (size_t)r*q*sizeof(int));
    cudaMalloc((void**)&dE, (size_t)p*r*sizeof(int));

    auto t0 = Clock::now();
    cudaMemcpy(dA, A, (size_t)q*p*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, (size_t)q*r*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, (size_t)p*q*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dD, D, (size_t)r*q*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t1 = Clock::now();

    dim3 block(TILE, TILE);
    dim3 grid((r + TILE - 1)/TILE, (p + TILE - 1)/TILE);
    if(variant == TILED) fusedTiled<<<grid, block>>>(p, q, r, dA, dB, dC, dD, dE);
    else if(variant == TILED4) origTiled <<<grid, block>>>(p, q, r, dA, dB, dC, dD, dE);
    else fusedNaive<<<grid, block>>>(p, q, r, dA, dB, dC, dD, dE);
    cudaDeviceSynchronize();          
    auto t2 = Clock::now();

    cudaMemcpy(E, dE, (size_t)p*r*sizeof(int), cudaMemcpyDeviceToHost);
    auto t3 = Clock::now();

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dE);
    return { ms(t0, t1), ms(t1, t2), ms(t2, t3) };
}

int main(int argc, char **argv){
    if(argc < 4){ printf("usage: %s p q r [--no-cpu]\n", argv[0]); return 1; }
    int p = atoi(argv[1]), q = atoi(argv[2]), r = atoi(argv[3]);
    bool timeCpu = !(argc > 4 && strcmp(argv[4], "--no-cpu") == 0);
    const int RUNS = 5;

    srand(42);
    auto gen = [](size_t n){ vector<int> v(n);
        for(auto &x : v) x = rand()%21 - 10; return v; };
    vector<int> A = gen((size_t)q*p), B = gen((size_t)q*r),
                C = gen((size_t)p*q), D = gen((size_t)r*q);
    vector<int> Eref((size_t)p*r), E((size_t)p*r);

    // warm-up
    { int *w; cudaMalloc((void**)&w, sizeof(int)); cudaFree(w); }
    {
        const int W = 64;
        vector<int> w1((size_t)W*W, 1), w2((size_t)W*W);
        for(int v = 0; v < 3; v++)
            runGpu(v, W, W, W, w1.data(), w1.data(), w1.data(), w1.data(), w2.data());
    }

    // CPU
    vector<double> cpu1;
    if(!timeCpu)
        cpuFused(p, q, r, A.data(), B.data(), C.data(), D.data(), Eref.data(), true);
    for(int i = 0; timeCpu && i < RUNS; i++){
        auto a = Clock::now();
        cpuFused(p, q, r, A.data(), B.data(), C.data(), D.data(), Eref.data(), false);
        cpu1.push_back(ms(a, Clock::now()));
    }

    // GPU
    const char *names[3] = { "naive", "tiled", "tiled4" };
    for(int v = 0; v < 3; v++){
        vector<double> h2d, ker, d2h;
        for(int i = 0; i < RUNS; i++){
            GpuTimes g = runGpu(v, p, q, r, A.data(), B.data(),
                                C.data(), D.data(), E.data());
            if(memcmp(E.data(), Eref.data(), E.size()*sizeof(int)) != 0){
                printf("MISMATCH in %s kernel\n", names[v]);
                return 1;
            }
            h2d.push_back(g.h2d); ker.push_back(g.kernel); d2h.push_back(g.d2h);
        }
        double k = median(ker), e2e = median(h2d) + k + median(d2h);
        printf("%-7s TILE=%2d  kernel %9.2f ms  end-to-end %9.2f ms  "
               "(transfer %.0f%%)  %.1f M elem-ops/s\n",
               names[v], TILE, k, e2e,
               100.0*(e2e - k)/e2e, 2.0*p*q*r/(k*1e3));
    }
    if(timeCpu)
        printf("cpu-1t %9.2f ms\n", median(cpu1));
    return 0;
}
