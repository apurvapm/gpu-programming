/* ==========================================================================
 *  CS6023: GPU Programming  --  Assignment 1
 *  "The Wizard's Lens": An Image Preprocessing Pipeline on the GPU
 *
 *  FILE: pipeline.cu
 *
 *  --------------------------------------------------------------------
 *   READ THIS FIRST
 *  --------------------------------------------------------------------
 *   1. DO NOT MODIFY any code outside the regions marked
 *          // ==== TODO n : ... ====
 *          ... your code here ...
 *          // ==== END TODO n ====
 *      The input parser, the output printer and the kernel signatures are
 *      checked automatically during grading.
 *
 *   2. DO NOT print anything extra to stdout. Whatever is on stdout is your
 *      answer. Use fprintf(stderr, ...) for debugging.
 *
 *   3. Everything must stay inside this single .cu file.
 *
 *   4. Rename this file to <YourRollNumber>.cu before submitting.
 	Example: CS24S009.cu
 *
 *  --------------------------------------------------------------------
 *   WHAT YOU HAVE TO WRITE  (9 TODOs)
 *  --------------------------------------------------------------------
 *      TODO 1 : grayscaleKernel   body
 *      TODO 2 : resizeKernel      body
 *      TODO 3 : cropKernel        body
 *      TODO 4 : normalizeKernel   body
 *      TODO 5 : allocate device memory
 *      TODO 6 : copy host -> device
 *      TODO 7 : configure and launch the four kernels
 *      TODO 8 : copy device -> host
 *      TODO 9 : free device memory
 *
 *   See README.md for how to build and test.
 * ========================================================================== */

/* ---------------- DO NOT MODIFY : INCLUDES AND MACROS ---------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Integer ceiling division: how many blocks of size b cover n items. */
#define CEIL_DIV(n, b) (((n) + (b) - (int)1) / (b))

/* Luma weights, ITU-R BT.601. Use exactly these constants. */
#define W_RED   0.299f
#define W_GREEN 0.587f
#define W_BLUE  0.114f
/* -------------------- END DO NOT MODIFY (macros) --------------------- */


/* ==========================================================================
 *  STAGE 1  --  GRAYSCALE
 *  --------------------------------------------------------------------
 *  d_rgb  : H*W*3 bytes, row-major, channel-interleaved (R,G,B,R,G,B,...)
 *           the pixel at (y, x) starts at index 3*(y*W + x)
 *  d_gray : H*W floats, row-major.  d_gray[y*W + x] is the pixel at (y, x)
 *
 *      gray(y,x) = 0.299*R(y,x) + 0.587*G(y,x) + 0.114*B(y,x)
 *
 *  The result stays on the 0 ... 255 scale (as a float, not rounded).
 * ========================================================================== */
__global__ void grayscaleKernel(const unsigned char *d_rgb,
                                float *d_gray,
                                int H, int W)
{
    // ==== TODO 1 : write the grayscale kernel ====
    //
    // Hints:
    //   * Cast the unsigned char values to float before doing arithmetic.
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id >= H*W){ return;
}
    
    //it has to write the grayscale value for cell (y, x)
    // the corresponding values are at

    float red_comp = 0.299f* (float)d_rgb[3*id];
    float green_comp = 0.587f* (float)d_rgb[3*id+1];
    float blue_comp = 0.114f* (float)d_rgb[3*id+2];
    d_gray[id] = red_comp+ green_comp+ blue_comp;
    // ==== END TODO 1 ====
}


/* ==========================================================================
 *  STAGE 2  --  BILINEAR RESIZE   (H x W)  ->  (Hr x Wr)
 *  --------------------------------------------------------------------
 *  Use the "align corners" mapping: the four corner pixels of the input
 *  land exactly on the four corner pixels of the output.
 *
 *      scaleY = (Hr > 1) ? (float)(H - 1) / (float)(Hr - 1) : 0.0f
 *      scaleX = (Wr > 1) ? (float)(W - 1) / (float)(Wr - 1) : 0.0f
 *
 *  For an output pixel (oy, ox):
 *
 *      fy = oy * scaleY          fx = ox * scaleX
 *      y0 = floor(fy)            x0 = floor(fx)
 *      y1 = min(y0 + 1, H - 1)   x1 = min(x0 + 1, W - 1)
 *      wy = fy - y0              wx = fx - x0
 *
 *      top    = in(y0,x0)*(1-wx) + in(y0,x1)*wx
 *      bottom = in(y1,x0)*(1-wx) + in(y1,x1)*wx
 *      out(oy,ox) = top*(1-wy) + bottom*wy
 * ========================================================================== */
__global__ void resizeKernel(const float *d_gray,
                             float *d_resized,
                             int H, int W,
                             int Hr, int Wr)
{
    // ==== TODO 2 : write the bilinear resize kernel ====
    //
    // Hints:
    //   * floorf() and min() are available inside device code.
    float sy, sx;
    if(Hr==1)sy=0;
    else {
        sy = (float)(H-1)/(Hr-1);
    }
    if(Wr ==1)sx = 0;
    else{
        sx = (float)(W-1)/(Wr-1);
    }
int id = blockIdx.x*blockDim.x + threadIdx.x;
if(id >= Hr*Wr) return;
    int ox = id % Wr;
    int oy = id / Wr;

    float fy = oy*sy;
    float fx = ox*sx;
    int y0 = floorf(fy);
    int x0 = floorf(fx);
    int y1 = min(y0 +1, H-1);
    int x1 = min(x0+1, W-1);
    float wy = fy - y0;
    float wx = fx - x0;

    float top = d_gray[W*y0 + x0]*(1-wx) + d_gray[y0*W+x1]*wx;
    float bottom = d_gray[W*y1 + x0]*(1-wx) + d_gray[y1*W+x1]*wx;
    d_resized[Wr*oy + ox] = top*(1-wy) + bottom*wy;
    // ==== END TODO 2 ====
}


/* ==========================================================================
 *  STAGE 3  --  CENTER CROP   (Hr x Wr)  ->  (Hc x Wc)
 *  --------------------------------------------------------------------
 *      offsetY = (Hr - Hc) / 2        <-- integer division, rounds down
 *      offsetX = (Wr - Wc) / 2
 *
 *      out(y, x) = in(y + offsetY, x + offsetX)
 *
 *  It is guaranteed that Hc <= Hr and Wc <= Wr.
 * ========================================================================== */
__global__ void cropKernel(const float *d_resized,
                           float *d_cropped,
                           int Hr, int Wr,
                           int Hc, int Wc)
{
    // ==== TODO 3 : write the center-crop kernel ====
    //
    // Hints:
    //   * Pure index arithmetic - no interpolation.
    //   * The input row stride is Wr, the output row stride is Wc.
int id = blockIdx.x*blockDim.x + threadIdx.x;
if(id >=Hc*Wc) return;
    int y = id/Wc; int x =id%Wc;
    int offy = (Hr - Hc)/2;

    int offx = (Wr - Wc)/2;

    d_cropped[y*Wc + x] = d_resized[Wr*(y+offy) + x+ offx];

    // ==== END TODO 3 ====
}


/* ==========================================================================
 *  STAGE 4  --  NORMALIZE
 *  --------------------------------------------------------------------
 *      out(y,x) = ( in(y,x) / 255.0 - mean ) / std
 *
 *  This is the last step: it turns 0..255 brightness values into the small
 *  zero-centred numbers a neural network expects.
 * ========================================================================== */
__global__ void normalizeKernel(const float *d_cropped,
                                float *d_out,
                                int Hc, int Wc,
                                float mean, float stdv)
{
    // ==== TODO 4 : write the normalize kernel ====
    //
    // Hints:
    //   * Element-wise, so a 1-D grid over Hc*Wc elements works.
    int id = blockIdx.x*blockDim.x + threadIdx.x;

if(id >= Wc*Hc) return;
    d_out[id] = (d_cropped[id]/255 - mean)/stdv;

    // ==== END TODO 4 ====
}


/* ==========================================================================
 *                                  MAIN
 * ========================================================================== */
int main(int argc, char **argv)
{
    /* ================= DO NOT MODIFY : INPUT PARSING ================== */
    FILE *fin = stdin;
    if (argc >= 2) {
        fin = fopen(argv[1], "r");
        if (!fin) {
            fprintf(stderr, "ERROR: cannot open input file '%s'\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    int H, W, Hr, Wr, Hc, Wc;
    float mean, stdv;

    if (fscanf(fin, "%d %d", &H, &W) != 2) {
        fprintf(stderr, "ERROR: malformed input (image dimensions)\n");
        return EXIT_FAILURE;
    }
    if (fscanf(fin, "%d %d", &Hr, &Wr) != 2) {
        fprintf(stderr, "ERROR: malformed input (resize dimensions)\n");
        return EXIT_FAILURE;
    }
    if (fscanf(fin, "%d %d", &Hc, &Wc) != 2) {
        fprintf(stderr, "ERROR: malformed input (crop dimensions)\n");
        return EXIT_FAILURE;
    }
    if (fscanf(fin, "%f %f", &mean, &stdv) != 2) {
        fprintf(stderr, "ERROR: malformed input (mean / std)\n");
        return EXIT_FAILURE;
    }

    if (H <= 0 || W <= 0 || Hr <= 0 || Wr <= 0 || Hc <= 0 || Wc <= 0) {
        fprintf(stderr, "ERROR: dimensions must be positive\n");
        return EXIT_FAILURE;
    }
    if (Hc > Hr || Wc > Wr) {
        fprintf(stderr, "ERROR: crop size must not exceed resize size\n");
        return EXIT_FAILURE;
    }

    const size_t nPixIn   = (size_t)H  * (size_t)W;
    const size_t nRgb     = nPixIn * 3u;
    const size_t nPixRes  = (size_t)Hr * (size_t)Wr;
    const size_t nPixCrop = (size_t)Hc * (size_t)Wc;

    unsigned char *h_rgb = (unsigned char *)malloc(nRgb * sizeof(unsigned char));
    float         *h_out = (float *)calloc(nPixCrop, sizeof(float));  
    if (!h_rgb || !h_out) {
        fprintf(stderr, "ERROR: host allocation failed\n");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < nRgb; ++i) {
        int v;
        if (fscanf(fin, "%d", &v) != 1) {
            fprintf(stderr, "ERROR: expected %zu pixel values, got %zu\n", nRgb, i);
            return EXIT_FAILURE;
        }
        h_rgb[i] = (unsigned char)v;
    }
    if (fin != stdin) fclose(fin);
    /* =============== END DO NOT MODIFY : INPUT PARSING ================ */


    /* ---- Device pointers. Allocate them in TODO 5, free them in TODO 9.
     *      Until you complete TODO 5 and TODO 6, the compiler will warn that
     *      these are unused. That is expected on a fresh checkout.          */
    unsigned char *d_rgb     = NULL;   /* H  x W  x 3  bytes   */
    float         *d_gray    = NULL;   /* H  x W       floats  */
    float         *d_resized = NULL;   /* Hr x Wr      floats  */
    float         *d_cropped = NULL;   /* Hc x Wc      floats  */
    float         *d_out     = NULL;   /* Hc x Wc      floats  */

    // ==== TODO 5 : allocate device memory ====
    //
    // Allocate all five buffers above with cudaMalloc.
    cudaMalloc(&d_rgb, H*W*3*sizeof(unsigned char));
    cudaMalloc(&d_gray, H*W*sizeof(float));
    cudaMalloc(&d_resized, Hr*Wr*sizeof(float));
    cudaMalloc(&d_cropped, Hc*Wc*sizeof(float));
    cudaMalloc(&d_out, Hc*Wc*sizeof(float));
    
    // ==== END TODO 5 ====


    // ==== TODO 6 : copy the input image from host to device ====
    //
    // Copy h_rgb (nRgb bytes) into d_rgb
    cudaMemcpy(d_rgb, h_rgb, nRgb * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // ==== END TODO 6 ====


    // ==== TODO 7 : configure and launch the four kernels, in order ====
    //
    // Launch order:  grayscale -> resize -> crop -> normalize
    // Data flow   :  d_rgb -> d_gray -> d_resized -> d_cropped -> d_out

	int blocksize = 1024;
    int nblocks1 = CEIL_DIV(W*H , blocksize);
    grayscaleKernel<<<nblocks1, blocksize>>>(d_rgb, d_gray, H, W);
     
    	
	int n_blocksr = CEIL_DIV(Wr*Hr, blocksize);
    resizeKernel<<<n_blocksr, blocksize>>>(d_gray, d_resized, H, W, Hr, Wr);
	
	int n_blocksc = CEIL_DIV(Wc*Hc, blocksize);
    cropKernel<<<n_blocksc, blocksize>>>(d_resized, d_cropped, Hr, Wr, Hc, Wc);

    normalizeKernel<<<n_blocksc, blocksize>>>(d_cropped, d_out,Hc, Wc, mean, stdv);

    // ==== END TODO 7 ====


    // ==== TODO 8 : copy the final result from device to host ====
    //
    // Copy nPixCrop floats from d_out into h_out.
    cudaMemcpy(h_out, d_out, nPixCrop*sizeof(float), cudaMemcpyDeviceToHost);

    // ==== END TODO 8 ====


    /* ================= DO NOT MODIFY : OUTPUT PRINTING ================ */
    printf("%d %d\n", Hc, Wc);
    for (int y = 0; y < Hc; ++y) {
        for (int x = 0; x < Wc; ++x) {
            printf("%.6f%c", h_out[(size_t)y * (size_t)Wc + (size_t)x],
                   (x == Wc - 1) ? '\n' : ' ');
        }
    }
    fflush(stdout);
    /* =============== END DO NOT MODIFY : OUTPUT PRINTING ============== */


    // ==== TODO 9 : free all device memory ====
    //
    // One cudaFree per cudaMalloc.
	cudaFree(d_gray);
cudaFree(d_rgb);
cudaFree(d_resized);
cudaFree(d_cropped);
cudaFree(d_out);

    // ==== END TODO 9 ====

    /* ---------------- DO NOT MODIFY : host cleanup ---------------- */
    free(h_rgb);
    free(h_out);
    return EXIT_SUCCESS;
}
