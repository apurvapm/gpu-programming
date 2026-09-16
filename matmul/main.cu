#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

#define TILE 16

__global__ void matmul(int p, int q, int r,  int *A, int *B,
	         int *C, int *D, int *E){
	unsigned tx = threadIdx.x;
	unsigned ty = threadIdx.y;

	//one thread computes 1 element in 1 block of E. 1 thread-block computes 1 block of E
	//funny that x runs in the x direction so its the col
	//and y runs in the y-direction so its the row
	//first get the block, then the 
	unsigned row = blockIdx.y*TILE + ty;
	unsigned col = blockIdx.x*TILE + tx;
	//shared memory for the 4 matrix blocks
	__shared__ int As[TILE][TILE];
	__shared__ int Bs[TILE][TILE];
	__shared__ int Cs[TILE][TILE];
	__shared__ int Ds[TILE][TILE];
	int sum =0;

	for(int tile = 0; tile<q; tile+=TILE){
		int A_row = tile + ty;
		int A_col = blockIdx.y*TILE + tx;
		if(A_row < q && A_col < p){
			As[ty][tx] = A[A_row*p + A_col];
		}else As[ty][tx] = 0;
		
		int B_row = tile + ty;
		int B_col = blockIdx.x*TILE + tx;
		if(B_row < q && B_col < r)Bs[ty][tx] = B[B_row*r + B_col];
		else Bs[ty][tx] = 0;

		__syncthreads();
		for(int k =0; k<TILE; k++){
			sum += As[k][ty]*Bs[k][tx];
		}

		__syncthreads();
	}

	for(int tile = 0; tile <q ; tile+= TILE){
		int C_row = blockIdx.y*TILE+ ty;
		int D_row= blockIdx.x*TILE +ty;
		int C_col= tile+ tx;
		int D_col = tile +tx;

		if(C_col < q && C_row < p)Cs[ty][tx] = C[C_row*q+C_col];
		else Cs[ty][tx] = 0;
		if(D_row<r && D_col<q)Ds[tx][ty] = D[D_row*q+D_col];
		else Ds[tx][ty]=0;

		__syncthreads();
		for(int k=0; k<TILE; k++){
			sum+= Cs[ty][k]*Ds[k][tx];
		}
		__syncthreads();
	}
	if(row<p && col<r){
		E[row*r+col] =sum;
	}

}

void compute(int p, int q, int r, int *h_matrixA, int *h_matrixB,
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;

	cudaMalloc(&d_matrixA, q * p * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	cudaMemcpy(d_matrixA, h_matrixA, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	dim3 myblock(TILE, TILE);
	dim3 mygrid((r+TILE-1)/TILE, (p+TILE-1)/TILE);
	matmul<<<mygrid, myblock>>>(p, q, r, d_matrixA, d_matrixB, d_matrixC, d_matrixD, d_matrixE);


	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}



int main(int argc, char **argv) {
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n");
		return 0;
	}

	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	matrixA = (int*) malloc(q * p * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, q, p);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	matrixE = (int*) malloc(p * r * sizeof(int));

	gettimeofday(&t1, NULL);
	compute(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	fclose(inputFilePtr);
	fclose(outputFilePtr);

	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
