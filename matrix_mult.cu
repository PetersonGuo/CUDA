#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <string.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void fill(double *A, int M, unsigned int seed) {
  int i = blockIdx.y * blockDim.y + threadIdx.y,
      j = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(seed, i * gridDim.x + j, 0, &state);

  double f = curand_uniform_double(&state);
  A[i*M + j] = FLT_MIN + f * (FLT_MAX - FLT_MIN);
  // double f = curand_uniform(&state);
  // A[i*M + j] = (int) (-5 + f * (10));
}

__global__ void multiply(double *A, double *B, double *C, size_t N, size_t M,
                         size_t O) {
  __shared__ double Asub[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Bsub[BLOCK_SIZE][BLOCK_SIZE];
  int i = blockIdx.y*blockDim.y + threadIdx.y, j = blockIdx.x*blockDim.x + threadIdx.x;
  for (int tileIdx = 0; tileIdx < (M + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
    int A_row = i, A_col = tileIdx * BLOCK_SIZE + threadIdx.x;
    int B_row = tileIdx * BLOCK_SIZE + threadIdx.y, B_col = j;
    Asub[threadIdx.y][threadIdx.x] = (A_row < N && A_col < M) ? A[A_row * M + A_col] : 0.0;
    Bsub[threadIdx.y][threadIdx.x] = (B_row < M && B_col < O) ? B[B_row * O + B_col] : 0.0;
    C[i*O + j] = 0;
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        C[i*O + j] += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];
    }
    __syncthreads();
  }
}

int main() {
  srand(time(NULL));
  int N = rand() % 20 + 1, M = rand() % 20 + 1, O = rand() % 20 + 1;
  double *h_A = (double*)malloc(N * M * sizeof(double));
  double *h_B = (double*)malloc(M * O * sizeof(double));
  double *h_C = (double*)malloc(N * O * sizeof(double));
  double *d_A, *d_B, *d_C;

  cudaMalloc((void**)&d_A, N * M * sizeof(double));
  cudaMalloc((void**)&d_B, M * O * sizeof(double));
  cudaMalloc((void **)&d_C, N * O * sizeof(double));
  cudaMemset(d_C, 0, N * O * sizeof(double));

  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSizeA((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 gridSizeB((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (O + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 gridSizeC((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (O + BLOCK_SIZE - 1) / BLOCK_SIZE);

  unsigned int seed = time(NULL);

  fill<<<gridSizeA, blockSize>>>(d_A, M, time(NULL));
  fill<<<gridSizeB, blockSize>>>(d_B, O, time(NULL));
  multiply<<<gridSizeC, blockSize>>>(d_A, d_B, d_C, N, M, O);

  cudaMemcpy(h_A, d_A, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, M * O * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, N * O * sizeof(double), cudaMemcpyDeviceToHost);

  printf("Matrix A:\n");
  for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
          printf("%lf ", h_A[i * M + j]);
      }
      printf("\n");
  }
  printf("\nMatrix B:\n");
  for (int i = 0; i < M; ++i) {
      for (int j = 0; j < O; ++j) {
          printf("%lf ", h_B[i * O + j]);
      }
      printf("\n");
  }
  printf("\nMatrix C (result):\n");
  for (int i = 0; i < N; ++i) {
      for (int j = 0; j < O; ++j) {
          printf("%lf ", h_C[i * O + j]);
      }
      printf("\n");
  }

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}