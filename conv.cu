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

__global__ void fill(double *mat, int w, unsigned int seed) {
  int i = blockIdx.y * blockDim.y + threadIdx.y,
      j = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(seed, i * gridDim.x + j, 0, &state);

  // double f = curand_uniform_double(&state);
  // mat[i*w + j] = FLT_MIN + f * (FLT_MAX - FLT_MIN);
  double f = curand_uniform(&state);
  mat[i*w + j] = (int) (-5 + f * (10));
}

__global__ void conv(double *in, size_t in_w, size_t in_h,
                      double *kernel, size_t kernel_w, size_t kernel_h,
                      double *out, size_t out_w, size_t out_h) {
  int center_x = kernel_w / 2;
  int center_y = kernel_h / 2;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < out_w && y < out_h) {
    double sum = 0.0;
    for (int ky = 0; ky < kernel_h; ++ky) {
      for (int kx = 0; kx < kernel_w; ++kx) {
        int in_x = x + kx - center_x;
        int in_y = y + ky - center_y;
        if (in_x >= 0 && in_x < in_w && in_y >= 0 && in_y < in_h) {
          sum += in[in_y * in_w + in_x] * kernel[ky * kernel_w + kx];
        }
      }
    }
    out[y * out_w + x] = sum;
  }
}

void print_matrix(double *mat, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      printf("%lf ", mat[i * w + j]);
    }
    printf("\n");
  }
}

int main() {
  srand(time(NULL));
  int in_w = rand() % 20 + 1, in_h = rand() % 20 + 1;
  int kernel_w = rand() % 5 + 1, kernel_h = kernel_w;
  double *h_in = (double*)malloc(in_w * in_h * sizeof(double));
  double *h_kernel = (double*)malloc(kernel_w * kernel_h * sizeof(double));
  double *h_out = (double*)malloc(in_w * in_h * sizeof(double));
  double *d_in, *d_kernel, *d_out;

  cudaMalloc((void**)&d_in, in_w * in_h * sizeof(double));
  cudaMalloc((void**)&d_kernel, kernel_h * kernel_w * sizeof(double));
  cudaMalloc((void **)&d_out, in_w * in_h * sizeof(double));
  cudaMemset(d_out, 0, in_w * in_h * sizeof(double));

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size_in((in_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (in_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 grid_size_kernel((kernel_h + BLOCK_SIZE - 1) / BLOCK_SIZE, (kernel_w + BLOCK_SIZE - 1) / BLOCK_SIZE);

  unsigned int seed = time(NULL);

  fill<<<grid_size_in, block_size>>>(d_in, in_w, seed);
  fill<<<grid_size_kernel, block_size>>>(d_kernel, kernel_w, seed+1);
  conv<<<grid_size_in, block_size>>>(d_in, in_w, in_h, d_kernel, kernel_w, kernel_h, d_out, in_w, in_h);

  cudaMemcpy(h_in, d_in, in_w * in_h * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_kernel, d_kernel, kernel_h * kernel_w * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out, d_out, in_w * in_h * sizeof(double), cudaMemcpyDeviceToHost);

  printf("Matrix A:\n");
  print_matrix(h_in, in_w, in_h);
  printf("\nMatrix B:\n");
  print_matrix(h_kernel, kernel_w, kernel_h);
  printf("\nMatrix C (result):\n");
  print_matrix(h_out, in_w, in_h);

  free(h_in);
  free(h_kernel);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_kernel);
  cudaFree(d_out);
}