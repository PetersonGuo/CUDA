// File: cuda_blur.cu
#include <cuda_runtime.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
// convolution.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16

// 3×3 box blur weights in row-major order
// You could also cudaMemcpyToSymbol a float d_kern[9] here if you prefer.
__constant__ float d_blur3x3[9] = {
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9,
    1.f/9, 1.f/9, 1.f/9
};

// in:  float image[h* w*3] in RGBRGB... order
// out: float image[h* w*3]
// w,h: image dimensions
__global__ void blur3x3_rgb(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // For each channel
    for (int c = 0; c < 3; ++c) {
        float sum = 0.f;
        // Apply 3×3 kernel centered at (x,y)
        for (int ky = -1; ky <= 1; ++ky) {
            int iy = y + ky;
            if (iy < 0 || iy >= height) continue;
            for (int kx = -1; kx <= 1; ++kx) {
                int ix = x + kx;
                if (ix < 0 || ix >= width) continue;
                // input index for channel c
                int in_idx = (iy * width + ix) * 3 + c;
                int kern_idx = (ky+1)*3 + (kx+1);
                sum += in[in_idx] * d_blur3x3[kern_idx];
            }
        }
        // write back
        out[(y * width + x) * 3 + c] = sum;
    }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s input.png output.png\n", argv[0]);
    return 1;
  }

  int w, h, c;
  unsigned char *img = stbi_load(argv[1], &w, &h, &c, 3);
  if (!img) { fprintf(stderr, "Error loading '%s'\n", argv[1]); return 1; }

  // --- host buffers (3 channels) ---
  float *h_in = (float*)malloc(w*h*3*sizeof(float));
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {              // <<-- FIXED bound here
      int idx = (y*w + x)*3;
      h_in[idx+0] = img[idx+0] / 255.0f;
      h_in[idx+1] = img[idx+1] / 255.0f;
      h_in[idx+2] = img[idx+2] / 255.0f;
    }
  }
  stbi_image_free(img);

  // --- device buffers (3 channels) ---
  float *d_in, *d_out;
  cudaMalloc(&d_in,  w*h*3*sizeof(float));
  cudaMalloc(&d_out, w*h*3*sizeof(float));
  cudaMemcpy(d_in, h_in, w*h*3*sizeof(float), cudaMemcpyHostToDevice);

  // --- launch blur ---
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((w+BLOCK_SIZE-1)/BLOCK_SIZE,
            (h+BLOCK_SIZE-1)/BLOCK_SIZE);
  blur3x3_rgb<<<grid, block>>>(d_in, d_out, w, h);
  cudaDeviceSynchronize();

  // --- copy back & clamp (3 channels) ---
  float *h_out = (float*)malloc(w*h*3*sizeof(float));
  cudaMemcpy(h_out, d_out, w*h*3*sizeof(float), cudaMemcpyDeviceToHost);

  unsigned char *out_img = (unsigned char*)malloc(w*h*3);
  for (int i = 0; i < w*h*3; ++i) {
    int v = (int)(h_out[i]*255.0f + 0.5f);
    if (v < 0)   v = 0;
    if (v > 255) v = 255;
    out_img[i] = (unsigned char)v;
  }

  // --- write RGB PNG ---
  stbi_write_png(argv[2], w, h, 3, out_img, w*3);

  // --- cleanup ---
  free(h_in); free(h_out); free(out_img);
  cudaFree(d_in); cudaFree(d_out);
  return 0;
}
