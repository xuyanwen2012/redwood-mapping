#include <stdio.h>

// CUDA kernel function
__global__ void helloCUDA() { printf("Hello, CUDA World!\n"); }

int main() {
  // Launch the CUDA kernel with one block containing one thread
  helloCUDA<<<1, 1>>>();

  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  // Check for any CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    return 1;
  }

  return 0;
}
