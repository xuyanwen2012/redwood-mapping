#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define HANDLE_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))

inline void handleCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s in %s at line %d\n",
            cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
