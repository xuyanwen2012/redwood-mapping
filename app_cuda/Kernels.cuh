#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <Eigen/Dense>

#include "CudaUtils.cuh"

__global__ void GenerateRandomVector3f(Eigen::Vector3f* output,
                                       const unsigned int seed, const int n) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandState state;
  curand_init(seed, idx, 0, &state);

  if (idx < n) {
    const float randomX = curand_uniform(&state) * 1024.0f;
    const float randomY = curand_uniform(&state) * 1024.0f;
    const float randomZ = curand_uniform(&state) * 1024.0f;

    output[idx] = Eigen::Vector3f(randomX, randomY, randomZ);
  }
}
