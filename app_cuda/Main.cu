
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iostream>

#include "Kernels.cuh"
#include "Morton.hpp"
#include "Usm.hpp"

__global__ void ConvertMortonOnly(const Eigen::Vector3f* input, Code_t* output,
                                  const int size, const float min_coord,
                                  const float range) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < size) {
    const auto x = input[tid].x();
    const auto y = input[tid].y();
    const auto z = input[tid].z();

    output[tid] = PointToCode(x, y, z, min_coord, range);

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void empty() {}

void CudaWarmUp() {
  for (auto i = 0; i < 10; ++i) {
    empty<<<1, 1>>>();
  }
}

int main() {
  constexpr auto n = 1280 * 720;
  // constexpr auto n = 1024 * 1024;

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    std::cout << "Compute Capability (SM version): " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
  }

  redwood::UsmVector<Eigen::Vector3f> u_input(n);
  redwood::UsmVector<Code_t> u_output(n);

  CudaWarmUp();

  int blockSize = 512;
  int numBlocks = (n + blockSize - 1) / blockSize;

  GenerateRandomVector3f<<<numBlocks, blockSize>>>(u_input.data(), 114514, n);
  HANDLE_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  TimeTask("ConvertMortonOnly", [&] {
    ConvertMortonOnly<<<numBlocks, blockSize>>>(u_input.data(), u_output.data(),
                                                n, 0.0f, 1024.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  });

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  for (auto i = 0; i < 10; ++i) {
    std::cout << "Element " << i << ": " << u_input[i].transpose() << "\t"
              << u_output[i] << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
