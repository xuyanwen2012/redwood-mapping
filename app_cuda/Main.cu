
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iostream>

#include "Kernels.cuh"
#include "Usm.hpp"

int main() {
  constexpr auto n = 1024 * 1024;

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    std::cout << "Compute Capability (SM version): " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
  }

  redwood::UsmVector<Eigen::Vector3f> a(n);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  GenerateRandomVector3f<<<numBlocks, blockSize>>>(a.data(), 114514, n);
  HANDLE_ERROR(cudaDeviceSynchronize());

  for (auto i = 0; i < 10; ++i) {
    std::cout << "Element " << i << ": " << a[i].transpose() << std::endl;
  }

  return 0;
}
