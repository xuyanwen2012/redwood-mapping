#pragma once

#include <iostream>

#define HANDLE_ERROR(err) (HandleCudaError(err, __FILE__, __LINE__))

inline void HandleCudaError(const cudaError_t err, const char *file,
                            const int line) {
  if (err != cudaSuccess) {
    const auto _ = fprintf(stderr, "CUDA Error: %s in %s at line %d\n",
                           cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
__global__ void EmptyKernel() {}

void GpuWarmUp() {
  EmptyKernel<<<1, 1>>>();
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void PrintCudaDeviceInfo() {
  int device_count;
  HANDLE_ERROR(cudaGetDeviceCount(&device_count));

  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);

    std::cout << "Device " << device << ": " << device_prop.name << std::endl;
    std::cout << "Compute Capability (SM version): " << device_prop.major << "."
              << device_prop.minor << std::endl;
  }
}

template <typename T> T *AllocateManaged(const size_t num_elements) {
  T *ptr;
  HANDLE_ERROR(cudaMallocManaged(&ptr, num_elements * sizeof(T)));
  return ptr;
}

std::ostream &operator<<(std::ostream &os, const float3 &p) {
  os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
  return os;
}