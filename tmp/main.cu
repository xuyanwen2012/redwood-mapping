#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>
#include <vector_types.h>

using Code_t = uint64_t;
constexpr int kCodeLen = 63;

__host__ __device__ __forceinline__ uint64_t
ExpandBits64(const uint32_t a) noexcept {
  uint64_t x = static_cast<uint64_t>(a) & 0x1fffff;
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

__host__ __device__ __forceinline__ uint64_t
Encode64(const uint32_t x, const uint32_t y, const uint32_t z) noexcept {
  return ExpandBits64(x) | (ExpandBits64(y) << 1) | (ExpandBits64(z) << 2);
}

__host__ __device__ __forceinline__ Code_t
PointToCode(const float x, const float y, const float z, const float min_coord,
            const float range) noexcept {
  constexpr uint32_t bit_scale = 0xFFFFFFFFu >> (32 - (kCodeLen / 3));
  const auto x_coord =
      static_cast<uint32_t>(bit_scale * ((x - min_coord) / range));
  const auto y_coord =
      static_cast<uint32_t>(bit_scale * ((y - min_coord) / range));
  const auto z_coord =
      static_cast<uint32_t>(bit_scale * ((z - min_coord) / range));
  return Encode64(x_coord, y_coord, z_coord);
}

struct Morton {
  constexpr Morton(const float min_coord, const float range)
      : min_coord(min_coord), range(range) {}

  __host__ __device__ __forceinline__ Code_t
  operator()(const float3 &point) const {
    return PointToCode(point.x, point.x, point.z, 0.0f, 1.0f);
  }

  float min_coord, range;
};

__global__ void convertMortonOnly(const float3 *input, Code_t *output,
                                  const int size, const float min_coord,
                                  const float range) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < size) {
    const auto x = input[tid].x;
    const auto y = input[tid].y;
    const auto z = input[tid].z;

    output[tid] = PointToCode(x, y, z, min_coord, range);

    tid += blockDim.x * gridDim.x;
  }
}

__global__ void warmup() {}

__global__ void convertMortonOnly_v2(const float3 *input, Code_t *output,
                                     const int size, Morton functor) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < size) {
    output[tid] = functor(input[tid]);
  }
}

int main() {
  thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<float> dis(0.0f, 1024.0f);

  constexpr auto min_coord = 0.0f;
  constexpr auto max_coord = 1024.0f;
  constexpr auto range = max_coord - min_coord;
  constexpr Morton morton_functor(min_coord, range);

  constexpr auto num_elements = 10'000'000;
  float3 *u_inputs;
  Code_t *u_outputs;
  cudaMallocManaged(&u_inputs, num_elements * sizeof(float3));
  cudaMallocManaged(&u_outputs, num_elements * sizeof(float3));

  std::generate_n(u_inputs, num_elements, [&] {
    const auto x = dis(gen);
    const auto y = dis(gen);
    const auto z = dis(gen);
    return make_float3(x, y, z);
  });

  // peek 10
  for (int i = 0; i < 10; ++i) {
    std::cout << u_inputs[i].x << ", " << u_inputs[i].y << ", " << u_inputs[i].z
              << std::endl;
  }

  warmup<<<1, 1>>>();
  cudaDeviceSynchronize();

  const auto start_time = std::chrono::high_resolution_clock::now();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  constexpr auto threadsPerBlock = 512;
  constexpr auto blocksPerGrid =
      (num_elements + threadsPerBlock - 1) / threadsPerBlock;
  convertMortonOnly_v2<<<blocksPerGrid, threadsPerBlock>>>(
      u_inputs, u_outputs, num_elements, morton_functor);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

  //   cudaDeviceSynchronize();

  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time);
  std::cout << "Execution time: " << duration_ms.count() << " ms" << std::endl;

  // peek 10 results
  for (int i = 0; i < 10; ++i) {
    std::cout << u_outputs[i] << std::endl;
  }

  cudaFree(u_inputs);
  return 0;
}
