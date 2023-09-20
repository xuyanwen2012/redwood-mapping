#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>

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
  Morton(const float min_coord, const float range)
      : min_coord(min_coord), range(range) {}

  __host__ __device__ __forceinline__ Code_t
  operator()(const float3& point) const {
    return PointToCode(point.x, point.x, point.z, 0.0f, 1.0f);
  }

  float min_coord, range;
};

struct CustomLess {
  template <typename DataType>
  __device__ bool operator()(const DataType& lhs, const DataType& rhs) {
    return lhs < rhs;
  }
};

__host__ __device__ __forceinline__ float3& operator++(float3& val) {
  ++val.x;
  ++val.y;
  ++val.z;
  return val;
}

int main() {
  constexpr int num_elements = 1024;
  float3* u_input;
  Code_t* u_morton_keys;
  Code_t* u_sorted_morton_keys;

  cudaMallocManaged(&u_input, num_elements * sizeof(float3));
  cudaMallocManaged(&u_morton_keys, num_elements * sizeof(Code_t));
  cudaMallocManaged(&u_sorted_morton_keys, num_elements * sizeof(Code_t));

  std::iota(u_input, u_input + num_elements, float3());
  std::reverse(u_input, u_input + num_elements);

  const Morton conversion_op(0.0f, num_elements);
  const cub::TransformInputIterator<Code_t, Morton, float3*> itr(u_input,
                                                                 conversion_op);
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, itr,
                                     u_sorted_morton_keys, num_elements,
                                     CustomLess());

  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, itr,
                                     u_sorted_morton_keys, num_elements,
                                     CustomLess());

  cudaDeviceSynchronize();

  for (int i = 0; i < num_elements; i++) {
    std::cout << i << ":\t" << itr[i] << "\n";
  }

  cudaFree(u_input);
  cudaFree(u_morton_keys);
  cudaFree(u_sorted_morton_keys);
  cudaFree(d_temp_storage);

  return 0;
}
