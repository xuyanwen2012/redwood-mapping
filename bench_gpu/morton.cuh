#pragma once

#include <cuda_runtime.h>
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
    return PointToCode(point.x, point.x, point.z, min_coord, range);
  }

  float min_coord, range;
};

__global__ void convertMortonOnly_v2(const size_t size, const float3 *input,
                                     Code_t *output, Morton functor) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    output[tid] = functor(input[tid]);
  }
}

// __global__ void convertMortonOnly_v2(const float3 *input, Code_t *output,
//                                      const int size, Morton functor) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tid < size) {
//     output[tid] = functor(input[tid]);
//   }
// }
