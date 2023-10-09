#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>
#include <random>

#include "sync.hpp"

namespace bm = benchmark;

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

struct CustomLess {
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs;
  }
};

std::ostream &operator<<(std::ostream &os, const float3 &v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}

__global__ void generateRandomFloat3(float3 *output, int size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  curandState state;
  curand_init(0, tid, 0, &state); // Initialize the random number generator

  while (tid < size) {
    float rand_x = curand_uniform(&state) * 1024.0f;
    float rand_y = curand_uniform(&state) * 1024.0f;
    float rand_z = curand_uniform(&state) * 1024.0f;

    output[tid] = make_float3(rand_x, rand_y, rand_z);

    tid += blockDim.x * gridDim.x;
  }
}

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

static void BM_rand_float3_gen_cpu(bm::State &state) {
  thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<float> dis(0.0f, 1024.0f);

  const auto num_elements = state.range(0);
  std::vector<float3> u_inputs(num_elements);

  for (auto _ : state) {
    std::generate(u_inputs.begin(), u_inputs.end(), [&] {
      const auto x = dis(gen);
      const auto y = dis(gen);
      const auto z = dis(gen);
      return make_float3(x, y, z);
    });
    bm::DoNotOptimize(u_inputs.size());
  }
}

static void BM_rand_float3_gen(bm::State &state) {
  const auto num_elements = state.range(0);
  float3 *u_input;
  BENCH_CUDA_TRY(cudaMallocManaged(&u_input, num_elements * sizeof(float3)));

  for (auto _ : state) {
    cuda_event_timer raii{state};
    const dim3 blockSize(1024);
    const dim3 gridSize((num_elements + blockSize.x - 1) / blockSize.x);
    generateRandomFloat3<<<gridSize, blockSize>>>(u_input, num_elements);
  }

  BENCH_CUDA_TRY(cudaFree(u_input));
}

static void BM_compute_morton_only(bm::State &state) {
  const auto num_elements = state.range(0);
  const auto num_threads = state.range(1);
  float3 *u_input;
  Code_t *u_output;
  cudaMallocManaged(&u_input, num_elements * sizeof(float3));
  cudaMallocManaged(&u_output, num_elements * sizeof(Code_t));

  for (auto _ : state) {
    cuda_event_timer raii{state, true};
    const dim3 blockSize(num_threads);
    const dim3 gridSize((num_elements + blockSize.x - 1) / blockSize.x);
    convertMortonOnly<<<gridSize, blockSize>>>(u_input, u_output, num_elements,
                                               0.0f, 1024.0f);
  }

  BENCH_CUDA_TRY(cudaFree(u_input));
  BENCH_CUDA_TRY(cudaFree(u_output));
}

static void BM_radixsort_morton_only(bm::State &state) {
  const auto num_elements = state.range(0);
  Code_t *u_mortons;
  Code_t *u_sorted_mortons;
  BENCH_CUDA_TRY(cudaMallocManaged(&u_mortons, num_elements * sizeof(Code_t)));
  BENCH_CUDA_TRY(
      cudaMallocManaged(&u_sorted_mortons, num_elements * sizeof(Code_t)));

  cub::CachingDeviceAllocator g_allocator(true);

  for (auto _ : state) {
    cuda_event_timer raii{state};

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    BENCH_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_mortons, u_sorted_mortons,
        num_elements));

    BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    BENCH_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, u_mortons, u_sorted_mortons,
        num_elements));

    bm::DoNotOptimize(u_sorted_mortons);
  }

  BENCH_CUDA_TRY(cudaFree(u_mortons));
  BENCH_CUDA_TRY(cudaFree(u_sorted_mortons));
}

// static void BM_transform_morton_and_merge_sort(bm::State &state) {
//   const auto num_elements = state.range(0);
//   float3 *u_input;
//   Code_t *u_sorted_mortons;
//   BENCH_CUDA_TRY(cudaMallocManaged(&u_input, num_elements * sizeof(float3)));
//   BENCH_CUDA_TRY(
//       cudaMallocManaged(&u_sorted_mortons, num_elements * sizeof(Code_t)));

//   cub::CachingDeviceAllocator g_allocator(true);

//   for (auto _ : state) {
//     cuda_event_timer raii{state};

//     // Transform and Sort together
//     constexpr Morton conversion_op(0.0f, 1024.0f);
//     const cub::TransformInputIterator<Code_t, Morton, float3 *> itr(
//         u_input, conversion_op);

//     void *d_temp_storage = nullptr;
//     size_t temp_storage_bytes = 0;

//     cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes,
//     itr,
//                                        u_sorted_mortons, num_elements,
//                                        CustomLess());

//     cudaMalloc(&d_temp_storage, temp_storage_bytes);

//     cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes,
//     itr,
//                                        u_sorted_mortons, num_elements,
//                                        CustomLess());

//     bm::DoNotOptimize(u_sorted_mortons);
//   }

//   BENCH_CUDA_TRY(cudaFree(u_input));
//   BENCH_CUDA_TRY(cudaFree(u_sorted_mortons));
// }

BENCHMARK(BM_rand_float3_gen_cpu)
    ->RangeMultiplier(10)
    ->Range(10'000, 1'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_rand_float3_gen)
    ->RangeMultiplier(10)
    ->Range(10'000, 1'000'000)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_compute_morton_only)
    ->ArgsProduct({{1'000'000}, {32, 64, 128, 256, 512, 1024}})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_radixsort_morton_only)
    ->RangeMultiplier(10)
    ->Range(10'000, 1'000'000)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

// BENCHMARK(BM_transform_morton_and_merge_sort)
//     ->RangeMultiplier(10)
//     ->Range(10'000, 1'000'000)
//     ->UseManualTime()
//     ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
