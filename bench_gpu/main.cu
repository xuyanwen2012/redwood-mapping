#include <benchmark/benchmark.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>

#include "morton.cuh"
#include "sync.hpp"

namespace bm = benchmark;

static void BM_compute_morton_v2_only(bm::State &state) {
  const auto num_elements = state.range(0);
  const auto num_threads = state.range(1);
  float3 *u_input;
  Code_t *u_output;
  cudaMallocManaged(&u_input, num_elements * sizeof(float3));
  cudaMallocManaged(&u_output, num_elements * sizeof(Code_t));

  for (auto _ : state) {
    cuda_event_timer raii{state, true};

    const auto num_blocks = (num_elements + num_threads - 1) / num_threads;
    convertMortonOnly_v2<<<num_blocks, num_threads>>>(
        u_input, u_output, num_elements, Morton(0.0f, 1024.0f));
  }

  BENCH_CUDA_TRY(cudaFree(u_input));
  BENCH_CUDA_TRY(cudaFree(u_output));
}

static void BM_radixsort_morton_only(bm::State &state) {
  const auto num_elements = state.range(0);
  Code_t *u_mortons;
  Code_t *u_sorted_mortons;
  cudaMallocManaged(&u_mortons, num_elements * sizeof(Code_t));
  cudaMallocManaged(&u_sorted_mortons, num_elements * sizeof(Code_t));
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
  }

  BENCH_CUDA_TRY(cudaFree(u_mortons));
  BENCH_CUDA_TRY(cudaFree(u_sorted_mortons));
}

static void BM_unique_morton(bm::State &state) {
  const auto num_elements = state.range(0);
  Code_t *u_input;
  Code_t *u_output;
  int *u_num_selected_out;
  cudaMallocManaged(&u_input, num_elements * sizeof(Code_t));
  cudaMallocManaged(&u_output, num_elements * sizeof(Code_t));
  cudaMallocManaged(&u_num_selected_out, 1 * sizeof(int));
  cub::CachingDeviceAllocator g_allocator(true);

  for (auto _ : state) {
    cuda_event_timer raii{state};

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    BENCH_CUDA_TRY(cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                                             u_input, u_output,
                                             u_num_selected_out, num_elements));

    BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    BENCH_CUDA_TRY(cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
                                             u_input, u_output,
                                             u_num_selected_out, num_elements));
  }

  BENCH_CUDA_TRY(cudaFree(u_input));
  BENCH_CUDA_TRY(cudaFree(u_output));
}

BENCHMARK(BM_compute_morton_v2_only)
    ->ArgsProduct({{10'000'000}, {32, 64, 128, 256, 512, 1024}})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_radixsort_morton_only)
    ->Args({1'000'000})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_unique_morton)
    ->Args({1'000'000})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
