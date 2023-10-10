#include <benchmark/benchmark.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>

#include "brt.cuh"
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

class RadixTreeFixture : public bm::Fixture {
public:
  void SetUp(const bm::State &state) override {
    const auto num_elements = state.range(0);
    cudaMallocManaged(&u_input, num_elements * sizeof(float3));
    cudaMallocManaged(&u_mortons, num_elements * sizeof(Code_t));
    BENCH_CUDA_TRY(cudaDeviceSynchronize());

    // init random inputs
    const auto min_coord = 0.0f;
    const auto range = 1024.0f;

    thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
    static std::uniform_real_distribution<float> dis(min_coord, range);

    std::generate_n(std::execution::par, u_input, num_elements, [&]() {
      return float3{dis(gen), dis(gen), dis(gen)};
    });

    std::transform(std::execution::par, u_input, u_input + num_elements,
                   u_mortons, [&](const auto &pt) {
                     return PointToCode(pt.x, pt.y, pt.z, min_coord, range);
                   });
    std::sort(std::execution::par, u_mortons, u_mortons + num_elements);

    const auto last_unique_it =
        std::unique(std::execution::par, u_mortons, u_mortons + num_elements);

    num_unique_keys = std::distance(u_mortons, last_unique_it);

    const auto num_brt_nodes = num_unique_keys - 1;
    cudaMallocManaged(&u_inner_nondes, num_brt_nodes * sizeof(brt::InnerNodes));

    // prepare for the later benchmark
    constexpr auto num_threads = 256;
    const auto num_blocks = (num_unique_keys + num_threads - 1) / num_threads;
    BuildRadixTreeKernel<<<num_blocks, num_threads>>>(u_mortons, u_inner_nondes,
                                                      num_unique_keys);
    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  void TearDown(const bm::State &state) override {
    BENCH_CUDA_TRY(cudaFree(u_input));
    BENCH_CUDA_TRY(cudaFree(u_mortons));
    BENCH_CUDA_TRY(cudaFree(u_inner_nondes));
  }

protected:
  float3 *u_input;
  Code_t *u_mortons;
  int num_unique_keys;
  brt::InnerNodes *u_inner_nondes;
};

BENCHMARK_DEFINE_F(RadixTreeFixture, FooTest)(bm::State &st) {
  const auto num_elements = st.range(0);
  const auto num_threads = st.range(1);

  brt::InnerNodes *inner_nondes;
  cudaMallocManaged(&inner_nondes, num_elements * sizeof(brt::InnerNodes));

  for (auto _ : st) {
    cuda_event_timer raii{st, true};

    const auto num_blocks = (num_elements + num_threads - 1) / num_threads;
    BuildRadixTreeKernel<<<num_blocks, num_threads>>>(u_mortons, inner_nondes,
                                                      num_unique_keys);
    bm::DoNotOptimize(inner_nondes);
  }

  BENCH_CUDA_TRY(cudaFree(inner_nondes));
}

BENCHMARK_REGISTER_F(RadixTreeFixture, FooTest)
    ->ArgsProduct({{10'000'000}, {32, 64, 128, 256, 512, 1024}})
    ->Unit(bm::kMillisecond);

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
