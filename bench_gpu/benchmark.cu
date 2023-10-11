#include <benchmark/benchmark.h>

#include <algorithm>
#include <bitset>
#include <cub/cub.cuh>
#include <random>

#include "brt.cuh"
#include "cuda_utils.cuh"
#include "morton.cuh"
#include "oct.cuh"
#include "sync.hpp"

namespace bm = benchmark;

namespace {

#define DEFINE_SYNC_KERNEL_WRAPPER(kernel_name, function_name, num_threads)    \
  template <typename... Args>                                                  \
  void function_name(const size_t num_items, Args... args) {                   \
    const auto num_blocks = (num_items + num_threads - 1) / num_threads;       \
    kernel_name<<<num_blocks, num_threads>>>(num_items, args...);              \
  }

#define DEFINE_CUB_WRAPPER(kernel, wrapper_name)                               \
  template <typename... Args> void wrapper_name(Args... args) {                \
    void *d_temp_storage = nullptr;                                            \
    size_t temp_storage_bytes = 0;                                             \
    kernel(d_temp_storage, temp_storage_bytes, args...);                       \
    cudaMalloc(&d_temp_storage, temp_storage_bytes);                           \
    kernel(d_temp_storage, temp_storage_bytes, args...);                       \
  }

} // namespace
// ---------------------
//        Kernels
// ---------------------

DEFINE_SYNC_KERNEL_WRAPPER(convertMortonOnly_v2, TransformMortonSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(BuildRadixTreeKernel, BuildRadixTreeSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(CalculateEdgeCountKernel, EdgeCountSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(MakeNodesKernel, MakeOctreeNodesSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(LinkNodesKernel, LinkNodesSync, 256)

DEFINE_CUB_WRAPPER(cub::DeviceRadixSort::SortKeys, CubRadixSort)
DEFINE_CUB_WRAPPER(cub::DeviceSelect::Unique, CubUnique)
DEFINE_CUB_WRAPPER(cub::DeviceScan::InclusiveSum, CubPrefixSum)

// ---------------------
//        Benchmarks
// ---------------------

constexpr float min_coord = 0.0f;
constexpr float max_coord = 1024.0f;
constexpr float range = max_coord - min_coord;
constexpr auto morton_encoder = MortonEncoder(min_coord, range);

// class MortonFixture : public bm::Fixture {
// public:
//   void SetUp(const bm::State &st) override {
//     const auto num_items = st.range(1);
//     std::cout << "SetUp::num_items: " << num_items << std::endl;
//     u_input = AllocateManaged<float3>(num_items);
//     u_mortons = AllocateManaged<Code_t>(num_items);
//     u_mortons_alt = AllocateManaged<Code_t>(num_items);
//     u_num_selected_out = AllocateManaged<int>(1);

//     // Init values
//     thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
//     std::uniform_real_distribution dis(min_coord, range); // <float>
//     std::generate_n(u_input, num_items,
//                     [&] { return make_float3(dis(gen), dis(gen), dis(gen));
//                     });
//   }

//   void TearDown(const bm::State &) override {
//     cudaFree(u_input);
//     cudaFree(u_mortons);
//     cudaFree(u_mortons_alt);
//     cudaFree(u_num_selected_out);
//   }

//   // int num_items;
//   float3 *u_input;
//   Code_t *u_mortons;
//   Code_t *u_mortons_alt;
//   int *u_num_selected_out;
// };

// BENCHMARK_DEFINE_F(MortonFixture, BM_ComputeMorton)(bm::State &st) {
//   const auto num_threads = st.range(0);
//   const auto num_items = st.range(1);
//   const auto u_output = AllocateManaged<Code_t>(num_items);
//   for (auto _ : st) {
//     cuda_event_timer raii{st, true};

//     const auto num_blocks = (num_items + num_threads - 1) / num_threads;
//     convertMortonOnly_v2<<<num_blocks, num_threads>>>(num_items, u_input,
//                                                       u_output,
//                                                       morton_encoder);
//     bm::DoNotOptimize(u_output);
//   }
//   cudaFree(u_output);
// }

// BENCHMARK_REGISTER_F(MortonFixture, BM_ComputeMorton)
//     ->ArgsProduct({{32, 64, 128, 256, 512, 1024}, {1280 * 720, 10'000'000}})
//     ->UseManualTime()
//     ->Unit(bm::kMillisecond);

static void BM_RadixSort(bm::State &st) {
  const auto num_items = st.range(0);

  const auto u_mortons = AllocateManaged<Code_t>(num_items);
  const auto u_mortons_alt = AllocateManaged<Code_t>(num_items);

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    CubRadixSort(u_mortons, u_mortons_alt, num_items);
  }
  cudaFree(u_mortons);
  cudaFree(u_mortons_alt);
}

static void BM_RemoveDuplicates(bm::State &st) {
  const auto num_items = st.range(0);

  const auto u_mortons = AllocateManaged<Code_t>(num_items);
  const auto u_mortons_alt = AllocateManaged<Code_t>(num_items);
  const auto u_num_selected_out = AllocateManaged<int>(1);

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    CubUnique(u_mortons, u_mortons_alt, u_num_selected_out, num_items);
  }
  cudaFree(u_mortons);
  cudaFree(u_mortons_alt);
  cudaFree(u_num_selected_out);
}

static void BM_PrefixSum(bm::State &st) {
  const auto num_items = st.range(0);
  const auto u_nums = AllocateManaged<int>(num_items);
  const auto u_sums = AllocateManaged<int>(num_items);

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    CubPrefixSum(u_nums, u_sums, num_items);
  }
  cudaFree(u_nums);
  cudaFree(u_sums);
}

BENCHMARK(BM_RadixSort)
    ->Args({1280 * 720})
    ->Args({10'000'000})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_RemoveDuplicates)
    ->Args({1280 * 720})
    ->Args({10'000'000})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_PrefixSum)
    ->Args({1280 * 720})
    ->Args({10'000'000})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
