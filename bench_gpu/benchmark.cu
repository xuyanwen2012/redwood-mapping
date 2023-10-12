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

// namespace {

// #define DEFINE_SYNC_KERNEL_WRAPPER(kernel_name, function_name, num_threads)    \
//   template <typename... Args>                                                  \
//   void function_name(const size_t num_items, Args... args) {                   \
//     const auto num_blocks = (num_items + num_threads - 1) / num_threads;       \
//     kernel_name<<<num_blocks, num_threads>>>(num_items, args...);              \
//     BENCH_CUDA_TRY(cudaDeviceSynchronize());                                   \
//   }

// #define DEFINE_CUB_WRAPPER(kernel, wrapper_name)                               \
//   template <typename... Args> void wrapper_name(Args... args) {                \
//     void *d_temp_storage = nullptr;                                            \
//     size_t temp_storage_bytes = 0;                                             \
//     kernel(d_temp_storage, temp_storage_bytes, args...);                       \
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);                           \
//     kernel(d_temp_storage, temp_storage_bytes, args...);                       \
//     BENCH_CUDA_TRY(cudaDeviceSynchronize());                                   \
//   }

// } // namespace

// ---------------------
//        Kernels
// ---------------------

// DEFINE_SYNC_KERNEL_WRAPPER(convertMortonOnly_v2, TransformMortonSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(BuildRadixTreeKernel, BuildRadixTreeSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(CalculateEdgeCountKernel, EdgeCountSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(MakeNodesKernel, MakeOctreeNodesSync, 256)
// DEFINE_SYNC_KERNEL_WRAPPER(LinkNodesKernel, LinkNodesSync, 256)

// DEFINE_CUB_WRAPPER(cub::DeviceRadixSort::SortKeys, CubRadixSort)
// DEFINE_CUB_WRAPPER(cub::DeviceSelect::Unique, CubUnique)
// DEFINE_CUB_WRAPPER(cub::DeviceScan::InclusiveSum, CubPrefixSum)

// ---------------------
//        Benchmarks
// ---------------------

constexpr float min_coord = 0.0f;
constexpr float max_coord = 1024.0f;
constexpr float range = max_coord - min_coord;
constexpr auto morton_encoder = MortonEncoder(min_coord, range);
constexpr auto morton_decoder = MortonDecoder(min_coord, range);

class SortedMortonFixture : public bm::Fixture {
public:
  void SetUp(const bm::State &st) override {
    num_items = 10'000'000;
    u_input = AllocateManaged<float3>(num_items);
    u_mortons = AllocateManaged<Code_t>(num_items);
    u_mortons_alt = AllocateManaged<Code_t>(num_items);
    u_num_selected_out = AllocateManaged<int>(1);

    // Init values
    thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution dis(min_coord, range); // <float>
    std::generate_n(u_input, num_items,
                    [&] { return make_float3(dis(gen), dis(gen), dis(gen)); });

    constexpr auto num_threads = 256;
    const auto num_blocks = (num_items + num_threads - 1) / num_threads;
    convertMortonOnly_v2<<<num_blocks, num_threads>>>(
        num_items, u_input, u_mortons, morton_encoder);
    BENCH_CUDA_TRY(cudaDeviceSynchronize());

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t last_temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   u_mortons, u_mortons_alt, num_items);
    BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   u_mortons, u_mortons_alt, num_items);

    BENCH_CUDA_TRY(cudaDeviceSynchronize());

    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons_alt,
                              u_mortons, u_num_selected_out, num_items);

    if (last_temp_storage_bytes < temp_storage_bytes) {
      BENCH_CUDA_TRY(cudaFree(d_temp_storage));
      BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      last_temp_storage_bytes = temp_storage_bytes;
    }

    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons,
                              u_mortons_alt, u_num_selected_out, num_items);

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
    num_unique = *u_num_selected_out;

    BENCH_CUDA_TRY(cudaFree(d_temp_storage));

    // for (auto i = 0; i < 10; ++i) {
    //   std::cout << i << ":\t" << u_mortons[i] << '\n';
    // }
    // std::cout << "num_unique: " << num_unique << std::endl;
  }

  void TearDown(const bm::State &) override {
    BENCH_CUDA_TRY(cudaFree(u_input));
    BENCH_CUDA_TRY(cudaFree(u_mortons));
    BENCH_CUDA_TRY(cudaFree(u_mortons_alt));
    BENCH_CUDA_TRY(cudaFree(u_num_selected_out));
  }

  size_t num_items;
  float3 *u_input;
  Code_t *u_mortons;
  Code_t *u_mortons_alt;
  int *u_num_selected_out;
  int num_unique;
};

BENCHMARK_DEFINE_F(SortedMortonFixture, BM_BuildRadixTree)(bm::State &st) {
  const auto num_threads = st.range(0);
  const auto u_inner_nodes = AllocateManaged<brt::InnerNodes>(num_unique);
  for (auto _ : st) {
    cuda_event_timer raii{st, true};

    const auto num_blocks = (num_unique + num_threads - 1) / num_threads;
    BuildRadixTreeKernel<<<num_blocks, num_threads>>>(num_unique, u_mortons,
                                                      u_inner_nodes);
  }
  BENCH_CUDA_TRY(cudaFree(u_inner_nodes));
}

class RadixTreeFixture : public SortedMortonFixture {
public:
  void SetUp(const bm::State &st) override {
    SortedMortonFixture::SetUp(st);
    u_inner_nodes =
        AllocateManaged<brt::InnerNodes>(SortedMortonFixture::num_unique);

    // Build Tree
    // BuildRadixTreeSync(SortedMortonFixture::num_unique, u_mortons,
    //                    u_inner_nodes);
    constexpr auto num_threads = 256;

    const auto num_blocks =
        (SortedMortonFixture::num_unique + num_threads - 1) / num_threads;
    BuildRadixTreeKernel<<<num_blocks, num_threads>>>(
        SortedMortonFixture::num_unique, u_mortons, u_inner_nodes);
    BENCH_CUDA_TRY(cudaDeviceSynchronize());

    // std::cout << "brt_nodes:" << std::endl;
    // for (auto i = 0; i < 10; ++i) {
    //   std::cout << i << ":\t" << u_inner_nodes[i].left << ", "
    //             << u_inner_nodes[i].right << "\t("
    //             << u_inner_nodes[i].delta_node << ")" << std::endl;
    // }
  }

  void TearDown(const bm::State &st) override {
    SortedMortonFixture::TearDown(st);
    BENCH_CUDA_TRY(cudaFree(u_inner_nodes));
  }

  brt::InnerNodes *u_inner_nodes;
};

BENCHMARK_DEFINE_F(RadixTreeFixture, BM_EdgeCount)(bm::State &st) {
  const auto num_threads = st.range(0);
  const auto num_brt_nodes = SortedMortonFixture::num_unique - 1;
  const auto u_edge_count = AllocateManaged<int>(num_brt_nodes);
  for (auto _ : st) {
    cuda_event_timer raii{st, true};

    constexpr auto num_threads = 256;
    const auto num_blocks = (num_brt_nodes + num_threads - 1) / num_threads;
    CalculateEdgeCountKernel<<<num_blocks, num_threads>>>(
        num_brt_nodes, u_edge_count, u_inner_nodes);
  }
  BENCH_CUDA_TRY(cudaFree(u_edge_count));
}

BENCHMARK_DEFINE_F(RadixTreeFixture, BM_MakeOctreeNodes)(bm::State &st) {
  const auto num_threads = st.range(0);

  const auto num_unique = SortedMortonFixture::num_unique;
  const auto num_brt_nodes = num_unique - 1;
  const auto u_edge_count = AllocateManaged<int>(num_brt_nodes);
  const auto u_oc_offset = AllocateManaged<int>(num_unique);
  const auto u_oc_nodes = AllocateManaged<oct::OctNode>(num_unique);

  constexpr auto num_threads_for_data_prep = 256;

  // EdgeCountSync(num_brt_nodes, u_edge_count, u_inner_nodes);
  const auto num_blocks = (num_brt_nodes + num_threads_for_data_prep - 1) /
                          num_threads_for_data_prep;
  CalculateEdgeCountKernel<<<num_blocks, num_threads_for_data_prep>>>(
      num_brt_nodes, u_edge_count, u_inner_nodes);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  u_edge_count[0] = 1; // Root node counts as 1

  // CubPrefixSum(u_edge_count, u_oc_offset + 1, num_brt_nodes);
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                u_edge_count, u_oc_offset + 1, num_items);
  BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                u_edge_count, u_oc_offset + 1, num_items);
  BENCH_CUDA_TRY(cudaDeviceSynchronize());

  u_oc_offset[0] = 0;

  // std::cout << "edge_count (oc_offset):" << std::endl;
  // for (auto i = 0; i < 10; ++i) {
  //   std::cout << i << ":\t" << u_edge_count[i] << "\t(" << u_oc_offset[i] <<
  //   ')'
  //             << std::endl;
  // }

  const auto num_oc_nodes = u_oc_offset[num_brt_nodes];
  const auto root_level = u_inner_nodes[0].delta_node / 3;
  Code_t root_prefix = u_mortons[0] >> (kCodeLen - (root_level * 3));

  for (auto _ : st) {
    cuda_event_timer raii{st, true};

    u_oc_nodes[0].cornor =
        morton_decoder(root_prefix << (kCodeLen - (root_level * 3)));
    u_oc_nodes[0].cell_size = range;

    const auto num_blocks = (num_brt_nodes + num_threads - 1) / num_threads;
    MakeNodesKernel<<<num_blocks, num_threads>>>(
        num_brt_nodes, u_oc_nodes, u_oc_offset, u_edge_count, u_mortons,
        u_inner_nodes, morton_decoder, root_level);
  }

  BENCH_CUDA_TRY(cudaFree(d_temp_storage));
  BENCH_CUDA_TRY(cudaFree(u_edge_count));
  BENCH_CUDA_TRY(cudaFree(u_oc_offset));
  BENCH_CUDA_TRY(cudaFree(u_oc_nodes));
}

static void BM_ComputeMorton(bm::State &st) {
  constexpr auto num_items = 10'000'000;
  const auto num_threads = st.range(0);
  const auto u_input = AllocateManaged<float3>(num_items);
  const auto u_output = AllocateManaged<Code_t>(num_items);

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    const auto num_blocks = (num_items + num_threads - 1) / num_threads;
    convertMortonOnly_v2<<<num_blocks, num_threads>>>(num_items, u_input,
                                                      u_output, morton_encoder);
  }

  BENCH_CUDA_TRY(cudaFree(u_input));
  BENCH_CUDA_TRY(cudaFree(u_output));
}

static void BM_RadixSort(bm::State &st) {
  const auto num_items = st.range(0);
  const auto u_mortons = AllocateManaged<Code_t>(num_items);
  const auto u_mortons_alt = AllocateManaged<Code_t>(num_items);

  // One time code to get the required temp storage size
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, u_mortons,
                                 u_mortons_alt, num_items);
  BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                   u_mortons, u_mortons_alt, num_items);
  }

  BENCH_CUDA_TRY(cudaFree(d_temp_storage));
  BENCH_CUDA_TRY(cudaFree(u_mortons));
  BENCH_CUDA_TRY(cudaFree(u_mortons_alt));
}

static void BM_RemoveDuplicates(bm::State &st) {
  const auto num_items = st.range(0);
  const auto u_mortons = AllocateManaged<Code_t>(num_items);
  const auto u_mortons_alt = AllocateManaged<Code_t>(num_items);
  const auto u_num_selected_out = AllocateManaged<int>(1);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons,
                            u_mortons_alt, u_num_selected_out, num_items);
  BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, u_mortons,
                              u_mortons_alt, u_num_selected_out, num_items);
  }

  BENCH_CUDA_TRY(cudaFree(d_temp_storage));
  BENCH_CUDA_TRY(cudaFree(u_mortons));
  BENCH_CUDA_TRY(cudaFree(u_mortons_alt));
  BENCH_CUDA_TRY(cudaFree(u_num_selected_out));
}

static void BM_PrefixSum(bm::State &st) {
  const auto num_items = st.range(0);
  const auto u_nums = AllocateManaged<int>(num_items);
  const auto u_sums = AllocateManaged<int>(num_items);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, u_nums,
                                u_sums, num_items);
  BENCH_CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  for (auto _ : st) {
    cuda_event_timer raii{st, true};
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, u_nums,
                                  u_sums, num_items);
  }
  BENCH_CUDA_TRY(cudaFree(d_temp_storage));
  BENCH_CUDA_TRY(cudaFree(u_nums));
  BENCH_CUDA_TRY(cudaFree(u_sums));
}

BENCHMARK(BM_ComputeMorton)
    ->RangeMultiplier(2)
    ->Range(32, 1024)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

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

BENCHMARK_REGISTER_F(SortedMortonFixture, BM_BuildRadixTree)
    ->RangeMultiplier(2)
    ->Range(32, 1024)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_REGISTER_F(RadixTreeFixture, BM_EdgeCount)
    ->RangeMultiplier(2)
    ->Range(32, 1024)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_PrefixSum)
    ->Args({1280 * 720})
    ->Args({10'000'000})
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_REGISTER_F(RadixTreeFixture, BM_MakeOctreeNodes)
    ->RangeMultiplier(2)
    ->Range(32, 1024)
    ->UseManualTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
