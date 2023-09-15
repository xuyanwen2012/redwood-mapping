#include <benchmark/benchmark.h>
#include <stdlib.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "brt/RadixTree.hpp"

namespace bm = benchmark;

static int RandomIntInRange(const int min, const int max) {
  static thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_int_distribution<int> dis(min, max);
  return dis(gen);
}

static void BM_count_lead_zeros_64(bm::State& state) {
  auto a = std::rand();
  for (auto _ : state) bm::DoNotOptimize(CLZ64(a));
}

static void BM_ProcessInternalNode(bm::State& state) {
  const auto input_size = state.range(0);
  std::vector<Code_t> u_sorted_morton_keys(input_size);
  std::iota(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end(), 0);

  const auto n_brt_nodes = input_size - 1;
  std::vector<brt::InnerNodes> brt_nodes(n_brt_nodes);

  for (auto _ : state) {
    for (int i = 0; i < n_brt_nodes; ++i) {
      brt::ProcessInternalNodesHelper(input_size, u_sorted_morton_keys.data(),
                                      i, brt_nodes.data());
    }
    bm::DoNotOptimize(brt_nodes.size());
  }
}

BENCHMARK(BM_count_lead_zeros_64);

BENCHMARK(BM_ProcessInternalNode)
    ->Arg(1280 * 720)
    ->Arg(640 * 480)
    ->ThreadRange(1, 8)
    ->UseRealTime()
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
