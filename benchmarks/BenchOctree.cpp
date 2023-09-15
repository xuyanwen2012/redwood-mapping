#include <benchmark/benchmark.h>
#include <stdlib.h>

#include <algorithm>
#include <execution>
#include <numeric>
#include <random>
#include <vector>

#include "octree/Octree.hpp"

namespace bm = benchmark;

static void BM_STD_PartialSum(bm::State& state) {
  std::vector<int> edge_count(state.range(0), 1);
  std::vector<int> oc_node_offsets(state.range(0) + 1);

  for (auto _ : state) {
    std::partial_sum(edge_count.begin(), edge_count.end(),
                     oc_node_offsets.begin() + 1);
    oc_node_offsets[0] = 0;
    bm::DoNotOptimize(oc_node_offsets.size());
  }
}

template <typename execution_policy_t>
static void BM_STD_InclusiveScan(bm::State& state,
                                 execution_policy_t&& policy) {
  const auto count = static_cast<size_t>(state.range(0));

  std::vector<int> edge_count(count, 1);
  std::vector<int> oc_node_offsets(count + 1);

  for (auto _ : state) {
    std::inclusive_scan(policy, edge_count.begin(), edge_count.end(),
                        oc_node_offsets.begin() + 1);
    oc_node_offsets[0] = 0;
    bm::DoNotOptimize(oc_node_offsets.size());
  }
}

BENCHMARK(BM_STD_PartialSum)->Arg(1280 * 720)->Arg(640 * 480);
BENCHMARK_CAPTURE(BM_STD_InclusiveScan, seq, std::execution::seq)
    ->Arg(1280 * 720);
BENCHMARK_CAPTURE(BM_STD_InclusiveScan, par, std::execution::par)
    ->Arg(1280 * 720);

BENCHMARK_MAIN();
