#include <benchmark/benchmark.h>
#include <stdlib.h>

#include <algorithm>
#include <execution>
#include <numeric>
#include <random>
#include <vector>

#include "Morton.hpp"
#include "octree/EdgeCount.hpp"
#include "octree/Octree.hpp"
#include "omp.h"

namespace bm = benchmark;

static int RandomIntInRange(const int min, const int max) {
  static thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_int_distribution<int> dis(min, max);
  return dis(gen);
}

static float RandomFloatInRange(const float min, const float max) {
  static thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(min, max);
  return dis(gen);
}

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

struct float3 {
  float x, y, z;
};

class MyFixture : public bm::Fixture {
 public:
  void SetUp(bm::State& state) {
    const auto input_size = state.range(0);

    std::vector<float3> u_inputs(input_size);
    std::vector<Code_t> u_sorted_morton_keys(input_size);

    std::generate(u_inputs.begin(), u_inputs.end(), [&] {
      const auto x = RandomFloatInRange(0.0f, 1024.0f);
      const auto y = RandomFloatInRange(0.0f, 1024.0f);
      const auto z = RandomFloatInRange(0.0f, 1024.0f);
      return float3{x, y, z};
    });

    // Parrallel
    std::transform(std::execution::par, u_inputs.begin(), u_inputs.end(),
                   u_sorted_morton_keys.begin(), [&](const auto& input) {
                     return PointToCode(input.x, input.y, input.z, 0.0f,
                                        1024.0f);
                   });

    std::sort(std::execution::par, u_sorted_morton_keys.begin(),
              u_sorted_morton_keys.end());

    u_sorted_morton_keys.erase(
        std::unique(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end()),
        u_sorted_morton_keys.end());

    const auto num_unique_keys = u_sorted_morton_keys.size();
    const auto num_brt_nodes = num_unique_keys - 1;

    u_brt_nodes.resize(num_brt_nodes);

#pragma omp parallel for
    for (int i = 0; i < num_brt_nodes; ++i) {
      ProcessInternalNodesHelper(num_unique_keys, u_sorted_morton_keys.data(),
                                 i, u_brt_nodes.data());
    }
  }

  std::vector<brt::InnerNodes> u_brt_nodes;
};

static void BM_CountEdges(benchmark::State& state, MyFixture& fixture) {
  std::vector<int> edge_count(state.range(0));

  for (auto _ : state) {
    for (int i = 0; i < state.range(0); ++i) {
      CalculateEdgeCountHelper(i, edge_count.data(),
                               fixture.u_brt_nodes.data());
      bm::DoNotOptimize(edge_count.size());
    }
  }
}

// Register the benchmark with the fixture.
BENCHMARK_DEFINE_F(MyFixture, BM_CountEdges)(benchmark::State& state) {
  BM_CountEdges(state, *this);
}

BENCHMARK(BM_STD_PartialSum)->Arg(1280 * 720)->Arg(640 * 480);
BENCHMARK_CAPTURE(BM_STD_InclusiveScan, seq, std::execution::seq)
    ->Arg(1280 * 720);
BENCHMARK_CAPTURE(BM_STD_InclusiveScan, par, std::execution::par)
    ->Arg(1280 * 720);

BENCHMARK_REGISTER_F(MyFixture, BM_CountEdges)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1280 * 720)
    ->Arg(640 * 480);

BENCHMARK_MAIN();
