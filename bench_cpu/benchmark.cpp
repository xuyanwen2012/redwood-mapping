#include <benchmark/benchmark.h>

#include <algorithm>
#include <execution>
#include <random>
#include <vector>

#include "Morton.hpp"
// #include "brt/RadixTree.hpp"
#include "brt.hpp"

namespace bm = benchmark;

struct float3 {
  float x, y, z;
};

constexpr float3 make_float3(const float x, const float y, const float z) {
  return {x, y, z};
}

template <typename execution_policy_t>
static void BM_transform_points_to_morton_tbb(bm::State &state,
                                              execution_policy_t &&policy) {
  std::vector<float3> inputs(state.range(0), {1.0f, 2.0f, 3.0f});
  std::vector<Code_t> outputs(state.range(0));

  for (auto _ : state) {
    std::transform(policy, inputs.begin(), inputs.end(), outputs.begin(),
                   [&](const auto &input) {
                     return PointToCode(input.x, input.y, input.z, 0.0f, 5.0f);
                   });
    bm::DoNotOptimize(outputs.size());
  }
}

static void BM_point3f_to_morton64(bm::State &state) {
  int32_t x = std::rand(), y = std::rand(), z = std::rand();
  for (auto _ : state) {
    auto result = PointToCode(x, y, z, 0.0f, 5.0f);
    bm::DoNotOptimize(result);
  }
}

// static void BM_point3f_to_morton64_v2(bm::State &state) {
//   std::vector<float3> inputs(state.range(0), {1.0f, 2.0f, 3.0f});
//   std::vector<Code_t> outputs(state.range(0));

//   for (auto _ : state) {
//     std::transform(inputs.begin(), inputs.end(), outputs.begin(),
//                    [&](const auto &input) {
//                      return PointToCode(input.x, input.y, input.z,
//                      0.0f, 5.0f);
//                    });
//     bm::DoNotOptimize(outputs.size());
//   }
// }

template <typename execution_policy_t>
static void BM_std_sort_tbb(bm::State &state, execution_policy_t &&policy) {
  const auto count = static_cast<size_t>(state.range(0));

  std::vector<Code_t> array(count);
  std::iota(array.begin(), array.end(), 1);
  std::reverse(policy, array.begin(), array.end());

  for (auto _ : state) {
    std::sort(policy, array.begin(), array.end());
    bm::DoNotOptimize(array.size());
  }
}

template <typename execution_policy_t>
static void BM_std_unique_tbb(bm::State &state, execution_policy_t &&policy) {
  const auto count = static_cast<size_t>(state.range(0));

  std::vector<Code_t> array(count);
  std::iota(array.begin(), array.end(), 1);

  for (auto _ : state) {
    const auto last_unique_it = std::unique(policy, array.begin(), array.end());
    const auto num_unique_keys = std::distance(array.begin(), last_unique_it);
    volatile auto res = num_unique_keys;
    (void)res;
  }
}

constexpr float min_coord = 0.0f;
constexpr float max_coord = 1024.0f;
constexpr float range = max_coord - min_coord;

static void BM_BuildRadixTree(bm::State &state) {
  const auto num_items = static_cast<size_t>(state.range(0));
  std::vector<int> indices(num_items);
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<float3> inputs(num_items);
  std::vector<Code_t> sorted_morton_keys(num_items);

  thread_local std::mt19937 gen(114514); // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, range);
  std::generate_n(inputs.data(), num_items,
                  [&] { return make_float3(dis(gen), dis(gen), dis(gen)); });

  std::transform(inputs.begin(), inputs.end(), sorted_morton_keys.begin(),
                 [&](const auto &pt) {
                   return PointToCode(pt.x, pt.y, pt.z, min_coord, range);
                 });

  std::sort(sorted_morton_keys.begin(), sorted_morton_keys.end());

  const auto last_unique_it =
      std::unique(sorted_morton_keys.begin(), sorted_morton_keys.end());

  const auto num_unique_keys =
      std::distance(sorted_morton_keys.begin(), last_unique_it);
  const auto num_brt_nodes = num_unique_keys - 1;

  std::vector<brt::InnerNodes> inner_nodes(num_brt_nodes);

  for (auto _ : state) {
    std::for_each_n(indices.begin(), num_brt_nodes, [&](const int i) {
      ProcessInternalNodesHelper(num_unique_keys, sorted_morton_keys.data(), i,
                                 inner_nodes.data());
    });
  }
}

BENCHMARK(BM_point3f_to_morton64)->ThreadRange(1, 8);
// BENCHMARK(BM_point3f_to_morton64_v2)
//     ->Arg(10'000'000)
//     ->ThreadRange(1, 8)
//     ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_transform_points_to_morton_tbb, seq, std::execution::seq)
    ->Arg(10'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_transform_points_to_morton_tbb, par, std::execution::par)
    ->Arg(10'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_std_sort_tbb, seq, std::execution::seq)
    ->Arg(10'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_std_sort_tbb, par, std::execution::par)
    ->Arg(10'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_std_unique_tbb, seq, std::execution::seq)
    ->Arg(10'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_std_unique_tbb, par, std::execution::par)
    ->Arg(10'000'000)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BuildRadixTree)->Arg(10'000'000)->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
