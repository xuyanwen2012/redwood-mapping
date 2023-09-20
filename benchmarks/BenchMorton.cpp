#include <benchmark/benchmark.h>

#include <algorithm>
#include <execution>
#include <random>
#include <vector>

#include "Morton.hpp"

namespace bm = benchmark;

constexpr size_t f32s_in_cacheline_k = 64 / sizeof(float);
constexpr size_t f32s_in_halfline_k = f32s_in_cacheline_k / 2;

struct alignas(64) f32_array_t {
  float raw[f32s_in_cacheline_k * 2];
};

static void f32_pairwise_accumulation(bm::State &state) {
  f32_array_t a, b, c;
  for (auto _ : state)
    for (size_t i = f32s_in_halfline_k; i != f32s_in_halfline_k * 3; ++i)
      bm::DoNotOptimize(c.raw[i] = a.raw[i] + b.raw[i]);
}

static void f32_pairwise_accumulation_aligned(bm::State &state) {
  f32_array_t a, b, c;
  for (auto _ : state)
    for (size_t i = 0; i != f32s_in_halfline_k; ++i)
      bm::DoNotOptimize(c.raw[i] = a.raw[i] + b.raw[i]);
}

static void i32_addition_semirandom(bm::State &state) {
  int32_t a = std::rand(), b = std::rand(), c = 0;
  for (auto _ : state) bm::DoNotOptimize(c = (++a) + (++b));
}

static void f64_sin(bm::State &state) {
  double argument = std::rand(), result = 0;
  for (auto _ : state) bm::DoNotOptimize(result = std::sin(argument += 1.0));
}

static void f64_sin_maclaurin(bm::State &state) {
  double argument = std::rand(), result = 0;
  for (auto _ : state) {
    argument += 1.0;
    result = argument - std::pow(argument, 3) / 6 + std::pow(argument, 5) / 120;
    bm::DoNotOptimize(result);
  }
}

static void f64_sin_maclaurin_powless(bm::State &state) {
  double argument = std::rand(), result = 0;
  for (auto _ : state) {
    argument += 1.0;
    result = argument - (argument * argument * argument) / 6.0 +
             (argument * argument * argument * argument * argument) / 120.0;
    bm::DoNotOptimize(result);
  }
}

static void BM_morton_encode_64(bm::State &state) {
  uint32_t x = std::rand(), y = std::rand(), z = std::rand();
  for (auto _ : state) {
    auto result = Encode64(x, y, z);
    bm::DoNotOptimize(result);
  }
}

static void BM_morton_decode_64(bm::State &state) {
  Code_t code = static_cast<Code_t>(std::rand());
  uint32_t x, y, z;
  for (auto _ : state) {
    Decode64(code, x, y, z);
    bm::DoNotOptimize(x);
  }
}

static void BM_point3f_to_morton64(bm::State &state) {
  int32_t x = std::rand(), y = std::rand(), z = std::rand();
  for (auto _ : state) {
    auto result = PointToCode(x, y, z, 0.0f, RAND_MAX);
    bm::DoNotOptimize(result);
  }
}

static void BM_morton64_to_point3f(bm::State &state) {
  Code_t code = static_cast<Code_t>(std::rand());
  float x, y, z;
  for (auto _ : state) {
    CodeToPoint(code, x, y, z, 0.0f, RAND_MAX);
    bm::DoNotOptimize(x);
  }
}

struct float3 {
  float x, y, z;
};

template <typename execution_policy_t>
static void BM_transform_points_to_morton(bm::State &state,
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

template <typename execution_policy_t>
static void supersort(bm::State &state, execution_policy_t &&policy) {
  const auto count = static_cast<size_t>(state.range(0));

  std::vector<Code_t> array(count);
  std::iota(array.begin(), array.end(), 1);
  std::reverse(policy, array.begin(), array.end());

  for (auto _ : state) {
    std::sort(policy, array.begin(), array.end());
    bm::DoNotOptimize(array.size());
  }
}

BENCHMARK(BM_morton_encode_64);
BENCHMARK(BM_morton_decode_64);
BENCHMARK(BM_point3f_to_morton64);
BENCHMARK(BM_morton64_to_point3f);

BENCHMARK_CAPTURE(supersort, seq, std::execution::seq)
    ->Arg(1280 * 720)
    ->Unit(bm::kMillisecond);
BENCHMARK_CAPTURE(supersort, par, std::execution::par)
    ->Arg(1280 * 720)
    ->Unit(bm::kMillisecond);

BENCHMARK_CAPTURE(BM_transform_points_to_morton, seq, std::execution::seq)
    ->Arg(1280 * 720)
    ->Unit(bm::kMillisecond);
BENCHMARK_CAPTURE(BM_transform_points_to_morton, par, std::execution::par)
    ->Arg(1280 * 720)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();