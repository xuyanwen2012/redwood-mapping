#include <omp.h>  // Include OpenMP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <execution>
#include <iostream>
#include <random>

#include "brt/RadixTree.hpp"

template <uint8_t Axis>
bool CompareAxis(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  }
  return a.z() < b.z();
}

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(0.0f, 1024.0f);

  constexpr auto input_size = 1280 * 720;

  std::vector<Eigen::Vector3f> u_inputs(input_size);
  //   std::vector<Code_t> u_morton_keys(input_size);
  std::vector<Code_t> u_sorted_morton_keys(input_size);
  std::vector<brt::InnerNodes> u_brt_nodes(input_size);

  std::generate(u_inputs.begin(), u_inputs.end(), [&] {
    const auto x = dis(gen);
    const auto y = dis(gen);
    const auto z = dis(gen);
    return Eigen::Vector3f(x, y, z);
  });

  float min_coord = 0.0f;
  float max_coord = 1.0f;

  const auto x_range =
      std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<0>);
  const auto y_range =
      std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<1>);
  const auto z_range =
      std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<2>);
  const auto x_min = x_range.first;
  const auto x_max = x_range.second;
  const auto y_min = y_range.first;
  const auto y_max = y_range.second;
  const auto z_min = z_range.first;
  const auto z_max = z_range.second;
  std::array<float, 3> mins{x_min->x(), y_min->y(), z_min->z()};
  std::array<float, 3> maxes{x_max->x(), y_max->y(), z_max->z()};
  min_coord = *std::min_element(mins.begin(), mins.end());
  max_coord = *std::max_element(maxes.begin(), maxes.end());

  const float range = max_coord - min_coord;

  std::cout << "Min: " << min_coord << "\n";
  std::cout << "Max: " << max_coord << "\n";
  std::cout << "Range: " << range << "\n";

  TimeTask("Compute & Sort", [&] {
    // Parrallel
    std::transform(std::execution::par_unseq, u_inputs.begin(), u_inputs.end(),
                   u_sorted_morton_keys.begin(), [&](const auto& input) {
                     return PointToCode(input.x(), input.y(), input.z(),
                                        min_coord, range);
                   });

    // Parallel sort using std::execution::par
    std::sort(std::execution::par, u_sorted_morton_keys.begin(),
              u_sorted_morton_keys.end());

    u_sorted_morton_keys.erase(
        std::unique(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end()),
        u_sorted_morton_keys.end());
  });

  TimeTask("Build Tree", [&] {
    const auto num_unique_keys = u_sorted_morton_keys.size();
    const auto num_brt_nodes = num_unique_keys - 1;

#pragma omp parallel for
    for (int i = 0; i < num_brt_nodes; ++i) {
      ProcessInternalNodesHelper(num_unique_keys, u_sorted_morton_keys.data(),
                                 i, u_brt_nodes.data());
    }
  });

  std::cout << "Unique keys: " << u_sorted_morton_keys.size() << "\n";

  return 0;
}