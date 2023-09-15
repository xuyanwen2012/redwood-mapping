#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

#include "brt/RadixTree.hpp"
#include "octree/EdgeCount.hpp"
#include "octree/Octree.hpp"

template <uint8_t Axis>
bool CompareAxis(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  }
  return a.z() < b.z();
}

_NODISCARD std::pair<float, float> FindMinMax(
    const std::vector<Eigen::Vector3f>& u_inputs) {
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

  return {min_coord, max_coord};
}

int main() {
  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(0.0f, 1024.0f);

  constexpr auto input_size = 1280 * 720;

  // Preallocate all the memory
  std::vector<Eigen::Vector3f> u_inputs(input_size);
  std::vector<Code_t> u_sorted_morton_keys(input_size);
  std::vector<brt::InnerNodes> u_brt_nodes(input_size);
  std::vector<int> u_edge_count(input_size);
  std::vector<int> u_oc_offset(input_size);

  std::generate(u_inputs.begin(), u_inputs.end(), [&] {
    const auto x = dis(gen);
    const auto y = dis(gen);
    const auto z = dis(gen);
    return Eigen::Vector3f(x, y, z);
  });

  std::vector<int> indices(input_size);
  std::iota(indices.begin(), indices.end(), 0);

  const auto [min_coord, max_coord] = FindMinMax(u_inputs);
  const auto range = max_coord - min_coord;
  std::cout << "Min: " << min_coord << "\n";
  std::cout << "Max: " << max_coord << "\n";
  std::cout << "Range: " << range << "\n";

  TimeTask("Compute", [&] {
    // Parrallel
    std::transform(std::execution::par_unseq, u_inputs.begin(), u_inputs.end(),
                   u_sorted_morton_keys.begin(), [&](const auto& input) {
                     return PointToCode(input.x(), input.y(), input.z(),
                                        min_coord, range);
                   });
  });

  TimeTask("Sort", [&] {
    // Parallel sort using std::execution::par
    std::sort(std::execution::par, u_sorted_morton_keys.begin(),
              u_sorted_morton_keys.end());

    u_sorted_morton_keys.erase(
        std::unique(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end()),
        u_sorted_morton_keys.end());
  });

  const auto num_unique_keys = u_sorted_morton_keys.size();
  const auto num_brt_nodes = num_unique_keys - 1;

  TimeTask("Build Radix Tree", [&] {
    std::for_each_n(std::execution::par, indices.begin(), num_brt_nodes,
                    [&](const int i) {
                      ProcessInternalNodesHelper(num_unique_keys,
                                                 u_sorted_morton_keys.data(), i,
                                                 u_brt_nodes.data());
                    });
  });

  TimeTask("Count & Prefix Sum", [&] {
    u_edge_count[0] = 1;
    std::for_each(std::execution::par, indices.begin() + 1,
                  indices.begin() + num_brt_nodes, [&](const int i) {
                    CalculateEdgeCountHelper(i, u_edge_count.data(),
                                             u_brt_nodes.data());
                  });

    std::partial_sum(u_edge_count.begin(), u_edge_count.end(),
                     u_oc_offset.begin() + 1);
    u_oc_offset[0] = 0;
  });

  std::cout << "Unique keys: " << u_sorted_morton_keys.size() << "\n";
  std::cout << "u_oc_offset.back(): " << u_oc_offset.back() << "\n";

  const auto num_oc_nodes = u_oc_offset.back();
  std::vector<oct::OctNode> oc_nodes(num_oc_nodes);

  TimeTask("Octree Nodes", [&] {
    const auto root_level = u_brt_nodes[0].delta_node / 3;
    const Code_t root_prefix =
        u_sorted_morton_keys[0] >> (kCodeLen - (root_level * 3));

    float dec_x, dec_y, dec_z;
    CodeToPoint(root_prefix << (kCodeLen - (root_level * 3)), dec_x, dec_y,
                dec_z, min_coord, range);
    oc_nodes[0].cornor = {dec_x, dec_y, dec_z};
    oc_nodes[0].cell_size = range;

    std::for_each(std::execution::par, indices.begin() + 1,
                  indices.begin() + num_brt_nodes, [&](const int i) {
                    MakeNodesHelper(
                        i, oc_nodes.data(), u_oc_offset.data(),
                        u_edge_count.data(), u_sorted_morton_keys.data(),
                        u_brt_nodes.data(), min_coord, range, root_level);
                  });
    // }
  });

  return 0;
}