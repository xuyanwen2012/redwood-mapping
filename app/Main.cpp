#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <execution>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "Common.hpp"
#include "Statistics.hpp"
#include "brt/RadixTree.hpp"
#include "octree/EdgeCount.hpp"
#include "octree/Octree.hpp"

// #define policy_t std::execution::seq
#define policy_t std::execution::par

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

  PrintMemoryUsage(VectorInfo<Eigen::Vector3f>{u_inputs, "Input Cloud"},
                   VectorInfo<Code_t>{u_sorted_morton_keys, "Morton Keys"},
                   VectorInfo<brt::InnerNodes>{u_brt_nodes, "Brt Nodes"},
                   VectorInfo<int>{u_edge_count, "Edge Counts"},
                   VectorInfo<int>{u_oc_offset, "Oct Offset"});

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
    std::transform(policy_t, u_inputs.begin(), u_inputs.end(),
                   u_sorted_morton_keys.begin(), [&](const auto& input) {
                     return PointToCode(input.x(), input.y(), input.z(),
                                        min_coord, range);
                   });
  });

  std::vector<Code_t>::iterator last_unique_it;

  TimeTask("Sort", [&] {
    std::sort(policy_t, u_sorted_morton_keys.begin(),
              u_sorted_morton_keys.end());

    last_unique_it = std::unique(policy_t, u_sorted_morton_keys.begin(),
                                 u_sorted_morton_keys.end());
  });

  const auto num_unique_keys =
      std::distance(u_sorted_morton_keys.begin(), last_unique_it);
  const auto num_brt_nodes = num_unique_keys - 1;

  TimeTask("Build Radix Tree", [&] {
    std::for_each_n(policy_t, indices.begin(), num_brt_nodes, [&](const int i) {
      ProcessInternalNodesHelper(num_unique_keys, u_sorted_morton_keys.data(),
                                 i, u_brt_nodes.data());
    });
  });

  TimeTask("Count & Prefix Sum", [&] {
    u_edge_count[0] = 1;
    std::for_each(policy_t, indices.begin() + 1,
                  indices.begin() + num_brt_nodes, [&](const int i) {
                    CalculateEdgeCountHelper(i, u_edge_count.data(),
                                             u_brt_nodes.data());
                  });

    // Can't parallel this
    std::partial_sum(u_edge_count.begin(), u_edge_count.end(),
                     u_oc_offset.begin() + 1);
    u_oc_offset[0] = 0;
  });

  std::cout << "Unique keys: " << num_unique_keys << "\n";
  std::cout << "u_oc_offset.back(): " << u_oc_offset.back() << "\n";

  const auto num_oc_nodes = u_oc_offset.back();
  std::vector<oct::OctNode> oc_nodes;  //(num_oc_nodes);

  TimeTask("Allocate Octree Nodes", [&] { oc_nodes.resize(num_oc_nodes); });

  PrintMemoryUsage(VectorInfo<oct::OctNode>{oc_nodes, "Octree Nodes"});

  Code_t root_prefix;
  int root_level;
  TimeTask("Octree Nodes", [&] {
    root_level = u_brt_nodes[0].delta_node / 3;
    root_prefix = u_sorted_morton_keys[0] >> (kCodeLen - (root_level * 3));

    float dec_x, dec_y, dec_z;
    CodeToPoint(root_prefix << (kCodeLen - (root_level * 3)), dec_x, dec_y,
                dec_z, min_coord, range);
    oc_nodes[0].cornor = {dec_x, dec_y, dec_z};
    oc_nodes[0].cell_size = range;

    std::for_each(policy_t, indices.begin() + 1,
                  indices.begin() + num_brt_nodes, [&](const int i) {
                    MakeNodesHelper(
                        i, oc_nodes.data(), u_oc_offset.data(),
                        u_edge_count.data(), u_sorted_morton_keys.data(),
                        u_brt_nodes.data(), min_coord, range, root_level);
                  });
  });

  TimeTask("Link Nodes", [&] {
    std::for_each(policy_t, indices.begin(), indices.begin() + num_brt_nodes,
                  [&](const int i) {
                    LinkNodesHelper(i, oc_nodes.data(), u_oc_offset.data(),
                                    u_edge_count.data(),
                                    u_sorted_morton_keys.data(),
                                    u_brt_nodes.data());
                  });
  });

  CheckTree(root_prefix, root_level * 3, oc_nodes.data(), 0,
            u_sorted_morton_keys.data());

  return 0;
}