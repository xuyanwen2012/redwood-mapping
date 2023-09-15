#pragma once

#include "../Morton.hpp"
#include "../brt/RadixTree.hpp"
#include "Node.hpp"

inline void MakeNodesHelper(const int i, oct::OctNode* nodes,
                            const int* node_offsets, const int* edge_count,
                            const Code_t* morton_keys,
                            const brt::InnerNodes* inners,
                            const float min_coord, const float tree_range,
                            const int root_level) {
  int oct_idx = node_offsets[i];
  const int n_new_nodes = edge_count[i];
  for (int j = 0; j < n_new_nodes - 1; ++j) {
    const int level = inners[i].delta_node / 3 - j;
    const Code_t node_prefix = morton_keys[i] >> (kCodeLen - (3 * level));
    const int child_idx = static_cast<int>(node_prefix & 0b111);
    const int parent = oct_idx + 1;

    nodes[parent].SetChild(oct_idx, child_idx);

    // calculate corner point (LSB have already been shifted off)
    float dec_x, dec_y, dec_z;
    CodeToPoint(node_prefix << (kCodeLen - (3 * level)), dec_x, dec_y, dec_z,
                min_coord, tree_range);
    nodes[oct_idx].cornor = {dec_x, dec_y, dec_z};

    // each cell is half the size of the level above it
    nodes[oct_idx].cell_size =
        tree_range / static_cast<float>(1 << (level - root_level));

    oct_idx = parent;
  }

  if (n_new_nodes > 0) {
    int rt_parent = inners[i].parent;
    while (edge_count[rt_parent] == 0) {
      rt_parent = inners[rt_parent].parent;
    }
    const int oct_parent = node_offsets[rt_parent];
    const int top_level = inners[i].delta_node / 3 - n_new_nodes + 1;
    const Code_t top_node_prefix =
        morton_keys[i] >> (kCodeLen - (3 * top_level));
    const int child_idx = static_cast<int>(top_node_prefix & 0b111);

    nodes[oct_parent].SetChild(oct_idx, child_idx);

    float dec_x, dec_y, dec_z;
    CodeToPoint(top_node_prefix << (kCodeLen - (3 * top_level)), dec_x, dec_y,
                dec_z, min_coord, tree_range);
    nodes[oct_idx].cornor = {dec_x, dec_y, dec_z};

    nodes[oct_idx].cell_size =
        tree_range / static_cast<float>(1 << (top_level - root_level));
  }
}

/**
 * @brief Link the octree nodes together.
 *
 * @param nodes
 * @param node_offsets
 * @param edge_count
 * @param sorted_morton
 * @param brt_nodes
 * @param num_brt_nodes
 */
void LinkOctreeNodes(oct::OctNode* nodes, const int* node_offsets,
                     const int* edge_count, const Code_t* sorted_morton,
                     const brt::InnerNodes* brt_nodes, size_t num_brt_nodes);
