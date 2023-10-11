#pragma once

#include "brt.cuh"

namespace oct {

struct Body {
  float mass;
};

struct OctNode {
  // Payload
  Body body;

  float3 cornor;
  float cell_size;

  // TODO: This is overkill number of pointers
  int children[8];
  // int* children_array;

  /**
   * @brief For bit position i (from the right): If 1, children[i] is the index
   * of a child octree node. If 0, the ith child is either absent, or
   * children[i] is the index of a leaf.
   */
  int child_node_mask;

  /**
   * @brief For bit position i (from the right): If 1, children[i] is the index
   * of a leaf (in the corresponding points array). If 0, the ith child is
   * either absent, or an octree node.
   */
  int child_leaf_mask;

  /**
   * @brief Set a child
   *
   * @param child: index of octree node that will become the child
   * @param my_child_idx: which of my children it will be [0-7]
   */
  __device__ __forceinline__ void SetChild(const int child,
                                           const int my_child_idx) {
    children[my_child_idx] = child;
    // child_node_mask |= (1 << my_child_idx);
    atomicOr(&child_node_mask, 1 << my_child_idx);
  }

  /**
   * @brief Set the Leaf object
   *
   * @param leaf: index of point that will become the leaf child
   * @param my_child_idx: which of my children it will be [0-7]
   */
  __device__ __forceinline__ void SetLeaf(const int leaf,
                                          const int my_child_idx) {
    children[my_child_idx] = leaf;
    // child_leaf_mask |= (1 << my_child_idx);
    atomicOr(&child_leaf_mask, 1 << my_child_idx);
  }
};

// --------------------------------------------------
//         Helpers (per data operations)
// --------------------------------------------------

__host__ __device__ __forceinline__ void
CalculateEdgeCountHelper(const int i, int *edge_count,
                         const brt::InnerNodes *inners) {
  const int my_depth = inners[i].delta_node / 3;
  const int parent_depth = inners[inners[i].parent].delta_node / 3;
  edge_count[i] = my_depth - parent_depth;
}

__device__ __forceinline__ void
MakeNodesHelper(const int i, OctNode *nodes, const int *node_offsets,
                const int *edge_count, const Code_t *morton_keys,
                const brt::InnerNodes *inners,
                const MortonDecoder morton_decoder, const int root_level) {
  int oct_idx = node_offsets[i];
  const int n_new_nodes = edge_count[i];
  const auto tree_range = morton_decoder.range;
  for (int j = 0; j < n_new_nodes - 1; ++j) {
    const int level = inners[i].delta_node / 3 - j;
    const Code_t node_prefix = morton_keys[i] >> (kCodeLen - (3 * level));
    const int child_idx = static_cast<int>(node_prefix & 0b111);
    const int parent = oct_idx + 1;

    nodes[parent].SetChild(oct_idx, child_idx);

    // calculate corner point (LSB have already been shifted off)
    // each cell is half the size of the level above it
    nodes[oct_idx].cornor =
        morton_decoder(node_prefix << (kCodeLen - (3 * level)));
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

    nodes[oct_idx].cornor =
        morton_decoder(top_node_prefix << (kCodeLen - (3 * top_level)));

    nodes[oct_idx].cell_size =
        tree_range / static_cast<float>(1 << (top_level - root_level));
  }
}

} // namespace oct

// --------------------------------------------------
//          Kernels (process all data)
// --------------------------------------------------

__global__ void CalculateEdgeCountKernel(const size_t num_brt_nodes,
                                         int *edge_count,
                                         const brt::InnerNodes *inners) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_brt_nodes) {
    oct::CalculateEdgeCountHelper(i, edge_count, inners);
  }
}

__global__ void MakeNodesKernel(const size_t num_brt_nodes, oct::OctNode *nodes,
                                const int *node_offsets, const int *edge_count,
                                const Code_t *morton_keys,
                                const brt::InnerNodes *inners,
                                const MortonDecoder morton_decoder,
                                const int root_level) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_brt_nodes) {
    oct::MakeNodesHelper(i, nodes, node_offsets, edge_count, morton_keys,
                         inners, morton_decoder, root_level);
  }
}
