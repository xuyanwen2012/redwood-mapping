#pragma once

#include "../brt/RadixTree.hpp"

inline void CalculateEdgeCountHelper(const int i, int* edge_count,
                                     const brt::InnerNodes* inners) {
  const int my_depth = inners[i].delta_node / 3;
  const int parent_depth = inners[inners[i].parent].delta_node / 3;
  edge_count[i] = my_depth - parent_depth;
}

_NODISCARD inline int CalculateEdgeCountHelper_v2(
    const int i, const brt::InnerNodes* inners) {
  const int my_depth = inners[i].delta_node / 3;
  const int parent_depth = inners[inners[i].parent].delta_node / 3;
  return my_depth - parent_depth;
}

inline void CalculateEdgeCountKernel(int* edge_count,
                                     const brt::InnerNodes* inners,
                                     const int num_brt_nodes) {
  // root has no parent, so don't do for index 0
  for (int i = 1; i < num_brt_nodes; ++i) {
    CalculateEdgeCountHelper(i, edge_count, inners);
  }
}
