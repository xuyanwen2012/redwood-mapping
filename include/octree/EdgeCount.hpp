#pragma once

#include "../brt/RadixTree.hpp"

inline void CalculateEdgeCountHelper(const int i, int* edge_count,
                                     const brt::InnerNodes* inners) {
  const int my_depth = inners[i].delta_node / 3;
  const int parent_depth = inners[inners[i].parent].delta_node / 3;
  edge_count[i] = my_depth - parent_depth;
}
