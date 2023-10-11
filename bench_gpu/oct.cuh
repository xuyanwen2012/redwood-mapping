#pragma once

#include "brt.cuh"

namespace oct {

__host__ __device__ __forceinline__ void
CalculateEdgeCountHelper(const int i, int *edge_count,
                         const brt::InnerNodes *inners) {
  const int my_depth = inners[i].delta_node / 3;
  const int parent_depth = inners[inners[i].parent].delta_node / 3;
  edge_count[i] = my_depth - parent_depth;
}

__global__ void CalculateEdgeCountKernel(const int num_nodes, int *edge_count,
                                         const brt::InnerNodes *inners) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_nodes) {
    CalculateEdgeCountHelper(i, edge_count, inners);
  }
}

} // namespace oct