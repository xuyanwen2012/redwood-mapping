// #pragma once

// #include <cuda_runtime.h>

// #include "CudaUtils.cuh"
// #include "Morton.hpp"

// __device__ __host__ __forceinline__ int Sign(const int val) noexcept {
//   return (val > 0) - (val < 0);
// }

// __device__ __host__ __forceinline__ int Divide2Ceil(const int x) noexcept {
//   return (x + 1) >> 1;
// }

// namespace brt_cuda {

// __device__ __host__ __forceinline__ int MakeLeaf(const int index) noexcept {
//   return index ^ ((-1 ^ index) & 1UL << (sizeof(int) * 8 - 1));
// }

// __device__ __host__ __forceinline__ int MakeInternal(const int index) noexcept {
//   return index;
// }

// struct InnerNodes {
//   // The number of bits in the morton code, this node represents in [Karras]
//   int delta_node;

//   // pointers
//   int left = -1;  // can be either inner or leaf
//   int right = -1;
//   int parent = -1;
// };

// __device__ __forceinline__ int Delta(const Code_t* morton_keys, const int i,
//                                      const int j) {
//   constexpr auto unused_bits = 1;
//   const auto li = morton_keys[i];
//   const auto lj = morton_keys[j];
//   return __clzll(li ^ lj) - unused_bits;
// }

// __device__ __forceinline__ int DeltaSafe(const int key_num,
//                                          const Code_t* morton_keys, const int i,
//                                          const int j) noexcept {
//   return (j < 0 || j >= key_num) ? -1 : Delta(morton_keys, i, j);
// }

// __device__ __forceinline__ void ProcessInternalNodesHelper(
//     const int key_num, const Code_t* morton_keys, const int i,
//     InnerNodes* brt_cuda_nodes) {
//   if (i == 0) printf("i == 0\n");

//   const auto direction{Sign(Delta(morton_keys, i, i + 1) -
//                             DeltaSafe(key_num, morton_keys, i, i - 1))};

//   const auto delta_min{DeltaSafe(key_num, morton_keys, i, i - direction)};

//   int I_max{2};
//   while (DeltaSafe(key_num, morton_keys, i, i + I_max * direction) > delta_min)
//     I_max <<= 2;  // aka, *= 2

//   // Find the other end using binary search.
//   int I{0};
//   for (int t{I_max / 2}; t; t /= 2)
//     if (DeltaSafe(key_num, morton_keys, i, i + (I + t) * direction) > delta_min)
//       I += t;

//   const int j{i + I * direction};

//   // Find the split position using binary search.
//   const auto delta_node{DeltaSafe(key_num, morton_keys, i, j)};
//   auto s{0};

//   int t{I};
//   do {
//     t = Divide2Ceil(t);
//     if (DeltaSafe(key_num, morton_keys, i, i + (s + t) * direction) >
//         delta_node)
//       s += t;
//   } while (t > 1);

//   const auto split{i + s * direction + min(direction, 0)};

//   const int left{min(i, j) == split ? MakeLeaf(split) : MakeInternal(split)};
//   const int right{max(i, j) == split + 1 ? MakeLeaf(split + 1)
//                                          : MakeInternal(split + 1)};

//   brt_cuda_nodes[i].delta_node = delta_node;
//   brt_cuda_nodes[i].left = left;
//   brt_cuda_nodes[i].right = right;

//   if (min(i, j) != split) brt_cuda_nodes[left].parent = i;
//   if (max(i, j) != split + 1) brt_cuda_nodes[right].parent = i;
// }

// }  // namespace brt_cuda