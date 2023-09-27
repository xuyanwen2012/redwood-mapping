#pragma once

#include <cassert>

#include "../Intrinsics.hpp"
// #include "Math.hpp"
#include "Morton.hpp"
#include "Node.hpp"

namespace brt {

struct InnerNodes {
  // The number of bits in the morton code, this node represents in [Karras]
  int delta_node;

  // pointers
  int left = -1;  // can be either inner or leaf
  int right = -1;
  int parent = -1;
};

inline int Delta(const Code_t* morton_keys, const int i, const int j) {
  constexpr auto unused_bits = 1;
  const auto li = morton_keys[i];
  const auto lj = morton_keys[j];
  return CLZ64(li ^ lj) - unused_bits;
}

inline int DeltaSafe(const int key_num, const Code_t* morton_keys, const int i,
                     const int j) noexcept {
  return (j < 0 || j >= key_num) ? -1 : Delta(morton_keys, i, j);
}

inline void ProcessInternalNodesHelper(const int key_num,
                                       const Code_t* morton_keys, const int i,
                                       InnerNodes* brt_nodes) {
  const auto direction{
      brt::math::sign<int>(Delta(morton_keys, i, i + 1) -
                           DeltaSafe(key_num, morton_keys, i, i - 1))};
  assert(direction != 0);

  const auto delta_min{DeltaSafe(key_num, morton_keys, i, i - direction)};

  int I_max{2};
  while (DeltaSafe(key_num, morton_keys, i, i + I_max * direction) > delta_min)
    I_max <<= 2;  // aka, *= 2

  // Find the other end using binary search.
  int I{0};
  for (int t{I_max / 2}; t; t /= 2)
    if (DeltaSafe(key_num, morton_keys, i, i + (I + t) * direction) > delta_min)
      I += t;

  const int j{i + I * direction};

  // Find the split position using binary search.
  const auto delta_node{DeltaSafe(key_num, morton_keys, i, j)};
  auto s{0};

  int t{I};
  do {
    t = math::divide2_ceil<int>(t);
    if (DeltaSafe(key_num, morton_keys, i, i + (s + t) * direction) >
        delta_node)
      s += t;
  } while (t > 1);

  const auto split{i + s * direction + math::min<int>(direction, 0)};

  // sanity check
  assert(Delta(morton_keys, i, j) > delta_min);
  assert(Delta(morton_keys, split, split + 1) == Delta(morton_keys, i, j));
  assert(!(split < 0 || split + 1 >= key_num));

  const int left{math::min<int>(i, j) == split
                     ? node::make_leaf<int>(split)
                     : node::make_internal<int>(split)};

  const int right{math::max<int>(i, j) == split + 1
                      ? node::make_leaf<int>(split + 1)
                      : node::make_internal<int>(split + 1)};

  brt_nodes[i].delta_node = delta_node;
  brt_nodes[i].left = left;
  brt_nodes[i].right = right;

  if (math::min<int>(i, j) != split) brt_nodes[left].parent = i;
  if (math::max<int>(i, j) != split + 1) brt_nodes[right].parent = i;
}

}  // namespace brt
