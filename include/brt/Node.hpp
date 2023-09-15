#pragma once

namespace brt {

namespace node {
template <typename T>
constexpr T make_leaf(const T& index) {
  return index ^ ((-1 ^ index) & 1UL << (sizeof(T) * 8 - 1));
}

template <typename T>
constexpr T make_internal(const T& index) {
  return index;
}
}  // namespace node

}  // namespace brt
