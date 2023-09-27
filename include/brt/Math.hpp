// #pragma once

// namespace brt {

// namespace math {
// template <typename T>
// constexpr int sign(T val) {
//   return (T(0) < val) - (val < T(0));
// }

// template <typename T>
// constexpr T min(const T& x, const T& y) {
//   return y ^ ((x ^ y) & -(x < y));
// }

// template <typename T>
// constexpr T max(const T& x, const T& y) {
//   return x ^ ((x ^ y) & -(x < y));
// }

// template <typename T>
// constexpr int divide_ceil(const T& x, const T& y) {
//   return (x + y - 1) / y;
// }

// /** Integer division by two, rounding up */
// template <typename T>
// constexpr int divide2_ceil(const T& x) {
//   return (x + 1) >> 1;
// }
// }  // namespace math

// }  // namespace brt
