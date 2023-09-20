#pragma once

#include <Eigen/Dense>
#include <algorithm>

#include "Utils.hpp"

template <uint8_t Axis>
bool CompareAxis(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  }
  return a.z() < b.z();
}

/**
 * @brief Find the minimum and maximum coordinate of the input points
 *
 * @param u_inputs point cloud
 * @return std::pair<float, float> min and max
 */
_NODISCARD inline std::pair<float, float> FindMinMax(
    const std::vector<Eigen::Vector3f>& u_inputs) {
  float min_coord = 0.0f;
  float max_coord = 1.0f;

  const auto x_range =
      std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<0>);
  const auto y_range =
      std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<1>);
  const auto z_range =
      std::minmax_element(u_inputs.begin(), u_inputs.end(), CompareAxis<2>);
  const auto x_min = x_range.first;
  const auto x_max = x_range.second;
  const auto y_min = y_range.first;
  const auto y_max = y_range.second;
  const auto z_min = z_range.first;
  const auto z_max = z_range.second;
  std::array<float, 3> mins{x_min->x(), y_min->y(), z_min->z()};
  std::array<float, 3> maxes{x_max->x(), y_max->y(), z_max->z()};
  min_coord = *std::min_element(mins.begin(), mins.end());
  max_coord = *std::max_element(maxes.begin(), maxes.end());

  return {min_coord, max_coord};
}
