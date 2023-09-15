#include <omp.h>  // Include OpenMP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <execution>
#include <iostream>
#include <random>
#include <thread>

#include "brt/RadixTree.hpp"

template <uint8_t Axis>
bool CompareAxis(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  if constexpr (Axis == 0) {
    return a.x() < b.x();
  } else if constexpr (Axis == 1) {
    return a.y() < b.y();
  }
  return a.z() < b.z();
}

void ProcessInput() {}

void UpdateGame() {}

// Function to render the game
void Render() { std::cout << "Rendering..." << std::endl; }

int main() {
  constexpr auto min_coord = 0.0f;
  constexpr auto range = 1024.0f;

  thread_local std::mt19937 gen(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> dis(min_coord, range);

  constexpr auto input_size = 1280 * 720;

  std::vector<Eigen::Vector3f> u_inputs(input_size);
  //   std::vector<Code_t> u_morton_keys(input_size);
  std::vector<Code_t> u_sorted_morton_keys(input_size);
  std::vector<brt::InnerNodes> u_brt_nodes(input_size);

  constexpr auto targetFPS = 30;
  const std::chrono::milliseconds frameDuration(1000 / targetFPS);

  while (true) {
    auto startTime = std::chrono::high_resolution_clock::now();

    // Your game logic goes here

    std::generate(u_inputs.begin(), u_inputs.end(), [&] {
      const auto x = dis(gen);
      const auto y = dis(gen);
      const auto z = dis(gen);
      return Eigen::Vector3f(x, y, z);
    });

    // Parrallel
    std::transform(std::execution::par, u_inputs.begin(), u_inputs.end(),
                   u_sorted_morton_keys.begin(), [&](const auto& input) {
                     return PointToCode(input.x(), input.y(), input.z(),
                                        min_coord, range);
                   });

    // Parallel sort using std::execution::par
    std::sort(std::execution::par, u_sorted_morton_keys.begin(),
              u_sorted_morton_keys.end());

    u_sorted_morton_keys.erase(
        std::unique(u_sorted_morton_keys.begin(), u_sorted_morton_keys.end()),
        u_sorted_morton_keys.end());

    const auto num_unique_keys = u_sorted_morton_keys.size();
    const auto num_brt_nodes = num_unique_keys - 1;

#pragma omp parallel for
    for (int i = 0; i < num_brt_nodes; ++i) {
      ProcessInternalNodesHelper(num_unique_keys, u_sorted_morton_keys.data(),
                                 i, u_brt_nodes.data());
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);

    // Calculate the actual FPS
    double fps = 1000.0 / elapsedTime.count();

    std::cout << "FPS: " << elapsedTime.count() << std::endl;

    // Sleep to maintain the target FPS
    if (elapsedTime < frameDuration) {
      std::this_thread::sleep_for(frameDuration - elapsedTime);
    }
  }

  //   std::generate(u_inputs.begin(), u_inputs.end(), [&] {
  //     const auto x = dis(gen);
  //     const auto y = dis(gen);
  //     const auto z = dis(gen);
  //     return Eigen::Vector3f(x, y, z);
  //   });

  //   TimeTask("Compute & Sort", [&] {
  //     // Parrallel
  //     std::transform(std::execution::par_unseq, u_inputs.begin(),
  //     u_inputs.end(),
  //                    u_sorted_morton_keys.begin(), [&](const auto& input) {
  //                      return PointToCode(input.x(), input.y(), input.z(),
  //                                         min_coord, range);
  //                    });

  //     // Parallel sort using std::execution::par
  //     std::sort(std::execution::par, u_sorted_morton_keys.begin(),
  //               u_sorted_morton_keys.end());

  //     u_sorted_morton_keys.erase(
  //         std::unique(u_sorted_morton_keys.begin(),
  //         u_sorted_morton_keys.end()), u_sorted_morton_keys.end());
  //   });

  //   TimeTask("Build Tree", [&] {
  //     const auto num_unique_keys = u_sorted_morton_keys.size();
  //     const auto num_brt_nodes = num_unique_keys - 1;

  // #pragma omp parallel for
  //     for (int i = 0; i < num_brt_nodes; ++i) {
  //       ProcessInternalNodesHelper(num_unique_keys,
  //       u_sorted_morton_keys.data(),
  //                                  i, u_brt_nodes.data());
  //     }
  //   });

  //   std::cout << "Unique keys: " << u_sorted_morton_keys.size() << "\n";

  return 0;
}