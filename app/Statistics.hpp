#pragma once

#include <iomanip>
#include <iostream>
#include <vector>

template <typename T>
struct VectorInfo {
  std::vector<T>& data;
  std::string name;
};

template <typename... T>
void PrintMemoryUsage(const VectorInfo<T>&... vectors) {
  size_t total_memory_used = 0;
  (..., (total_memory_used += vectors.data.size() * sizeof(T)));

  // Convert total_memory_used to megabytes
  double total_memory_mb =
      static_cast<double>(total_memory_used) / (1024 * 1024);

  // Print table header
  std::cout << std::left << std::setw(30) << "Vector Name" << std::setw(20)
            << "Memory Used (MB)" << std::setw(15) << "Percentage (%)"
            << std::endl;
  std::cout << std::setw(30) << "-----------" << std::setw(20)
            << "-----------------" << std::setw(15) << "------------"
            << std::endl;

  // Calculate and print memory usage in megabytes and percentages for each
  // vector
  (..., (std::cout << std::setw(30) << vectors.name << std::fixed
                   << std::setprecision(2) << std::setw(20)
                   << (static_cast<double>(vectors.data.size() * sizeof(T)) /
                       (1024 * 1024))
                   << std::fixed << std::setprecision(2) << std::setw(15)
                   << (static_cast<double>(vectors.data.size() * sizeof(T)) /
                       total_memory_used) *
                          100
                   << std::endl));

  // Print total memory used in megabytes
  std::cout << std::setw(30) << "Total Memory Used" << std::fixed
            << std::setprecision(2) << std::setw(20) << total_memory_mb
            << std::setw(15) << "100.00" << '\n'
            << std::endl;
}
