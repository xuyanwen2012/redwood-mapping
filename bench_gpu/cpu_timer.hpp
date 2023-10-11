#pragma once

#include <chrono>
#include <iostream>

template <typename Func> void TimeTask(const std::string &task_name, Func &&f) {
  const auto t0 = std::chrono::high_resolution_clock::now();

  std::forward<Func>(f)();

  const auto t1 = std::chrono::high_resolution_clock::now();
  const auto time_span =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  std::cout << "Finished " << task_name << "! Time took: " << time_span.count()
            << "ms. " << std::endl;
}
