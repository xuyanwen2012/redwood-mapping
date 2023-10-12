cmake_minimum_required(VERSION 3.10)
project(BenchCpu LANGUAGES CXX)

find_package(benchmark REQUIRED)
find_package(TBB REQUIRED)

add_executable(BenchCpu benchmark.cpp)

target_compile_features(BenchCpu PUBLIC cxx_std_17)
target_link_libraries(BenchCpu PUBLIC benchmark::benchmark TBB::tbb)
