cmake_minimum_required(VERSION 3.10)
project(BenchGpu LANGUAGES CXX CUDA)

add_executable(BenchGpu main.cu)

target_compile_features(BenchGpu PUBLIC cxx_std_17)
target_compile_options(BenchGpu PUBLIC -arch=sm_75 --expt-relaxed-constexpr)
set_target_properties(BenchGpu PROPERTIES CUDA_ARCHITECTURES 75)

target_include_directories(BenchGpu
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(BenchGpu PUBLIC ${CUDA_LIBRARIES})

# Google Benchmark
find_package(benchmark REQUIRED)
target_link_libraries(BenchGpu PUBLIC benchmark::benchmark)

find_package(TBB REQUIRED)
target_link_libraries(BenchGpu PUBLIC TBB::tbb)
