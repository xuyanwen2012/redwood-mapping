cmake_minimum_required(VERSION 3.10)
project(BenchGpu LANGUAGES CXX CUDA)

find_package(benchmark REQUIRED)
# find_package(TBB REQUIRED)

add_executable(BenchGpu benchmark.cu)

target_compile_features(BenchGpu PUBLIC cxx_std_17)
target_compile_options(BenchGpu PUBLIC -arch=sm_75 --expt-relaxed-constexpr)
set_target_properties(BenchGpu PROPERTIES CUDA_ARCHITECTURES 75)
target_include_directories(BenchGpu
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(BenchGpu PUBLIC ${CUDA_LIBRARIES} benchmark::benchmark)

# TMP

add_executable(Tmp tmp.cu)
target_compile_features(Tmp PUBLIC cxx_std_17)
target_compile_options(Tmp PUBLIC -arch=sm_75 --expt-relaxed-constexpr)
set_target_properties(Tmp PROPERTIES CUDA_ARCHITECTURES 75)
target_include_directories(Tmp PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Tmp PUBLIC ${CUDA_LIBRARIES})
