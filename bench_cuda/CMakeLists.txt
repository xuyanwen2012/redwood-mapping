cmake_minimum_required(VERSION 3.16)
project(MyCudaApp LANGUAGES CXX CUDA)

add_executable(MyCudaApp Main.cu)

set(CMAKE_CXX_STANDARD 17)

# target_compile_features(MyCudaApp PRIVATE cxx_std_14)
target_include_directories(MyCudaApp PRIVATE ${CMAKE_SOURCE_DIR}/include)

# CUDA
target_compile_options(MyCudaApp PRIVATE -arch=sm_75 --expt-relaxed-constexpr)
set_target_properties(MyCudaApp PROPERTIES CUDA_ARCHITECTURES 87)
target_include_directories(MyCudaApp
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(MyCudaApp PUBLIC ${CUDA_LIBRARIES})

# Add Google Benchmark
find_package(benchmark REQUIRED)
target_link_libraries(MyCudaApp PRIVATE benchmark::benchmark)
