cmake_minimum_required(VERSION 3.16)
project(MyAppCuda LANGUAGES CXX CUDA)

add_executable(MyAppCuda Main.cu Usm.cu)

set(CMAKE_CXX_STANDARD 14)

target_include_directories(MyAppCuda PRIVATE ${CMAKE_SOURCE_DIR}/include)

# CUDA
target_compile_options(MyAppCuda PRIVATE -arch=sm_75 --expt-relaxed-constexpr)
set_target_properties(MyAppCuda PROPERTIES CUDA_ARCHITECTURES "70;75;80;87")
target_include_directories(MyAppCuda
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(MyAppCuda PUBLIC ${CUDA_LIBRARIES})

# Add Eigen
find_package(Eigen3 REQUIRED)
target_include_directories(MyAppCuda PUBLIC ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
# Ok so using Eigen + CUDA has a bunch of warnings, so we suppress them
target_compile_options(
  MyAppCuda PRIVATE -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored)
