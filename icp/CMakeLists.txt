cmake_minimum_required(VERSION 3.16)
project(ICP LANGUAGES CXX)

add_executable(ICP Main.cpp PointCloud.cpp PointCloud.hpp SimpleICP.hpp SimpleICP.cpp)

target_compile_features(ICP PRIVATE cxx_std_17)

# Add Eigen
find_package(Eigen3 REQUIRED)
target_include_directories(ICP PUBLIC ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
