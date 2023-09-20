cmake_minimum_required(VERSION 3.16)
project(redwood_map LANGUAGES CXX)

add_library(redwood_map)

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/include/*.hpp")

target_sources(
  redwood_map
  PRIVATE Test.cpp
  PUBLIC ${HEADER_LIST})

target_include_directories(redwood_map PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_features(redwood_map PUBLIC cxx_std_14)

# Add TBB
find_package(TBB REQUIRED)
target_link_libraries(redwood_map PRIVATE TBB::tbb)

# Add Eigen
find_package(Eigen3 REQUIRED)
target_include_directories(redwood_map PUBLIC ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
