cmake_minimum_required(VERSION 3.16)
project(MyApp LANGUAGES CXX)

add_executable(MyApp Main.cpp)

target_compile_features(MyApp PRIVATE cxx_std_17)

target_include_directories(MyApp PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(MyApp PRIVATE redwood_map)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(MyApp PUBLIC OpenMP::OpenMP_CXX)
endif()
