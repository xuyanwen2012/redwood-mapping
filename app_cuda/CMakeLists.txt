cmake_minimum_required(VERSION 3.16)
project(MyAppCu LANGUAGES CXX)

add_executable(MyAppCu Main.cpp)

# target_compile_features(MyAppCu PRIVATE cxx_std_17)

target_include_directories(MyAppCu PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(MyAppCu PRIVATE redwood_map)

# CUDA target_compile_features(MyCudaApp PRIVATE cxx_std_14)

target_compile_options(MyAppCu PRIVATE -arch=sm_75 --expt-relaxed-constexpr)
set_target_properties(MyAppCu PROPERTIES CUDA_ARCHITECTURES 87)
target_include_directories(MyAppCu
                           PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(MyAppCu PUBLIC ${CUDA_LIBRARIES})
