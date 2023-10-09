cmake_minimum_required(VERSION 3.10)
project(Tmp LANGUAGES CXX CUDA)

add_executable(Tmp main.cu)

target_compile_features(Tmp PUBLIC cxx_std_14)
target_compile_options(Tmp PUBLIC -arch=sm_75 --expt-relaxed-constexpr)

target_include_directories(Tmp PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Tmp PUBLIC ${CUDA_LIBRARIES})
