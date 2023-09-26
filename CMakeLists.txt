cmake_minimum_required(VERSION 3.16)
project(Redwood LANGUAGES CXX)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(app_cuda)
add_subdirectory(benchmarks)
