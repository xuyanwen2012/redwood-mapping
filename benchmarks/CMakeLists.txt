cmake_minimum_required(VERSION 3.16)
project(my_bench LANGUAGES CXX)

file(GLOB BENCH_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add Google Benchmark
find_package(benchmark REQUIRED)

foreach(testsourcefile ${BENCH_SOURCE})
  get_filename_component(testname ${testsourcefile} NAME_WE)

  add_executable(${testname} ${testsourcefile})

  target_compile_features(${testname} PRIVATE cxx_std_17)
  target_link_libraries(${testname} PRIVATE redwood_map benchmark::benchmark)

endforeach(testsourcefile ${BENCH_SOURCE})
