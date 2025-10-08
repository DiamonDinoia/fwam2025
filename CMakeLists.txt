cmake_minimum_required(VERSION 3.10)
project(simd101)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Fetch nanobench
FetchContent_Declare(
  nanobench
  GIT_REPOSITORY https://github.com/martinus/nanobench.git
  GIT_TAG        v4.3.11
)
FetchContent_MakeAvailable(nanobench)

# Fetch xsimd
FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(xsimd)

add_executable(simd101 main.cpp)
target_include_directories(simd101 PRIVATE include)
target_link_libraries(simd101 PRIVATE nanobench xsimd)
target_compile_options(simd101 PRIVATE -march=native
)
