cmake_minimum_required (VERSION 3.1)
project(top LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++17 -march=native -fpic -fopenmp -ftree-vectorize -fno-rtti" )

include_directories(${CMAKE_CURRENT_LIST_DIR})
add_executable(main examples/main.cc)

