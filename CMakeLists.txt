cmake_minimum_required(VERSION 3.1)
project(unity LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++17 -march=native -fpic -fopenmp -ftree-vectorize -fno-rtti" )
set(CMAKE_CXX_FLAGS "-march=native")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

# Build faiss static libraries to reduce compilation time for exampels and tests
add_library(faiss_blas STATIC src/faiss_blas.cc)
add_library(faiss STATIC src/faiss.cc)
add_library(unity STATIC src/unity.cc)

target_link_libraries(faiss faiss_blas)

file(GLOB examples ${CMAKE_CURRENT_LIST_DIR}/examples/*.cc)

foreach(path ${examples})
    get_filename_component(name ${path} NAME_WE)
    add_executable(${name} ${path})
    target_link_libraries(${name} faiss pthread)
endforeach()
