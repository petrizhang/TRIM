cmake_minimum_required(VERSION 3.1)
project(top LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++17 -march=native -fpic -fopenmp -ftree-vectorize -fno-rtti" )

include_directories(${CMAKE_CURRENT_LIST_DIR}
        /data/home/petrizhang/.conda/envs/faiss/lib/python3.12/site-packages/pybind11/include
)
add_executable(hnsw examples/hnsw.cc)
target_link_libraries(hnsw pthread)

add_executable(pq examples/pq.cc)
target_link_libraries(pq pthread)