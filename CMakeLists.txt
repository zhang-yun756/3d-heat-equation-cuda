cmake_minimum_required(VERSION 3.18)
project(HeatEquation3DCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall>)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_PATH}
)

set(SOURCES
    src/main.cpp
    src/stencil_kernel.cu
    src/cuda_graph_solver.cpp
)

set(HEADERS
    include/memory_manager.hpp
    include/field.hpp
    include/heat_solver.hpp
    include/stencil_kernel.cuh
    include/cuda_graph_solver.hpp
)

add_executable(heat_solver_cuda ${SOURCES} ${HEADERS})

target_link_libraries(heat_solver_cuda
    PRIVATE
        MPI::MPI_CXX
        CUDA::cudart
)

set_target_properties(heat_solver_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
