#ifndef STENCIL_KERNEL_CUH
#define STENCIL_KERNEL_CUH

#include <cuda_runtime.h>

constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

__global__ void evolve_interior_stencil_kernel(
    double* __restrict__ curr_data,
    const double* __restrict__ prev_data,
    int nx, int ny, int nz,
    double alpha, double dt,
    double inv_dx2, double inv_dy2, double inv_dz2
);

__global__ void evolve_tiled_2d5_stencil_kernel(
    double* __restrict__ curr_data,
    const double* __restrict__ prev_data,
    int nx, int ny, int nz,
    double alpha, double dt,
    double inv_dx2, double inv_dy2, double inv_dz2
);

void launch_evolve_interior(
    double* curr_data,
    const double* prev_data,
    int nx, int ny, int nz,
    double alpha, double dt,
    double dx, double dy, double dz,
    cudaStream_t stream = 0
);

#endif // STENCIL_KERNEL_CUH
