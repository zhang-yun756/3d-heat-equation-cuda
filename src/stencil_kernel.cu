#include "stencil_kernel.cuh"
#include "memory_manager.hpp"
#include <cuda_runtime.h>

__global__ void evolve_interior_stencil_kernel(
    double* __restrict__ curr_data,
    const double* __restrict__ prev_data,
    int nx, int ny, int nz,
    double alpha, double dt,
    double inv_dx2, double inv_dy2, double inv_dz2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int ny_ghost = ny + 2;
    int nz_ghost = nz + 2;

    if (i <= nx && j <= ny && k <= nz) {
        size_t center = static_cast<size_t>(i) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + k;
        size_t ip = static_cast<size_t>(i + 1) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + k;
        size_t im = static_cast<size_t>(i - 1) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + k;
        size_t jp = static_cast<size_t>(i) * ny_ghost * nz_ghost + static_cast<size_t>(j + 1) * nz_ghost + k;
        size_t jm = static_cast<size_t>(i) * ny_ghost * nz_ghost + static_cast<size_t>(j - 1) * nz_ghost + k;
        size_t kp = static_cast<size_t>(i) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + (k + 1);
        size_t km = static_cast<size_t>(i) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + (k - 1);

        double val = prev_data[center];
        double d2x = (prev_data[ip] - 2.0 * val + prev_data[im]) * inv_dx2;
        double d2y = (prev_data[jp] - 2.0 * val + prev_data[jm]) * inv_dy2;
        double d2z = (prev_data[kp] - 2.0 * val + prev_data[km]) * inv_dz2;

        curr_data[center] = val + alpha * dt * (d2x + d2y + d2z);
    }
}

__global__ void evolve_tiled_2d5_stencil_kernel(
    double* __restrict__ curr_data,
    const double* __restrict__ prev_data,
    int nx, int ny, int nz,
    double alpha, double dt,
    double inv_dx2, double inv_dy2, double inv_dz2)
{
    __shared__ double s_tile[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * BLOCK_DIM_X + tx + 1;
    int gy = blockIdx.y * BLOCK_DIM_Y + ty + 1;

    int ny_ghost = ny + 2;
    int nz_ghost = nz + 2;

    if (gx > nx || gy > ny) return;

    double r_prev = 0.0;
    double r_curr = 0.0;
    double r_next = 0.0;

    size_t base_idx = static_cast<size_t>(gy) * nz_ghost + gx;
    size_t stride_z = static_cast<size_t>(ny_ghost) * nz_ghost;

    if (1 <= nz) {
        r_curr = prev_data[base_idx + 0 * stride_z];
        r_next = prev_data[base_idx + 1 * stride_z];
    }

    for (int k = 1; k <= nz; ++k) {
        r_prev = r_curr;
        r_curr = r_next;
        r_next = prev_data[base_idx + static_cast<size_t>(k + 1) * stride_z];

        s_tile[ty + 1][tx + 1] = r_curr;

        if (tx == 0) {
            size_t idx_left = static_cast<size_t>(gy) * nz_ghost + (gx - 1) + static_cast<size_t>(k) * stride_z;
            s_tile[ty + 1][0] = prev_data[idx_left];
        }
        if (tx == BLOCK_DIM_X - 1 || gx == nx) {
            size_t idx_right = static_cast<size_t>(gy) * nz_ghost + (gx + 1) + static_cast<size_t>(k) * stride_z;
            s_tile[ty + 1][tx + 2] = prev_data[idx_right];
        }
        if (ty == 0) {
            size_t idx_top = static_cast<size_t>(gy - 1) * nz_ghost + gx + static_cast<size_t>(k) * stride_z;
            s_tile[0][tx + 1] = prev_data[idx_top];
        }
        if (ty == BLOCK_DIM_Y - 1 || gy == ny) {
            size_t idx_bottom = static_cast<size_t>(gy + 1) * nz_ghost + gx + static_cast<size_t>(k) * stride_z;
            s_tile[ty + 2][tx + 1] = prev_data[idx_bottom];
        }

        __syncthreads();

        double d2x = (s_tile[ty + 1][tx + 2] - 2.0 * r_curr + s_tile[ty + 1][tx]) * inv_dx2;
        double d2y = (s_tile[ty + 2][tx + 1] - 2.0 * r_curr + s_tile[ty][tx + 1]) * inv_dy2;
        double d2z = (r_next - 2.0 * r_curr + r_prev) * inv_dz2;

        size_t write_idx = base_idx + static_cast<size_t>(k) * stride_z;
        curr_data[write_idx] = r_curr + alpha * dt * (d2x + d2y + d2z);

        __syncthreads();
    }
}

void launch_evolve_interior(
    double* curr_data,
    const double* prev_data,
    int nx, int ny, int nz,
    double alpha, double dt,
    double dx, double dy, double dz,
    cudaStream_t stream)
{
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = 1.0 / (dz * dz);

    dim3 block_dim(16, 8, 8);
    dim3 grid_dim(
        (nz + block_dim.x - 1) / block_dim.x,
        (ny + block_dim.y - 1) / block_dim.y,
        (nx + block_dim.z - 1) / block_dim.z
    );

    evolve_interior_stencil_kernel<<<grid_dim, block_dim, 0, stream>>>(
        curr_data, prev_data, nx, ny, nz, alpha, dt, inv_dx2, inv_dy2, inv_dz2
    );
    CUDA_CHECK(cudaGetLastError());
}
