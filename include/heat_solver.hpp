#ifndef HEAT_SOLVER_HPP
#define HEAT_SOLVER_HPP

#include "field.hpp"
#include "stencil_kernel.cuh"
#include "cuda_graph_solver.hpp"
#include <mpi.h>
#include <iostream>
#include <utility>

struct SolverConfig {
    int nx = 256;
    int ny = 256;
    int nz = 256;
    int steps = 1000;
    double alpha = 0.5;
    double dx = 0.01;
    double dy = 0.01;
    double dz = 0.01;
    bool use_cuda_graph = false;
};

class HeatSolver3D {
private:
    SolverConfig config;
    Field3D current_field;
    Field3D previous_field;
    int rank = 0;
    int num_ranks = 1;
    double dt = 0.0001;
    cudaStream_t compute_stream = 0;
    CudaGraphExecutor graph_executor;

public:
    HeatSolver3D(const SolverConfig& cfg, int mpi_rank, int mpi_size)
        : config(cfg), rank(mpi_rank), num_ranks(mpi_size) {}

    ~HeatSolver3D() {
        if (compute_stream != 0) {
            cudaStreamDestroy(compute_stream);
        }
    }

    void setup() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

        current_field.initialize(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz);
        previous_field.initialize(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz);

        double dx2 = config.dx * config.dx;
        double dy2 = config.dy * config.dy;
        double dz2 = config.dz * config.dz;
        dt = (dx2 * dy2 * dz2) / (2.0 * config.alpha * (dx2 + dy2 + dz2));

        int center_i = config.nx / 2;
        int center_j = config.ny / 2;
        int center_k = config.nz / 2;

        for (int i = 1; i <= config.nx; ++i) {
            for (int j = 1; j <= config.ny; ++j) {
                for (int k = 1; k <= config.nz; ++k) {
                    double dist_sq = (i - center_i) * (i - center_i) +
                                     (j - center_j) * (j - center_j) +
                                     (k - center_k) * (k - center_k);
                    if (dist_sq < 400.0) {
                        current_field.host_at(i, j, k) = 100.0;
                        previous_field.host_at(i, j, k) = 100.0;
                    }
                }
            }
        }

        current_field.sync_to_device(compute_stream);
        previous_field.sync_to_device(compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }

    void run() {
        if (rank == 0) {
            std::cout << "3D Heat Equation Solver Initialized [" 
                      << config.nx << "x" << config.ny << "x" << config.nz 
                      << "], Steps: " << config.steps << ", dt: " << dt << std::endl;
        }

        if (config.use_cuda_graph) {
            graph_executor.begin_capture(compute_stream);
            launch_evolve_interior(
                current_field.d_data.get(),
                previous_field.d_data.get(),
                config.nx, config.ny, config.nz,
                config.alpha, dt,
                config.dx, config.dy, config.dz,
                compute_stream
            );
            graph_executor.end_capture();
        }

        for (int step = 1; step <= config.steps; ++step) {
            if (config.use_cuda_graph) {
                graph_executor.launch();
            } else {
                launch_evolve_interior(
                    current_field.d_data.get(),
                    previous_field.d_data.get(),
                    config.nx, config.ny, config.nz,
                    config.alpha, dt,
                    config.dx, config.dy, config.dz,
                    compute_stream
                );
            }
            std::swap(current_field, previous_field);
        }

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        if (rank == 0) {
            std::cout << "Solver execution completed successfully." << std::endl;
        }
    }
};

#endif // HEAT_SOLVER_HPP
