#ifndef HEAT_SOLVER_HPP
#define HEAT_SOLVER_HPP

#include "field.hpp"
#include "stencil_kernel.cuh"
#include "cuda_graph_solver.hpp"
#include "domain_decomp.hpp"
#include "halo_exchange.hpp"
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
    DomainDecomposition3D domain;
    HaloExchangePipeline halo_pipeline;

    Field3D current_field;
    Field3D previous_field;
    
    double dt = 0.0001;
    cudaStream_t compute_stream = 0;
    CudaGraphExecutor graph_executor;

public:
    HeatSolver3D(const SolverConfig& cfg, MPI_Comm comm = MPI_COMM_WORLD)
        : config(cfg) {
        domain.initialize(config.nx, config.ny, config.nz, comm);
        domain.setup_p2p_nvlink();
    }

    ~HeatSolver3D() {
        if (compute_stream != 0) {
            cudaStreamDestroy(compute_stream);
        }
    }

    void setup() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

        current_field.initialize(domain.local_x(), domain.local_y(), domain.local_z(), config.dx, config.dy, config.dz);
        previous_field.initialize(domain.local_x(), domain.local_y(), domain.local_z(), config.dx, config.dy, config.dz);

        halo_pipeline.initialize(domain);

        double dx2 = config.dx * config.dx;
        double dy2 = config.dy * config.dy;
        double dz2 = config.dz * config.dz;
        dt = (dx2 * dy2 * dz2) / (2.0 * config.alpha * (dx2 + dy2 + dz2));

        int center_i = config.nx / 2;
        int center_j = config.ny / 2;
        int center_k = config.nz / 2;

        for (int i = 1; i <= domain.local_x(); ++i) {
            for (int j = 1; j <= domain.local_y(); ++j) {
                for (int k = 1; k <= domain.local_z(); ++k) {
                    int global_i = domain.start_x() + i;
                    int global_j = domain.start_y() + j;
                    int global_k = domain.start_z() + k;

                    double dist_sq = (global_i - center_i) * (global_i - center_i) +
                                     (global_j - center_j) * (global_j - center_j) +
                                     (global_k - center_k) * (global_k - center_k);
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
        if (domain.rank() == 0) {
            std::cout << "3D Distributed Heat Equation Solver Initialized [" 
                      << config.nx << "x" << config.ny << "x" << config.nz 
                      << "], Local sub-grid: [" << domain.local_x() << "x" << domain.local_y() << "x" << domain.local_z()
                      << "], MPI ranks: " << domain.size() << ", Steps: " << config.steps << std::endl;
        }

        for (int step = 1; step <= config.steps; ++step) {
            halo_pipeline.pack_halos(previous_field, compute_stream);
            halo_pipeline.start_exchange(previous_field);

            launch_evolve_interior(
                current_field.d_data.get(),
                previous_field.d_data.get(),
                domain.local_x(), domain.local_y(), domain.local_z(),
                config.alpha, dt,
                config.dx, config.dy, config.dz,
                compute_stream
            );

            halo_pipeline.finish_exchange(previous_field, compute_stream);
            std::swap(current_field, previous_field);
        }

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        if (domain.rank() == 0) {
            std::cout << "Distributed multi-GPU execution completed successfully." << std::endl;
        }
    }
};

#endif // HEAT_SOLVER_HPP
