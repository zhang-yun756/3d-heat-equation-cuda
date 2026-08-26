#ifndef HEAT_SOLVER_HPP
#define HEAT_SOLVER_HPP

#include "field.hpp"
#include <mpi.h>
#include <iostream>

struct SolverConfig {
    int nx = 256;
    int ny = 256;
    int nz = 256;
    int steps = 1000;
    double alpha = 0.5;
    double dx = 0.01;
    double dy = 0.01;
    double dz = 0.01;
};

class HeatSolver3D {
private:
    SolverConfig config;
    Field3D current_field;
    Field3D previous_field;
    int rank = 0;
    int num_ranks = 1;

public:
    HeatSolver3D(const SolverConfig& cfg, int mpi_rank, int mpi_size)
        : config(cfg), rank(mpi_rank), num_ranks(mpi_size) {}

    void setup() {
        current_field.initialize(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz);
        previous_field.initialize(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz);

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

        current_field.sync_to_device();
        previous_field.sync_to_device();
    }

    void run() {
        if (rank == 0) {
            std::cout << "3D Heat Equation Solver Initialized [" 
                      << config.nx << "x" << config.ny << "x" << config.nz 
                      << "], Steps: " << config.steps << std::endl;
        }
    }
};

#endif // HEAT_SOLVER_HPP
