#include "heat_solver.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count > 0) {
        CUDA_CHECK(cudaSetDevice(rank % device_count));
    }

    SolverConfig config;
    if (argc >= 4) {
        config.nx = std::atoi(argv[1]);
        config.ny = std::atoi(argv[2]);
        config.nz = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        config.steps = std::atoi(argv[4]);
    }

    HeatSolver3D solver(config, rank, size);
    solver.setup();
    solver.run();

    MPI_Finalize();
    return 0;
}
