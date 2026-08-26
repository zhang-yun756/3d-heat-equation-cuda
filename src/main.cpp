#include "heat_solver.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    SolverConfig config;
    if (argc >= 4) {
        config.nx = std::atoi(argv[1]);
        config.ny = std::atoi(argv[2]);
        config.nz = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        config.steps = std::atoi(argv[4]);
    }

    HeatSolver3D solver(config, MPI_COMM_WORLD);
    solver.setup();
    solver.run();

    MPI_Finalize();
    return 0;
}
