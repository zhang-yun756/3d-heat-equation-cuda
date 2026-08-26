#ifndef DOMAIN_DECOMP_HPP
#define DOMAIN_DECOMP_HPP

#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include "memory_manager.hpp"

enum NeighborDir {
    X_MINUS = 0,
    X_PLUS  = 1,
    Y_MINUS = 2,
    Y_PLUS  = 3,
    Z_MINUS = 4,
    Z_PLUS  = 5
};

class DomainDecomposition3D {
private:
    MPI_Comm cart_comm = MPI_COMM_NULL;
    int world_rank = 0;
    int world_size = 1;
    
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};
    int neighbors[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    int global_nx = 256;
    int global_ny = 256;
    int global_nz = 256;

    int local_nx = 256;
    int local_ny = 256;
    int local_nz = 256;

    int offset_x = 0;
    int offset_y = 0;
    int offset_z = 0;

    int device_id = 0;

public:
    DomainDecomposition3D() = default;
    ~DomainDecomposition3D();

    void initialize(int nx, int ny, int nz, MPI_Comm parent_comm = MPI_COMM_WORLD);
    void setup_p2p_nvlink();

    MPI_Comm communicator() const { return cart_comm; }
    int rank() const { return world_rank; }
    int size() const { return world_size; }
    int device() const { return device_id; }

    int neighbor(NeighborDir dir) const { return neighbors[static_cast<int>(dir)]; }
    
    int local_x() const { return local_nx; }
    int local_y() const { return local_ny; }
    int local_z() const { return local_nz; }

    int start_x() const { return offset_x; }
    int start_y() const { return offset_y; }
    int start_z() const { return offset_z; }
};

#endif // DOMAIN_DECOMP_HPP
