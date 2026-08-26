#include "domain_decomp.hpp"
#include <iostream>

DomainDecomposition3D::~DomainDecomposition3D() {
    if (cart_comm != MPI_COMM_NULL && cart_comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&cart_comm);
    }
}

void DomainDecomposition3D::initialize(int nx, int ny, int nz, MPI_Comm parent_comm) {
    global_nx = nx;
    global_ny = ny;
    global_nz = nz;

    MPI_Comm_rank(parent_comm, &world_rank);
    MPI_Comm_size(parent_comm, &world_size);

    int periods[3] = {0, 0, 0};
    dims[0] = 0; dims[1] = 0; dims[2] = 0;
    MPI_Dims_create(world_size, 3, dims);

    MPI_Cart_create(parent_comm, 3, dims, periods, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &world_rank);
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);

    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[X_MINUS], &neighbors[X_PLUS]);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[Y_MINUS], &neighbors[Y_PLUS]);
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[Z_MINUS], &neighbors[Z_PLUS]);

    local_nx = (global_nx + dims[0] - 1) / dims[0];
    local_ny = (global_ny + dims[1] - 1) / dims[1];
    local_nz = (global_nz + dims[2] - 1) / dims[2];

    offset_x = coords[0] * local_nx;
    offset_y = coords[1] * local_ny;
    offset_z = coords[2] * local_nz;

    if (coords[0] == dims[0] - 1) local_nx = global_nx - offset_x;
    if (coords[1] == dims[1] - 1) local_ny = global_ny - offset_y;
    if (coords[2] == dims[2] - 1) local_nz = global_nz - offset_z;

    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count > 0) {
        device_id = world_rank % dev_count;
        CUDA_CHECK(cudaSetDevice(device_id));
    }
}

void DomainDecomposition3D::setup_p2p_nvlink() {
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 1) return;

    for (int i = 0; i < dev_count; ++i) {
        if (i == device_id) continue;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_id, i));
        if (can_access) {
            cudaError_t err = cudaDeviceEnablePeerAccess(i, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
            }
        }
    }
}
