#include "halo_exchange.hpp"
#include <cuda_runtime.h>

__global__ void pack_x_face_kernel(double* __restrict__ dst, const double* __restrict__ src, int ny_ghost, int nz_ghost, int i_index) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < ny_ghost && k < nz_ghost) {
        size_t src_idx = static_cast<size_t>(i_index) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + k;
        size_t dst_idx = static_cast<size_t>(j) * nz_ghost + k;
        dst[dst_idx] = src[src_idx];
    }
}

__global__ void unpack_x_face_kernel(double* __restrict__ dst, const double* __restrict__ src, int ny_ghost, int nz_ghost, int i_index) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < ny_ghost && k < nz_ghost) {
        size_t dst_idx = static_cast<size_t>(i_index) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + k;
        size_t src_idx = static_cast<size_t>(j) * nz_ghost + k;
        dst[dst_idx] = src[src_idx];
    }
}

HaloExchangePipeline::~HaloExchangePipeline() {
    if (stream_x) cudaStreamDestroy(stream_x);
    if (stream_y) cudaStreamDestroy(stream_y);
    if (stream_z) cudaStreamDestroy(stream_z);
}

void HaloExchangePipeline::initialize(const DomainDecomposition3D& decomp) {
    domain = decomp;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_x, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_y, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_z, cudaStreamNonBlocking));

    int nx = domain.local_x();
    int ny = domain.local_y();
    int nz = domain.local_z();

    size_t size_x = static_cast<size_t>(ny + 2) * (nz + 2);
    size_t size_y = static_cast<size_t>(nx + 2) * (nz + 2);
    size_t size_z = static_cast<size_t>(nx + 2) * (ny + 2);

    send_buffers[X_MINUS].allocate(size_x, MemoryType::Pinned);
    send_buffers[X_PLUS].allocate(size_x, MemoryType::Pinned);
    recv_buffers[X_MINUS].allocate(size_x, MemoryType::Pinned);
    recv_buffers[X_PLUS].allocate(size_x, MemoryType::Pinned);

    send_buffers[Y_MINUS].allocate(size_y, MemoryType::Pinned);
    send_buffers[Y_PLUS].allocate(size_y, MemoryType::Pinned);
    recv_buffers[Y_MINUS].allocate(size_y, MemoryType::Pinned);
    recv_buffers[Y_PLUS].allocate(size_y, MemoryType::Pinned);

    send_buffers[Z_MINUS].allocate(size_z, MemoryType::Pinned);
    send_buffers[Z_PLUS].allocate(size_z, MemoryType::Pinned);
    recv_buffers[Z_MINUS].allocate(size_z, MemoryType::Pinned);
    recv_buffers[Z_PLUS].allocate(size_z, MemoryType::Pinned);
}

void HaloExchangePipeline::pack_halos(Field3D& field, cudaStream_t stream) {
    int nx = domain.local_x();
    int ny = domain.local_y();
    int nz = domain.local_z();

    int ny_g = ny + 2;
    int nz_g = nz + 2;

    dim3 block(16, 16);
    dim3 grid_x((nz_g + block.x - 1) / block.x, (ny_g + block.y - 1) / block.y);

    if (domain.neighbor(X_MINUS) != MPI_PROC_NULL) {
        pack_x_face_kernel<<<grid_x, block, 0, stream>>>(send_buffers[X_MINUS].get(), field.d_data.get(), ny_g, nz_g, 1);
    }
    if (domain.neighbor(X_PLUS) != MPI_PROC_NULL) {
        pack_x_face_kernel<<<grid_x, block, 0, stream>>>(send_buffers[X_PLUS].get(), field.d_data.get(), ny_g, nz_g, nx);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void HaloExchangePipeline::start_exchange(Field3D& field) {
    (void)field;
    int req_count = 0;
    MPI_Comm comm = domain.communicator();

    size_t size_x = send_buffers[X_MINUS].size();

    int left = domain.neighbor(X_MINUS);
    int right = domain.neighbor(X_PLUS);

    if (left != MPI_PROC_NULL) {
        MPI_Isend(send_buffers[X_MINUS].get(), size_x, MPI_DOUBLE, left, 101, comm, &requests[req_count++]);
        MPI_Irecv(recv_buffers[X_MINUS].get(), size_x, MPI_DOUBLE, left, 102, comm, &requests[req_count++]);
    }
    if (right != MPI_PROC_NULL) {
        MPI_Isend(send_buffers[X_PLUS].get(), size_x, MPI_DOUBLE, right, 102, comm, &requests[req_count++]);
        MPI_Irecv(recv_buffers[X_PLUS].get(), size_x, MPI_DOUBLE, right, 101, comm, &requests[req_count++]);
    }

    if (req_count > 0) {
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
}

void HaloExchangePipeline::finish_exchange(Field3D& field, cudaStream_t stream) {
    int nx = domain.local_x();
    int ny_g = domain.local_y() + 2;
    int nz_g = domain.local_z() + 2;

    dim3 block(16, 16);
    dim3 grid_x((nz_g + block.x - 1) / block.x, (ny_g + block.y - 1) / block.y);

    if (domain.neighbor(X_MINUS) != MPI_PROC_NULL) {
        unpack_x_face_kernel<<<grid_x, block, 0, stream>>>(field.d_data.get(), recv_buffers[X_MINUS].get(), ny_g, nz_g, 0);
    }
    if (domain.neighbor(X_PLUS) != MPI_PROC_NULL) {
        unpack_x_face_kernel<<<grid_x, block, 0, stream>>>(field.d_data.get(), recv_buffers[X_PLUS].get(), ny_g, nz_g, nx + 1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
