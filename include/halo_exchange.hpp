#ifndef HALO_EXCHANGE_HPP
#define HALO_EXCHANGE_HPP

#include "domain_decomp.hpp"
#include "field.hpp"
#include "memory_manager.hpp"
#include <mpi.h>
#include <vector>

class HaloExchangePipeline {
private:
    DomainDecomposition3D domain;
    
    // Halo buffers for 6 faces: X_MINUS, X_PLUS, Y_MINUS, Y_PLUS, Z_MINUS, Z_PLUS
    GpuBuffer<double> send_buffers[6];
    GpuBuffer<double> recv_buffers[6];

    MPI_Request requests[12];
    
    cudaStream_t stream_x = 0;
    cudaStream_t stream_y = 0;
    cudaStream_t stream_z = 0;

public:
    HaloExchangePipeline() = default;
    ~HaloExchangePipeline();

    void initialize(const DomainDecomposition3D& decomp);

    void pack_halos(Field3D& field, cudaStream_t stream = 0);
    void start_exchange(Field3D& field);
    void finish_exchange(Field3D& field, cudaStream_t stream = 0);
};

#endif // HALO_EXCHANGE_HPP
