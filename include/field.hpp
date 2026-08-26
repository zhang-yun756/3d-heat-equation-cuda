#ifndef FIELD_HPP
#define FIELD_HPP

#include "memory_manager.hpp"
#include <vector>
#include <cassert>

struct Field3D {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    
    int nx_ghost = 0;
    int ny_ghost = 0;
    int nz_ghost = 0;

    double dx = 0.01;
    double dy = 0.01;
    double dz = 0.01;

    std::vector<double> h_data;
    GpuBuffer<double> d_data;

    void initialize(int local_nx, int local_ny, int local_nz, double spacing_x = 0.01, double spacing_y = 0.01, double spacing_z = 0.01) {
        nx = local_nx;
        ny = local_ny;
        nz = local_nz;

        nx_ghost = nx + 2;
        ny_ghost = ny + 2;
        nz_ghost = nz + 2;

        dx = spacing_x;
        dy = spacing_y;
        dz = spacing_z;

        size_t total_elements = static_cast<size_t>(nx_ghost) * ny_ghost * nz_ghost;
        h_data.resize(total_elements, 0.0);
        d_data.allocate(total_elements, MemoryType::Device);
    }

    inline size_t flat_index(int i, int j, int k) const {
        assert(i >= 0 && i < nx_ghost);
        assert(j >= 0 && j < ny_ghost);
        assert(k >= 0 && k < nz_ghost);
        return static_cast<size_t>(i) * ny_ghost * nz_ghost + static_cast<size_t>(j) * nz_ghost + k;
    }

    double& host_at(int i, int j, int k) {
        return h_data[flat_index(i, j, k)];
    }

    const double& host_at(int i, int j, int k) const {
        return h_data[flat_index(i, j, k)];
    }

    void sync_to_device(cudaStream_t stream = 0) {
        d_data.copy_to_device(h_data.data(), stream);
    }

    void sync_to_host(cudaStream_t stream = 0) {
        d_data.copy_to_host(h_data.data(), stream);
    }
};

#endif // FIELD_HPP
