#ifndef MEMORY_MANAGER_HPP
#define MEMORY_MANAGER_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + \
                                     cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

enum class MemoryType {
    Device,
    Pinned,
    Unified
};

template <typename T>
class GpuBuffer {
private:
    T* d_ptr = nullptr;
    size_t count = 0;
    MemoryType type = MemoryType::Device;

public:
    GpuBuffer() = default;

    GpuBuffer(size_t n, MemoryType mem_type = MemoryType::Device) 
        : count(n), type(mem_type) {
        allocate(n, mem_type);
    }

    ~GpuBuffer() {
        free();
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept 
        : d_ptr(other.d_ptr), count(other.count), type(other.type) {
        other.d_ptr = nullptr;
        other.count = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            free();
            d_ptr = other.d_ptr;
            count = other.count;
            type = other.type;
            other.d_ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    void allocate(size_t n, MemoryType mem_type = MemoryType::Device) {
        free();
        count = n;
        type = mem_type;
        if (count == 0) return;

        size_t size_bytes = count * sizeof(T);
        switch (type) {
            case MemoryType::Device:
                CUDA_CHECK(cudaMalloc(&d_ptr, size_bytes));
                break;
            case MemoryType::Pinned:
                CUDA_CHECK(cudaMallocHost(&d_ptr, size_bytes));
                break;
            case MemoryType::Unified:
                CUDA_CHECK(cudaMallocManaged(&d_ptr, size_bytes));
                break;
        }
    }

    void free() {
        if (d_ptr) {
            if (type == MemoryType::Pinned) {
                cudaFreeHost(d_ptr);
            } else {
                cudaFree(d_ptr);
            }
            d_ptr = nullptr;
            count = 0;
        }
    }

    void copy_to_device(const T* host_src, cudaStream_t stream = 0) {
        if (type == MemoryType::Pinned || type == MemoryType::Unified) {
            std::copy(host_src, host_src + count, d_ptr);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr, host_src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
        }
    }

    void copy_to_host(T* host_dst, cudaStream_t stream = 0) const {
        if (type == MemoryType::Pinned || type == MemoryType::Unified) {
            std::copy(d_ptr, d_ptr + count, host_dst);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(host_dst, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }
    }

    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }
    size_t size() const { return count; }
    MemoryType memory_type() const { return type; }
};

#endif // MEMORY_MANAGER_HPP
