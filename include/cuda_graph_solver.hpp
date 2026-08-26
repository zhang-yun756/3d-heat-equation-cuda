#ifndef CUDA_GRAPH_SOLVER_HPP
#define CUDA_GRAPH_SOLVER_HPP

#include <cuda_runtime.h>
#include "memory_manager.hpp"

class CudaGraphExecutor {
private:
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool is_initialized = false;
    cudaStream_t stream = 0;

public:
    CudaGraphExecutor() = default;
    ~CudaGraphExecutor();

    void begin_capture(cudaStream_t capture_stream);
    void end_capture();
    void launch();
    bool initialized() const { return is_initialized; }
};

#endif // CUDA_GRAPH_SOLVER_HPP
