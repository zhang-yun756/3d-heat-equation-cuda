#include "cuda_graph_solver.hpp"

CudaGraphExecutor::~CudaGraphExecutor() {
    if (is_initialized) {
        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
        is_initialized = false;
    }
}

void CudaGraphExecutor::begin_capture(cudaStream_t capture_stream) {
    stream = capture_stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

void CudaGraphExecutor::end_capture() {
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    is_initialized = true;
}

void CudaGraphExecutor::launch() {
    if (is_initialized) {
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
    }
}
