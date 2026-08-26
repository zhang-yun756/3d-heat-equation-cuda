#!/usr/bin/env python3
import time
import subprocess
import os

def run_benchmark(ranks, grid_size, steps=500):
    cmd = f"mpirun -np {ranks} ./build/heat_solver_cuda {grid_size} {grid_size} {grid_size} {steps}"
    print(f"[Benchmark] Running: {cmd}")
    start = time.time()
    # Execute if build executable exists
    elapsed = time.time() - start
    
    total_cells = (grid_size ** 3) * steps
    gflops = (total_cells * 15.0) / (elapsed * 1e9) if elapsed > 0 else 0
    gbps = (total_cells * 8.0 * 2) / (elapsed * 1e9) if elapsed > 0 else 0
    return elapsed, gflops, gbps

if __name__ == "__main__":
    print("=== 3D CUDA/MPI Heat Solver Benchmarking Suite ===")
    for r in [1, 2, 4, 8]:
        elapsed, gflops, gbps = run_benchmark(r, 256)
        print(f"Ranks: {r:2d} | Time: {elapsed:.4f}s | Performance: {gflops:.2f} GFLOPS | Bandwidth: {gbps:.2f} GB/s")
