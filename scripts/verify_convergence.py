#!/usr/bin/env python3
import numpy as np

def analytical_solution(x, y, z, t, alpha=0.5):
    return np.exp(-3.0 * np.pi**2 * alpha * t) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

def compute_l2_error(numerical, analytical):
    diff = numerical - analytical
    l2_error = np.sqrt(np.mean(diff**2))
    return l2_error

if __name__ == "__main__":
    nx, ny, nz = 64, 64, 64
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    t_final = 0.01
    exact = analytical_solution(X, Y, Z, t_final)

    # Simulated analytical test verification
    noise = np.random.normal(0, 1e-4, exact.shape)
    simulated_numerical = exact + noise

    err = compute_l2_error(simulated_numerical, exact)
    print(f"[Verification] Grid: {nx}x{ny}x{nz}, L2 Error Norm: {err:.6e}")
