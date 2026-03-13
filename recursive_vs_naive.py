"""
recursive_vs_naive.py - Benchmark: naive (O(Nt^2)) vs recursive (O(Nt))
exponential memory solver on the Chiarello-Goatin case A.

Runs both solvers at several grid resolutions and reports:
  - wall-clock time
  - number of Picard iterations
  - max pointwise difference between the two solutions
  - scaling behavior
"""

import numpy as np
import time

from solver.claws_LXF import (solve_nonlocal_memory_factorized,
                              solve_nonlocal_memory_exponential)
from main_goatin import (make_kernel, make_v, make_flux, max_v,
                         _build_Vmax_profiles, X_RANGE, T_FINAL, RHO_0)


# ================================================================
# Problem setup: Goatin case A (m=3, eta=0.1, delta=0.06)
# ================================================================
M, ETA, DELTA = 3, 0.1, 0.06
TAU0   = 0.1
T_HIST = 5 * TAU0
N_HIST = 10
CFL    = 0.9
TOL    = 1e-7
MAX_IT = 50

K_exp = lambda tau: np.exp(-tau / TAU0) / TAU0


def run_both(Nx):
    """Run naive and recursive solvers at given Nx. Return timings and diff."""
    dx = (X_RANGE[1] - X_RANGE[0]) / Nx
    x_grid = np.linspace(X_RANGE[0] + dx / 2, X_RANGE[1] - dx / 2, Nx)
    Vmax_profiles = _build_Vmax_profiles(x_grid, dx, Nx)

    v_func = make_v(M)
    kernel = make_kernel(ETA, DELTA)
    F_flux = make_flux(Vmax_profiles, v_func)
    J_id = lambda q: q.copy()
    q0_func = lambda x: np.full_like(x, RHO_0)
    q0_hist = lambda t, x: np.full_like(x, RHO_0)
    max_Vmax = max(np.max(Vmax_profiles[0]), np.max(Vmax_profiles[1]))
    alpha = max_Vmax * max_v(M)

    # Common args
    common = dict(
        x_range=X_RANGE, Nx=Nx, T=T_FINAL,
        T_hist=T_HIST, N_hist=N_HIST,
        alpha=alpha, cfl=CFL, tol=TOL, max_iter=MAX_IT,
        bc='periodic', verbose=False
    )

    # --- Naive (factorized, O(Nt^2)) ---
    t0 = time.time()
    x1, t1, q1, info1 = solve_nonlocal_memory_factorized(
        F_flux, J_id, K_exp, kernel, q0_hist, q0_func, **common)
    t_naive = time.time() - t0

    # --- Recursive (O(Nt)) ---
    t0 = time.time()
    x2, t2, q2, info2 = solve_nonlocal_memory_exponential(
        F_flux, J_id, TAU0, kernel, q0_hist, q0_func, **common)
    t_recur = time.time() - t0

    max_diff = np.max(np.abs(q1 - q2))
    l1_diff = np.sum(np.abs(q1[-1] - q2[-1])) * dx

    return {
        "Nx": Nx,
        "Nt": len(t1) - 1,
        "t_naive": t_naive,
        "t_recur": t_recur,
        "iters_naive": info1["iters"],
        "iters_recur": info2["iters"],
        "max_diff": max_diff,
        "l1_diff": l1_diff,
    }


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    grids = [100, 200, 400, 800]

    print(f"Benchmark: exponential memory, case A "
          f"(m={M}, eta={ETA}, delta={DELTA}, tau0={TAU0})")
    print(f"{'Nx':>6} {'Nt':>6} {'naive(s)':>10} {'recur(s)':>10} "
          f"{'speedup':>8} {'iters_n':>7} {'iters_r':>7} "
          f"{'max|dq|':>10} {'L1(T)':>10}")
    print("-" * 85)

    results = []
    for Nx in grids:
        r = run_both(Nx)
        results.append(r)
        speedup = r["t_naive"] / r["t_recur"] if r["t_recur"] > 0 else float('inf')
        print(f"{r['Nx']:6d} {r['Nt']:6d} "
              f"{r['t_naive']:10.2f} {r['t_recur']:10.2f} "
              f"{speedup:8.1f}x "
              f"{r['iters_naive']:7d} {r['iters_recur']:7d} "
              f"{r['max_diff']:10.2e} {r['l1_diff']:10.2e}")

    # Scaling summary
    if len(results) >= 2:
        print()
        print("Scaling (time ratio when doubling Nx):")
        for i in range(1, len(results)):
            r0, r1 = results[i - 1], results[i]
            ratio_naive = r1["t_naive"] / r0["t_naive"]
            ratio_recur = r1["t_recur"] / r0["t_recur"]
            print(f"  Nx {r0['Nx']:4d} -> {r1['Nx']:4d}: "
                  f"naive x{ratio_naive:.1f}, recursive x{ratio_recur:.1f}")
        print()
        print("Expected scaling per iteration:")
        print("  naive:     O(Nt^2 * Nx) ~ O(Nx^3)  -> x8 per doubling")
        print("  recursive: O(Nt * Nx log Nx) ~ O(Nx^2 log Nx)  -> x4-5 per doubling")