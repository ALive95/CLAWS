"""
compare_direct_vs_fp.py

Compare two solvers for the nonlocal conservation law with exponential memory:

  d_t q + d_x(F(t, x, W, q)) = 0
  W(t,x) = int_{-inf}^{t} (1/tau0) exp(-(t-s)/tau0) [gamma * J(q(s,.))](x) ds

  (A) Fixed-point (Picard) iteration  — solve_nonlocal_memory_exponential
  (B) Direct LxF marching scheme      — solve_direct_lxf_memory_exponential

Both use the same grid, kernel, and initial data. Since both discretize the
same causal sum with the same recursive formula, they should yield identical
(or near-identical) discrete solutions.

Setup: simple test case with
  F(t, x, W, q) = q * V(W),  V(W) = 1 - W  (linear speed reduction)
  J(q) = q
  gamma(z) = quintic kernel on [-eta, eta]  (from Chiarello-Goatin paper)
  q0(x)    = 0.5 + 0.2 sin(2*pi*x),  x in [0,1] (periodic)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.claws_LXF import (solve_nonlocal_memory_exponential,
                       solve_direct_lxf_memory_exponential)

# ================================================================
# Problem parameters
# ================================================================
X_RANGE = (0.0, 1.0)
T_FINAL = 0.5
NX      = 1000
CFL     = 0.45
ALPHA   = 1.0          # max wave speed bound
TAU0    = 1.0         # memory time scale
ETA     = 0.3          # spatial kernel support half-width
T_HIST  = 5 * TAU0     # history window
N_HIST  = 200          # history grid points
MAX_ITER = 100
TOL      = 1e-10

# ================================================================
# Model functions
# ================================================================

def quintic_kernel(z, eta):
    """
    Quintic spatial kernel on [-eta, eta], normalized to integrate to 1.
    From Chiarello-Goatin: w(z) = (1 - |z|/eta)^5 / (eta/3) for |z|<=eta.
    """
    r = np.abs(z) / eta
    return np.where(r <= 1.0, (1.0 - r)**5 * (3.0 / eta), 0.0)

def gamma(z):
    return quintic_kernel(z, ETA)

def J(q):
    """Nonlinear weight in the convolution."""
    return q

def V(W):
    """Speed function: decreasing in traffic density."""
    return np.maximum(0.0, 1.0 - W)

def F(t, x, W, q):
    """Physical flux: f(q) * V(W), f(q) = q."""
    return q * V(W)

def q0_func(x):
    """Smooth periodic initial condition."""
    return 0.5 + 0.2 * np.sin(2 * np.pi * x)

def q0_hist_func(t, x):
    """
    Historical data for t <= 0: constant in time (steady state).
    """
    return q0_func(x)

# ================================================================
# Run both solvers
# ================================================================
import time

print("=" * 60)
print("Fixed-point (Picard) solver")
print("=" * 60)
t0 = time.perf_counter()
x_fp, t_fp, q_fp, info_fp = solve_nonlocal_memory_exponential(
    F, J, TAU0, gamma, q0_hist_func, q0_func,
    X_RANGE, NX, T_FINAL, T_HIST, N_HIST,
    ALPHA, cfl=CFL, max_iter=MAX_ITER, tol=TOL,
    bc='periodic', verbose=True
)
time_fp = time.perf_counter() - t0
print(f"  converged={info_fp['converged']}, iters={info_fp['iters']}")
print(f"  final residual: {info_fp['residuals'][-1]:.3e}")
print(f"  wall time: {time_fp:.3f} s\n")

print("=" * 60)
print("Direct LxF marching solver")
print("=" * 60)
t0 = time.perf_counter()
x_d, t_d, q_d, info_d = solve_direct_lxf_memory_exponential(
    F, J, TAU0, gamma, q0_hist_func, q0_func,
    X_RANGE, NX, T_FINAL, T_HIST, N_HIST,
    ALPHA, cfl=CFL, bc='periodic', verbose=True
)
time_d = time.perf_counter() - t0
print(f"  wall time: {time_d:.3f} s\n")

# ================================================================
# Quantify difference
# ================================================================
dx = x_fp[1] - x_fp[0]

# L1 and Linf differences at each time level
Nt = len(t_fp) - 1
l1_diff   = np.array([dx * np.sum(np.abs(q_fp[n] - q_d[n])) for n in range(Nt + 1)])
linf_diff = np.array([np.max(np.abs(q_fp[n] - q_d[n])) for n in range(Nt + 1)])

print("Difference between FP and direct solvers:")
print(f"  max L1   over time: {l1_diff.max():.3e}")
print(f"  max Linf over time: {linf_diff.max():.3e}")
print(f"\nWall-clock times:")
print(f"  Fixed-point : {time_fp:.3f} s  ({info_fp['iters']} iters)")
print(f"  Direct LxF  : {time_d:.3f} s")
print(f"  Speedup     : {time_fp / time_d:.2f}x")

# ================================================================
# Plots
# ================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Fixed-Point vs Direct LxF — Exponential Memory", fontsize=13)

time_indices = [0, Nt // 4, Nt // 2, Nt]
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

# Panel 1: solution snapshots — FP
ax = axes[0, 0]
for i, n in enumerate(time_indices):
    ax.plot(x_fp, q_fp[n], color=colors[i], label=f"t={t_fp[n]:.3f}")
ax.set_title("Fixed-point solver  q(t,x)")
ax.set_xlabel("x")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: solution snapshots — direct
ax = axes[0, 1]
for i, n in enumerate(time_indices):
    ax.plot(x_d, q_d[n], color=colors[i], label=f"t={t_d[n]:.3f}")
ax.set_title("Direct LxF solver  q(t,x)")
ax.set_xlabel("x")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: wall-clock time comparison
ax = axes[0, 2]
labels  = [f"Fixed-point\n({info_fp['iters']} iters)", "Direct LxF"]
times   = [time_fp, time_d]
bars    = ax.bar(labels, times, color=["steelblue", "darkorange"], width=0.4)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
            f"{t:.3f} s", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Wall time (s)")
ax.set_title(f"Computation time  (speedup {time_fp/time_d:.2f}x)")
ax.grid(True, axis="y", alpha=0.3)

# Panel 4: pointwise difference at final time
ax = axes[1, 0]
ax.plot(x_fp, q_fp[-1] - q_d[-1], color="steelblue")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_title(f"Pointwise difference  q_FP - q_direct  at t={t_fp[-1]:.3f}")
ax.set_xlabel("x")
ax.grid(True, alpha=0.3)

# Panel 5: L1 and Linf differences over time
ax = axes[1, 1]
ax.semilogy(t_fp, l1_diff,   label=r"$L^1$ diff")
ax.semilogy(t_fp, linf_diff, label=r"$L^\infty$ diff", ls="--")
ax.set_title("||q_FP - q_direct|| over time")
ax.set_xlabel("t")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 6: FP residuals per iteration
ax = axes[1, 2]
ax.semilogy(range(1, len(info_fp['residuals']) + 1), info_fp['residuals'],
            marker="o", ms=4, color="steelblue")
ax.set_title("Fixed-point residuals")
ax.set_xlabel("iteration")
ax.set_ylabel(r"$\|q^{(k+1)} - q^{(k)}\|_\infty$")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = "figures/fp_vs_direct/fp_vs_direct.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {outpath}")