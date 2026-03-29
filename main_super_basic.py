"""
main_super_basic.py - Run CLAWS solver on four test cases and compare.

Problem: 1D nonlocal LWR traffic model with constant speed limit
  d_t q + d_x(V0 * q(1-q)(1-W)) = 0

  Case 1 (spatial)     :  W = gamma * q  (no memory)
  Case 2 (exponential) :  K(tau) = (1/tau0) exp(-tau/tau0)
  Case 3 (Erlang)      :  K(tau) = (tau/tau0^2) exp(-tau/tau0)
  Case 4 (triangular)  :  K(tau) = (2/tau0)(1 - tau/tau0),  tau in [0, tau0]

  All memory cases: W(t,x) = int_{-inf}^{t} K(t-s) [gamma * q(s,.)](x) ds

  V(t,x) = V0  (constant)
  J(q) = q,  gamma: one-sided cosine bump on [-R, 0]  (cars look ahead)

Initial condition: two rectangular plateaux -> two shocks.
History (t <= 0): q_hist(t,x) = q0(x) exp(t/T_decay)  [L^1 in time].
Boundary conditions: outflow (zero-gradient).
Picard residual: L^inf norm.

Exponential case uses the recursive O(Nt) solver; Erlang and triangular
use the generic factorized solver O(Nt^2).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from solver.claws_LXF import (solve_nonlocal_space,
                               solve_nonlocal_memory_exponential,
                               solve_nonlocal_memory_factorized)

os.makedirs("figures/super_basic", exist_ok=True)
os.makedirs("solutions/super_basic", exist_ok=True)

# ================================================================
# Save / load flags
# ================================================================
SAVE_SOLUTIONS = True
LOAD_SOLUTIONS = True


def _save(name, x, t, q, info):
    path_ = f"solutions/super_basic/{name}.npz"
    np.savez(path_, x=x, t=t, q=q,
             residuals=info["residuals"],
             converged=info["converged"],
             iters=info["iters"])
    print(f"  Saved: {path_}")


def _load(name):
    path_ = f"solutions/super_basic/{name}.npz"
    if not os.path.exists(path_):
        return None
    d = np.load(path_, allow_pickle=True)
    info = {"residuals": list(d["residuals"]),
            "converged": bool(d["converged"]),
            "iters":     int(d["iters"])}
    print(f"  Loaded: {path_}")
    return d["x"], d["t"], d["q"], info

# ================================================================
# Common parameters
# ================================================================
V0      = 1.0    # constant speed
L       = 4.0    # domain length
X_RANGE = (0.0, L)
NX      = 1000

# alpha >= max_q |dF/dq| = V0 * max_q |1 - 2q| = V0
ALPHA = V0
CFL   = 0.45
TOL   = 1e-8
T     = 5.0

TAU0   = 1.0
T_HIST = 5.0
N_HIST = 200

SNAP_TIMES = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

# ================================================================
# Problem setup
# ================================================================

def F(t, x, w, q):
    return V0 * q * (1.0 - q) * (1.0 - w)

def J(q):
    return q.copy()

def q0_func(x):
    q = np.zeros_like(x)
    q[(x >= 0.5) & (x <= 1.2)] = 0.8
    q[(x >= 2.2) & (x <= 2.9)] = 0.7
    return q

T_DECAY = 1.0
def q0_hist(t, x):
    return q0_func(x) * np.exp(t / T_DECAY)

R = 0.4
def gamma(z):
    """One-sided kernel: cars look ahead (y > x, i.e. z = x-y < 0).
    Support [-R, 0], cosine shape, normalized to integrate to 1.
    int_{-R}^{0} cos^2(pi z / (2R)) dz = R/2, so divide by R/2.
    """
    vals = np.zeros_like(z, dtype=float)
    m    = (z >= -R) & (z <= 0)
    vals[m] = np.cos(np.pi * z[m] / (2 * R))**2
    return vals / (R / 2)

# ================================================================
# Temporal kernels (each normalized: int_0^inf K(tau) dtau = 1)
# ================================================================

def K_exponential(tau):
    """(1/tau0) exp(-tau/tau0)."""
    return np.exp(-tau / TAU0) / TAU0

def K_erlang(tau):
    """(tau/tau0^2) exp(-tau/tau0)  [Erlang-2]."""
    return (tau / TAU0**2) * np.exp(-tau / TAU0)

def K_triangular(tau):
    """(2/tau0)(1 - tau/tau0) on [0, tau0], zero elsewhere."""
    if tau < 0.0 or tau > TAU0:
        return 0.0
    return (2.0 / TAU0) * (1.0 - tau / TAU0)

# ================================================================
# Figure: model components
# ================================================================
dx_plot = L / NX
x_plot  = np.linspace(X_RANGE[0] + dx_plot / 2, X_RANGE[1] - dx_plot / 2, NX)

fig0, axes0 = plt.subplots(2, 2, figsize=(12, 8))

# --- Top-left: memory kernels ---
ax = axes0[0, 0]
tau_arr = np.linspace(0, 3.5 * TAU0, 500)
ax.plot(tau_arr, [K_exponential(t) for t in tau_arr],
        color="tomato",      lw=2.0, label="Exponential")
ax.plot(tau_arr, [K_erlang(t)      for t in tau_arr],
        color="forestgreen", lw=2.0, label="Erlang",     ls="--")
ax.plot(tau_arr, [K_triangular(t)  for t in tau_arr],
        color="darkorange",  lw=2.0, label="Triangular", ls="-.")
ax.axvline(TAU0, color="gray", lw=0.8, ls=":", label=r"$\tau_0$")
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$K(\tau)$")
ax.set_title("Memory kernels")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Top-right: spatial kernel gamma(z) ---
ax = axes0[0, 1]
z_arr = np.linspace(-1.5 * R, 0.5 * R, 500)
ax.plot(z_arr, gamma(z_arr), color="steelblue", lw=2.0)
ax.axvline(-R, color="gray", lw=0.8, ls=":", label="$-R$")
ax.axvline( 0, color="gray", lw=0.8, ls="--", label="$0$")
ax.fill_between(z_arr, gamma(z_arr), alpha=0.15, color="steelblue")
ax.set_xlabel("$z$")
ax.set_ylabel(r"$\gamma(z)$")
ax.set_title("Spatial kernel $\\gamma$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Bottom-left: historical datum q0_hist(t, x) ---
ax   = axes0[1, 0]
hist_times = [0.0, -0.5, -1.0, -2.0, -4.0]
cols = plt.cm.Blues_r(np.linspace(0.1, 0.7, len(hist_times)))
for tc, col in zip(hist_times, cols):
    ax.plot(x_plot, q0_hist(tc, x_plot), color=col, lw=1.8,
            label=f"$t = {tc:.1f}$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$q_0(t, x)$")
ax.set_title("Historical datum $q_0(t,x)$, $t \\leq 0$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Bottom-right: initial condition q0(x) ---
ax = axes0[1, 1]
ax.plot(x_plot, q0_func(x_plot), color="darkorchid", lw=2.0)
ax.set_xlabel("$x$")
ax.set_ylabel("$q_0(x)$")
ax.set_title("Initial condition $q_0(x)$")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

fig0.suptitle("Model components ($V = V_0$ constant)", fontsize=13)
fig0.tight_layout()
fig0.savefig("figures/super_basic/model_components.png", dpi=150)
print("Saved: figures/super_basic/model_components.png")
print()

# ================================================================
# Shared solver kwargs
# ================================================================
common = dict(alpha=ALPHA, cfl=CFL, tol=TOL, bc='outflow')
mem_common = dict(**common,
                  x_range=X_RANGE, Nx=NX, T=T,
                  T_hist=T_HIST, N_hist=N_HIST)

# ================================================================
# Run all four cases  (or load from disk if LOAD_SOLUTIONS=True)
# ================================================================

def _run_or_load(name, solver_fn, *args, **kwargs):
    if LOAD_SOLUTIONS:
        result = _load(name)
        if result is not None:
            return result
    print(f"  Running solver...")
    result = solver_fn(*args, **kwargs)
    if SAVE_SOLUTIONS:
        _save(name, *result)
    return result

print("=" * 60)
print("CASE 1: Spatial nonlocality (no memory)")
print("=" * 60)
x1, t1, q1, info1 = _run_or_load(
    "spatial", solve_nonlocal_space,
    F, J, gamma, q0_func, X_RANGE, NX, T, **common)

print()
print("=" * 60)
print("CASE 2: Exponential memory  K(tau) = exp(-tau/tau0)/tau0")
print("=" * 60)
x2, t2, q2, info2 = _run_or_load(
    "exponential", solve_nonlocal_memory_exponential,
    F, J, TAU0, gamma, q0_hist, q0_func, **mem_common)

print()
print("=" * 60)
print("CASE 3: Erlang memory  K(tau) = (tau/tau0^2) exp(-tau/tau0)")
print("=" * 60)
x3, t3, q3, info3 = _run_or_load(
    "erlang", solve_nonlocal_memory_factorized,
    F, J, K_erlang, gamma, q0_hist, q0_func, **mem_common)

print()
print("=" * 60)
print("CASE 4: Triangular memory  K(tau) = (2/tau0)(1-tau/tau0)")
print("=" * 60)
x4, t4, q4, info4 = _run_or_load(
    "triangular", solve_nonlocal_memory_factorized,
    F, J, K_triangular, gamma, q0_hist, q0_func, **mem_common)

# ================================================================
# Collect results for plotting
# ================================================================
cases = [
    (x1, t1, q1, info1, "Spatial",     "steelblue",   "-"),
    (x2, t2, q2, info2, "Exponential", "tomato",      "--"),
    (x3, t3, q3, info3, "Erlang",      "forestgreen", "-."),
    (x4, t4, q4, info4, "Triangular",  "darkorange",  ":"),
]

# ================================================================
# (a) Overlaid profiles at each snapshot time
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for i, tc in enumerate(SNAP_TIMES):
    ax = axes[i]
    for x_, t_, q_, _, label, color, ls in cases:
        n = np.argmin(np.abs(t_ - tc))
        ax.plot(x_, q_[n], color=color, lw=1.8, ls=ls, label=label)
    ax.set_title(f"$t = {tc:.1f}$", fontsize=11)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$q$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=9)

fig.suptitle(
    r"Nonlocal CL: spatial vs. memory kernels — $V = V_0$ constant, outflow BC",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("figures/super_basic/comparison_snapshots.png", dpi=150)
print("\nSaved: figures/super_basic/comparison_snapshots.png")

# ================================================================
# (b) Space-time plots, 4 panels
# ================================================================
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
vmin, vmax_ = 0.0, 0.85

for ax, (x_, t_, q_, _, label, _, _ls) in zip(axes2, cases):
    T_mesh, X_mesh = np.meshgrid(t_, x_, indexing="ij")
    pcm = ax.pcolormesh(X_mesh, T_mesh, q_, shading="auto",
                        cmap="jet", vmin=vmin, vmax=vmax_)
    fig2.colorbar(pcm, ax=ax, label="$q$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title(label)

fig2.suptitle(r"Space-time density ($V = V_0$ constant, outflow BC)", fontsize=12)
fig2.tight_layout()
fig2.savefig("figures/super_basic/comparison_spacetime.png", dpi=150)
print("Saved: figures/super_basic/comparison_spacetime.png")

# ================================================================
# (c) Picard convergence, all four cases
# ================================================================
fig3, ax3 = plt.subplots(figsize=(7, 4))
for _, _, _, info, label, color, ls in cases:
    ax3.semilogy(info["residuals"], color=color, ls=ls,
                 marker="o", ms=4, label=label)
ax3.set_xlabel("Picard iteration")
ax3.set_ylabel(r"$\|q^{(k+1)} - q^{(k)}\|_{L^\infty}$")
ax3.set_title("Picard convergence")
ax3.legend()
ax3.grid(True, alpha=0.3, which="both")
fig3.tight_layout()
fig3.savefig("figures/super_basic/convergence.png", dpi=150)
print("Saved: figures/super_basic/convergence.png")

print()
print("Done.")