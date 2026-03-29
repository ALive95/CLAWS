"""
main_basic.py - Run CLAWS solver on four test cases and compare.

Problem: 1D nonlocal LWR traffic model with space-time varying speed limit
  d_t q + d_x(Vmax(t,x) * q(1-q)(1-W)) = 0

  Case 1 (spatial)     :  W = gamma * q  (no memory)
  Case 2 (exponential) :  K(tau) = (1/tau0) exp(-tau/tau0)
  Case 3 (Erlang)      :  K(tau) = (tau/tau0^2) exp(-tau/tau0)
  Case 4 (triangular)  :  K(tau) = (2/tau0)(1 - tau/tau0),  tau in [0, tau0]

  All memory cases: W(t,x) = int_{-inf}^{t} K(t-s) [gamma * q(s,.)](x) ds

  Vmax(t,x) = V0 (1 + eps * sin(2pi x/L) * cos(2pi t/T_per))
  J(q) = q,  gamma: one-sided cosine bump on [-R, 0]  (cars look ahead)

Initial condition: two rectangular plateaux -> two shocks.
History (t <= 0): q_hist(t,x) = q0(x) exp(t/T_decay)  [L^1 in time].
Boundary conditions: Dirichlet (q = 0 at both ends).
Picard residual: L^1 norm.

Exponential case uses the recursive O(Nt) solver; Erlang and triangular
use the generic factorized solver O(Nt^2).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from solver.claws_LXF import (solve_nonlocal_space,
                        solve_nonlocal_memory_exponential,
                        solve_nonlocal_memory_factorized)

os.makedirs("figures/basic", exist_ok=True)
os.makedirs("solutions/basic", exist_ok=True)

# ================================================================
# Save / load flags
# ================================================================
SAVE_SOLUTIONS = True   # save each solved case to solutions/basic/<name>.npz
LOAD_SOLUTIONS = True   # load from disk if file exists (skip solver)


def _save(name, x, t, q, info):
    path_ = f"solutions/basic/{name}.npz"
    np.savez(path_, x=x, t=t, q=q,
             residuals=info["residuals"],
             converged=info["converged"],
             iters=info["iters"])
    print(f"  Saved: {path_}")


def _load(name):
    path_ = f"solutions/basic/{name}.npz"
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
V0    = 1.0    # base speed
EPS   = 0.4    # oscillation amplitude (< 1)
L     = 4.0    # domain length / spatial period
T_PER = 1.0    # temporal period of Vmax oscillation
X_RANGE = (0.0, L)
NX    = 1000

# alpha >= max_{t,x} Vmax(t,x) * max_q |1-2q| = V0*(1+EPS)*1
ALPHA = V0 * (1.0 + EPS)
CFL   = 0.45
TOL   = 1e-8
T     = 5.0

TAU0   = 1.0   # shared memory time scale
T_HIST = 5.0   # history window (>> T_DECAY so tail is negligible)
N_HIST = 200    # history quadrature steps

SNAP_TIMES = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]

# ================================================================
# Problem setup
# ================================================================

def V_max(t, x):
    return V0 * (1.0 + EPS * np.sin(2 * np.pi * x / L)
                           * np.cos(2 * np.pi * t / T_PER))

def F(t, x, w, q):
    return V_max(t, x) * q * (1.0 - q) * (1.0 - w)

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
# Layout: 1x3
#   [left]   memory kernels K(tau)
#   [center] spatial kernel gamma(z)
#   [right]  historical datum q0_hist(t, x) as 3D surface, t=0 in red

dx_plot = L / NX
x_plot  = np.linspace(X_RANGE[0] + dx_plot / 2, X_RANGE[1] - dx_plot / 2, NX)

fig0 = plt.figure(figsize=(16, 5))
ax0  = fig0.add_subplot(1, 3, 1)           # memory kernels
ax1  = fig0.add_subplot(1, 3, 2)           # spatial kernel
ax2  = fig0.add_subplot(1, 3, 3, projection="3d")  # historical datum (3D)

# --- Left: memory kernels ---
tau_arr = np.linspace(0, 3.5 * TAU0, 500)
ax0.plot(tau_arr, [K_exponential(t) for t in tau_arr],
         color="tomato",      lw=2.0, label="Exponential")
ax0.plot(tau_arr, [K_erlang(t)      for t in tau_arr],
         color="forestgreen", lw=2.0, label="Erlang",     ls="--")
ax0.plot(tau_arr, [K_triangular(t)  for t in tau_arr],
         color="darkorange",  lw=2.0, label="Triangular", ls="-.")
ax0.axvline(TAU0, color="gray", lw=0.8, ls=":", label=r"$\tau_0$")
ax0.set_xlabel(r"$\tau$")
ax0.set_ylabel(r"$K(\tau)$")
ax0.set_title("Memory kernels")
ax0.legend(fontsize=9)
ax0.grid(True, alpha=0.3)

# --- Center: spatial kernel gamma(z) ---
z_arr = np.linspace(-1.5 * R, 0.5 * R, 500)
ax1.plot(z_arr, gamma(z_arr), color="steelblue", lw=2.0)
ax1.axvline(-R, color="gray", lw=0.8, ls=":", label="$-R$")
ax1.axvline( 0, color="gray", lw=0.8, ls="--", label="$0$")
ax1.fill_between(z_arr, gamma(z_arr), alpha=0.15, color="steelblue")
ax1.set_xlabel("$z$")
ax1.set_ylabel(r"$\gamma(z)$")
ax1.set_title("Spatial kernel $\\gamma$")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Right: historical datum as 3D surface ---
# Use a coarser grid for the surface to keep rendering fast
x_3d   = np.linspace(X_RANGE[0], X_RANGE[1], 400)
t_3d   = np.linspace(-T_HIST, 0.0, 200)
X3, T3 = np.meshgrid(x_3d, t_3d)           # shape (100, 200)
Q3     = q0_hist(T3, X3)

ax2.plot_surface(X3, T3, Q3, cmap="Blues", alpha=1.0,
                 linewidth=0, antialiased=True)

# Highlight t=0 profile in red
ax2.plot(x_3d, np.zeros_like(x_3d), q0_func(x_3d),
         color="red", lw=2.5, zorder=5, label="$t=0$")

ax2.set_xlabel("$x$", labelpad=6)
ax2.set_ylabel("$t$", labelpad=6)
ax2.set_zlabel(r"$q_{\mathrm{hist}}$", labelpad=6)
ax2.set_title("Historical datum $q_{\\mathrm{hist}}(t,x)$, $t \\leq 0$")
ax2.legend(fontsize=9)
ax2.view_init(elev=25, azim=60)

fig0.suptitle("Model components", fontsize=13)
fig0.tight_layout()
fig0.savefig("figures/basic/model_components.png", dpi=300)
print("Saved: figures/basic/model_components.png")
print()

# ================================================================
# Shared solver kwargs
# ================================================================
common = dict(alpha=ALPHA, cfl=CFL, tol=TOL,
              norm='L1', bc='dirichlet', bc_val=(0.0, 0.0))
mem_common = dict(**common,
                  x_range=X_RANGE, Nx=NX, T=T,
                  T_hist=T_HIST, N_hist=N_HIST)

# ================================================================
# Run all four cases  (or load from disk if LOAD_SOLUTIONS=True)
# ================================================================

def _run_or_load(name, solver_fn, *args, **kwargs):
    """Run solver or load cached result depending on flags."""
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
    r"Nonlocal CL: spatial vs. memory kernels — $V_{\max}(t,x)$ flux, Dirichlet BC",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("figures/basic/comparison_snapshots.png", dpi=150)
print("\nSaved: figures/basic/comparison_snapshots.png")

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

fig2.suptitle("Space-time density (Dirichlet BC)", fontsize=12)
fig2.tight_layout()
fig2.savefig("figures/basic/comparison_spacetime.png", dpi=150)
print("Saved: figures/basic/comparison_spacetime.png")

# ================================================================
# (c) Picard convergence, all four cases
# ================================================================
fig3, ax3 = plt.subplots(figsize=(7, 4))
for _, _, _, info, label, color, ls in cases:
    ax3.semilogy(info["residuals"], color=color, ls=ls,
                 marker="o", ms=4, label=label)
ax3.set_xlabel("Picard iteration")
ax3.set_ylabel(r"$\|q^{(k+1)} - q^{(k)}\|_{L^1}$")
ax3.set_title("Picard convergence")
ax3.legend()
ax3.grid(True, alpha=0.3, which="both")
fig3.tight_layout()
fig3.savefig("figures/basic/convergence.png", dpi=150)
print("Saved: figures/basic/convergence.png")

print()
print("Done.")