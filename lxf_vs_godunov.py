"""
compare_lxf_godunov.py

Compare LxF and Godunov solvers on two nonlocal problems:
  Case 1: d_t q + d_x(q(1-q) V(W)) = 0,  W = gamma * J(q)         (space only)
  Case 2: same flux,  W = int K(t-s) [gamma * J(q(s,.))](x) ds    (space + memory)

For each case: run both solvers on grids Nx in NX_LIST, compute the
pairwise L1 and Linf differences, and plot:
  (a) overlaid snapshots at final time (finest grid)
  (b) point-wise difference |q_LXF - q_G| at final time for each grid
  (c) ||q_LXF - q_G||_1  and  ||q_LXF - q_G||_inf  vs Nx

F(t, x, w, q) = q (1-q) (1 - w)  works for scalar and array inputs (numpy
broadcasts), so it is valid for both the LxF (array) and Godunov (scalar) calls.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.insert(0, os.path.dirname(__file__))

import solver.claws_LXF as lxf
import solver.claws_G   as gov

# ================================================================
# Problem parameters
# ================================================================

X_RANGE  = (0.0, 1.0)   # periodic domain [0, 1]
T        = 0.5          # final time
ALPHA    = 1.4           # wave speed bound: max|dF/dq| = max|(1-2q)V(t,x)(1-w)| <= 1.4
CFL      = 0.45          # CFL < 0.5 for LxF stability; Godunov allows up to 1
TOL      = 1e-8          # Picard tolerance
MAX_ITER = 60

# Grid sweep
NX_LIST  = [50, 100, 200, 400]

# Memory case extra parameters
TAU0   = 0.5   # exponential memory decay time
T_HIST = 2.0   # history window length
N_HIST = 40    # steps in history window

SNAP_TIMES = [0.0, T / 4, T / 2]   # snapshot times for overlaid plots

os.makedirs("figures/lxf_vs_godunov", exist_ok=True)

# ================================================================
# Problem definition
# ================================================================

def V(t, x):
    """Space-time dependent speed: oscillates in space, decays in time."""
    return 1.0 + 0.4 * np.sin(2.0 * np.pi * x) * np.exp(-t)

def F(t, x, w, q):
    """Flux: q(1-q) V(t,x) (1-w). Works for scalar and array q, x, w."""
    return q * (1.0 - q) * V(t, x) * (1.0 - w)

def J(q_arr):
    return q_arr.copy()

def gamma(z_arr):
    """Bump kernel on [-0.15, 0.15], L1-normalized on the periodic domain."""
    R = 0.15
    mask = np.abs(z_arr) < R
    vals = np.zeros_like(np.asarray(z_arr, dtype=float))
    vals[mask] = np.cos(np.pi * z_arr[mask] / (2 * R)) ** 2
    return vals / R  # int cos^2(pi z / 2R) dz from -R to R = R

def K_time(tau):
    """Exponential temporal kernel: (1/tau0) exp(-tau/tau0)."""
    return np.exp(-tau / TAU0) / TAU0

def q0_func(x):
    """Smooth periodic initial condition (roughly centered bump)."""
    return 0.5 * np.exp(-20.0 * (x - 0.5) ** 2)

def q0_hist(t, x):
    """Constant history: q(t, x) = q0(x) for t <= 0."""
    return q0_func(x)


# ================================================================
# Utility: interpolate q to a common fine grid for fair comparison
# ================================================================

def interp_to(x_src, q_row, x_dst):
    """Piecewise-constant (nearest-cell) interpolation from x_src to x_dst."""
    return np.interp(x_dst, x_src, q_row)


# ================================================================
# Run both solvers for a given Nx, return solutions on their own grids
# ================================================================

def run_space(Nx):
    """Run space-only case with LxF and Godunov at resolution Nx."""
    print(f"  [space] Nx={Nx} -- LxF")
    x_l, t_l, q_l, _ = lxf.solve_nonlocal_space(
        F, J, gamma, q0_func, X_RANGE, Nx, T,
        alpha=ALPHA, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', verbose=False)

    print(f"  [space] Nx={Nx} -- Godunov")
    x_g, t_g, q_g, _ = gov.solve_nonlocal_space(
        F, J, gamma, q0_func, X_RANGE, Nx, T,
        alpha=ALPHA, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', riemann='concave', verbose=False)

    return x_l, t_l, q_l, x_g, t_g, q_g


def run_memory(Nx):
    """Run memory (exponential) case with LxF and Godunov at resolution Nx."""
    print(f"  [memory] Nx={Nx} -- LxF")
    x_l, t_l, q_l, _ = lxf.solve_nonlocal_memory_exponential(
        F, J, TAU0, gamma, q0_hist, q0_func,
        X_RANGE, Nx, T, T_HIST, N_HIST,
        alpha=ALPHA, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', verbose=False)

    print(f"  [memory] Nx={Nx} -- Godunov")
    x_g, t_g, q_g, _ = gov.solve_nonlocal_memory_exponential(
        F, J, TAU0, gamma, q0_hist, q0_func,
        X_RANGE, Nx, T, T_HIST, N_HIST,
        alpha=ALPHA, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', riemann='concave', verbose=False)

    return x_l, t_l, q_l, x_g, t_g, q_g


# ================================================================
# Difference metrics at final time (solutions on same grid)
# ================================================================

def diff_metrics(x, q_l, q_g):
    """
    Compute L1 and Linf differences between two solutions on the same grid.
    dx is uniform so L1 ~ dx * sum|diff|.
    """
    dx = x[1] - x[0]
    diff = np.abs(q_l[-1] - q_g[-1])   # final time
    return dx * np.sum(diff), np.max(diff)


# ================================================================
# Plotting
# ================================================================

def plot_snapshots_compare(x_l, t_l, q_l, x_g, t_g, q_g,
                           snap_times, title, filename):
    """Overlay LxF and Godunov snapshots at selected times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.viridis(np.linspace(0.15, 0.85, len(snap_times)))

    for tc, col in zip(snap_times, colors):
        n_l = np.argmin(np.abs(t_l - tc))
        n_g = np.argmin(np.abs(t_g - tc))
        t_label = f"t={t_l[n_l]:.2f}"
        ax.plot(x_l, q_l[n_l], color=col, lw=2.0,
                label=f"LxF  {t_label}")
        ax.plot(x_g, q_g[n_g], color=col, lw=1.5,
                ls='--', label=f"Godunov  {t_label}")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$q$")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_diff_profile(results, snap_time, title, filename):
    """
    Plot |q_LxF - q_G|(x) at snap_time for each Nx.
    results: list of (Nx, x_l, t_l, q_l, x_g, t_g, q_g)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.plasma(np.linspace(0.15, 0.85, len(results)))

    for (Nx, x_l, t_l, q_l, x_g, t_g, q_g), col in zip(results, colors):
        n_l = np.argmin(np.abs(t_l - snap_time))
        n_g = np.argmin(np.abs(t_g - snap_time))
        # Both grids identical (same Nx, same x_range), no interpolation needed
        diff = np.abs(q_l[n_l] - q_g[n_g])
        ax.plot(x_l, diff, color=col, lw=1.5, label=f"Nx={Nx}")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$|q_{LxF} - q_G|$")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_metrics_vs_nx(nx_list, l1_list, linf_list, title, filename):
    """Plot L1 and Linf ||q_LxF - q_G|| vs Nx on a log-log scale."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(nx_list, l1_list,   'o-', lw=1.8, label="$L^1$")
    ax.loglog(nx_list, linf_list, 's--', lw=1.8, label="$L^\\infty$")

    # Reference slope -1 (first-order convergence guide)
    ref_x = np.array([nx_list[0], nx_list[-1]], dtype=float)
    ref_y = l1_list[0] * (ref_x[0] / ref_x) ** 1
    ax.loglog(ref_x, ref_y, 'k:', lw=1.0, label="$O(h)$")

    ax.set_xlabel("$N_x$")
    ax.set_ylabel("$\\|q_{LxF} - q_G\\|$  at $T$")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close(fig)


# ================================================================
# Main
# ================================================================

for case_name, run_fn in [("space", run_space), ("memory", run_memory)]:
    print()
    print("=" * 60)
    print(f"CASE: {case_name}")
    print("=" * 60)

    results  = []   # (Nx, x_l, t_l, q_l, x_g, t_g, q_g)
    nx_vals  = []
    l1_vals  = []
    linf_vals = []

    for Nx in NX_LIST:
        x_l, t_l, q_l, x_g, t_g, q_g = run_fn(Nx)
        results.append((Nx, x_l, t_l, q_l, x_g, t_g, q_g))

        l1, linf = diff_metrics(x_l, q_l, q_g)
        nx_vals.append(Nx)
        l1_vals.append(l1)
        linf_vals.append(linf)
        print(f"    Nx={Nx:4d}  ||diff||_1={l1:.3e}  ||diff||_inf={linf:.3e}")

    # --- Plot 1: snapshot overlay at finest grid ---
    Nx_fine, x_l, t_l, q_l, x_g, t_g, q_g = results[-1]
    plot_snapshots_compare(
        x_l, t_l, q_l, x_g, t_g, q_g,
        SNAP_TIMES,
        title=f"LxF vs Godunov snapshots ({case_name}, Nx={Nx_fine})",
        filename=f"figures/lxf_vs_godunov/{case_name}_snapshots.png")

    # --- Plot 2: pointwise difference profile at T ---
    plot_diff_profile(
        results, snap_time=T,
        title=f"|q_LxF - q_G| at T={T} ({case_name})",
        filename=f"figures/lxf_vs_godunov/{case_name}_diff_profile.png")

    # --- Plot 3: metrics vs Nx ---
    plot_metrics_vs_nx(
        nx_vals, l1_vals, linf_vals,
        title=f"Scheme difference vs grid ({case_name})",
        filename=f"figures/lxf_vs_godunov/{case_name}_metrics_nx.png")

print()
print("Done. Figures saved to figures/lxf_vs_godunov/")