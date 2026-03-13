"""
main_goatin_comparison.py - Compare the Chiarello-Goatin spatial nonlocal model
against the same model augmented with memory in time.

Two parameter sets from the paper:
  (A) m=3, eta=0.1, delta=0.06
  (B) m=3, eta=1.0, delta=0.0

Memory kernels (factorized: kappa(tau,z) = K(tau) * w_{eta,delta}(z)):
  1. Exponential:  K(tau) = (1/tau0) exp(-tau/tau0)
  2. Erlang:       K(tau) = (tau/tau0^2) exp(-tau/tau0)
  3. Triangular:   K(tau) = (2/tau0)(1 - tau/tau0) on [0, tau0]

All normalized to integrate to 1.  As tau0 -> 0, all reduce to delta(t)
and we recover the memoryless case.
"""

import numpy as np
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver.claws_LXF import (solve_nonlocal_space, solve_nonlocal_memory_factorized,
                              solve_nonlocal_memory_exponential)
from main_goatin import (make_kernel, make_v, make_flux, max_v,
                         _build_Vmax_profiles, X_RANGE, T_FINAL, RHO_0)


# ================================================================
# Discretization parameters
# ================================================================
NX  = 2000
DX  = (X_RANGE[1] - X_RANGE[0]) / NX
CFL = 0.9
TOL = 1e-7
MAX_ITER = 50

# Memory time scale
TAU0 = 0.1

# History: 5*tau0 for exponential/Erlang, tau0 for triangular.
# We use max to cover all kernels.
T_HIST = max(5 * TAU0, TAU0)
N_HIST = 50

# If True, save each solution to solutions/ as compressed .npz files.
# If files already exist, load them instead of re-solving.
SAVE_SOLUTIONS = True
SOL_DIR = "solutions"


# ================================================================
# Temporal memory kernels (all normalized to integrate to 1)
# ================================================================

def K_exponential(tau):
    """K(tau) = (1/tau0) exp(-tau/tau0)."""
    return np.exp(-tau / TAU0) / TAU0


def K_erlang(tau):
    """K(tau) = (tau/tau0^2) exp(-tau/tau0)."""
    return (tau / TAU0**2) * np.exp(-tau / TAU0)


def K_triangular(tau):
    """K(tau) = (2/tau0)(1 - tau/tau0) for tau in [0, tau0], 0 otherwise."""
    if tau < 0 or tau > TAU0:
        return 0.0
    return (2.0 / TAU0) * (1.0 - tau / TAU0)


KERNELS = {
    "exponential": K_exponential,
    "Erlang":      K_erlang,
    "triangular":  K_triangular,
}


# ================================================================
# Solver wrappers
# ================================================================

def solve_spatial(eta, delta, m, Vmax_profiles):
    """Solve the memoryless (spatial-only) problem."""
    v_func = make_v(m)
    kernel = make_kernel(eta, delta)
    F_flux = make_flux(Vmax_profiles, v_func)
    J_id = lambda q: q.copy()
    q0_func = lambda x: np.full_like(x, RHO_0)

    max_Vmax = max(np.max(Vmax_profiles[0]), np.max(Vmax_profiles[1]))
    alpha = max_Vmax * max_v(m)

    return solve_nonlocal_space(
        F_flux, J_id, kernel, q0_func, X_RANGE, NX, T_FINAL,
        alpha=alpha, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', verbose=True
    )


def solve_with_memory(eta, delta, m, Vmax_profiles, K_time, label=""):
    """Solve with factorized memory: kappa = K_time * w_{eta,delta}.

    Uses the recursive O(Nt) solver for the exponential kernel,
    and the generic O(Nt^2) factorized solver for the rest.
    """
    v_func = make_v(m)
    kernel = make_kernel(eta, delta)
    F_flux = make_flux(Vmax_profiles, v_func)
    J_id = lambda q: q.copy()
    q0_func = lambda x: np.full_like(x, RHO_0)

    # History: constant rho_0
    q0_hist = lambda t, x: np.full_like(x, RHO_0)

    max_Vmax = max(np.max(Vmax_profiles[0]), np.max(Vmax_profiles[1]))
    alpha = max_Vmax * max_v(m)

    if label:
        print(f"  [{label}]")

    common = dict(
        x_range=X_RANGE, Nx=NX, T=T_FINAL,
        T_hist=T_HIST, N_hist=N_HIST,
        alpha=alpha, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', verbose=True
    )

    if label == "exponential":
        # Recursive solver: O(Nt * Nx log Nx) per iteration
        return solve_nonlocal_memory_exponential(
            F_flux, J_id, TAU0, kernel, q0_hist, q0_func, **common)
    else:
        # Generic factorized solver: O(Nt^2 * Nx) per iteration
        return solve_nonlocal_memory_factorized(
            F_flux, J_id, K_time, kernel, q0_hist, q0_func, **common)


# ================================================================
# Plotting
# ================================================================

def plot_comparison_grid(results, case_label, filename):
    """
    Plot (t,x) density for spatial + 3 memory kernels side by side.

    Parameters
    ----------
    results : dict {label: (x, t, q)}
    case_label : string for suptitle
    filename : output file
    """
    labels = list(results.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, lab in zip(axes, labels):
        x, t, q = results[lab]
        T_mesh, X_mesh = np.meshgrid(t, x, indexing="ij")
        pcm = ax.pcolormesh(X_mesh, T_mesh, q, shading="auto",
                            cmap="jet", vmin=0, vmax=1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_title(lab, fontsize=10)
        fig.colorbar(pcm, ax=ax, shrink=0.85)

    fig.suptitle(case_label, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_snapshots_overlay(results, snap_times, case_label, filename):
    """
    Overlay solution snapshots at selected times for all models.

    Parameters
    ----------
    results : dict {label: (x, t, q)}
    snap_times : list of floats
    case_label : string for suptitle
    filename : output file
    """
    n_snaps = len(snap_times)
    fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 4),
                             squeeze=False)
    axes = axes[0]
    colors = {"spatial (no memory)": "k",
              "exponential": "tab:blue",
              "Erlang": "tab:red",
              "triangular": "tab:green"}

    for ax, ts in zip(axes, snap_times):
        for lab, (x, t, q) in results.items():
            n_idx = np.argmin(np.abs(t - ts))
            ls = "-" if lab == "spatial (no memory)" else "--"
            lw = 2.0 if lab == "spatial (no memory)" else 1.5
            ax.plot(x, q[n_idx], ls, color=colors.get(lab, "gray"),
                    lw=lw, label=lab)
        ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(f"$t = {ts}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(case_label, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_l1_difference(results_mem, x_ref, t_ref, q_ref,
                       case_label, filename):
    """
    Plot ||rho_memory(t) - rho_spatial(t)||_{L1} vs time for each kernel.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    dx = x_ref[1] - x_ref[0]
    colors = {"exponential": "tab:blue",
              "Erlang": "tab:red",
              "triangular": "tab:green"}

    for lab, (x, t, q) in results_mem.items():
        # Both should share the same grid, but be safe
        diff_l1 = np.sum(np.abs(q - q_ref), axis=1) * dx
        ax.plot(t, diff_l1, color=colors.get(lab, "gray"),
                lw=1.5, label=lab)

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|\rho_{\mathrm{mem}} - \rho_{\mathrm{space}}\|_{L^1}$")
    ax.set_title(case_label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close(fig)


# ================================================================
# Save / load solutions
# ================================================================

def _sol_path(case_tag, model_name):
    """Build the .npz file path for a given case and model."""
    # Sanitize: "spatial (no memory)" -> "spatial_no_memory"
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    return os.path.join(SOL_DIR, f"case{case_tag}_{safe}.npz")


def save_solution(case_tag, model_name, x, t, q):
    """Save (x, t, q) to a compressed .npz file."""
    os.makedirs(SOL_DIR, exist_ok=True)
    path = _sol_path(case_tag, model_name)
    np.savez_compressed(path, x=x, t=t, q=q)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved: {path} ({size_mb:.1f} MB)")


def load_solution(case_tag, model_name):
    """Load (x, t, q) from a .npz file, or return None if missing."""
    path = _sol_path(case_tag, model_name)
    if os.path.exists(path):
        data = np.load(path)
        print(f"  Loaded: {path}")
        return data["x"], data["t"], data["q"]
    return None


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    t_start_total = time.time()

    # Precompute Vmax profiles
    x_grid = np.linspace(X_RANGE[0] + DX / 2, X_RANGE[1] - DX / 2, NX)
    Vmax_profiles = _build_Vmax_profiles(x_grid, DX, NX)

    # Two parameter sets
    cases = {
        "A": {"m": 3, "eta": 0.1, "delta": 0.06},
        "B": {"m": 3, "eta": 1.0, "delta": 0.0},
    }

    snap_times = [0.1, 0.25, 0.4]

    # All model names in order
    model_names = ["spatial (no memory)"] + list(KERNELS.keys())

    for tag, params in cases.items():
        m, eta, delta = params["m"], params["eta"], params["delta"]
        case_label = (rf"Case {tag}: $m={m}$, $\eta={eta}$, "
                      rf"$\delta={delta}$, $\tau_0={TAU0}$")

        print("=" * 60)
        print(f"CASE {tag}: m={m}, eta={eta}, delta={delta}")
        print("=" * 60)

        all_results = {}
        t_case = time.time()

        # --- Spatial-only (reference) ---
        cached = load_solution(tag, "spatial (no memory)") if SAVE_SOLUTIONS else None
        if cached is not None:
            x_s, t_s, q_s = cached
        else:
            print("\n--- Spatial (no memory) ---")
            t0 = time.time()
            x_s, t_s, q_s, info_s = solve_spatial(
                eta, delta, m, Vmax_profiles)
            print(f"  Time: {time.time() - t0:.1f}s")
            if SAVE_SOLUTIONS:
                save_solution(tag, "spatial (no memory)", x_s, t_s, q_s)
        all_results["spatial (no memory)"] = (x_s, t_s, q_s)

        # --- Memory variants ---
        for kname, K_func in KERNELS.items():
            cached = load_solution(tag, kname) if SAVE_SOLUTIONS else None
            if cached is not None:
                x_m, t_m, q_m = cached
            else:
                print(f"\n--- Memory: {kname} ---")
                t0 = time.time()
                x_m, t_m, q_m, info_m = solve_with_memory(
                    eta, delta, m, Vmax_profiles, K_func, label=kname)
                print(f"  Time: {time.time() - t0:.1f}s")
                if SAVE_SOLUTIONS:
                    save_solution(tag, kname, x_m, t_m, q_m)
            all_results[kname] = (x_m, t_m, q_m)

        print(f"\n  Case {tag} total: {time.time() - t_case:.1f}s")

        # --- Plots ---
        results_mem = {k: v for k, v in all_results.items()
                       if k != "spatial (no memory)"}

        plot_comparison_grid(
            all_results, case_label,
            filename=f"density_case{tag}.png")

        plot_snapshots_overlay(
            all_results, snap_times, case_label,
            filename=f"snapshots_case{tag}.png")

        plot_l1_difference(
            results_mem, x_s, t_s, q_s, case_label,
            filename=f"l1diff_case{tag}.png")

    print(f"\nTotal time: {time.time() - t_start_total:.1f}s")
    print("All comparison plots saved.")