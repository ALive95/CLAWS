"""
plot_claws.py - Plotting utilities for CLAWS solver output.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_snapshots(x, t, q, times, title="", filename=None):
    """
    Plot solution snapshots at selected times.

    Parameters
    ----------
    x      : (Nx,) cell centers
    t      : (Nt+1,) time levels
    q      : (Nt+1, Nx) solution
    times  : list of floats, desired snapshot times
    title  : figure title
    filename : if given, save to file
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.viridis(np.linspace(0.1, 0.9, len(times)))

    for tc, col in zip(times, colors):
        n = np.argmin(np.abs(t - tc))
        ax.plot(x, q[n], color=col, lw=1.8, label=f"$t = {t[n]:.3f}$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$q(t, x)$")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
    return fig, ax


def plot_spacetime(x, t, q, title="", filename=None,
                   vmin=None, vmax=None):
    """
    Space-time contour plot of the solution.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    T_mesh, X_mesh = np.meshgrid(t, x, indexing="ij")
    pcm = ax.pcolormesh(X_mesh, T_mesh, q, shading="auto",
                        cmap="jet", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="$\\rho$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title(title)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
    return fig, ax


def plot_convergence(residuals, title="", filename=None):
    """
    Plot fixed-point residuals vs iteration.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(residuals) + 1), residuals, "o-", ms=4)
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel(r"$\|q^{(k)} - q^{(k-1)}\|_\infty$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
    return fig, ax


def plot_comparison(x, t, q_space, q_memory, snap_time, filename=None):
    """
    Side-by-side comparison of spatial-only vs memory solutions at a given time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    n_s = np.argmin(np.abs(t - snap_time))
    axes[0].plot(x, q_space[n_s], "k-", lw=1.8)
    axes[0].set_title(f"Nonlocal in space, $t={t[n_s]:.3f}$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$q(t, x)$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, q_memory[n_s], "k-", lw=1.8)
    axes[1].set_title(f"Nonlocal + memory, $t={t[n_s]:.3f}$")
    axes[1].set_xlabel("$x$")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
    return fig, axes


# ================================================================
# Chiarello-Goatin-Rossi experiment plots
# ================================================================

def plot_functional(params, values, xlabel="", ylabel="",
                    filename=None):
    """
    Plot a cost functional (J or Psi) vs a swept parameter.
    Reproduces Figures 2-4 of Chiarello et al.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(params, values, "o-", color="tab:blue", ms=5, lw=1.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
    return fig, ax


def plot_spacetime_grid(solutions, labels, suptitle="",
                        filename=None):
    """
    Row of (t,x) density plots for multiple solutions.
    Reproduces Figures 5-7 of Chiarello et al.

    Parameters
    ----------
    solutions : list of (x, t, q) tuples
    labels    : list of subplot titles
    suptitle  : overall figure title
    filename  : if given, save to file
    """
    n = len(solutions)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (x, t, q), lab in zip(axes, solutions, labels):
        T_mesh, X_mesh = np.meshgrid(t, x, indexing="ij")
        pcm = ax.pcolormesh(X_mesh, T_mesh, q, shading="auto",
                            cmap="jet", vmin=0, vmax=1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        ax.set_title(lab)
        fig.colorbar(pcm, ax=ax, shrink=0.85)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
    return fig, axes