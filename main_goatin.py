"""
main_goatin.py - Replicate the numerical experiment from:
  Chiarello, Goatin, Rossi (2018), "Stability estimates for non-local
  scalar conservation laws", Section 4.

Problem (4.5)-(4.8): nonlocal traffic flow on a circular road.
  d_t rho + d_x(f(t,x,rho) v(w_{eta,delta} * rho)) = 0,
    x in (-1, 1), periodic BCs, T = 0.5, rho_0 = 0.6.

  f(t,x,rho) = Vmax(t,x) rho(1-rho)
  v(rho)      = (1-rho)^{m-1} (1+rho)^m
  w_{eta,delta} = quintic kernel with support radius eta, shift delta

We use the fixed-point solver from claws_LXF.py.
"""

import numpy as np
import time
from solver.claws_LXF import solve_nonlocal_space
from utils.plot_claws import (plot_spacetime_grid, plot_functional)


# ================================================================
# Physical setup
# ================================================================
X_RANGE = (-1.0, 1.0)
T_FINAL = 0.5
RHO_0   = 0.6     # constant initial density

# Spatial discretization (paper: dx = 0.001 => Nx = 2000)
NX  = 2000
DX  = (X_RANGE[1] - X_RANGE[0]) / NX
CFL = 0.9
TOL = 1e-7
MAX_ITER = 50


# ================================================================
# Vmax(t, x): space-time dependent speed limit (eq. 4.6)
# ================================================================
# Vmax = g * phi, where g is a Gaussian smoother and phi is
# piecewise constant.  The paper states sigma=10 for g, but this
# yields a nearly flat Vmax (sigma >> domain size).  From Figure 1
# the effective smoothing width is ~0.01, consistent with sigma=0.01
# (likely sigma^{-1}=100 in the original notation).
SIGMA_GAUSS = 0.01


def _phi_speed(t, x):
    """Piecewise constant speed limit (before Gaussian smoothing)."""
    phi = np.full_like(x, 7.0)
    mid = (x > -1.0/3) & (x <= 1.0/3)
    if t <= 1.0/6 or t > 1.0/3:
        phi[mid] = 3.0
    else:
        phi[mid] = 1.5
    return phi


def _build_Vmax_profiles(x, dx, Nx):
    """Precompute the two distinct Vmax(x) profiles (smoothed by Gaussian).

    Profile 0: phi middle = 3.0   (t in [0,1/6] and (1/3,1/2])
    Profile 1: phi middle = 1.5   (t in (1/6,1/3])

    Uses periodic FFT convolution with a Gaussian kernel.
    """
    # Gaussian kernel on the periodic grid
    k_idx = np.arange(Nx)
    k_dist = np.where(k_idx <= Nx // 2,
                      k_idx * dx, (k_idx - Nx) * dx)
    g = np.exp(-0.5 * (k_dist / SIGMA_GAUSS)**2)
    g /= (g.sum() * dx)   # normalize so integral = 1
    g_fft = np.fft.fft(g)

    profiles = []
    for t_rep in [0.0, 0.25]:   # representatives for each regime
        phi_vals = _phi_speed(t_rep, x)
        vmax = np.real(np.fft.ifft(np.fft.fft(phi_vals) * g_fft)) * dx
        profiles.append(vmax)
    return profiles


# ================================================================
# Kernel w_{eta, delta} (eq. 4.8)
# ================================================================
# w(x) = (16 / (5 pi eta^6)) (eta^2 - (x-delta)^2)^{5/2}
#         on [-eta+delta, eta+delta], zero elsewhere.
# Normalized to integrate to 1.

def make_kernel(eta, delta):
    """Return a callable kernel w_{eta,delta}(z) for the given parameters."""
    C = 16.0 / (5.0 * np.pi * eta**6)

    def w(z):
        u = z - delta
        mask = np.abs(u) < eta
        vals = np.zeros_like(z, dtype=float)
        vals[mask] = C * (eta**2 - u[mask]**2)**2.5
        return vals
    return w


# ================================================================
# Velocity function v (eq. 4.7)
# ================================================================

def make_v(m):
    """Return v(rho) = (1-rho)^{m-1} (1+rho)^m."""
    def v(rho):
        return (1.0 - rho)**(m - 1) * (1.0 + rho)**m
    return v


def max_v(m):
    """Upper bound of |v(w)| for w in [0, 1].

    Critical point at w* = 1/(2m-1).
    For m=1, v(w)=1+w, max at w=1 is 2.
    """
    if m == 1:
        return 2.0
    w_star = 1.0 / (2 * m - 1)
    return (1.0 - w_star)**(m - 1) * (1.0 + w_star)**m


# ================================================================
# Flux and alpha
# ================================================================
# Total flux: Phi(t,x,rho) = Vmax(t,x) rho(1-rho) v(W)
#
# Frozen-flux derivative: d_rho Phi = Vmax(t,x) (1-2rho) v(W)
# => alpha >= max(Vmax) * max|v|   (CFL condition 4.4)


def make_flux(Vmax_profiles, v_func):
    """Build the flux function F(t, x, w, q) for the solver."""
    def F(t, x, w, q):
        # Select the correct Vmax profile based on time
        if 1.0/6 < t <= 1.0/3:
            vmax = Vmax_profiles[1]
        else:
            vmax = Vmax_profiles[0]
        return vmax * q * (1.0 - q) * v_func(w)
    return F


# ================================================================
# Cost functionals (eqs. 4.9, 4.10)
# ================================================================

def compute_J(q, dx, dt):
    """J(T) = int_0^T TV_x(rho(t,.)) dt  (eq. 4.9).

    Spatial TV with periodic wrapping; trapezoidal rule in time.
    """
    tv = np.sum(np.abs(np.diff(q, axis=1, append=q[:, :1])), axis=1)
    return np.trapz(tv, dx=dt)


def compute_Psi(q, x, dx, dt, a=-4.0/5, b=-1.0/3):
    """Psi(T; a, b) = int_0^T int_a^b phi(rho) dx dt  (eq. 4.10).

    phi(r) = 0 if r<0.75, 10r-7.5 if 0.75<=r<=0.85, 1 if r>0.85.
    """
    def phi_queue(r):
        out = np.zeros_like(r)
        mid = (r >= 0.75) & (r <= 0.85)
        out[mid] = 10.0 * r[mid] - 7.5
        out[r > 0.85] = 1.0
        return out

    mask = (x >= a) & (x <= b)
    integrand = np.sum(phi_queue(q[:, mask]) * dx, axis=1)
    return np.trapz(integrand, dx=dt)


# ================================================================
# Single solve wrapper
# ================================================================

def run_single(eta, delta, m, Vmax_profiles, verbose=False):
    """Solve the nonlocal traffic problem for given (eta, delta, m)."""
    v_func = make_v(m)
    kernel = make_kernel(eta, delta)
    F_flux = make_flux(Vmax_profiles, v_func)
    J_id = lambda q: q.copy()
    q0_func = lambda x: np.full_like(x, RHO_0)

    # Viscosity coefficient (CFL condition 4.4)
    max_Vmax = max(np.max(Vmax_profiles[0]), np.max(Vmax_profiles[1]))
    alpha = max_Vmax * max_v(m)

    x, t, q, info = solve_nonlocal_space(
        F_flux, J_id, kernel, q0_func, X_RANGE, NX, T_FINAL,
        alpha=alpha, cfl=CFL, tol=TOL, max_iter=MAX_ITER,
        bc='periodic', verbose=verbose
    )
    return x, t, q, info


def run_single_direct(eta, delta, m, Vmax_profiles, verbose=False):
    """
    Direct (explicit) LxF march for the Goatin problem — no fixed-point loop.

    At each time step n:
      1. Compute W^n = kernel * q^n  (current-step density, no iteration).
      2. Advance q^n -> q^{n+1} with one periodic LxF step.
    """
    v_func   = make_v(m)
    kernel   = make_kernel(eta, delta)
    F_flux   = make_flux(Vmax_profiles, v_func)

    max_Vmax = max(np.max(Vmax_profiles[0]), np.max(Vmax_profiles[1]))
    alpha    = max_Vmax * max_v(m)

    # --- Grid ---
    dx = (X_RANGE[1] - X_RANGE[0]) / NX
    x  = np.linspace(X_RANGE[0] + dx / 2, X_RANGE[1] - dx / 2, NX)
    dt = CFL * dx / alpha
    Nt = max(1, int(np.ceil(T_FINAL / dt)))
    dt = T_FINAL / Nt
    t  = np.linspace(0, T_FINAL, Nt + 1)

    if verbose:
        print(f"Direct LxF: Nx={NX}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}")

    # FFT-based periodic spatial convolution
    k_idx  = np.arange(NX)
    k_dist = np.where(k_idx <= NX // 2, k_idx * dx, (k_idx - NX) * dx)
    kfft   = np.fft.fft(kernel(k_dist))

    def conv(q):
        return np.real(np.fft.ifft(np.fft.fft(q) * kfft)) * dx

    lam   = dt / dx
    q     = np.full(NX, RHO_0)
    q_out = np.zeros((Nt + 1, NX))
    q_out[0] = q.copy()

    for n in range(Nt):
        W_n  = conv(q)                          # W^n from current density
        flux = F_flux(t[n], x, W_n, q)

        # Periodic LxF step
        q_r   = np.roll(q, -1)
        fl_r  = np.roll(flux, -1)
        F_hat = 0.5 * (flux + fl_r) - 0.5 * alpha * (q_r - q)
        F_l   = np.roll(F_hat, 1)
        q     = q - lam * (F_hat - F_l)

        q_out[n + 1] = q.copy()

    return x, t, q_out


# ================================================================
# Parameter sweeps
# ================================================================

def sweep_eta(Vmax_profiles):
    """Sweep eta = 0.1:0.1:1, with m=3, delta=0."""
    etas = np.arange(0.1, 1.01, 0.1)
    m, delta = 3, 0.0
    Js, Psis = [], []
    solutions = {}

    for eta in etas:
        print(f"  eta={eta:.1f}", end=" ", flush=True)
        x, t, q, info = run_single(eta, delta, m, Vmax_profiles)
        dt = t[1] - t[0]
        Js.append(compute_J(q, DX, dt))
        Psis.append(compute_Psi(q, x, DX, dt))
        if np.isclose(eta, 0.2) or np.isclose(eta, 0.5) or np.isclose(eta, 1.0):
            solutions[round(eta, 1)] = (x, t, q)
        print(f"iters={info['iters']}, J={Js[-1]:.4f}, Psi={Psis[-1]:.4f}")

    return etas, np.array(Js), np.array(Psis), solutions


def sweep_delta(Vmax_profiles):
    """Sweep delta = -0.1:0.02:0.1, with m=3, eta=0.1."""
    deltas = np.arange(-0.1, 0.101, 0.02)
    m, eta = 3, 0.1
    Js, Psis = [], []
    solutions = {}

    for delta in deltas:
        print(f"  delta={delta:.2f}", end=" ", flush=True)
        x, t, q, info = run_single(eta, delta, m, Vmax_profiles)
        dt = t[1] - t[0]
        Js.append(compute_J(q, DX, dt))
        Psis.append(compute_Psi(q, x, DX, dt))
        if np.isclose(delta, -0.04) or np.isclose(delta, 0.06) \
           or np.isclose(delta, 0.08):
            solutions[round(delta, 2)] = (x, t, q)
        print(f"iters={info['iters']}, J={Js[-1]:.4f}, Psi={Psis[-1]:.4f}")

    return deltas, np.array(Js), np.array(Psis), solutions


def sweep_m(Vmax_profiles):
    """Sweep m = 1:1:10, with eta=0.1, delta=0."""
    ms = np.arange(1, 11)
    eta, delta = 0.1, 0.0
    Js, Psis = [], []
    solutions = {}

    for m in ms:
        print(f"  m={m}", end=" ", flush=True)
        x, t, q, info = run_single(eta, delta, int(m), Vmax_profiles)
        dt = t[1] - t[0]
        Js.append(compute_J(q, DX, dt))
        Psis.append(compute_Psi(q, x, DX, dt))
        if m in [3, 10]:
            solutions[m] = (x, t, q)
        print(f"iters={info['iters']}, J={Js[-1]:.4f}, Psi={Psis[-1]:.4f}")

    return ms, np.array(Js), np.array(Psis), solutions


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    t_start_total = time.time()

    # --- Precompute Vmax profiles ---
    x_grid = np.linspace(X_RANGE[0] + DX/2, X_RANGE[1] - DX/2, NX)
    Vmax_profiles = _build_Vmax_profiles(x_grid, DX, NX)

    # --- Sweep eta (Figure 2 & 5) ---
    print("=" * 60)
    print("SWEEP: eta in [0.1, 1.0], m=3, delta=0")
    print("=" * 60)
    t0 = time.time()
    etas, J_eta, Psi_eta, sol_eta = sweep_eta(Vmax_profiles)
    print(f"  Sweep eta: {time.time() - t0:.1f}s")

    plot_functional(etas, J_eta, xlabel=r"$\eta$",
                    ylabel=r"$J(T)$", filename="figures/Goatin/J_vs_eta.png")
    plot_functional(etas, Psi_eta, xlabel=r"$\eta$",
                    ylabel=r"$\Psi(T)$", filename="figures/Goatin/Psi_vs_eta.png")
    fig5_data = {k: sol_eta[k] for k in [0.2, 0.5, 1.0] if k in sol_eta}
    if fig5_data:
        plot_spacetime_grid(
            list(fig5_data.values()),
            [rf"$\eta={k}$" for k in fig5_data],
            suptitle=r"$m=3$, $\delta=0$",
            filename="figures/Goatin/density_eta.png")

    # --- Sweep delta (Figure 3 & 6) ---
    print()
    print("=" * 60)
    print("SWEEP: delta in [-0.1, 0.1], m=3, eta=0.1")
    print("=" * 60)
    t0 = time.time()
    deltas, J_delta, Psi_delta, sol_delta = sweep_delta(Vmax_profiles)
    print(f"  Sweep delta: {time.time() - t0:.1f}s")

    plot_functional(deltas, J_delta, xlabel=r"$\delta$",
                    ylabel=r"$J(T)$", filename="figures/Goatin/J_vs_delta.png")
    plot_functional(deltas, Psi_delta, xlabel=r"$\delta$",
                    ylabel=r"$\Psi(T)$", filename="figures/Goatin/Psi_vs_delta.png")
    fig6_data = {k: sol_delta[k] for k in [-0.04, 0.06, 0.08]
                 if k in sol_delta}
    if fig6_data:
        plot_spacetime_grid(
            list(fig6_data.values()),
            [rf"$\delta={k}$" for k in fig6_data],
            suptitle=r"$m=3$, $\eta=0.1$",
            filename="figures/Goatin/density_delta.png")

    # --- Sweep m (Figure 4 & 7) ---
    print()
    print("=" * 60)
    print("SWEEP: m in [1, 10], eta=0.1, delta=0")
    print("=" * 60)
    t0 = time.time()
    ms, J_m, Psi_m, sol_m = sweep_m(Vmax_profiles)
    print(f"  Sweep m: {time.time() - t0:.1f}s")

    plot_functional(ms, J_m, xlabel=r"$m$",
                    ylabel=r"$J(T)$", filename="figures/Goatin/J_vs_m.png")
    plot_functional(ms, Psi_m, xlabel=r"$m$",
                    ylabel=r"$\Psi(T)$", filename="figures/Goatin/Psi_vs_m.png")
    fig7_data = {k: sol_m[k] for k in [3, 10] if k in sol_m}
    if fig7_data:
        plot_spacetime_grid(
            list(fig7_data.values()),
            [rf"$m={k}$" for k in fig7_data],
            suptitle=r"$\eta=0.1$, $\delta=0$",
            filename="figures/Goatin/density_m.png")

    print()
    print(f"Total time: {time.time() - t_start_total:.1f}s")
    print("All plots saved.")

    # ================================================================
    # FP vs direct comparison (eta=0.5, delta=0, m=3)
    # ================================================================
    ETA_CMP, DELTA_CMP, M_CMP = 0.5, 0.0, 3

    print()
    print("=" * 60)
    print(f"FP vs Direct comparison: eta={ETA_CMP}, delta={DELTA_CMP}, m={M_CMP}")
    print("=" * 60)

    x_fp, t_fp, q_fp, _ = run_single(ETA_CMP, DELTA_CMP, M_CMP,
                                      Vmax_profiles, verbose=True)
    x_dir, t_dir, q_dir = run_single_direct(ETA_CMP, DELTA_CMP, M_CMP,
                                            Vmax_profiles, verbose=True)

    # Interpolate direct solution onto FP time grid if grids differ
    # (they use the same alpha/CFL so grids should match; assert for safety)
    assert q_fp.shape == q_dir.shape, \
        f"Grid mismatch: FP {q_fp.shape} vs direct {q_dir.shape}"

    err = np.abs(q_fp - q_dir)   # pointwise absolute error

    fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(15, 4))

    T_mesh, X_mesh = np.meshgrid(t_fp, x_fp, indexing="ij")
    vmin, vmax_ = 0.0, 1.0

    # --- Left: fixed-point solution ---
    pcm0 = axes_cmp[0].pcolormesh(X_mesh, T_mesh, q_fp,
                                   shading="auto", cmap="jet",
                                   vmin=vmin, vmax=vmax_)
    fig_cmp.colorbar(pcm0, ax=axes_cmp[0], label=r"$\rho$")
    axes_cmp[0].set_xlabel("$x$")
    axes_cmp[0].set_ylabel("$t$")
    axes_cmp[0].set_title("Fixed-point solver")

    # --- Center: direct LxF solution ---
    pcm1 = axes_cmp[1].pcolormesh(X_mesh, T_mesh, q_dir,
                                   shading="auto", cmap="jet",
                                   vmin=vmin, vmax=vmax_)
    fig_cmp.colorbar(pcm1, ax=axes_cmp[1], label=r"$\rho$")
    axes_cmp[1].set_xlabel("$x$")
    axes_cmp[1].set_ylabel("$t$")
    axes_cmp[1].set_title("Direct LxF")

    # --- Right: pointwise error ---
    pcm2 = axes_cmp[2].pcolormesh(X_mesh, T_mesh, err,
                                   shading="auto", cmap="Reds")
    fig_cmp.colorbar(pcm2, ax=axes_cmp[2], label=r"$|\rho_{\mathrm{FP}} - \rho_{\mathrm{direct}}|$")
    axes_cmp[2].set_xlabel("$x$")
    axes_cmp[2].set_ylabel("$t$")
    axes_cmp[2].set_title("Pointwise error")

    fig_cmp.suptitle(
        rf"Chiarello--Goatin: $\eta={ETA_CMP}$, $\delta={DELTA_CMP}$, $m={M_CMP}$",
        fontsize=12)
    fig_cmp.tight_layout()
    fig_cmp.savefig("figures/Goatin/fp_vs_direct.png", dpi=150)
    print("Saved: figures/Goatin/fp_vs_direct.png")