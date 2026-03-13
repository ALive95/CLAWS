"""
claws_LXF.py - Conservation Laws with Memory (CLAWS)

Numerical solver for 1D nonlocal conservation laws via fixed-point iteration.

Models:
  [Space]   d_t q + d_x(F(t, x, W, q)) = 0,   W = int gamma(x-y) J(q(t,y)) dy
  [Memory]  d_t q + d_x(F(t, x, W, q)) = 0,   W = int_{-inf}^{t} int kappa(t-s, x-y) J(q(s,y)) dy ds

Numerical methods:
  Inner solver : Lax-Friedrichs finite volume scheme (conservative, entropy-satisfying)
  Outer loop   : Picard fixed-point iteration (contraction guaranteed for small T)
"""

import numpy as np


# ================================================================
# Nonlocal operators
# ================================================================

def eval_W_space(J_q, gamma, x, dx):
    """
    Evaluate the spatial nonlocal operator via midpoint quadrature:
      W(x_i) = int gamma(x_i - y) J(q(y)) dy
             ~ dx * sum_j gamma(x_i - x_j) * J(q_j)

    Parameters
    ----------
    J_q   : (Nx,) J(q) evaluated at cell centers
    gamma : callable(z_arr) -> kernel values (vectorized)
    x     : (Nx,) cell centers
    dx    : cell width

    Returns
    -------
    W : (Nx,) nonlocal operator at cell centers
    """
    # Kernel matrix: G[i,j] = gamma(x_i - x_j)
    G = gamma(x[:, None] - x[None, :])
    return dx * (G @ J_q)


def eval_W_memory(J_q_all, kappa, x, dx, t_levels, n):
    """
    Evaluate the nonlocal operator with memory at time level n:
      W(t_n, x_i) = int_{-inf}^{t_n} int kappa(t_n - s, x_i - y) J(q(s,y)) dy ds

    Discretized via rectangle rule in time, midpoint in space:
      W_i ~ sum_{k < n} dt_k * dx * sum_j kappa(t_n - t_k, x_i - x_j) * J(q(t_k, x_j))

    Parameters
    ----------
    J_q_all  : (M, Nx) J(q) at all time levels 0..M-1
    kappa    : callable(tau, dz_arr) -> kernel (vectorized)
    x        : (Nx,) cell centers
    dx       : cell width
    t_levels : (M,) time grid
    n        : current time index (W is evaluated at t_levels[n])

    Returns
    -------
    W : (Nx,) nonlocal operator at cell centers
    """
    Nx = len(x)
    W = np.zeros(Nx)
    t_n = t_levels[n]
    diffs_x = x[:, None] - x[None, :]  # (Nx, Nx)

    for k in range(n):
        tau = t_n - t_levels[k]              # time lag > 0
        dt_k = t_levels[k + 1] - t_levels[k] # quadrature weight
        K = kappa(tau, diffs_x)              # (Nx, Nx) kernel matrix
        W += dt_k * dx * (K @ J_q_all[k])

    return W


# ================================================================
# Local conservation law solver: Lax-Friedrichs finite volume
# ================================================================

def solve_local_lxf(phi, q_init, dx, dt, Nt, alpha, bc='outflow'):
    """
    Solve  d_t q + d_x(Phi(n, q)) = 0  on a uniform 1D grid
    using the (global) Lax-Friedrichs scheme.

    Numerical flux at cell interface j+1/2:
      F_hat_{j+1/2} = 1/2 (Phi_j + Phi_{j+1}) - alpha/2 (q_{j+1} - q_j)

    Conservative update:
      q_j^{n+1} = q_j^n - (dt/dx) (F_hat_{j+1/2} - F_hat_{j-1/2})

    Stability requires: alpha >= max|d_q Phi|  and  alpha * dt / dx <= 1.

    Parameters
    ----------
    phi    : callable(n, q_arr) -> flux array at time index n
             (encodes the frozen nonlocal term)
    q_init : (Nx,) initial condition
    dx, dt : spatial and temporal step sizes
    Nt     : number of time steps
    alpha  : numerical viscosity (max wave speed bound)
    bc     : 'outflow' (zero-gradient) or 'periodic'

    Returns
    -------
    q_hist : (Nt+1, Nx) solution at all time levels
    """
    Nx = len(q_init)
    lam = dt / dx  # mesh ratio

    q_hist = np.zeros((Nt + 1, Nx))
    q_hist[0] = q_init.copy()
    q = q_init.copy()

    for n in range(Nt):
        # Flux at cell centers
        flux = phi(n, q)

        if bc == 'periodic':
            # Periodic LxF: all Nx interfaces, wrapping at boundaries.
            # F_hat[j] = numerical flux at interface j+1/2.
            q_right = np.roll(q, -1)        # q_{j+1}, periodic
            flux_right = np.roll(flux, -1)   # Phi_{j+1}
            F_hat = 0.5 * (flux + flux_right) \
                  - 0.5 * alpha * (q_right - q)
            # Conservative update: q_j -= lam * (F_{j+1/2} - F_{j-1/2})
            F_hat_left = np.roll(F_hat, 1)   # F_{j-1/2}
            q -= lam * (F_hat - F_hat_left)
        else:
            # LxF numerical flux at interfaces j+1/2, j = 0,...,Nx-2
            F_hat = 0.5 * (flux[:-1] + flux[1:]) \
                  - 0.5 * alpha * (q[1:] - q[:-1])
            # Update interior cells j = 1,...,Nx-2
            q[1:-1] -= lam * (F_hat[1:] - F_hat[:-1])
            # Outflow (zero-gradient) boundary conditions
            q[0]  = q[1]
            q[-1] = q[-2]

        q_hist[n + 1] = q.copy()

    return q_hist


# ================================================================
# Fixed-point solver: nonlocal in space only
# ================================================================

def solve_nonlocal_space(F, J, gamma, q0_func, x_range, Nx, T,
                         alpha, cfl=0.5, max_iter=50, tol=1e-8,
                         bc='outflow', verbose=True):
    """
    Solve the 1D nonlocal conservation law (spatial nonlocality) via
    Picard fixed-point iteration.

      d_t q + d_x(F(t, x, W[J(q), gamma], q)) = 0,   t in (0, T)
      q(0, x) = q0(x)

    Fixed-point procedure (cf. paper, Definition 3.5):
      1. Given iterate q~, compute  w(t,x) = W[J(q~), gamma](t,x)
      2. Solve the LOCAL conservation law  d_t q + d_x(F(t,x,w(t,x),q)) = 0
         with the Lax-Friedrichs scheme (w is frozen / prescribed)
      3. Update q~ <- q, check contraction

    Convergence is guaranteed for small T (paper, Lemma 3.10).

    Parameters
    ----------
    F        : callable(t, x_arr, w_arr, q_arr) -> flux array
               The full flux F(t, x, w, q) where w is the nonlocal term.
    J        : callable(q_arr) -> array, the nonlocal nonlinearity
    gamma    : callable(z_arr) -> array, the spatial kernel
    q0_func  : callable(x_arr) -> array, initial condition
    x_range  : (x_left, x_right)
    Nx       : number of spatial cells
    T        : final time
    alpha    : max wave speed bound, i.e. alpha >= max|d_q F|
    cfl      : CFL number (0 < cfl <= 1)
    max_iter : maximum Picard iterations
    tol      : convergence tolerance (L^inf norm)
    bc       : 'outflow' or 'periodic'
    verbose  : print iteration info

    Returns
    -------
    x    : (Nx,) cell centers
    t    : (Nt+1,) time levels
    q    : (Nt+1, Nx) converged solution
    info : dict with 'residuals', 'converged', 'iters'
    """
    # --- Spatial grid ---
    dx = (x_range[1] - x_range[0]) / Nx
    x = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, Nx)

    # --- Time grid (from CFL) ---
    dt = cfl * dx / alpha
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt  # adjust to hit T exactly
    t = np.linspace(0, T, Nt + 1)

    # --- Initial condition ---
    q0 = q0_func(x)

    if verbose:
        print(f"Grid: Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}, "
              f"CFL={alpha * dt / dx:.3f}")

    # --- Precompute kernel for convolution ---
    if bc == 'periodic':
        # FFT-based periodic convolution.
        # Kernel on shifted grid: k_dist[j] = j*dx for j<=Nx/2,
        #   (j-Nx)*dx for j>Nx/2  (wrapping to negative distances).
        k_idx = np.arange(Nx)
        k_dist = np.where(k_idx <= Nx // 2,
                          k_idx * dx, (k_idx - Nx) * dx)
        gamma_fft = np.fft.fft(gamma(k_dist))

    # --- Fixed-point iteration ---
    # Initial guess: q~(t, .) = q0 for all t  (constant extension)
    q_prev = np.tile(q0, (Nt + 1, 1))
    residuals = []

    for it in range(max_iter):

        # Step 1: evaluate w(t_n, x) = W[J(q~^n), gamma] at every time level
        w = np.zeros((Nt + 1, Nx))
        for n in range(Nt + 1):
            Jq = J(q_prev[n])
            if bc == 'periodic':
                # W_i = dx * sum_j gamma(x_i - x_j) J(q_j), periodic
                w[n] = np.real(np.fft.ifft(
                    np.fft.fft(Jq) * gamma_fft)) * dx
            else:
                w[n] = eval_W_space(Jq, gamma, x, dx)

        # Step 2: build the frozen flux phi(n, q) = F(t_n, x, w^n, q)
        #         This is what the LxF solver will use as its flux function.
        def phi(n, q_arr, _w=w, _t=t, _x=x):
            return F(_t[n], _x, _w[n], q_arr)

        # Step 3: solve the local conservation law with LxF
        q_new = solve_local_lxf(phi, q0, dx, dt, Nt, alpha, bc=bc)

        # Step 4: convergence check (L^inf over all of space-time)
        diff = np.max(np.abs(q_new - q_prev))
        residuals.append(diff)

        if verbose:
            print(f"  iter {it + 1:3d}: ||dq||_inf = {diff:.3e}")

        if diff < tol:
            if verbose:
                print(f"  Converged in {it + 1} iterations.")
            return x, t, q_new, {
                "residuals": residuals, "converged": True, "iters": it + 1
            }

        q_prev = q_new.copy()

    if verbose:
        print(f"  Not converged (residual={diff:.3e})")
    return x, t, q_new, {
        "residuals": residuals, "converged": False, "iters": max_iter
    }


# ================================================================
# Fixed-point solver: nonlocal in space + memory in time
# ================================================================

def solve_nonlocal_memory(F, J, kappa, q0_hist_func, q0_init_func,
                          x_range, Nx, T, T_hist, N_hist,
                          alpha, cfl=0.5, max_iter=50, tol=1e-8,
                          bc='outflow', verbose=True):
    """
    Solve the 1D nonlocal conservation law with memory via Picard
    fixed-point iteration.

      d_t q + d_x(F(t, x, W, q)) = 0,            t in (0, T)
      W(t,x) = int_{-inf}^{t} int kappa(t-s, x-y) J(q(s,y)) dy ds
      q(t,x) = q0(t,x),                           t <= 0

    The W operator integrates over ALL past times s < t, including both
    the prescribed history (t <= 0) and the forward solution (0 < s < t).
    This couples W to the solution itself, necessitating fixed-point iteration.

    The historical contribution to W is precomputed once since q0 is fixed.

    Parameters
    ----------
    F             : callable(t, x_arr, w_arr, q_arr) -> flux array
    J             : callable(q_arr) -> array
    kappa         : callable(tau, dz_arr) -> kernel kappa(tau, z)
    q0_hist_func  : callable(t, x_arr) -> historical data for t <= 0
    q0_init_func  : callable(x_arr) -> initial condition at t = 0
    x_range       : (x_left, x_right)
    Nx            : spatial cells
    T             : final time
    T_hist        : history window length (history on [-T_hist, 0])
    N_hist        : time steps in the historical window
    alpha         : max wave speed
    cfl, max_iter, tol, verbose : solver parameters

    Returns
    -------
    x, t_fwd, q_fwd, info
    """
    # --- Spatial grid ---
    dx = (x_range[1] - x_range[0]) / Nx
    x = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, Nx)

    # --- Forward time grid ---
    dt = cfl * dx / alpha
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt
    t_fwd = np.linspace(0, T, Nt + 1)

    # --- Historical time grid ---
    dt_hist = T_hist / max(N_hist, 1)
    t_hist = np.linspace(-T_hist, 0, N_hist + 1)

    # --- Initial condition ---
    q0 = q0_init_func(x)

    if verbose:
        print(f"Grid: Nx={Nx}, Nt={Nt}, N_hist={N_hist}")
        print(f"  dx={dx:.4e}, dt={dt:.4e}, dt_hist={dt_hist:.4e}")

    # --- Precompute J(q0) on historical grid ---
    J_q_hist = np.zeros((N_hist + 1, Nx))
    for k in range(N_hist + 1):
        J_q_hist[k] = J(q0_hist_func(t_hist[k], x))

    # --- Precompute historical contribution to W ---
    # W_hist(t_n, x) = sum over historical levels k:
    #   dt_k * dx * sum_j kappa(t_n - t_hist[k], x_i - x_j) * J(q0(t_k, x_j))
    # This does not change across fixed-point iterations.
    diffs_x = x[:, None] - x[None, :]  # (Nx, Nx), reused below
    W_hist = np.zeros((Nt + 1, Nx))

    for n in range(Nt + 1):
        t_n = t_fwd[n]
        for k in range(N_hist):
            tau = t_n - t_hist[k]                  # time lag
            dtk = t_hist[k + 1] - t_hist[k]        # quadrature weight
            K = kappa(tau, diffs_x)                 # (Nx, Nx)
            W_hist[n] += dtk * dx * (K @ J_q_hist[k])

    if verbose:
        print(f"  Historical W precomputed.")

    # --- Fixed-point iteration ---
    q_prev = np.tile(q0, (Nt + 1, 1))
    residuals = []

    for it in range(max_iter):

        # Step 1: total W = W_hist + W_fwd
        # W_fwd(t_n, x) = sum_{k=0}^{n-1} dt * dx * sum_j kappa(t_n - t_fwd[k], x_i-x_j) J(q~_k_j)
        w = W_hist.copy()
        for n in range(1, Nt + 1):
            t_n = t_fwd[n]
            for k in range(n):
                tau = t_n - t_fwd[k]               # time lag
                K = kappa(tau, diffs_x)
                w[n] += dt * dx * (K @ J(q_prev[k]))

        # Step 2: frozen flux
        def phi(n, q_arr, _w=w, _t=t_fwd, _x=x):
            return F(_t[n], _x, _w[n], q_arr)

        # Step 3: solve local CL
        q_new = solve_local_lxf(phi, q0, dx, dt, Nt, alpha, bc=bc)

        # Step 4: convergence
        diff = np.max(np.abs(q_new - q_prev))
        residuals.append(diff)

        if verbose:
            print(f"  iter {it + 1:3d}: ||dq||_inf = {diff:.3e}")

        if diff < tol:
            if verbose:
                print(f"  Converged in {it + 1} iterations.")
            return x, t_fwd, q_new, {
                "residuals": residuals, "converged": True, "iters": it + 1
            }

        q_prev = q_new.copy()

    if verbose:
        print(f"  Not converged (residual={diff:.3e})")
    return x, t_fwd, q_new, {
        "residuals": residuals, "converged": False, "iters": max_iter
    }


# ================================================================
# Fixed-point solver: factorized memory kappa(tau,z) = K(tau)*gamma(z)
# ================================================================

def solve_nonlocal_memory_factorized(
        F, J, K_time, gamma, q0_hist_func, q0_init_func,
        x_range, Nx, T, T_hist, N_hist,
        alpha, cfl=0.5, max_iter=50, tol=1e-8,
        bc='periodic', verbose=True):
    """
    Solve the 1D nonlocal conservation law with factorized memory:

      d_t q + d_x(F(t, x, W, q)) = 0,   t in (0, T)
      W(t,x) = int_{-inf}^{t} K(t-s) [gamma * J(q(s,.))](x) ds

    where kappa(tau, z) = K(tau) * gamma(z).  The spatial convolution
    gamma * J(q) is computed via FFT (periodic), then the temporal
    integral is discretized via rectangle rule.

    Parameters
    ----------
    F             : callable(t, x_arr, w_arr, q_arr) -> flux array
    J             : callable(q_arr) -> array
    K_time        : callable(tau) -> scalar, temporal kernel
    gamma         : callable(z_arr) -> array, spatial kernel
    q0_hist_func  : callable(t, x_arr) -> historical data for t <= 0
    q0_init_func  : callable(x_arr) -> initial condition at t = 0
    x_range       : (x_left, x_right)
    Nx, T, T_hist, N_hist : grid sizes
    alpha, cfl, max_iter, tol, bc, verbose : solver params

    Returns
    -------
    x, t_fwd, q_fwd, info
    """
    # --- Spatial grid ---
    dx = (x_range[1] - x_range[0]) / Nx
    x = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, Nx)

    # --- Forward time grid ---
    dt = cfl * dx / alpha
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt
    t_fwd = np.linspace(0, T, Nt + 1)

    # --- Historical time grid ---
    t_hist = np.linspace(-T_hist, 0, N_hist + 1)

    # --- Initial condition ---
    q0 = q0_init_func(x)

    if verbose:
        print(f"Grid: Nx={Nx}, Nt={Nt}, N_hist={N_hist}")
        print(f"  dx={dx:.4e}, dt={dt:.4e}")

    # --- Precompute FFT of spatial kernel (periodic) ---
    k_idx = np.arange(Nx)
    k_dist = np.where(k_idx <= Nx // 2,
                      k_idx * dx, (k_idx - Nx) * dx)
    gamma_fft = np.fft.fft(gamma(k_dist))

    def convolve_space(Jq):
        """Periodic convolution: (gamma * Jq)(x), returns (Nx,) array."""
        return np.real(np.fft.ifft(np.fft.fft(Jq) * gamma_fft)) * dx

    # --- Precompute spatial convolutions of historical data ---
    # C_hist[k] = gamma * J(q0(t_hist[k], .))
    C_hist = np.zeros((N_hist + 1, Nx))
    for k in range(N_hist + 1):
        C_hist[k] = convolve_space(J(q0_hist_func(t_hist[k], x)))

    # --- Precompute historical contribution to W ---
    # W_hist(t_n) = sum_{k} dt_k * K(t_n - t_hist[k]) * C_hist[k]
    W_hist = np.zeros((Nt + 1, Nx))
    for n in range(Nt + 1):
        t_n = t_fwd[n]
        for k in range(N_hist):
            tau = t_n - t_hist[k]
            dtk = t_hist[k + 1] - t_hist[k]
            W_hist[n] += dtk * K_time(tau) * C_hist[k]

    if verbose:
        print(f"  Historical W precomputed.")

    # --- Fixed-point iteration ---
    q_prev = np.tile(q0, (Nt + 1, 1))
    residuals = []

    for it in range(max_iter):

        # Step 1: compute spatial convolutions at each forward time level
        C_fwd = np.zeros((Nt + 1, Nx))
        for n in range(Nt + 1):
            C_fwd[n] = convolve_space(J(q_prev[n]))

        # Step 2: total W = W_hist + W_fwd
        # W_fwd(t_n) = sum_{k=0}^{n-1} dt * K(t_n - t_fwd[k]) * C_fwd[k]
        w = W_hist.copy()
        for n in range(1, Nt + 1):
            t_n = t_fwd[n]
            for k in range(n):
                tau = t_n - t_fwd[k]
                w[n] += dt * K_time(tau) * C_fwd[k]

        # Step 3: frozen flux
        def phi(n, q_arr, _w=w, _t=t_fwd, _x=x):
            return F(_t[n], _x, _w[n], q_arr)

        # Step 4: solve local CL
        q_new = solve_local_lxf(phi, q0, dx, dt, Nt, alpha, bc=bc)

        # Step 5: convergence
        diff = np.max(np.abs(q_new - q_prev))
        residuals.append(diff)

        if verbose:
            print(f"  iter {it + 1:3d}: ||dq||_inf = {diff:.3e}")

        if diff < tol:
            if verbose:
                print(f"  Converged in {it + 1} iterations.")
            return x, t_fwd, q_new, {
                "residuals": residuals, "converged": True, "iters": it + 1
            }

        q_prev = q_new.copy()

    if verbose:
        print(f"  Not converged (residual={diff:.3e})")
    return x, t_fwd, q_new, {
        "residuals": residuals, "converged": False, "iters": max_iter
    }

# ================================================================
# Fixed-point solver: exponential memory with recursive update
# ================================================================

def solve_nonlocal_memory_exponential(
        F, J, tau0, gamma, q0_hist_func, q0_init_func,
        x_range, Nx, T, T_hist, N_hist,
        alpha, cfl=0.5, max_iter=50, tol=1e-8,
        bc='periodic', verbose=True):
    """
    Solve the 1D nonlocal conservation law with exponential memory:

      d_t q + d_x(F(t, x, W, q)) = 0,   t in (0, T)
      W(t,x) = int_{-inf}^{t} K(t-s) [gamma * J(q(s,.))](x) ds

    where K(tau) = (1/tau0) exp(-tau/tau0).

    Exploits the recursive structure of the exponential kernel:
      S^{n+1} = exp(-dt/tau0) * S^n + dt * C^n
      W^n     = S^n / tau0
    where C^n = gamma * J(q^n) is the spatial convolution at level n.

    This reduces the temporal sum from O(Nt^2 * Nx) to O(Nt * Nx),
    so the total per-iteration cost is O(Nt * Nx log Nx) (dominated
    by the FFT convolutions).

    Parameters
    ----------
    F             : callable(t, x_arr, w_arr, q_arr) -> flux array
    J             : callable(q_arr) -> array
    tau0          : float, memory decay time scale
    gamma         : callable(z_arr) -> array, spatial kernel
    q0_hist_func  : callable(t, x_arr) -> historical data for t <= 0
    q0_init_func  : callable(x_arr) -> initial condition at t = 0
    x_range       : (x_left, x_right)
    Nx, T, T_hist, N_hist : grid sizes
    alpha, cfl, max_iter, tol, bc, verbose : solver params

    Returns
    -------
    x, t_fwd, q_fwd, info
    """
    # --- Spatial grid ---
    dx = (x_range[1] - x_range[0]) / Nx
    x = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, Nx)

    # --- Forward time grid ---
    dt = cfl * dx / alpha
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt
    t_fwd = np.linspace(0, T, Nt + 1)

    # --- Historical time grid ---
    t_hist = np.linspace(-T_hist, 0, N_hist + 1)

    # --- Initial condition ---
    q0 = q0_init_func(x)

    # Exponential decay factor per forward time step
    decay = np.exp(-dt / tau0)

    if verbose:
        print(f"Grid: Nx={Nx}, Nt={Nt}, N_hist={N_hist}")
        print(f"  dx={dx:.4e}, dt={dt:.4e}, decay={decay:.6f}")

    # --- Precompute FFT of spatial kernel (periodic) ---
    k_idx = np.arange(Nx)
    k_dist = np.where(k_idx <= Nx // 2,
                      k_idx * dx, (k_idx - Nx) * dx)
    gamma_fft = np.fft.fft(gamma(k_dist))

    def convolve_space(Jq):
        return np.real(np.fft.ifft(np.fft.fft(Jq) * gamma_fft)) * dx

    # --- Precompute historical accumulator S at t=0 ---
    # S_hist_0 = sum_{k} dt_k * exp(|t_hist[k]| / tau0) * C_hist[k]
    # At forward time t_n, the full historical part is
    #   exp(-t_n / tau0) * S_hist_0,
    # which the recursion S^{n+1} = decay * S^n + ... handles
    # automatically since S^0 = S_hist_0 and each step multiplies
    # by decay = exp(-dt/tau0).
    S_hist_0 = np.zeros(Nx)
    for k in range(N_hist):
        tau_k = -t_hist[k]                         # lag from t=0
        dtk = t_hist[k + 1] - t_hist[k]
        C_k = convolve_space(J(q0_hist_func(t_hist[k], x)))
        S_hist_0 += dtk * np.exp(-tau_k / tau0) * C_k

    if verbose:
        print(f"  Historical S precomputed.")

    # --- Fixed-point iteration ---
    q_prev = np.tile(q0, (Nt + 1, 1))
    residuals = []

    for it in range(max_iter):

        # Build W at all time levels via recursion on q_prev.
        # The naive temporal sum is:
        #   tau0 * W^n = sum_{k=0}^{n-1} dt * decay^{n-k} * C^k
        # The recursion maintains S^n = sum_{k=0}^{n-1} dt * decay^{n-1-k} * C^k,
        # so tau0 * W^n = decay * S^n.  Hence W^n = decay * S^n / tau0.
        #
        # Update rule:  S^{n+1} = decay * S^n + dt * C^n
        w = np.zeros((Nt + 1, Nx))
        S = S_hist_0.copy()

        for n in range(Nt + 1):
            w[n] = decay * S / tau0
            C_n = convolve_space(J(q_prev[n]))
            S = decay * S + dt * C_n

        # Frozen flux
        def phi(n, q_arr, _w=w, _t=t_fwd, _x=x):
            return F(_t[n], _x, _w[n], q_arr)

        # Solve local CL
        q_new = solve_local_lxf(phi, q0, dx, dt, Nt, alpha, bc=bc)

        # Convergence check
        diff = np.max(np.abs(q_new - q_prev))
        residuals.append(diff)

        if verbose:
            print(f"  iter {it + 1:3d}: ||dq||_inf = {diff:.3e}")

        if diff < tol:
            if verbose:
                print(f"  Converged in {it + 1} iterations.")
            return x, t_fwd, q_new, {
                "residuals": residuals, "converged": True, "iters": it + 1
            }

        q_prev = q_new.copy()

    if verbose:
        print(f"  Not converged (residual={diff:.3e})")
    return x, t_fwd, q_new, {
        "residuals": residuals, "converged": False, "iters": max_iter
    }

# ================================================================
# Direct LxF solvers: no fixed-point, march forward step by step
# ================================================================

def _lxf_step(q, flux, dx, dt, alpha, bc):
    """
    One Lax-Friedrichs update step.

    Parameters
    ----------
    q    : (Nx,) solution at current time level
    flux : (Nx,) physical flux at cell centers
    dx, dt, alpha : grid and viscosity parameters
    bc   : 'periodic' or 'outflow'

    Returns
    -------
    q_new : (Nx,) solution at next time level
    """
    lam = dt / dx
    q_new = q.copy()

    if bc == 'periodic':
        q_r   = np.roll(q, -1)
        fl_r  = np.roll(flux, -1)
        F_hat = 0.5 * (flux + fl_r) - 0.5 * alpha * (q_r - q)
        F_l   = np.roll(F_hat, 1)
        q_new -= lam * (F_hat - F_l)
    else:
        F_hat = 0.5 * (flux[:-1] + flux[1:]) - 0.5 * alpha * (q[1:] - q[:-1])
        q_new[1:-1] -= lam * (F_hat[1:] - F_hat[:-1])
        q_new[0]  = q_new[1]
        q_new[-1] = q_new[-2]

    return q_new


def solve_direct_lxf_memory_factorized(
        F, J, K_time, gamma, q0_hist_func, q0_init_func,
        x_range, Nx, T, T_hist, N_hist,
        alpha, cfl=0.5, bc='periodic', verbose=True):
    """
    Solve the 1D nonlocal conservation law with factorized memory via a
    direct (explicit) Lax-Friedrichs marching scheme — NO fixed-point loop.

      d_t q + d_x(F(t, x, W, q)) = 0,   t in (0, T)
      W(t,x) = int_{-inf}^{t} K(t-s) [gamma * J(q(s,.))](x) ds

    At each time step n:
      1. W^n is assembled from the already-computed C^0,...,C^{n-1}
         (causal: uses only past data, no iteration needed).
      2. A single LxF step advances q^n -> q^{n+1}.
      3. C^n = gamma * J(q^n) is stored for future steps.

    Cost: O(Nt^2 * Nx log Nx)  — compare O(Nt * Nx log Nx) for the
    exponential-specific solver below.

    Parameters
    ----------
    Same as solve_nonlocal_memory_factorized (fixed-point version).

    Returns
    -------
    x, t_fwd, q_hist, info
      info contains 'method': 'direct'
    """
    # --- Spatial grid ---
    dx = (x_range[1] - x_range[0]) / Nx
    x  = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, Nx)

    # --- Forward time grid ---
    dt = cfl * dx / alpha
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt
    t_fwd = np.linspace(0, T, Nt + 1)

    # --- Historical time grid ---
    t_hist = np.linspace(-T_hist, 0, N_hist + 1)

    q0 = q0_init_func(x)

    if verbose:
        print(f"Direct LxF: Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}")

    # --- FFT spatial kernel ---
    k_idx  = np.arange(Nx)
    k_dist = np.where(k_idx <= Nx // 2, k_idx * dx, (k_idx - Nx) * dx)
    gfft   = np.fft.fft(gamma(k_dist))

    def conv(Jq):
        """Periodic spatial convolution gamma * Jq."""
        return np.real(np.fft.ifft(np.fft.fft(Jq) * gfft)) * dx

    # --- Precompute historical contribution to W ---
    # W_hist[n] = sum_{k=0}^{N_hist-1} dt_k * K(t_n - t_hist[k]) * C_hist[k]
    C_hist   = np.array([conv(J(q0_hist_func(t_hist[k], x)))
                         for k in range(N_hist + 1)])
    W_hist   = np.zeros((Nt + 1, Nx))
    for n in range(Nt + 1):
        t_n = t_fwd[n]
        for k in range(N_hist):
            dtk = t_hist[k + 1] - t_hist[k]
            W_hist[n] += dtk * K_time(t_n - t_hist[k]) * C_hist[k]

    if verbose:
        print(f"  Historical W precomputed.")

    # --- Direct marching ---
    q_hist = np.zeros((Nt + 1, Nx))
    q_hist[0] = q0

    # C[k] = gamma * J(q^k), accumulated as we march forward
    C_fwd    = np.zeros((Nt + 1, Nx))
    C_fwd[0] = conv(J(q0))

    for n in range(Nt):
        # W^n = W_hist[n] + sum_{k=0}^{n-1} dt * K(t_n - t_k) * C^k
        # (causal sum: excludes k=n, exactly matching FP solver at convergence)
        W_n = W_hist[n].copy()
        t_n = t_fwd[n]
        for k in range(n):
            W_n += dt * K_time(t_n - t_fwd[k]) * C_fwd[k]

        # One explicit LxF step
        flux    = F(t_n, x, W_n, q_hist[n])
        q_hist[n + 1] = _lxf_step(q_hist[n], flux, dx, dt, alpha, bc)

        # Store spatial convolution for future W computations
        C_fwd[n + 1] = conv(J(q_hist[n + 1]))

        if verbose and (n + 1) % max(1, Nt // 5) == 0:
            print(f"  step {n + 1}/{Nt}")

    return x, t_fwd, q_hist, {"method": "direct"}


def solve_direct_lxf_memory_exponential(
        F, J, tau0, gamma, q0_hist_func, q0_init_func,
        x_range, Nx, T, T_hist, N_hist,
        alpha, cfl=0.5, bc='periodic', verbose=True):
    """
    Direct (explicit) LxF marching for exponential memory K(tau)=(1/tau0)exp(-tau/tau0).

    Uses the same recursive accumulator as the fixed-point exponential solver,
    but advances one step at a time with no outer iteration:

      S^{n+1} = decay * S^n + dt * C^n,    C^n = gamma * J(q^n)
      W^n     = decay * S^n / tau0           (causal: excludes contribution of q^n itself)
      q^{n+1} = LxF(q^n, W^n)

    Cost: O(Nt * Nx log Nx) — optimal for exponential kernel.

    Parameters
    ----------
    Same as solve_nonlocal_memory_exponential (fixed-point version).

    Returns
    -------
    x, t_fwd, q_hist, info
    """
    # --- Spatial grid ---
    dx = (x_range[1] - x_range[0]) / Nx
    x  = np.linspace(x_range[0] + dx / 2, x_range[1] - dx / 2, Nx)

    # --- Forward time grid ---
    dt = cfl * dx / alpha
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt
    t_fwd = np.linspace(0, T, Nt + 1)

    # --- Historical time grid ---
    t_hist = np.linspace(-T_hist, 0, N_hist + 1)

    q0    = q0_init_func(x)
    decay = np.exp(-dt / tau0)  # exp(-dt/tau0): factor per forward step

    if verbose:
        print(f"Direct LxF (exp): Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}, "
              f"decay={decay:.6f}")

    # --- FFT spatial kernel ---
    k_idx  = np.arange(Nx)
    k_dist = np.where(k_idx <= Nx // 2, k_idx * dx, (k_idx - Nx) * dx)
    gfft   = np.fft.fft(gamma(k_dist))

    def conv(Jq):
        return np.real(np.fft.ifft(np.fft.fft(Jq) * gfft)) * dx

    # --- Historical accumulator at t=0 ---
    # S_hist_0 = sum_{k=0}^{N_hist-1} dt_k * exp(-|t_hist[k]|/tau0) * C_hist[k]
    S = np.zeros(Nx)
    for k in range(N_hist):
        dtk   = t_hist[k + 1] - t_hist[k]
        tau_k = -t_hist[k]          # lag from t=0 (positive)
        C_k   = conv(J(q0_hist_func(t_hist[k], x)))
        S    += dtk * np.exp(-tau_k / tau0) * C_k   # S_hist_0

    if verbose:
        print(f"  Historical S precomputed.")

    # --- Direct marching ---
    q_hist    = np.zeros((Nt + 1, Nx))
    q_hist[0] = q0

    for n in range(Nt):
        # W^n = decay * S^n / tau0
        # S^n accumulates contributions from history and q^0,...,q^{n-1}.
        # (Same formula as in the FP exponential solver's inner loop.)
        W_n = decay * S / tau0

        # One explicit LxF step
        flux          = F(t_fwd[n], x, W_n, q_hist[n])
        q_hist[n + 1] = _lxf_step(q_hist[n], flux, dx, dt, alpha, bc)

        # Recursive accumulator update: S^{n+1} = decay*S^n + dt*C^n
        C_n = conv(J(q_hist[n]))
        S   = decay * S + dt * C_n

        if verbose and (n + 1) % max(1, Nt // 5) == 0:
            print(f"  step {n + 1}/{Nt}")

    return x, t_fwd, q_hist, {"method": "direct_exponential"}