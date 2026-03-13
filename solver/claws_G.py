"""
claws_G.py - Conservation Laws with Memory (CLAWS)

Numerical solver for 1D nonlocal conservation laws via fixed-point iteration.

Models:
  [Space]   d_t q + d_x(F(t, x, W, q)) = 0,   W = int gamma(x-y) J(q(t,y)) dy
  [Memory]  d_t q + d_x(F(t, x, W, q)) = 0,   W = int_{-inf}^{t} int kappa(t-s, x-y) J(q(s,y)) dy ds

Numerical methods:
  Inner solver : Godunov finite volume scheme (conservative, entropy-satisfying)
  Outer loop   : Picard fixed-point iteration (contraction guaranteed for small T)

Note on F: the physical flux F(t, x, w, q) must support scalar (x, w, q)
arguments, as the Godunov Riemann solver evaluates it pointwise at interfaces.
"""

import numpy as np
from scipy.optimize import minimize_scalar


# ================================================================
# Nonlocal operators  (unchanged from claws_LXF)
# ================================================================

def eval_W_space(J_q, gamma, x, dx):
    """
    Evaluate the spatial nonlocal operator via midpoint quadrature:
      W(x_i) = int gamma(x_i - y) J(q(y)) dy
             ~ dx * sum_j gamma(x_i - x_j) * J(q_j)
    """
    G = gamma(x[:, None] - x[None, :])
    return dx * (G @ J_q)


def eval_W_memory(J_q_all, kappa, x, dx, t_levels, n):
    """
    Evaluate the nonlocal operator with memory at time level n:
      W(t_n, x_i) = int_{-inf}^{t_n} int kappa(t_n - s, x_i - y) J(q(s,y)) dy ds

    Discretized via rectangle rule in time, midpoint in space.
    """
    Nx = len(x)
    W = np.zeros(Nx)
    t_n = t_levels[n]
    diffs_x = x[:, None] - x[None, :]

    for k in range(n):
        tau = t_n - t_levels[k]
        dt_k = t_levels[k + 1] - t_levels[k]
        K = kappa(tau, diffs_x)
        W += dt_k * dx * (K @ J_q_all[k])

    return W


# ================================================================
# Godunov Riemann flux
# ================================================================

def godunov_flux(u_L, u_R, f_loc, riemann='general', q_star=None):
    """
    Godunov numerical flux for a scalar conservation law.

    Resolves the Riemann problem (u_L, u_R) for the frozen local flux f_loc(u).

    Two modes:
      'general' : bounded Brent minimization — works for any flux shape,
                  but calls minimize_scalar at every interface (slow).
      'concave' : analytical Riemann solver for concave fluxes f(q) = q(1-q)*C
                  where C > 0 depends on frozen (t, x, w) but not on q.
                  The critical point q* = argmax f is fixed regardless of C,
                  so it only needs to be specified once (default 0.5).
                  - u_L <= u_R (rarefaction): entropy solution is min over
                    endpoints (concave f has no interior minimum)
                  - u_L >  u_R (shock): entropy solution is f(q*) if q* in
                    [u_R, u_L], otherwise max over endpoints

    Parameters
    ----------
    u_L, u_R : left and right states (scalars)
    f_loc    : callable u -> R, physical flux with (x, t, w) frozen at interface
    riemann  : 'general' or 'concave'
    q_star   : critical point of f for 'concave' mode (default 0.5)
    """
    if u_L == u_R:
        return f_loc(u_L)

    if riemann == 'concave':
        q_star = 0.5 if q_star is None else q_star
        if u_L < u_R:
            # Concave f has no interior minimum: entropy solution at an endpoint
            return min(f_loc(u_L), f_loc(u_R))
        else:
            # Entropy solution: f(q*) if q* in [u_R, u_L], else max at endpoint
            if u_R <= q_star <= u_L:
                return f_loc(q_star)
            else:
                return max(f_loc(u_L), f_loc(u_R))
    else:
        if u_L < u_R:
            # Rarefaction: minimize f over [u_L, u_R]
            res = minimize_scalar(f_loc, bounds=(u_L, u_R), method='bounded')
            return res.fun
        else:
            # Shock: maximize f over [u_R, u_L]
            res = minimize_scalar(lambda u: -f_loc(u), bounds=(u_R, u_L), method='bounded')
            return -res.fun


# ================================================================
# Local conservation law solver: Godunov finite volume
# ================================================================

def solve_local_godunov(F, t_arr, x, w, q_init, dx, dt, Nt,
                        bc='outflow', riemann='general', q_star=None):
    """
    Solve  d_t q + d_x(F(t, x, w(t,x), q)) = 0  on a uniform 1D grid
    using the Godunov scheme.

    At each interface i+1/2, the numerical flux is the Godunov flux for the
    Riemann problem (q_i, q_{i+1}) with F frozen at (t^n, x_{i+1/2}, w_{i+1/2}):
      x_{i+1/2} = x[i] + dx/2
      w_{i+1/2} = (w^n_i + w^n_{i+1}) / 2   (arithmetic average)

    Conservative update:
      q_i^{n+1} = q_i^n - (dt/dx) (F_hat_{i+1/2} - F_hat_{i-1/2})

    Stability: dt/dx * max|dF/dq| <= 1  (CFL must be enforced externally).

    Parameters
    ----------
    F      : callable(t, x_s, w_s, u) -> scalar flux (scalar x_s, w_s, u)
    t_arr  : (Nt+1,) time levels
    x      : (Nx,) cell centers
    w      : (Nt+1, Nx) frozen nonlocal field at each time level
    q_init : (Nx,) initial condition
    dx, dt : spatial and temporal step sizes
    Nt     : number of time steps
    bc     : 'outflow' (zero-gradient) or 'periodic'
    riemann : 'general' or 'concave' (see godunov_flux)
    q_star  : critical point for 'concave' mode (default 0.5)

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
        t_n = t_arr[n]

        if bc == 'periodic':
            # Nx interfaces: F_hat[i] = numerical flux at interface i+1/2
            F_hat = np.zeros(Nx)
            for i in range(Nx):
                i_R = (i + 1) % Nx
                u_L, u_R = q[i], q[i_R]
                x_h = x[i] + dx / 2               # interface position
                w_h = 0.5 * (w[n, i] + w[n, i_R]) # averaged nonlocal field
                # Freeze (t_n, x_h, w_h) and pass 1d flux to Riemann solver
                f_loc = lambda u, _t=t_n, _x=x_h, _w=w_h: F(_t, _x, _w, u)
                F_hat[i] = godunov_flux(u_L, u_R, f_loc, riemann, q_star)
            # Conservative update: q[i] -= lam * (F_{i+1/2} - F_{i-1/2})
            q -= lam * (F_hat - np.roll(F_hat, 1))

        else:
            # Nx-1 interior interfaces: F_hat[i] = flux at interface i+1/2, i=0..Nx-2
            F_hat = np.zeros(Nx - 1)
            for i in range(Nx - 1):
                u_L, u_R = q[i], q[i + 1]
                x_h = x[i] + dx / 2
                w_h = 0.5 * (w[n, i] + w[n, i + 1])
                f_loc = lambda u, _t=t_n, _x=x_h, _w=w_h: F(_t, _x, _w, u)
                F_hat[i] = godunov_flux(u_L, u_R, f_loc, riemann, q_star)
            # Update interior cells j=1..Nx-2
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
                         bc='outflow', riemann='general', q_star=None,
                         verbose=True):
    """
    Solve the 1D nonlocal conservation law (spatial nonlocality) via
    Picard fixed-point iteration.

      d_t q + d_x(F(t, x, W[J(q), gamma], q)) = 0,   t in (0, T)
      q(0, x) = q0(x)

    Fixed-point procedure (cf. paper, Definition 3.5):
      1. Given iterate q~, compute  w(t,x) = W[J(q~), gamma](t,x)
      2. Solve the LOCAL conservation law  d_t q + d_x(F(t,x,w(t,x),q)) = 0
         with the Godunov scheme (w is frozen / prescribed)
      3. Update q~ <- q, check contraction

    Parameters
    ----------
    F        : callable(t, x_s, w_s, q_s) -> scalar flux (scalar inputs)
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
    dt = T / Nt
    t = np.linspace(0, T, Nt + 1)

    # --- Initial condition ---
    q0 = q0_func(x)

    if verbose:
        print(f"Grid: Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}, "
              f"CFL={alpha * dt / dx:.3f}")

    # --- Precompute kernel for convolution ---
    if bc == 'periodic':
        k_idx = np.arange(Nx)
        k_dist = np.where(k_idx <= Nx // 2,
                          k_idx * dx, (k_idx - Nx) * dx)
        gamma_fft = np.fft.fft(gamma(k_dist))

    # --- Fixed-point iteration ---
    q_prev = np.tile(q0, (Nt + 1, 1))
    residuals = []

    for it in range(max_iter):

        # Step 1: evaluate w(t_n, x) = W[J(q~^n), gamma] at every time level
        w = np.zeros((Nt + 1, Nx))
        for n in range(Nt + 1):
            Jq = J(q_prev[n])
            if bc == 'periodic':
                # FFT-based periodic convolution
                w[n] = np.real(np.fft.ifft(
                    np.fft.fft(Jq) * gamma_fft)) * dx
            else:
                w[n] = eval_W_space(Jq, gamma, x, dx)

        # Step 2: solve the local conservation law with Godunov
        q_new = solve_local_godunov(F, t, x, w, q0, dx, dt, Nt,
                                    bc=bc, riemann=riemann, q_star=q_star)

        # Step 3: convergence check (L^inf over all of space-time)
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
                          bc='outflow', riemann='general', q_star=None,
                          verbose=True):
    """
    Solve the 1D nonlocal conservation law with memory via Picard
    fixed-point iteration.

      d_t q + d_x(F(t, x, W, q)) = 0,            t in (0, T)
      W(t,x) = int_{-inf}^{t} int kappa(t-s, x-y) J(q(s,y)) dy ds
      q(t,x) = q0(t,x),                           t <= 0

    Parameters
    ----------
    F             : callable(t, x_s, w_s, q_s) -> scalar flux
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

    # --- Precompute historical contribution to W (fixed across iterations) ---
    diffs_x = x[:, None] - x[None, :]
    W_hist = np.zeros((Nt + 1, Nx))

    for n in range(Nt + 1):
        t_n = t_fwd[n]
        for k in range(N_hist):
            tau = t_n - t_hist[k]
            dtk = t_hist[k + 1] - t_hist[k]
            K = kappa(tau, diffs_x)
            W_hist[n] += dtk * dx * (K @ J_q_hist[k])

    if verbose:
        print(f"  Historical W precomputed.")

    # --- Fixed-point iteration ---
    q_prev = np.tile(q0, (Nt + 1, 1))
    residuals = []

    for it in range(max_iter):

        # Step 1: total W = W_hist + W_fwd
        w = W_hist.copy()
        for n in range(1, Nt + 1):
            t_n = t_fwd[n]
            for k in range(n):
                tau = t_n - t_fwd[k]
                K = kappa(tau, diffs_x)
                w[n] += dt * dx * (K @ J(q_prev[k]))

        q_new = solve_local_godunov(F, t_fwd, x, w, q0, dx, dt, Nt,
                                    bc=bc, riemann=riemann, q_star=q_star)

        # Step 3: convergence
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
        bc='periodic', riemann='general', q_star=None, verbose=True):
    """
    Solve the 1D nonlocal conservation law with factorized memory:

      d_t q + d_x(F(t, x, W, q)) = 0,   t in (0, T)
      W(t,x) = int_{-inf}^{t} K(t-s) [gamma * J(q(s,.))](x) ds

    where kappa(tau, z) = K(tau) * gamma(z).  Spatial convolution via FFT.

    Parameters
    ----------
    F             : callable(t, x_s, w_s, q_s) -> scalar flux
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
    C_hist = np.zeros((N_hist + 1, Nx))
    for k in range(N_hist + 1):
        C_hist[k] = convolve_space(J(q0_hist_func(t_hist[k], x)))

    # --- Precompute historical contribution to W ---
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
        w = W_hist.copy()
        for n in range(1, Nt + 1):
            t_n = t_fwd[n]
            for k in range(n):
                tau = t_n - t_fwd[k]
                w[n] += dt * K_time(tau) * C_fwd[k]

        # Step 3: solve local CL with Godunov
        q_new = solve_local_godunov(F, t_fwd, x, w, q0, dx, dt, Nt,
                                    bc=bc, riemann=riemann, q_star=q_star)

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
# Fixed-point solver: exponential memory with recursive update
# ================================================================

def solve_nonlocal_memory_exponential(
        F, J, tau0, gamma, q0_hist_func, q0_init_func,
        x_range, Nx, T, T_hist, N_hist,
        alpha, cfl=0.5, max_iter=50, tol=1e-8,
        bc='periodic', riemann='general', q_star=None, verbose=True):
    """
    Solve the 1D nonlocal conservation law with exponential memory:

      d_t q + d_x(F(t, x, W, q)) = 0,   t in (0, T)
      W(t,x) = int_{-inf}^{t} K(t-s) [gamma * J(q(s,.))](x) ds

    where K(tau) = (1/tau0) exp(-tau/tau0).

    Exploits the recursive structure of the exponential kernel:
      S^{n+1} = exp(-dt/tau0) * S^n + dt * C^n
      W^n     = S^n / tau0
    where C^n = gamma * J(q^n). Complexity O(Nt * Nx log Nx) per iteration.

    Parameters
    ----------
    F             : callable(t, x_s, w_s, q_s) -> scalar flux
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
    # S_hist_0 = sum_{k} dt_k * exp(-|t_hist[k]| / tau0) * C_hist[k]
    # The full historical contribution at t_n is exp(-t_n/tau0) * S_hist_0,
    # handled automatically by the recursion since S^0 = S_hist_0.
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
        # Recursion: S^{n+1} = decay * S^n + dt * C^n
        # W^n = decay * S^n / tau0  (see claws_LXF for derivation)
        w = np.zeros((Nt + 1, Nx))
        S = S_hist_0.copy()

        for n in range(Nt + 1):
            w[n] = decay * S / tau0
            C_n = convolve_space(J(q_prev[n]))
            S = decay * S + dt * C_n

        # Solve local CL with Godunov
        q_new = solve_local_godunov(F, t_fwd, x, w, q0, dx, dt, Nt,
                                    bc=bc, riemann=riemann, q_star=q_star)

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