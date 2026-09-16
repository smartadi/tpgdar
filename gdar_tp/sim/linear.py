"""Linear graph-supported GDAR data generator.

Used to (a) sanity-check the fitter with exact ground truth and (b) drive the
TP-decomposition tests where we need a coefficient tensor that is *truly*
low-rank in the scheduling parameter.
"""

from __future__ import annotations

import numpy as np


def stable_diffusion_coeffs(laplacian, a, b, k_lags=1):
    """Build ``A_k = a_k I - b_k L`` diffusion coefficient matrices.

    Parameters
    ----------
    laplacian : (N, N) ndarray
    a, b : sequence of length ``k_lags``
        Per-lag diffusion coefficients.

    Returns
    -------
    list of (N, N) ndarray
    """
    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    n = laplacian.shape[0]
    eye = np.eye(n)
    return [a[k] * eye - b[k] * laplacian for k in range(k_lags)]


def simulate_gdar(coeffs, n_steps, x0=None, B=None, U=None, noise_std=0.0,
                  rng=None, burn_in=0):
    """Simulate ``x(t+1) = sum_k A_k x(t-k) + B u(t) + e(t)``.

    Parameters
    ----------
    coeffs : list of (N, N) ndarray
        ``[A_0, A_1, ..., A_{K-1}]`` (lag 0 is the most recent sample).
    n_steps : int
        Number of samples to return (after burn-in).
    x0 : (K, N) ndarray, optional
        Initial history, newest row last. Zeros if omitted.
    B : (N, M) ndarray, optional
    U : (n_steps + burn_in, M) ndarray, optional
        Input sequence. Zeros if omitted.
    noise_std : float
    burn_in : int
        Discarded leading samples to wash out the initial transient.

    Returns
    -------
    X : (n_steps, N) ndarray
    U_out : (n_steps, M) ndarray  (empty M=0 array if no input)
    """
    rng = np.random.default_rng(rng)
    k = len(coeffs)
    n = coeffs[0].shape[0]
    total = n_steps + burn_in

    m = 0 if B is None else B.shape[1]
    if U is None:
        U = np.zeros((total, m))
    else:
        U = np.asarray(U, dtype=float)
        if U.shape[0] < total:
            raise ValueError("U must have at least n_steps + burn_in rows")

    hist = np.zeros((k, n)) if x0 is None else np.array(x0, dtype=float)
    if hist.shape != (k, n):
        raise ValueError(f"x0 must be shape ({k}, {n})")

    out = np.zeros((total, n))
    buf = list(hist)  # buf[-1] is the newest past sample x(t), buf[-1-k'] is x(t-k')
    # Convention (matches the plan and GDARModel.predict_one_step):
    #   x(t+1) = sum_k A_k x(t-k) + B u(t)
    # so the input that drives the state produced at loop index `t` is U[t-1].
    for t in range(total):
        nxt = np.zeros(n)
        for lag, A in enumerate(coeffs):
            nxt += A @ buf[-1 - lag]
        if m and t >= 1:
            nxt += B @ U[t - 1]
        if noise_std:
            nxt += noise_std * rng.standard_normal(n)
        out[t] = nxt
        buf.append(nxt)
        buf.pop(0)

    X = out[burn_in:]
    U_out = U[burn_in:total]
    return X, U_out
