"""Latent-population GDAR: a constrained linear-Gaussian state-space model.

Motivation (see docs/PLAN.md and the project notes): the observed channels are a
*marginalization* of a richer multi-population system (a Wilson-Cowan node is an
E and an I population; we usually record something close to E).  A single-
population GDAR has to spend extra lags mimicking the hidden populations.  This
module fits a latent model with **two populations per node** and asks whether the
hidden latent recovers the true inhibitory dynamics.

Identifiability is anchored, not hoped for:

* **Fixed observation.** ``C`` selects the E-slot of each node, so latent-E is
  clamped to the observed channel.  Only the hidden I-slot is free -- this kills
  the rotational ambiguity that sinks unconstrained factor models.
* **Graph structure.** Between-node coupling follows the array graph only.
* **Dale's law.** Inter-node projections originate from E populations and are
  non-negative; I populations do not project between nodes.  (The within-node
  E-I block is left free -- that is the local circuit.)

Model::

    x(t+1) = A x(t) + w,   w ~ N(0, Q)      x in R^{2N}  (per node: [E, I])
    y(t)   = C x(t) + v,   v ~ N(0, R)      y in R^{N}   (observed E channels)

Fit by EM (Kalman filter + RTS smoother E-step; projected M-step for the
structure and sign constraints).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Structure / sign masks for the 2-population-per-node layout
# ---------------------------------------------------------------------------
# latent index: node i -> E at 2*i, I at 2*i+1

def population_masks(adjacency, allow_e_to_i_between=True):
    """Build the A-structure mask and Dale sign mask for a 2-pop-per-node model.

    Returns
    -------
    struct : (2N, 2N) bool   -- True where A may be nonzero
    sign   : (2N, 2N) int    -- +1 entry must be >=0, -1 must be <=0, 0 free
    C      : (N, 2N) float    -- observation (selects each node's E-slot)
    """
    A = np.asarray(adjacency)
    N = A.shape[0]
    d = 2 * N
    struct = np.zeros((d, d), dtype=bool)
    sign = np.zeros((d, d), dtype=int)

    def E(i): return 2 * i
    def I(i): return 2 * i + 1

    for i in range(N):
        # within-node local E-I circuit: fully free (any sign)
        for a in (E(i), I(i)):
            for b in (E(i), I(i)):
                struct[a, b] = True
    for i in range(N):
        for j in range(N):
            if i == j or A[i, j] == 0:
                continue
            # between-node: only the source E projects, non-negative (Dale)
            struct[E(i), E(j)] = True; sign[E(i), E(j)] = +1
            if allow_e_to_i_between:
                struct[I(i), E(j)] = True; sign[I(i), E(j)] = +1
    C = np.zeros((N, d))
    for i in range(N):
        C[i, E(i)] = 1.0
    return struct, sign, C


def project_A(A, struct, sign):
    """Project a dynamics matrix onto the structure + Dale-sign constraints."""
    A = A * struct
    A = np.where(sign > 0, np.maximum(A, 0.0), A)
    A = np.where(sign < 0, np.minimum(A, 0.0), A)
    return A


# ---------------------------------------------------------------------------
# Kalman filter + RTS smoother
# ---------------------------------------------------------------------------

@dataclass
class LDS:
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    mu0: np.ndarray
    P0: np.ndarray

    def filter(self, Y):
        T, n = Y.shape
        d = self.A.shape[0]
        xf = np.zeros((T, d)); Pf = np.zeros((T, d, d))
        xp = np.zeros((T, d)); Pp = np.zeros((T, d, d))
        loglik = 0.0
        x_pred, P_pred = self.mu0.copy(), self.P0.copy()
        for t in range(T):
            xp[t] = x_pred; Pp[t] = P_pred
            S = self.C @ P_pred @ self.C.T + self.R
            S = 0.5 * (S + S.T)
            innov = Y[t] - self.C @ x_pred
            Sinv = np.linalg.inv(S)
            Kg = P_pred @ self.C.T @ Sinv
            xf[t] = x_pred + Kg @ innov
            Pf[t] = (np.eye(d) - Kg @ self.C) @ P_pred
            Pf[t] = 0.5 * (Pf[t] + Pf[t].T)
            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (innov @ Sinv @ innov + logdet + n * np.log(2 * np.pi))
            x_pred = self.A @ xf[t]
            P_pred = self.A @ Pf[t] @ self.A.T + self.Q
            P_pred = 0.5 * (P_pred + P_pred.T)
        return dict(xf=xf, Pf=Pf, xp=xp, Pp=Pp, loglik=loglik)

    def smooth(self, Y):
        f = self.filter(Y)
        xf, Pf, Pp = f["xf"], f["Pf"], f["Pp"]
        T, d = xf.shape
        xs = np.zeros_like(xf); Ps = np.zeros_like(Pf)
        xs[-1] = xf[-1]; Ps[-1] = Pf[-1]
        J = np.zeros((T, d, d))
        for t in range(T - 2, -1, -1):
            J[t] = Pf[t] @ self.A.T @ np.linalg.inv(Pp[t + 1])
            xs[t] = xf[t] + J[t] @ (xs[t + 1] - self.A @ xf[t])
            Ps[t] = Pf[t] + J[t] @ (Ps[t + 1] - Pp[t + 1]) @ J[t].T
            Ps[t] = 0.5 * (Ps[t] + Ps[t].T)
        # lag-one smoothed covariance (Shumway-Stoffer)
        Plag = np.zeros((T, d, d))  # Plag[t] = Cov(x_t, x_{t-1} | Y), t = 1..T-1
        for t in range(T - 1, 0, -1):
            Plag[t] = Ps[t] @ J[t - 1].T
        return dict(xs=xs, Ps=Ps, Plag=Plag, loglik=f["loglik"], J=J)


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

def fit_em(Y, struct, sign, C, n_iter=30, update_R=True, r_floor=1e-4,
           q_floor=1e-6, rng=0, verbose=False, init_A=None):
    """Fit the constrained latent LDS by EM.

    Parameters
    ----------
    Y : (T, N) observed (E) channels, mean-centered.
    struct, sign, C : from :func:`population_masks`.
    update_R : also re-estimate (diagonal) observation noise.

    Returns
    -------
    lds : LDS       -- fitted model
    info : dict     -- {"loglik": trace, "xs": final smoothed states}
    """
    rng = np.random.default_rng(rng)
    T, n = Y.shape
    d = struct.shape[0]

    # init
    if init_A is None:
        A = 0.7 * np.eye(d) + 0.02 * rng.standard_normal((d, d))
        A = project_A(A, struct, sign)
    else:
        A = project_A(init_A.copy(), struct, sign)
    Q = 0.05 * np.eye(d)
    R = 0.01 * np.eye(n)
    mu0 = np.zeros(d)
    P0 = np.eye(d)

    ll_trace = []
    for it in range(n_iter):
        lds = LDS(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0)
        sm = lds.smooth(Y)
        ll_trace.append(sm["loglik"])
        xs, Ps, Plag = sm["xs"], sm["Ps"], sm["Plag"]

        # second-moment sums
        Exx = Ps + np.einsum("ti,tj->tij", xs, xs)          # E[x_t x_t^T]
        S00 = Exx[:-1].sum(axis=0)                           # sum x_{t-1}x_{t-1}
        S11 = Exx[1:].sum(axis=0)                            # sum x_t x_t
        S10 = (Plag[1:] + np.einsum("ti,tj->tij", xs[1:], xs[:-1])).sum(axis=0)

        # A: unconstrained LS then project
        A_unc = S10 @ np.linalg.inv(S00 + 1e-8 * np.eye(d))
        A = project_A(A_unc, struct, sign)

        # Q (diagonal, floored)
        Qfull = (S11 - A @ S10.T - S10 @ A.T + A @ S00 @ A.T) / (T - 1)
        Q = np.diag(np.maximum(np.diag(Qfull), q_floor))

        # R (diagonal, floored)
        if update_R:
            resid = Y - xs @ C.T
            Rfull = (resid.T @ resid) / T + C @ (Ps.mean(axis=0)) @ C.T
            R = np.diag(np.maximum(np.diag(Rfull), r_floor))

        mu0 = xs[0]
        P0 = Ps[0] + 1e-6 * np.eye(d)
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f"  EM iter {it:3d}  loglik={ll_trace[-1]:.1f}")

    lds = LDS(A=A, C=C, Q=Q, R=R, mu0=mu0, P0=P0)
    sm = lds.smooth(Y)
    return lds, {"loglik": np.array(ll_trace), "xs": sm["xs"]}


def one_step_pred_nrmse(lds, Y):
    """One-step-ahead predictive NRMSE of the observed channels under the LDS."""
    f = lds.filter(Y)
    xp = f["xp"]                       # one-step predicted state (before update)
    Yhat = xp @ lds.C.T
    rmse = np.sqrt(np.mean((Yhat[1:] - Y[1:]) ** 2, axis=0))
    scale = np.std(Y[1:], axis=0)
    return float(np.mean(rmse / (scale + 1e-12)))


def recovered_inhibitory(info, N):
    """Extract the smoothed hidden I-slot trajectory, shape (T, N)."""
    xs = info["xs"]
    return xs[:, 1::2]
