"""TP-GDAR inside Felix Schwock's GDAR model class.

Felix's GDAR (``gdar.gdar_model.CVARModel.fit_gdar``) is a VAR whose lag matrices are
symmetric with support on the graph plus the diagonal: one coefficient per edge per
lag, one per node per lag, ``K (N + E)`` in total.  It is estimated by GLS: symmetric
OLS, residual covariance Sigma, refit weighted by Sigma^-1.

:mod:`gdar_tp.tpgdar` fits edge-wise (non-symmetric) operators, so comparing it to
Felix's model also changes the model class.  Here every TP component is a symmetric
graph-supported operator, ``Theta(p_g) = sum_j W[g, j] Theta_j``, fitted by the same
GLS metric.  With ``vec Theta_j = R phi_j`` the loss is quadratic per grid point,

    L_g(phi) = phi' H_g phi - 2 phi' h_g + const,    H_g = R' (G_g kron Sigma^-1) R,

so alternating least squares is the same pair of exact block solves as
:func:`gdar_tp.tpgdar.fit_joint`, and the objective never increases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .tpgdar import resid_cov_sum


def symmetric_basis(adjacency, K):
    """Sparse R with ``vec(Theta) = R phi`` (column-major, Theta is N x KN).

    ``phi`` holds, per lag, the N self terms and then one coefficient per edge i < j.
    """
    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]
    iu, ju = np.where(np.triu(A, 1))
    E = len(iu)
    n_lag = N + E
    rows, cols = [], []
    for k in range(K):
        off = k * N * N                 # column k N of Theta starts at N * (k N) in vec
        c0 = k * n_lag
        rows.append(off + np.arange(N) * N + np.arange(N))
        cols.append(c0 + np.arange(N))
        rows.append(off + ju * N + iu)  # entry (i, j)
        cols.append(c0 + N + np.arange(E))
        rows.append(off + iu * N + ju)  # entry (j, i)
        cols.append(c0 + N + np.arange(E))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N * N * K, K * n_lag))


def theta_of(phi, R, N):
    """phi -> Theta (N, D)."""
    return (R @ phi).reshape(N, -1, order="F")


def _quad(Gzz, Gzy, R, Sinv):
    """H, h of the GLS loss on phi for one condition."""
    N = Sinv.shape[0]
    used = np.unique(R.nonzero()[0])
    i, c = used % N, used // N
    M = Sinv[np.ix_(i, i)] * Gzz[np.ix_(c, c)]
    Rs = R[used]
    H = Rs.T @ (Rs.T @ M.T).T
    F = (Sinv @ Gzy.T).ravel(order="F")
    return np.asarray(H), np.asarray(R.T @ F).ravel()


def _solve(H, h, jitter):
    d = np.trace(H) / len(H) if len(H) else 1.0
    return np.linalg.solve(H + jitter * d * np.eye(len(H)), h)


def fit_gls_pooled(Gzz, Gzy, Syy, n, R, jitter=1e-10):
    """Felix's estimator on summed statistics.  Returns phi, Sigma (from the sym-OLS fit)."""
    N = Syy.shape[0]
    H, h = _quad(Gzz, Gzy, R, np.eye(N))
    phi = _solve(H, h, jitter)
    Sigma = resid_cov_sum(theta_of(phi, R, N), Gzz, Gzy, Syy) / n
    H, h = _quad(Gzz, Gzy, R, np.linalg.pinv(Sigma))
    return _solve(H, h, jitter), Sigma


@dataclass
class SymTP:
    """``Theta(p_g) = sum_j W[g, j] Theta_j``, every Theta_j symmetric on the graph."""

    W: np.ndarray               # (P, r)
    phi: np.ndarray             # (r, n_phi)
    R: sp.csr_matrix
    N: int
    p_grid: np.ndarray
    objective: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def theta_from_weights(self, w):
        return theta_of(w @ self.phi, self.R, self.N)

    def theta_grid(self):
        return np.stack([self.theta_from_weights(w) for w in self.W])

    def theta_at(self, p):
        """Operator at an off-grid ``p`` by linear interpolation of the weights."""
        w = np.array([np.interp(p, self.p_grid, self.W[:, j]) for j in range(self.W.shape[1])])
        return self.theta_from_weights(w)

    @property
    def n_params(self):
        """Free numbers: r operators plus the weights, less the r^2 of the gauge."""
        P, r = self.W.shape
        return r * self.phi.shape[1] + P * r - r * r


def fit_sym_tp(stats, R, Sigma, r, p_grid, n_iter=60, tol=1e-7, jitter=1e-10):
    """Joint GLS fit of a rank-``r`` TP-GDAR on per-condition ``stats`` (see stack_stats)."""
    P = len(stats["n"])
    N = stats["Syy"].shape[1]
    Sinv = np.linalg.pinv(Sigma)
    quads = [_quad(stats["Gzz"][g], stats["Gzy"][g], R, Sinv) for g in range(P)]
    Hs = np.stack([q[0] for q in quads])
    hs = np.stack([q[1] for q in quads])
    n_phi = hs.shape[1]

    # Start: pooled operator with constant weight, then the leading per-condition
    # deviations from it in the pooled metric, so rank r starts at the pooled model.
    Hbar, hbar = Hs.sum(0), hs.sum(0)
    phi0 = _solve(Hbar, hbar, jitter)
    phi = np.zeros((r, n_phi))
    phi[0] = phi0
    if r > 1:
        L = np.linalg.cholesky(Hbar + jitter * np.trace(Hbar) / n_phi * np.eye(n_phi))
        dev = np.stack([_solve(Hs[g], hs[g], jitter) - phi0 for g in range(P)])
        _, _, Vt = np.linalg.svd(dev @ L, full_matrices=False)
        phi[1:] = np.linalg.solve(L.T, Vt[:r - 1].T).T
    W = np.zeros((P, r))
    W[:, 0] = 1.0

    def objective():
        return sum((W[g] @ phi) @ Hs[g] @ (W[g] @ phi) - 2 * (W[g] @ phi) @ hs[g] for g in range(P))

    obj = [objective()]
    for _ in range(n_iter):
        for g in range(P):                                  # W-step
            Hw = phi @ Hs[g] @ phi.T
            W[g] = np.linalg.lstsq(Hw, phi @ hs[g], rcond=1e-12)[0]
        A = np.zeros((r * n_phi, r * n_phi))                # phi-step
        b = np.zeros(r * n_phi)
        for g in range(P):
            A += np.kron(np.outer(W[g], W[g]), Hs[g])
            b += np.kron(W[g], hs[g])
        phi = _solve(A, b, jitter).reshape(r, n_phi)
        # gauge: unit weight columns, leading (largest) operator first
        scale = np.linalg.norm(W, axis=0) / np.sqrt(P)
        W, phi = W / scale, phi * scale[:, None]
        obj.append(objective())
        if abs(obj[-2] - obj[-1]) <= tol * abs(obj[-1]):
            break
    return SymTP(W=W, phi=phi, R=R, N=N, p_grid=np.asarray(p_grid, float), objective=np.array(obj))
