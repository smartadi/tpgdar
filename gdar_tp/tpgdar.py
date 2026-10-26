"""Stacked TP-GDAR: the TP structure fitted on the whole one-step operator.

Implements docs/MODEL.md sections 1-3.  The one-step operator at grid point ``g`` is

    Theta(p_g) = [ A_0 .. A_{K-1} | B_0 .. B_{Q-1} ]  in R^{N x D},    x(t+1) = Theta z(t) + e

and the TP form, per column block ``b`` (e.g. b = A columns, b = B columns), is

    Theta_b(p) = sum_j W_b[g, j] Theta_b^(j).

Everything is computed from per-condition sufficient statistics
``Gzz = sum z z^T``, ``Gzy = sum z x(t+1)^T``, ``Syy = sum x(t+1) x(t+1)^T``.  Two
learners:

* :func:`fit_two_stage` -- independent per-condition fits, then an SVD over the
  p mode per block (the HOSVD route the scripts used up to 2026-10-26);
* :func:`fit_joint` -- alternating least squares on the actual data loss, i.e. a
  G_g-weighted low-rank fit; each step is an exact block minimiser, so the
  objective never increases.  Initialised from the two-stage solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------

def trial_stats(X, U=None, K=1, Q=1):
    """Sufficient statistics of one trial (never crosses a trial boundary).

    ``z(t) = [x(t); ..; x(t-K+1); u(t); ..; u(t-Q+1)]`` predicts ``x(t+1)``.

    Returns
    -------
    dict with Gzz (D, D), Gzy (D, N), Syy (N, N), n (int)
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    lag0 = max(K, Q if U is not None else 1) - 1
    ts = np.arange(lag0, T - 1)
    blocks = [X[ts - k] for k in range(K)]
    if U is not None:
        U = np.asarray(U, dtype=float).reshape(T, -1)
        blocks += [U[ts - q] for q in range(Q)]
    Z = np.concatenate(blocks, axis=1)
    Y = X[ts + 1]
    return dict(Gzz=Z.T @ Z, Gzy=Z.T @ Y, Syy=Y.T @ Y, n=len(ts))


def stack_stats(per_condition):
    """List (one per grid point) of lists of trial stats -> stacked summed arrays."""
    out = {k: [] for k in ("Gzz", "Gzy", "Syy", "n")}
    for trials in per_condition:
        for k in out:
            out[k].append(sum(s[k] for s in trials))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def row_columns(support, K, n_inputs=0):
    """Allowed regressor columns per output row: graph-supported lags + all inputs."""
    support = np.asarray(support, dtype=bool)
    N = support.shape[0]
    u_cols = np.arange(K * N, K * N + n_inputs)
    return [np.concatenate([k * N + np.where(support[i])[0] for k in range(K)] + [u_cols])
            for i in range(N)]


def _ridge(stats_Gzz, cols, ridge):
    """Levenberg-style Tikhonov weights, one per (row, allowed column).

    ``lam[i][c] = ridge * Gzz[c, c]``: each coefficient is shrunk in proportion to
    how strongly its own regressor is excited.  This is scale-free per column (A
    lags and input lags have different units) and is what keeps the badly
    conditioned lag Grams of real ECoG (cond ~ 5e10) from overfitting; a single
    per-row weight does not.
    """
    d = np.diagonal(stats_Gzz)
    return [ridge * d[c] + 1e-300 for c in cols]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def resid_cov_sum(Theta, Gzz, Gzy, Syy):
    """Residual outer-product sum ``sum_t e e^T`` for operator Theta (N, D)."""
    TG = Theta @ Gzy
    return Syy - TG - TG.T + Theta @ Gzz @ Theta.T


def data_loss(Theta, Gzz, Gzy, Syy):
    return float(np.trace(resid_cov_sum(Theta, Gzz, Gzy, Syy)))


def gaussian_ll(Theta, Gzz, Gzy, Syy, n, Sigma):
    """Total one-step Gaussian log-likelihood of a data set under (Theta, Sigma)."""
    S = resid_cov_sum(Theta, Gzz, Gzy, Syy)
    _, ld = np.linalg.slogdet(Sigma)
    N = Sigma.shape[0]
    return float(-0.5 * (np.trace(np.linalg.solve(Sigma, S)) + n * (N * np.log(2 * np.pi) + ld)))


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------

def fit_static(Gzz, Gzy, cols, ridge=1e-6):
    """Graph-supported ridge LS for one operator from summed statistics."""
    N = Gzy.shape[1]
    lam = _ridge(Gzz, cols, ridge)
    Theta = np.zeros((N, Gzz.shape[0]))
    for i, c in enumerate(cols):
        R = Gzz[np.ix_(c, c)] + np.diag(lam[i])
        Theta[i, c] = np.linalg.solve(R, Gzy[c, i])
    return Theta


def fit_per_condition(stats, cols, ridge=1e-6):
    """Independent operator per grid point, (P, N, D)."""
    return np.stack([fit_static(stats["Gzz"][g], stats["Gzy"][g], cols, ridge)
                     for g in range(len(stats["n"]))])


def fit_pooled(stats, cols, ridge=1e-6):
    """One p-blind operator for all grid points, (N, D)."""
    return fit_static(stats["Gzz"].sum(0), stats["Gzy"].sum(0), cols, ridge)


@dataclass
class TPGDAR:
    """TP form per column block: ``Theta(p_g)[:, blocks[b]] = sum_j W[b][g, j] Theta[b][j]``."""

    blocks: list            # column index arrays, a partition of 0..D-1
    W: list                 # per block (P, r_b)
    Theta: list             # per block (r_b, N, |block_b|)
    p_grid: np.ndarray
    D: int
    objective: np.ndarray = field(default_factory=lambda: np.zeros(0))
    method: str = ""
    row_cols: list = field(default_factory=list)

    @property
    def ranks(self):
        return [w.shape[1] for w in self.W]

    def theta_from_weights(self, w_list):
        N = self.Theta[0].shape[1]
        out = np.zeros((N, self.D))
        for b, cb in enumerate(self.blocks):
            out[:, cb] = np.tensordot(w_list[b], self.Theta[b], axes=1)
        return out

    def theta_grid(self):
        P = self.W[0].shape[0]
        return np.stack([self.theta_from_weights([w[g] for w in self.W]) for g in range(P)])

    def weights_at(self, p, log=False):
        x = np.log(self.p_grid) if log else self.p_grid
        x0 = np.log(p) if log else p
        order = np.argsort(x)
        return [np.array([np.interp(x0, x[order], w[order, j]) for j in range(w.shape[1])])
                for w in self.W]

    def theta_at(self, p, log=False):
        """Operator at an off-grid ``p`` by linear interpolation of the weights."""
        return self.theta_from_weights(self.weights_at(p, log=log))


def _pooled_start(stats, cols, blocks, ranks, p_grid, ridge):
    """Initial point that already equals the pooled fit: component 1 is the pooled
    operator with constant weight, the rest span the per-condition deviations."""
    pooled = fit_pooled(stats, cols, ridge)
    Th = fit_per_condition(stats, cols, ridge)
    P, N, D = Th.shape
    W, Theta = [], []
    for cb, r in zip(blocks, ranks):
        w = [np.ones(P)]
        th = [pooled[:, cb]]
        if r > 1:
            U, sv, Vt = np.linalg.svd((Th[:, :, cb] - pooled[None, :, cb]).reshape(P, -1),
                                      full_matrices=False)
            for j in range(r - 1):
                w.append(U[:, j])
                th.append((sv[j] * Vt[j]).reshape(N, len(cb)))
        W.append(np.stack(w, axis=1))
        Theta.append(np.stack(th))
    return TPGDAR(blocks=list(blocks), W=W, Theta=Theta, p_grid=np.asarray(p_grid, float),
                  D=D, method="pooled-start", row_cols=list(cols))


def fit_two_stage(stats, cols, blocks, ranks, p_grid, ridge=1e-6):
    """Per-condition fits, then a rank-``r_b`` SVD over the p mode of each block."""
    Th = fit_per_condition(stats, cols, ridge)          # (P, N, D)
    P, N, D = Th.shape
    W, Theta = [], []
    for cb, r in zip(blocks, ranks):
        M = Th[:, :, cb].reshape(P, -1)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        W.append(U[:, :r].copy())
        Theta.append((s[:r, None] * Vt[:r]).reshape(r, N, len(cb)))
    return TPGDAR(blocks=list(blocks), W=W, Theta=Theta, p_grid=np.asarray(p_grid, float),
                  D=D, method="two-stage", row_cols=list(cols))


def augment(stats, ridge):
    """Ridge as an augmented Gram per grid point: ``Gzz_g + ridge * diag(Gzz_g)``.

    The penalty ``sum_g sum_ic ridge * Gzz_g[c,c] * Theta(p_g)[i,c]^2`` sits on the
    operators at the grid points, not on the TP factors, so it cannot be escaped by
    rescaling W against Theta_j (the gauge).  At full rank it reproduces
    :func:`fit_per_condition` exactly, and a constant rank-1 weight reproduces
    :func:`fit_pooled` (whose ridge is built from the summed Gram).
    """
    out = dict(stats)
    d = np.einsum("gcc->gc", stats["Gzz"])
    out["Gzz"] = stats["Gzz"] + ridge * np.einsum("gc,cd->gcd", d, np.eye(d.shape[1]))
    return out


def joint_objective(model, stats_aug):
    """J = sum_g L~_g(Theta(p_g)), the data loss with the ridge folded into each Gram."""
    return float(sum(data_loss(Th, stats_aug["Gzz"][g], stats_aug["Gzy"][g], stats_aug["Syy"][g])
                     for g, Th in enumerate(model.theta_grid())))


def _theta_step(model, stats, cols):
    """Exact minimiser over all Theta_b^(j) with W fixed (row-wise normal equations)."""
    Gzz, Gzy = stats["Gzz"], stats["Gzy"]
    N = Gzy.shape[2]
    nb = len(model.blocks)
    # position of every column inside its block
    where = np.empty(model.D, dtype=int); owner = np.empty(model.D, dtype=int)
    for b, cb in enumerate(model.blocks):
        where[cb] = np.arange(len(cb)); owner[cb] = b
    # block-pair normal matrices over full block columns: R[b][b'] (r_b, nb, r_b', nb')
    R = [[np.einsum("gj,gk,gcd->jckd", model.W[b], model.W[b2],
                    Gzz[:, model.blocks[b]][:, :, model.blocks[b2]], optimize=True)
          for b2 in range(nb)] for b in range(nb)]
    rhs = [np.einsum("gj,gci->jci", model.W[b], Gzy[:, model.blocks[b], :], optimize=True)
           for b in range(nb)]
    new = [np.zeros_like(t) for t in model.Theta]
    for i in range(N):
        # unknowns: for each block b, for each j, the allowed columns of row i in block b
        sel = [where[cols[i][owner[cols[i]] == b]] for b in range(nb)]
        idx = [(b, j) for b in range(nb) for j in range(model.W[b].shape[1])]
        sizes = [len(sel[b]) for b, _ in idx]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        A = np.zeros((offs[-1], offs[-1])); y = np.zeros(offs[-1])
        for m, (b, j) in enumerate(idx):
            sm = slice(offs[m], offs[m + 1])
            y[sm] = rhs[b][j, sel[b], i]
            for m2, (b2, j2) in enumerate(idx):
                A[sm, offs[m2]:offs[m2 + 1]] = R[b][b2][j][np.ix_(sel[b], [j2], sel[b2])][:, 0, :]
        beta = np.linalg.solve(A, y)
        for m, (b, j) in enumerate(idx):
            new[b][j, i, sel[b]] = beta[offs[m]:offs[m + 1]]
    model.Theta = new


def _w_step(model, stats):
    """Exact minimiser over the weights of every grid point with Theta fixed."""
    Gzz, Gzy = stats["Gzz"], stats["Gzy"]
    N = Gzy.shape[2]
    Psi = []                                            # padded (N, D) per (b, j)
    for b, cb in enumerate(model.blocks):
        for j in range(model.W[b].shape[1]):
            full = np.zeros((N, model.D)); full[:, cb] = model.Theta[b][j]
            Psi.append(full)
    Psi = np.stack(Psi)                                 # (m, N, D)
    rb = model.ranks
    for g in range(Gzz.shape[0]):
        PG = Psi @ Gzz[g]                               # (m, N, D)
        H = np.einsum("mnd,knd->mk", PG, Psi)
        h = np.einsum("mnd,dn->m", Psi, Gzy[g])
        H[np.diag_indices_from(H)] += 1e-12 * (np.trace(H) / len(H) + 1e-300)   # rank-deficient guard
        w = np.linalg.solve(H, h)
        o = 0
        for b, r in enumerate(rb):
            model.W[b][g] = w[o:o + r]; o += r


def _normalize_gauge(model):
    """Rotate each block so its W has orthonormal columns (Theta(p) unchanged)."""
    for b in range(len(model.blocks)):
        Q, Rm = np.linalg.qr(model.W[b])
        model.W[b] = Q
        model.Theta[b] = np.tensordot(Rm, model.Theta[b], axes=1)


def fit_joint(stats, cols, blocks, ranks, p_grid, ridge=1e-6, n_iter=200, tol=1e-8,
              init=None):
    """Joint (G-weighted) TP-GDAR fit by alternating least squares.

    Minimises ``J = sum_g L_g(Theta(p_g)) + ridge * sum_g ||Theta(p_g)||^2_{diag G_g}``,
    i.e. the data loss on Grams augmented by :func:`augment`.  The ridge is on the
    grid-point operators, so it is invariant to the W/Theta gauge.  Each sweep is a
    Theta-step then a W-step, both exact, so J is non-increasing.

    Default start: the pooled operator plus the leading per-condition deviations
    (:func:`_pooled_start`), so the fit can only improve on pooled.  (Starting from
    the two-stage HOSVD is valid but can sit far from the optimum in the data
    metric on badly conditioned real data.)
    """
    model = init if init is not None else _pooled_start(stats, cols, blocks, ranks,
                                                        p_grid, ridge)
    model = TPGDAR(blocks=model.blocks, W=[w.copy() for w in model.W],
                   Theta=[t.copy() for t in model.Theta], p_grid=model.p_grid,
                   D=model.D, method="joint", row_cols=list(cols))
    aug = augment(stats, ridge)
    trace = [joint_objective(model, aug)]
    for _ in range(n_iter):
        _theta_step(model, aug, cols)
        _w_step(model, aug)
        trace.append(joint_objective(model, aug))
        if trace[-2] - trace[-1] <= tol * abs(trace[-2]):
            break
    _normalize_gauge(model)
    model.objective = np.array(trace)
    return model
