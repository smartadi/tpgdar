"""Graph Diffusion AutoRegression (GDAR) fit.

Fits the lag-K model

    x(t+1) = sum_{k=0}^{K-1} A_k x(t-k) + B u(t) + e(t)

under a graph-structural constraint on the ``A_k``.  Two parameterizations
(the open decision in Stage 1 of the plan):

* ``"edgewise"`` -- each ``A_k`` is free on the support ``E ∪ diag`` and zero
  elsewhere, fit by masked ridge least squares, one row at a time.  This matches
  GDAR's flow-signal derivation most directly.

* ``"diffusion"`` -- strict ``A_k = a_k I - b_k L`` (two scalars per lag), fit
  jointly across channels.  Tiny parameter count, clean for the TP/LMI stages.

Both share the same design-matrix construction and prediction/flow API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GDARModel:
    A: list[np.ndarray]           # K matrices, A[k] multiplies x(t-k)
    B: np.ndarray                 # (N, M); shape (N, 0) if no input
    K: int
    param: str                    # "edgewise" | "diffusion"
    support: np.ndarray           # (N, N) bool
    diffusion: dict | None = None  # {"a": (K,), "b": (K,)} for the strict form

    @property
    def n(self) -> int:
        return self.A[0].shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    def predict_one_step(self, X, U=None):
        """One-step-ahead prediction over a trajectory.

        Given the true past at every ``t`` (teacher forcing), predict x(t+1).

        Returns
        -------
        pred : (T - K, N)  predictions for t = K-1 .. T-2
        true : (T - K, N)  the matching x(t+1)
        """
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        preds, trues = [], []
        has_u = self.m > 0 and U is not None
        for t in range(self.K - 1, T - 1):
            nxt = np.zeros(self.n)
            for k in range(self.K):
                nxt += self.A[k] @ X[t - k]
            if has_u:
                nxt += self.B @ U[t]
            preds.append(nxt)
            trues.append(X[t + 1])
        return np.array(preds), np.array(trues)

    def rollout(self, x_hist, n_steps, U=None):
        """Free-run the model from an initial history.

        Parameters
        ----------
        x_hist : (K, N) history, newest row last (x(t-K+1) .. x(t)).
        n_steps : int
        U : (n_steps, M) optional inputs applied at each generated step.

        Returns
        -------
        (n_steps, N) predicted trajectory.
        """
        buf = [np.asarray(row, dtype=float) for row in x_hist]
        out = np.zeros((n_steps, self.n))
        has_u = self.m > 0 and U is not None
        for s in range(n_steps):
            nxt = np.zeros(self.n)
            for k in range(self.K):
                nxt += self.A[k] @ buf[-1 - k]
            if has_u:
                nxt += self.B @ U[s]
            out[s] = nxt
            buf.append(nxt)
            buf.pop(0)
        return out


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def _build_design(X, U, K):
    """Assemble regressor blocks.

    Returns
    -------
    Zlags : list length K of (T_eff, N) arrays; Zlags[k] = x(t-k)
    Ublk  : (T_eff, M) inputs at time t   (M may be 0)
    Y     : (T_eff, N) targets x(t+1)
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    M = 0 if U is None else np.asarray(U).shape[1]
    T_eff = T - K
    Zlags = [np.zeros((T_eff, N)) for _ in range(K)]
    Ublk = np.zeros((T_eff, M))
    Y = np.zeros((T_eff, N))
    for idx, t in enumerate(range(K - 1, T - 1)):
        for k in range(K):
            Zlags[k][idx] = X[t - k]
        if M:
            Ublk[idx] = U[t]
        Y[idx] = X[t + 1]
    return Zlags, Ublk, Y


# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------

def _fit_edgewise(Zlags, Ublk, Y, support, ridge):
    K = len(Zlags)
    T_eff, N = Y.shape
    M = Ublk.shape[1]
    A = [np.zeros((N, N)) for _ in range(K)]
    B = np.zeros((N, M))
    # Full lagged regressor stack, shared across rows: [x(t) | x(t-1) | ... | u(t)]
    Zfull = np.hstack(Zlags + ([Ublk] if M else []))  # (T_eff, K*N + M)
    for i in range(N):
        allowed_j = np.where(support[i])[0]
        # columns for this row: allowed j at every lag, then all inputs
        cols = []
        col_map = []  # (kind, k, j) for placing back
        for k in range(K):
            for j in allowed_j:
                cols.append(k * N + j)
                col_map.append(("A", k, j))
        for mm in range(M):
            cols.append(K * N + mm)
            col_map.append(("B", None, mm))
        Zi = Zfull[:, cols]
        # ridge normal equations
        G = Zi.T @ Zi + ridge * np.eye(Zi.shape[1])
        rhs = Zi.T @ Y[:, i]
        beta = np.linalg.solve(G, rhs)
        for c, (kind, k, j) in enumerate(col_map):
            if kind == "A":
                A[k][i, j] = beta[c]
            else:
                B[i, j] = beta[c]
    return A, B


def _fit_diffusion(Zlags, Ublk, Y, L, ridge):
    """Jointly fit A_k = a_k I - b_k L plus a per-channel input matrix B.

    Parameters a_k, b_k are global scalars; B is (N, M).
    Design rows are indexed by (t, channel i).
    """
    K = len(Zlags)
    T_eff, N = Y.shape
    M = Ublk.shape[1]
    n_glob = 2 * K
    n_par = n_glob + N * M

    # Precompute L @ x(t-k) for each lag.
    LZ = [(Zlags[k] @ L.T) for k in range(K)]  # (Lx)_i(t-k) = (L @ x)_i

    rows = T_eff * N
    D = np.zeros((rows, n_par))
    y = Y.reshape(-1)  # channel-fastest? we need consistent ordering
    # Order rows as (t outer, i inner): flatten Y row-major gives [t0: i0..iN, t1:...]
    for k in range(K):
        # a_k column: value x_i(t-k) at row (t,i)
        D[:, k] = Zlags[k].reshape(-1)
        # b_k column: coeff is -b_k, so store column as (Lx) and negate later
        D[:, K + k] = LZ[k].reshape(-1)
    if M:
        # B block: row (t,i) uses u(t) in channel i's own M columns
        for i in range(N):
            base = n_glob + i * M
            # rows where channel == i are indices t*N + i
            ridx = np.arange(T_eff) * N + i
            D[ridx, base:base + M] = Ublk
    G = D.T @ D + ridge * np.eye(n_par)
    beta = np.linalg.solve(G, D.T @ y)
    a = beta[:K]
    b_coeff = beta[K:2 * K]            # this is the coefficient on (Lx) = -b_k
    b = -b_coeff
    eye = np.eye(N)
    A = [a[k] * eye - b[k] * L for k in range(K)]
    B = np.zeros((N, M))
    if M:
        B = beta[n_glob:].reshape(N, M)
    return A, B, a, b


def fit_gdar(X, U=None, K=1, support=None, laplacian=None,
             param="edgewise", ridge=1e-4):
    """Fit a GDAR model.

    Parameters
    ----------
    X : (T, N) channel time series.
    U : (T, M) inputs, optional.
    K : int, number of lags.
    support : (N, N) bool, required for ``param="edgewise"``.
    laplacian : (N, N), required for ``param="diffusion"``.
    param : "edgewise" | "diffusion".
    ridge : float, Tikhonov weight.

    Returns
    -------
    GDARModel
    """
    Zlags, Ublk, Y = _build_design(X, U, K)
    N = X.shape[1]
    if param == "edgewise":
        if support is None:
            support = np.ones((N, N), dtype=bool)
        A, B = _fit_edgewise(Zlags, Ublk, Y, support, ridge)
        return GDARModel(A=A, B=B, K=K, param=param, support=support.astype(bool))
    elif param == "diffusion":
        if laplacian is None:
            raise ValueError("diffusion parameterization needs a Laplacian")
        A, B, a, b = _fit_diffusion(Zlags, Ublk, Y, laplacian, ridge)
        supp = (np.abs(laplacian) > 0) | np.eye(N, dtype=bool)
        return GDARModel(A=A, B=B, K=K, param=param, support=supp,
                         diffusion={"a": a, "b": b})
    raise ValueError(f"unknown param {param!r}")


# ---------------------------------------------------------------------------
# Flow signal & coupling
# ---------------------------------------------------------------------------

def coupling_matrix(model: GDARModel) -> np.ndarray:
    """Static off-diagonal coupling strength ``C[i,j] = ||A_.[i,j]||_2`` over lags.

    Diagonal zeroed.  This is the quantity we correlate against ground-truth
    connectivity (the "flow-signal correlation" metric).
    """
    stack = np.stack(model.A, axis=0)  # (K, N, N)
    C = np.sqrt((stack ** 2).sum(axis=0))
    np.fill_diagonal(C, 0.0)
    return C


def edge_flow(model: GDARModel, X, edges):
    """Directed edge-flow time series.

    For each edge (i, j) returns both directed flows:
    phi_{i<-j}(t) = sum_k A_k[i,j] x_j(t-k)  and its reverse.

    Returns
    -------
    flows : (T - K, n_edges, 2) array; [:, e, 0] = i<-j, [:, e, 1] = j<-i
    t_index : indices of t (K-1 .. T-2)
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    t_range = range(model.K - 1, T - 1)
    flows = np.zeros((len(list(range(model.K - 1, T - 1))), len(edges), 2))
    for idx, t in enumerate(t_range):
        for e, (i, j) in enumerate(edges):
            fij = sum(model.A[k][i, j] * X[t - k, j] for k in range(model.K))
            fji = sum(model.A[k][j, i] * X[t - k, i] for k in range(model.K))
            flows[idx, e, 0] = fij
            flows[idx, e, 1] = fji
    return flows, np.array(list(t_range))
