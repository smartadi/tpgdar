"""Parameter-varying (quasi-LPV) GDAR fitting.

Fit ``A_k(p)`` on a grid of scheduling values by fitting an independent GDAR
model in each parameter bin, then stack the coefficients into a tensor

    coeffs[p1, ..., pd, N, N, K]

that the TP layer decomposes.  Two entry points:

* :func:`fit_grid`      -- one segment (trajectory) per grid point of ``p``.
* :func:`fit_windowed`  -- a single long recording where ``p(t)`` varies; slide
  windows, bin each window by its mean ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gdar import fit_gdar


@dataclass
class QLPVModel:
    """A grid of GDAR coefficient matrices indexed by scheduling value(s)."""

    p_grid: list[np.ndarray]      # list of d 1-D arrays, the axis ticks per param
    coeffs: np.ndarray            # (P1,...,Pd, N, N, K)
    B: np.ndarray                 # (P1,...,Pd, N, M)
    K: int
    param: str
    support: np.ndarray

    @property
    def param_shape(self):
        return tuple(len(g) for g in self.p_grid)

    def coeff_tensor(self):
        """Return (coeffs, B) with parameter modes leading."""
        return self.coeffs, self.B


def _fit_one(X, U, K, support, laplacian, param, ridge):
    m = fit_gdar(X, U=U, K=K, support=support, laplacian=laplacian,
                 param=param, ridge=ridge)
    A = np.stack(m.A, axis=-1)              # (N, N, K)
    return A, m.B


def fit_grid(segments, p_values, K=1, support=None, laplacian=None,
             param="edgewise", ridge=1e-4):
    """Fit one GDAR model per grid point (1-D schedule).

    Parameters
    ----------
    segments : list of (X, U) tuples (U may be None), one per grid point.
    p_values : (P,) array of the scheduling value at each segment.
    """
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    A_list, B_list = [], []
    for idx in order:
        X, U = segments[idx]
        A, B = _fit_one(X, U, K, support, laplacian, param, ridge)
        A_list.append(A)
        B_list.append(B)
    coeffs = np.stack(A_list, axis=0)       # (P, N, N, K)
    Bt = np.stack(B_list, axis=0)           # (P, N, M)
    supp = _support_of(support, laplacian, coeffs.shape[-2])
    return QLPVModel(p_grid=[p_sorted], coeffs=coeffs, B=Bt, K=K,
                     param=param, support=np.asarray(supp, dtype=bool))


def fit_grid_2d(segment_grid, p1_values, p2_values, K=1, support=None,
                laplacian=None, param="edgewise", ridge=1e-4):
    """Fit a 2-D schedule ``(p1, p2)``.

    Parameters
    ----------
    segment_grid : nested list ``segment_grid[i][j] = (X, U)`` of shape (P1, P2).
    """
    p1 = np.asarray(p1_values, dtype=float)
    p2 = np.asarray(p2_values, dtype=float)
    P1, P2 = len(p1), len(p2)
    A0, B0 = _fit_one(*_pad(segment_grid[0][0]), K, support, laplacian, param, ridge)
    N = A0.shape[0]
    M = B0.shape[1]
    coeffs = np.zeros((P1, P2, N, N, K))
    Bt = np.zeros((P1, P2, N, M))
    coeffs[0, 0] = A0
    Bt[0, 0] = B0
    for i in range(P1):
        for j in range(P2):
            if i == 0 and j == 0:
                continue
            X, U = _pad(segment_grid[i][j])
            A, B = _fit_one(X, U, K, support, laplacian, param, ridge)
            coeffs[i, j] = A
            Bt[i, j] = B
    supp = _support_of(support, laplacian, coeffs.shape[-2])
    return QLPVModel(p_grid=[p1, p2], coeffs=coeffs, B=Bt, K=K,
                     param=param, support=np.asarray(supp, dtype=bool))


def _support_of(support, laplacian, N):
    """The support mask stored on the model, mirroring ``fit_gdar``'s defaults."""
    if support is not None:
        return support
    if laplacian is not None:
        return (np.abs(laplacian) > 0) | np.eye(N, dtype=bool)
    return np.ones((N, N), dtype=bool)


def _pad(seg):
    if isinstance(seg, tuple):
        return seg
    return seg, None


def fit_windowed(X, p_series, n_bins=8, win=None, hop=None, U=None, K=1,
                 support=None, laplacian=None, param="edgewise", ridge=1e-4,
                 p_range=None):
    """Windowed qLPV fit from one long recording with time-varying ``p``.

    Slides windows of length ``win`` (default T/(2 n_bins)) with step ``hop``,
    computes each window's mean ``p``, bins windows into ``n_bins`` equal-width
    bins over ``p_range``, and fits one GDAR model per non-empty bin by
    concatenating that bin's windows.

    Returns
    -------
    QLPVModel with ``p_grid`` = bin centres of the *occupied* bins.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    p_series = np.asarray(p_series, dtype=float)
    if win is None:
        win = max(K + 5, T // (2 * n_bins))
    if hop is None:
        hop = max(1, win // 2)
    lo, hi = (p_series.min(), p_series.max()) if p_range is None else p_range
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # collect window slices per bin
    bins_X = [[] for _ in range(n_bins)]
    bins_U = [[] for _ in range(n_bins)]
    for start in range(0, T - win + 1, hop):
        sl = slice(start, start + win)
        pbar = p_series[sl].mean()
        b = min(n_bins - 1, max(0, int(np.searchsorted(edges, pbar, side="right") - 1)))
        bins_X[b].append(X[sl])
        if U is not None:
            bins_U[b].append(np.asarray(U)[sl])

    p_occ, A_list, B_list = [], [], []
    for b in range(n_bins):
        if not bins_X[b]:
            continue
        # fit per window then average coefficients (keeps each window stationary)
        As, Bs = [], []
        for w, Xw in enumerate(bins_X[b]):
            Uw = bins_U[b][w] if U is not None else None
            A, B = _fit_one(Xw, Uw, K, support, laplacian, param, ridge)
            As.append(A)
            Bs.append(B)
        A_list.append(np.mean(As, axis=0))
        B_list.append(np.mean(Bs, axis=0))
        p_occ.append(centres[b])

    coeffs = np.stack(A_list, axis=0)
    Bt = np.stack(B_list, axis=0)
    supp = _support_of(support, laplacian, coeffs.shape[-2])
    return QLPVModel(p_grid=[np.array(p_occ)], coeffs=coeffs, B=Bt, K=K,
                     param=param, support=np.asarray(supp, dtype=bool))
