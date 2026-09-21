"""TP (Tensor Product) model transformation of a qLPV coefficient tensor.

Two related jobs:

1. **HOSVD over the parameter modes** (:func:`hosvd`) -- compress the dependence
   of the GDAR coefficients on the scheduling variable(s) into a small core of
   *vertex systems* plus orthonormal *weighting functions* ``w_j(p)``.  The
   singular-value spectrum tells you how many modes the parameter dependence
   really has.

2. **Convex-hull / polytopic form** (:func:`convex_hull`) -- re-express the
   coefficient trajectory ``A(p)`` as a convex combination of a finite set of
   vertex LTI systems, ``A(p) = sum_v lambda_v(p) A_v`` with ``lambda_v >= 0`` and
   ``sum_v lambda_v = 1``.  This polytope is exactly what the LMI control
   synthesis (``control.py``) consumes.  Two hull types are offered:

   * ``method="tight"`` -- the exact convex hull of the trajectory (QuickHull);
     the fewest, tightest vertices (CNO-like: least conservative for control).
   * ``method="box"``   -- an axis-aligned bounding box in the affine PCA
     coordinates (``2**q`` vertices); looser but always valid (SNNN-like).

Reducing vertices vs. hull tightness is the feasibility knob explored in Stage 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# HOSVD over parameter modes
# ---------------------------------------------------------------------------

@dataclass
class TPModel:
    factors: list[np.ndarray]      # U_m per parameter mode, (P_m, r_m)
    core: np.ndarray               # (r_1,...,r_d, *sys_shape)
    singular_values: list[np.ndarray]
    ranks: list[int]
    sys_shape: tuple
    n_param_modes: int

    @property
    def n_vertices(self) -> int:
        return int(np.prod(self.ranks))

    def vertex_systems(self):
        """Return the ``prod(ranks)`` vertex systems, each shaped ``sys_shape``.

        Ordered by the flattened core parameter indices (row-major over ranks).
        """
        R = self.n_vertices
        return self.core.reshape(R, *self.sys_shape)

    def reconstruct(self):
        """Reconstruct the full tensor ``core x_1 U_1 ... x_d U_d``."""
        T = self.core
        for m in range(self.n_param_modes):
            T = _mode_mult(T, self.factors[m], m)
        return T


def _unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def _mode_mult(T, M, mode):
    """Mode-``mode`` product ``T x_mode M`` (M is (J, r_mode))."""
    Tn = np.moveaxis(T, mode, 0)          # (r_mode, ...)
    out = M @ Tn.reshape(Tn.shape[0], -1)  # (J, ...)
    out = out.reshape((M.shape[0],) + Tn.shape[1:])
    return np.moveaxis(out, 0, mode)


def hosvd(tensor, n_param_modes, energy=0.9999, max_rank=None, min_sv=1e-9):
    """Higher-order SVD truncated along the leading ``n_param_modes`` modes.

    The trailing modes (the "system" dimensions, e.g. ``N, N, K``) are *not*
    compressed; only the parameter dependence is.

    Parameters
    ----------
    tensor : ndarray
        Shape ``(P_1, ..., P_d, *sys_shape)`` with ``d = n_param_modes``.
    energy : float
        Keep the fewest singular values per mode capturing this energy fraction.
    max_rank : int or None
        Optional hard cap on retained rank per parameter mode.
    """
    tensor = np.asarray(tensor, dtype=float)
    d = n_param_modes
    sys_shape = tensor.shape[d:]
    factors, svals, ranks = [], [], []
    core = tensor
    for m in range(d):
        Um_full, s, _ = np.linalg.svd(_unfold(tensor, m), full_matrices=False)
        e = s ** 2
        cum = np.cumsum(e) / e.sum()
        r = int(np.searchsorted(cum, energy) + 1)
        r = max(1, min(r, np.sum(s > min_sv)))
        if max_rank is not None:
            r = min(r, max_rank)
        Um = Um_full[:, :r]
        factors.append(Um)
        svals.append(s)
        ranks.append(r)
    # project onto the retained factors to form the core
    for m in range(d):
        core = _mode_mult(core, factors[m].T, m)
    return TPModel(factors=factors, core=core, singular_values=svals,
                   ranks=ranks, sys_shape=sys_shape, n_param_modes=d)


# ---------------------------------------------------------------------------
# Convex-hull / polytopic form
# ---------------------------------------------------------------------------

@dataclass
class Polytope:
    vertex_systems: np.ndarray     # (V, *sys_shape) LTI vertex matrices
    weights: np.ndarray            # (G, V) convex weights per grid point
    vertex_grid_index: np.ndarray  # (V,) index into the flattened grid
    method: str
    affine_dim: int
    residual: float = 0.0          # max reconstruction error of A(p) over grid
    grid_shape: tuple = ()
    meta: dict = field(default_factory=dict)

    @property
    def n_vertices(self):
        return self.vertex_systems.shape[0]


def _barycentric(Y_vertices, y_points, ridge=1e-10):
    """Nonneg, sum-to-one weights of points w.r.t. vertices in affine coords.

    Solves, per point, ``min_{lambda>=0} ||[Y_v; 1] lambda - [y; 1]||`` and
    renormalizes.  ``Y_vertices`` is (V, q), ``y_points`` is (G, q).
    """
    from scipy.optimize import nnls

    V = Y_vertices.shape[0]
    G = y_points.shape[0]
    # augment with a sum-to-one row (weighted large to enforce it)
    w_sum = 1e3
    A_aug = np.vstack([Y_vertices.T, w_sum * np.ones((1, V))])  # (q+1, V)
    lam = np.zeros((G, V))
    for i in range(G):
        b = np.concatenate([y_points[i], [w_sum]])
        sol, _ = nnls(A_aug, b)
        s = sol.sum()
        lam[i] = sol / s if s > ridge else np.ones(V) / V
    return lam


def convex_hull(coeff_tensor, n_param_modes, method="tight", energy=0.9999,
                max_vertices=None, n_vertices=None):
    """Build a polytopic representation of the coefficient trajectory.

    Parameters
    ----------
    coeff_tensor : ndarray  ``(P_1,...,P_d, *sys_shape)``.
    n_param_modes : int
    method : "tight" | "box"
    energy : float
        Affine dimensionality is chosen to capture this fraction of the
        coefficient variance across the grid.

    Returns
    -------
    Polytope
    """
    T = np.asarray(coeff_tensor, dtype=float)
    d = n_param_modes
    grid_shape = T.shape[:d]
    sys_shape = T.shape[d:]
    G = int(np.prod(grid_shape))
    S = int(np.prod(sys_shape))
    M = T.reshape(G, S)                     # coefficient vector per grid point

    # affine embedding via PCA
    mean = M.mean(axis=0)
    Mc = M - mean
    U, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    if s[0] < 1e-12:
        # constant over the grid: single vertex
        vs = M[:1].reshape(1, *sys_shape)
        return Polytope(vertex_systems=vs, weights=np.ones((G, 1)),
                        vertex_grid_index=np.array([0]), method=method,
                        affine_dim=0, residual=0.0, grid_shape=grid_shape)
    e = s ** 2
    cum = np.cumsum(e) / e.sum()
    q = int(np.searchsorted(cum, energy) + 1)
    q = max(1, q)
    Y = U[:, :q] * s[:q]                    # (G, q) affine coordinates

    if method == "tight":
        idx = _tight_hull_vertices(Y)
    elif method == "box":
        idx = _box_vertices(Y)
    elif method == "spa":
        if n_vertices is None:
            raise ValueError("method='spa' needs n_vertices")
        idx = _spa_vertices(Y, n_vertices)
    else:
        raise ValueError(f"unknown hull method {method!r}")

    if max_vertices is not None and len(idx) > max_vertices:
        # keep the vertices with the largest spread (farthest from centroid)
        c = Y.mean(axis=0)
        order = np.argsort(-np.linalg.norm(Y[idx] - c, axis=1))
        idx = np.array(idx)[order[:max_vertices]]

    idx = np.asarray(sorted(set(int(i) for i in idx)))
    vertex_systems = M[idx].reshape(len(idx), *sys_shape)
    lam = _barycentric(Y[idx], Y)
    recon = lam @ M[idx]                    # (G, S)
    residual = float(np.max(np.linalg.norm(recon - M, axis=1)))
    return Polytope(vertex_systems=vertex_systems, weights=lam,
                    vertex_grid_index=idx, method=method, affine_dim=q,
                    residual=residual, grid_shape=grid_shape)


def _tight_hull_vertices(Y):
    """Indices of the exact convex-hull vertices of the point cloud ``Y``."""
    q = Y.shape[1]
    if q == 1:
        return [int(np.argmin(Y[:, 0])), int(np.argmax(Y[:, 0]))]
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = ConvexHull(Y)
        return list(np.unique(hull.vertices))
    except (QhullError, Exception):
        # degenerate (points nearly on a lower-dim flat): fall back per-axis
        return _box_vertices(Y)


def _spa_vertices(Y, r):
    """Indices of ``r`` simplex corners via farthest-from-affine-hull greedy.

    Endmember/vertex finder (the robust N-FINDR-style variant): the first corner
    is the point farthest from the centroid; each subsequent corner is the point
    farthest from the affine subspace spanned by those already chosen.  Recovers
    the true pure vertices when the data visits them, even from mixed samples.
    """
    G = Y.shape[0]
    idx = [int(np.argmax(np.linalg.norm(Y - Y.mean(axis=0), axis=1)))]
    while len(idx) < min(r, G):
        anchor = Y[idx[0]]
        rel = Y - anchor
        if len(idx) > 1:
            diffs = (Y[idx[1:]] - anchor).T          # (q, m)
            Qb, _ = np.linalg.qr(diffs)               # orthonormal span (q, m')
            resid = rel - rel @ Qb @ Qb.T
        else:
            resid = rel
        dist = np.linalg.norm(resid, axis=1)
        j = int(np.argmax(dist))
        if j in idx:
            break
        idx.append(j)
    return idx


def _box_vertices(Y):
    """Grid indices closest to each corner of the bounding box in affine coords."""
    q = Y.shape[1]
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    corners = []
    for mask in range(2 ** q):
        corner = np.array([hi[a] if (mask >> a) & 1 else lo[a] for a in range(q)])
        j = int(np.argmin(np.linalg.norm(Y - corner, axis=1)))
        corners.append(j)
    return list(np.unique(corners))


def weight_functions(tp: TPModel):
    """Per-parameter-mode weighting functions (the HOSVD factor columns)."""
    return [f.copy() for f in tp.factors]
