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
from scipy.linalg import cho_factor, cho_solve

from .tp import _barycentric
from .tpgdar import resid_cov_sum


def symmetric_basis(adjacency, K, node_inputs=False):
    """Sparse R with ``vec(Theta) = R phi`` (column-major, Theta is N x KN).

    ``phi`` holds, per lag, the N self terms and then one coefficient per edge i < j.
    With ``node_inputs`` each listed node also has its own input, ``x_i(t+1) += b_i u_m(t)``
    (``True`` lists every node): Theta gains one column per input and phi gains one number
    per input at the end.  B is then diagonal for every node, or one entry per column.
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
    n_phi = K * n_lag
    D = K * N
    nodes = np.arange(N) if node_inputs is True else np.asarray([] if node_inputs is False else node_inputs, dtype=int)
    if len(nodes):
        rows.append((K * N + np.arange(len(nodes))) * N + nodes)
        cols.append(n_phi + np.arange(len(nodes)))
        n_phi += len(nodes)
        D += len(nodes)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N * D, n_phi))


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
    """``(H + jitter d I) x = h``, d the mean diagonal.  H is a Gram-weighted form, so Cholesky; LU if rounding breaks it."""
    M = np.array(H, dtype=float)
    d = np.trace(M) / len(M) if len(M) else 1.0
    M[np.diag_indices(len(M))] += jitter * d
    try:
        return cho_solve(cho_factor(M, lower=True, check_finite=False), h, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.solve(M, h)


def gls_quads(stats, R, Sigma):
    """Stacked ``H_g, h_g`` of every condition, so that fits with the same Sigma can share them."""
    Sinv = np.linalg.pinv(Sigma)
    quads = [_quad(stats["Gzz"][g], stats["Gzy"][g], R, Sinv) for g in range(len(stats["n"]))]
    return np.stack([q[0] for q in quads]), np.stack([q[1] for q in quads])


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


def polytope_vertices(model, Hbar):
    """Convex form of a rank-r fit: ``Theta(p_g) = sum_v lam[g, v] Theta_v``, lam >= 0, sum 1.

    Distances are in the pooled data metric ``Hbar``, not the unweighted coefficients.
    Rank 2: the two extreme grid points, which contain the trajectory.  Rank 3: the
    smallest triangle on three convex-hull edges, which contains it too, but whose corners
    are generally not grid points (``index`` then gives the nearest one, ``dist`` how far,
    as a fraction of the spread).  Higher ranks: successive projection onto grid points,
    which need not contain it; ``outside`` reports the miss and ``affine_err`` the cost of
    sum-to-one.  Vertices stay symmetric and graph-supported.
    """
    W, phi = model.W, model.phi
    P, r = W.shape
    a = np.linalg.lstsq(W, np.ones(P), rcond=None)[0]
    affine_err = float(np.max(np.abs(W @ a - 1)))
    Wa = W + np.outer(1 - W @ a, a) / (a @ a)
    L = np.linalg.cholesky(phi @ Hbar @ phi.T + 1e-12 * np.eye(r))
    Y = Wa @ L
    c = Y.mean(0)
    spread = np.max(np.linalg.norm(Y - c, axis=1))
    if r == 3:
        _, _, Vt = np.linalg.svd(Y - c, full_matrices=False)
        tri = _enclosing_triangle((Y - c) @ Vt[:2].T)
        Yv = tri @ Vt[:2] + c
        lam = _barycentric(Yv, Y)
        dist = np.linalg.norm(Y[:, None] - Yv[None], axis=2)
        return dict(index=dist.argmin(0), dist=dist.min(0) / spread, lam=lam,
                    phi=np.linalg.solve(L.T, Yv.T).T @ phi, affine_err=affine_err,
                    outside=float(np.max(np.linalg.norm(lam @ Yv - Y, axis=1)) / spread))
    idx = [int(np.argmax(np.linalg.norm(Y - c, axis=1)))]
    while len(idx) < r:
        basis = Y[idx[1:]] - Y[idx[0]] if len(idx) > 1 else np.zeros((0, r))
        res = Y - Y[idx[0]]
        if len(basis):
            Q, _ = np.linalg.qr(basis.T)
            res = res - res @ Q @ Q.T
        idx.append(int(np.argmax(np.linalg.norm(res, axis=1))))
    lam = _barycentric(Y[idx], Y)
    outside = float(np.max(np.linalg.norm(lam @ Y[idx] - Y, axis=1)) / spread)
    return dict(index=np.array(idx), dist=np.zeros(r), lam=lam, phi=Wa[idx] @ phi,
                affine_err=affine_err, outside=outside)


def regular_simplex_normals(r):
    """Outward unit facet normals ``(r + 1, r)`` of a regular r-simplex centred at 0; they sum to zero.

    Facet k is opposite vertex k, so vertex k lies in direction ``-normals[k]``.  r = 1: the two ends of a segment;
    r = 2: an equilateral triangle with normals at 0, 120 and 240 degrees; r = 3: a regular tetrahedron.  MATH §9.6.
    """
    if r == 1:
        return np.array([[1.0], [-1.0]])
    if r == 2:
        a = 2 * np.pi * np.arange(3) / 3
        return np.stack([np.cos(a), np.sin(a)], 1)
    if r == 3:
        return -np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)
    raise ValueError("regular_simplex_normals: r must be 1, 2 or 3")


def rotation_matrix(r, angles):
    """Rotation of R^r: none at r = 1, one angle (radians) at r = 2, a rotation vector (3 numbers) at r = 3."""
    if r == 1:
        return np.eye(1)
    if r == 2:
        c, s = np.cos(float(np.ravel(angles)[0])), np.sin(float(np.ravel(angles)[0]))
        return np.array([[c, -s], [s, c]])
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(angles, float).reshape(3)).as_matrix()


def hull_frame(model, Hbar):
    """Bins of a free rank-r fit in prediction coordinates, centred and on their principal axes.

    ``y_g = W_g L`` with ``L L' = Phi Hbar Phi'`` (Euclidean distance = distance in the pooled data metric, MATH §9.2);
    ``yt_g = (y_g - ybar) U`` with U the principal axes, each signed so that its largest coordinate is positive.  The
    frame doesn't depend on the gauge W -> W M, Phi -> M^-1 Phi.  Returns (Y, ybar, U, Yt, L).
    """
    W, phi = model.W, model.phi
    r = W.shape[1]
    L = np.linalg.cholesky(phi @ Hbar @ phi.T + 1e-12 * np.eye(r))
    Y = W @ L
    ybar = Y.mean(0)
    _, _, Vt = np.linalg.svd(Y - ybar, full_matrices=False)
    U = Vt.T
    Yt = (Y - ybar) @ U
    flip = np.sign(Yt[np.argmax(np.abs(Yt), 0), np.arange(r)])
    flip[flip == 0] = 1.0
    return Y, ybar, U * flip, Yt * flip, L


def tight_regular_simplex(Yt, rot):
    """Tightest regular simplex of a given rotation around the rows of ``Yt (P, r)``.

    Facet normals n_k = rot nu_k; m_k = max_g n_k . yt_g; h = mean_k m_k; centre c from n_k . c = m_k - h (consistent
    because the normals sum to zero).  Every facet touches a bin.  Returns (normals, h, c).  MATH §9.6.
    """
    r = Yt.shape[1]
    n = regular_simplex_normals(r) @ rot.T
    m = (Yt @ n.T).max(0)
    h = float(m.mean())
    c = np.linalg.lstsq(n, m - h, rcond=None)[0]
    return n, h, c


def _rotation_grid(r, n_random=2000, seed=0):
    if r == 1:
        return [np.zeros(1)]
    if r == 2:
        return [np.array([np.deg2rad(d)]) for d in range(120)]
    from scipy.spatial.transform import Rotation
    return list(Rotation.random(n_random, random_state=seed).as_rotvec())


def smallest_rotation(Yt):
    """Rotation angles whose tight regular simplex is smallest (least h): grid, then a local refinement."""
    from scipy.optimize import minimize, minimize_scalar
    r = Yt.shape[1]
    if r == 1:
        return np.zeros(1)

    def size(a):
        return tight_regular_simplex(Yt, rotation_matrix(r, a))[1]

    grid = _rotation_grid(r)
    hs = np.array([size(a) for a in grid])
    if r == 2:
        d = float(grid[int(hs.argmin())][0])
        res = minimize_scalar(lambda x: size([x]), bounds=(d - np.deg2rad(1), d + np.deg2rad(1)), method="bounded",
                              options=dict(xatol=1e-10))
        best = np.array([res.x % (2 * np.pi / 3)])
        return best if size(best) <= hs.min() else grid[int(hs.argmin())]
    best, best_h = grid[int(hs.argmin())], hs.min()
    for i in np.argsort(hs)[:5]:
        res = minimize(size, grid[i], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
        if res.fun < best_h:
            best, best_h = res.x, res.fun
    return np.asarray(best, float)


def regular_hull(model, Hbar, angles=None, tau=1.0):
    """Exact convex form of a free rank-r fit on r + 1 vertices forming a regular simplex (MATH §9.6).

    ``Theta(p_g) = sum_v lam[g, v] Theta_v`` holds exactly for the fitted model: no sum-to-one projection, no NNLS.
    The simplex is regular in the pooled data metric (every pair of vertices differs equally in prediction).  Its
    parameters are the rotation ``angles`` (None: the rotation with the smallest tight simplex) and the tightness
    ``tau >= 1`` (1: every facet touches a bin; larger values inflate it about its centre, which moves every weight
    toward 1 / (r + 1)).  Angles are measured in the bins' principal-axis frame (``hull_frame``).

    Returns a dict like ``polytope_vertices``: ``phi`` (r + 1, n_phi) vertices, ``lam`` (P, r + 1) weights, ``index``
    and ``dist`` (nearest bin of each vertex and its distance as a fraction of the bins' spread), ``outside`` (the
    largest miss of the rebuilt bins, ~0), ``affine_err`` (0), plus ``h`` (centre-to-facet distance), ``h_tight``,
    ``tau``, ``angles``, ``Yv`` (vertices in prediction coordinates) and ``Y`` (bins in them).
    """
    if tau < 1:
        raise ValueError("regular_hull: tau must be >= 1")
    phi = model.phi
    r = phi.shape[0]
    Y, ybar, U, Yt, L = hull_frame(model, Hbar)
    if angles is None:
        angles = smallest_rotation(Yt)
    angles = np.atleast_1d(np.asarray(angles, float))
    n, h, c = tight_regular_simplex(Yt, rotation_matrix(r, angles))
    At = c - tau * r * h * n                                   # vertex v opposite facet v
    lam = (tau * h - (Yt - c) @ n.T) / ((r + 1) * tau * h)
    Yv = At @ U.T + ybar
    spread = float(np.max(np.linalg.norm(Y - ybar, axis=1)))
    dist = np.linalg.norm(Y[:, None] - Yv[None], axis=2)
    return dict(phi=np.linalg.solve(L.T, Yv.T).T @ phi, lam=lam, index=dist.argmin(0), dist=dist.min(0) / spread,
                outside=float(np.max(np.linalg.norm(lam @ Yv - Y, axis=1)) / spread), affine_err=0.0,
                h=tau * h, h_tight=h, tau=float(tau), angles=angles, Yv=Yv, Y=Y)


def _enclosing_triangle(Y2):
    """Smallest triangle containing the 2-D points whose sides lie on convex-hull edges."""
    from itertools import combinations

    from scipy.spatial import ConvexHull

    eq = ConvexHull(Y2).equations           # n . x + b <= 0 inside
    best, best_area = None, np.inf
    for tr in combinations(range(len(eq)), 3):
        E = eq[list(tr)]
        try:
            V = np.stack([np.linalg.solve(E[[a, b], :2], -E[[a, b], 2])
                          for a, b in ((1, 2), (0, 2), (0, 1))])
        except np.linalg.LinAlgError:
            continue
        # bounded only if the outward normals positively span the plane
        M = np.vstack([E[:, :2].T, np.ones(3)])
        if np.linalg.cond(M) > 1e12 or np.any(np.linalg.solve(M, [0, 0, 1]) <= 0):
            continue
        u, v = V[1] - V[0], V[2] - V[0]
        area = abs(u[0] * v[1] - u[1] * v[0]) / 2
        if area < best_area:
            best, best_area = V, area
    return best


def fit_sym_tp(stats, R, Sigma, r, p_grid, n_iter=60, tol=1e-7, jitter=1e-10, starts=("pooled",), quads=None):
    """Joint GLS fit of a rank-``r`` TP-GDAR on per-condition ``stats`` (see stack_stats).

    ``starts`` lists the initialisations to run; the lowest final objective wins.
      pooled  the pooled operator plus the leading per-condition deviations in the pooled metric
      svd     the leading singular vectors of the per-condition fits, unweighted

    ``pooled`` can stall when a few conditions hold nearly all the energy (a condition with an
    unstable mode, say): the pooled metric is then set by those few directions.  ``tol`` is
    relative to the whole objective, so such a fit can also stop early; lower it for those data.

    ``quads`` is :func:`gls_quads` of the same stats and Sigma, for callers that fit several ranks.
    """
    P = len(stats["n"])
    N = stats["Syy"].shape[1]
    Hs, hs = gls_quads(stats, R, Sigma) if quads is None else quads
    n_phi = hs.shape[1]

    def objective(W, phi):
        return sum((W[g] @ phi) @ Hs[g] @ (W[g] @ phi) - 2 * (W[g] @ phi) @ hs[g] for g in range(P))

    def start_pooled():
        # rank r starts at the pooled model, with constant weight
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
        return W, phi

    def start_svd():
        fits = np.stack([_solve(Hs[g], hs[g], jitter) for g in range(P)])
        _, S, Vt = np.linalg.svd(fits, full_matrices=False)
        phi = S[:r, None] * Vt[:r]
        return np.stack([np.linalg.lstsq(phi @ Hs[g] @ phi.T, phi @ hs[g], rcond=1e-12)[0] for g in range(P)]), phi

    def alternate(W, phi):
        obj = [objective(W, phi)]
        for _ in range(n_iter):
            for g in range(P):                              # W-step
                Hw = phi @ Hs[g] @ phi.T
                W[g] = np.linalg.lstsq(Hw, phi @ hs[g], rcond=1e-12)[0]
            A = np.zeros((r * n_phi, r * n_phi))            # phi-step: A_jk = sum_g W[g, j] W[g, k] H_g
            for j in range(r):
                for k in range(j, r):
                    block = np.tensordot(W[:, j] * W[:, k], Hs, axes=1)
                    A[j * n_phi:(j + 1) * n_phi, k * n_phi:(k + 1) * n_phi] = block
                    A[k * n_phi:(k + 1) * n_phi, j * n_phi:(j + 1) * n_phi] = block
            phi = _solve(A, (W.T @ hs).ravel(), jitter).reshape(r, n_phi)
            # gauge: unit weight columns
            scale = np.linalg.norm(W, axis=0) / np.sqrt(P)
            W, phi = W / scale, phi * scale[:, None]
            obj.append(objective(W, phi))
            if abs(obj[-2] - obj[-1]) <= tol * abs(obj[-1]):
                break
        return W, phi, np.array(obj)

    makers = {"pooled": start_pooled, "svd": start_svd}
    runs = [alternate(*makers[s]()) for s in starts]
    W, phi, obj = min(runs, key=lambda run: run[2][-1])
    return SymTP(W=W, phi=phi, R=R, N=N, p_grid=np.asarray(p_grid, float), objective=obj)


# ------------------------------------------------------------------ vertex form (MATH §9.5)
@dataclass
class VertexTP(SymTP):
    """TP-GDAR in vertex form: ``W`` holds the convex weights lam (P, V), each row on the simplex; ``phi`` the vertices.

    ``objective`` is the penalised objective per sweep; ``data_loss`` and ``penalty`` its two parts at the end.
    """

    mu: float = 0.0
    data_loss: float = np.nan
    penalty: float = np.nan

    @property
    def n_params(self):
        """Free numbers: V vertex operators plus V - 1 weights per bin (no gauge left but the vertex order)."""
        P, V = self.W.shape
        return V * self.phi.shape[1] + P * (V - 1)


def simplex_qp(Q, b):
    """``argmin_{lam >= 0, sum lam = 1}  lam' Q lam - 2 b' lam`` for a small positive semidefinite ``Q (V, V)``.

    Exact by enumerating the faces of the simplex: the minimum of a convex quadratic over the simplex is the minimiser
    over the affine hull of some face, lying in that face.  For each support S the equality-constrained problem is
    solved through its KKT system; infeasible solutions are dropped and the best feasible one is returned.  2^V - 1
    small solves, so V is kept small.  MATH §9.5.
    """
    from itertools import combinations

    Q = np.asarray(Q, float)
    b = np.asarray(b, float)
    V = len(b)
    if V > 8:
        raise ValueError("simplex_qp enumerates 2^V faces; V > 8 is not supported")
    best, best_obj = None, np.inf
    for size in range(1, V + 1):
        for S in combinations(range(V), size):
            S = list(S)
            if size == 1:
                lam_S = np.ones(1)
            else:
                M = np.zeros((size + 1, size + 1))
                M[:size, :size] = Q[np.ix_(S, S)]
                M[:size, size] = 1.0
                M[size, :size] = 1.0
                lam_S = np.linalg.lstsq(M, np.r_[b[S], 1.0], rcond=None)[0][:size]
                if np.any(lam_S < -1e-12):
                    continue
                lam_S = np.clip(lam_S, 0.0, None)
                lam_S = lam_S / lam_S.sum()
            lam = np.zeros(V)
            lam[S] = lam_S
            obj = lam @ Q @ lam - 2 * b @ lam
            if obj < best_obj:
                best, best_obj = lam, obj
    return best


def fit_sym_tp_vertex(stats, R, Sigma, V, p_grid, mu=0.0, start=None, n_iter=200, tol=1e-7, jitter=1e-10, quads=None):
    """TP-GDAR fitted in vertex form, with the convex weights inside the fit.  MATH §9.5, approved by Aditya 2026-11-21.

        min  J = sum_g L_g( sum_v lam[g, v] phi_v ) + mu n sum_v || phi_v - phibar ||^2_Hbar,    lam[g] >= 0, sum_v lam[g, v] = 1

    L_g(phi) = phi' H_g phi - 2 phi' h_g is bin g's GLS loss with ``Sigma`` fixed (as in :func:`fit_sym_tp`),
    Hbar = sum_g H_g / n the pooled data metric per sample, n the number of targets, phibar the mean vertex.

    Alternates two exact block solves, so J never increases:
      lam-step  per bin, ``simplex_qp(Phi H_g Phi', Phi h_g)``
      phi-step  ( sum_g lam_g lam_g' kron H_g + mu n (I - 11'/V) kron Hbar ) vec Phi = sum_g lam_g kron h_g

    With mu = 0 any simplex containing the fitted family fits equally well, so the vertices stay where the start
    put them; mu > 0 pulls them together as far as the data allow.  ``start`` is (lam (P, V), phi (V, n_phi)); by
    default the convex form (:func:`polytope_vertices`) of the free rank-V fit.  V = 1 is the pooled GLS fit (GDAR).
    """
    P = len(stats["n"])
    N = stats["Syy"].shape[1]
    Hs, hs = gls_quads(stats, R, Sigma) if quads is None else quads
    n = float(np.sum(stats["n"]))
    Hsum = Hs.sum(0)                                   # = n Hbar
    C = np.eye(V) - 1.0 / V                            # centring: sum_v ||phi_v - phibar||^2_M = tr(C Phi M Phi')

    if start is not None:
        lam, phi = (np.array(a, dtype=float) for a in start)
    elif V == 1:
        lam, phi = np.ones((P, 1)), _solve(Hsum, hs.sum(0), jitter)[None]
    else:
        free = fit_sym_tp(stats, R, Sigma, V, p_grid, jitter=jitter, quads=(Hs, hs))
        poly = polytope_vertices(free, Hsum / n)
        lam, phi = poly["lam"], poly["phi"]

    def parts(lam, phi):
        data = 0.0
        for g in range(P):
            f = lam[g] @ phi
            data += f @ Hs[g] @ f - 2 * f @ hs[g]
        return float(data), float(mu * np.trace(C @ phi @ Hsum @ phi.T))

    obj = [sum(parts(lam, phi))]
    n_phi = phi.shape[1]
    for _ in range(n_iter):
        for g in range(P):                                         # lam-step, exact per bin
            Q, b = phi @ Hs[g] @ phi.T, phi @ hs[g]
            new = simplex_qp(Q, b)
            if new @ Q @ new - 2 * b @ new <= lam[g] @ Q @ lam[g] - 2 * b @ lam[g]:
                lam[g] = new
        A = np.zeros((V * n_phi, V * n_phi))                       # phi-step, one linear solve
        for j in range(V):
            for k in range(j, V):
                block = np.tensordot(lam[:, j] * lam[:, k], Hs, axes=1) + mu * C[j, k] * Hsum
                A[j * n_phi:(j + 1) * n_phi, k * n_phi:(k + 1) * n_phi] = block
                A[k * n_phi:(k + 1) * n_phi, j * n_phi:(j + 1) * n_phi] = block
        phi = _solve(A, (lam.T @ hs).ravel(), jitter).reshape(V, n_phi)
        obj.append(sum(parts(lam, phi)))
        if abs(obj[-2] - obj[-1]) <= tol * abs(obj[-1]):
            break
    data, pen = parts(lam, phi)
    return VertexTP(W=lam, phi=phi, R=R, N=N, p_grid=np.asarray(p_grid, float), objective=np.array(obj),
                    mu=float(mu), data_loss=data, penalty=pen)
