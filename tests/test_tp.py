import numpy as np

from gdar_tp.tp import hosvd, convex_hull
from gdar_tp import metrics


def _low_rank_tensor(n_modes=2, P=11, N=4, K=1, seed=0):
    """Build coeffs[p] = sum_j g_j(p) V_j with convex weights g (rank n_modes)."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n_modes, N, N, K))
    p = np.linspace(0, 1, P)
    # convex weight functions: a partition of unity of polynomials in p
    raw = np.stack([p ** j for j in range(n_modes)], axis=1)  # (P, n_modes)
    W = raw / raw.sum(axis=1, keepdims=True)
    T = np.einsum("pj,jabk->pabk", W, V)
    return p, T, V, W


def test_hosvd_rank_matches_mode_count():
    _, T, V, W = _low_rank_tensor(n_modes=3, P=15, N=5, K=2)
    tp = hosvd(T, n_param_modes=1, energy=0.99999)
    assert tp.ranks[0] == 3


def test_hosvd_reconstruction_exact():
    _, T, V, W = _low_rank_tensor(n_modes=2, P=11, N=4, K=1)
    tp = hosvd(T, n_param_modes=1, energy=0.99999)
    recon = tp.reconstruct()
    assert np.abs(recon - T).max() < 1e-9


def test_convex_hull_weights_valid():
    _, T, V, W = _low_rank_tensor(n_modes=2, P=11, N=4, K=1)
    poly = convex_hull(T, n_param_modes=1, method="tight")
    # weights non-negative and sum to one
    assert poly.weights.min() > -1e-8
    assert np.allclose(poly.weights.sum(axis=1), 1.0, atol=1e-6)
    # rank-2 (1-D affine) trajectory -> exactly two hull vertices
    assert poly.n_vertices == 2
    # reconstruction of A(p) from the polytope is essentially exact
    assert poly.residual < 1e-6


def test_convex_hull_recovers_endpoints():
    p, T, V, W = _low_rank_tensor(n_modes=2, P=11, N=4, K=1)
    poly = convex_hull(T, n_param_modes=1, method="tight")
    # the two vertex systems should match T at p=0 and p=1 (the extremes)
    verts = poly.vertex_systems.reshape(2, -1)
    endpoints = np.stack([T[0].ravel(), T[-1].ravel()])
    # match up to ordering
    d00 = np.abs(verts[0] - endpoints[0]).max()
    d01 = np.abs(verts[0] - endpoints[1]).max()
    assert min(d00, d01) < 1e-6


def test_energy_spectrum_and_rank():
    s = np.array([10.0, 1.0, 0.01, 1e-6])
    cum = metrics.energy_spectrum(s)
    assert cum[-1] == 1.0
    assert metrics.rank_at_energy(s, thresh=0.99) <= 2


def test_box_hull_valid_2d():
    # a genuinely 2-D parameter dependence
    rng = np.random.default_rng(3)
    P1, P2, N = 7, 6, 3
    V = rng.standard_normal((3, N, N, 1))
    p1 = np.linspace(0, 1, P1); p2 = np.linspace(0, 1, P2)
    T = np.zeros((P1, P2, N, N, 1))
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            w = np.array([(1 - a) * (1 - b), a, b])
            w = w / w.sum()
            T[i, j] = np.einsum("j,jabk->abk", w, V)
    poly = convex_hull(T, n_param_modes=2, method="box")
    assert np.allclose(poly.weights.sum(axis=1), 1.0, atol=1e-6)
    assert poly.weights.min() > -1e-6
