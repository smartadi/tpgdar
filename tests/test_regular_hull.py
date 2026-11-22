"""Regular-simplex hull of a free rank-r fit (MATH §9.6): the checks Aditya approved on 2026-11-22.

The vertex model rebuilds the free fit in every bin; weights are >= 0 and sum to one; at tau = 1 every vertex has a
bin with zero weight; at rank 1 the vertices are the end bins; the same vertices after a change of gauge.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from gdar_tp.sym_tp import (SymTP, regular_hull, regular_simplex_normals, rotation_matrix, smallest_rotation,
                            hull_frame, tight_regular_simplex)

P, NPHI = 16, 12


def _model(r, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((P, r)) + np.array([3.0] + [0.0] * (r - 1))
    phi = rng.standard_normal((r, NPHI))
    B = rng.standard_normal((NPHI, NPHI))
    Hbar = B @ B.T + NPHI * np.eye(NPHI)
    return SymTP(W=W, phi=phi, R=sp.eye(NPHI, format="csr"), N=1, p_grid=np.arange(float(P))), Hbar


def _hdist(a, b, Hbar):
    d = a - b
    return np.sqrt(d @ Hbar @ d)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_normals_are_a_regular_simplex(r):
    n = regular_simplex_normals(r)
    assert n.shape == (r + 1, r)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0)
    np.testing.assert_allclose(n.sum(0), 0.0, atol=1e-15)
    if r > 1:
        G = n @ n.T
        off = G[~np.eye(r + 1, dtype=bool)]
        np.testing.assert_allclose(off, -1.0 / r)


@pytest.mark.parametrize("r", [1, 2, 3])
@pytest.mark.parametrize("tau", [1.0, 1.3, 2.0])
def test_exact_convex_form_of_the_free_fit(r, tau):
    m, Hbar = _model(r, seed=r)
    hull = regular_hull(m, Hbar, tau=tau)
    lam, phi_v = hull["lam"], hull["phi"]
    assert lam.shape == (P, r + 1) and phi_v.shape == (r + 1, NPHI)
    assert lam.min() >= -1e-12
    np.testing.assert_allclose(lam.sum(1), 1.0, atol=1e-12)
    free = m.W @ m.phi
    assert np.abs(lam @ phi_v - free).max() <= 1e-10 * np.abs(free).max()
    assert hull["outside"] < 1e-10


@pytest.mark.parametrize("r", [2, 3])
def test_vertices_are_equidistant_in_the_data_metric(r):
    m, Hbar = _model(r, seed=10 + r)
    phi_v = regular_hull(m, Hbar)["phi"]
    d = [_hdist(phi_v[i], phi_v[j], Hbar) for i in range(r + 1) for j in range(i + 1, r + 1)]
    np.testing.assert_allclose(d, d[0], rtol=1e-8)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_tight_hull_touches_a_bin_on_every_facet_and_tau_shrinks_toward_uniform(r):
    m, Hbar = _model(r, seed=20 + r)
    tight = regular_hull(m, Hbar, tau=1.0)
    assert np.all(np.abs(tight["lam"].min(0)) < 1e-12)
    loose = regular_hull(m, Hbar, angles=tight["angles"], tau=1.5)
    np.testing.assert_allclose(loose["lam"], tight["lam"] / 1.5 + (1 - 1 / 1.5) / (r + 1), atol=1e-12)
    assert loose["lam"].min() >= (1 - 1 / 1.5) / (r + 1) - 1e-12


def test_rank_one_vertices_are_the_end_bins():
    m, Hbar = _model(1, seed=3)
    hull = regular_hull(m, Hbar)
    free = m.W @ m.phi
    ends = {int(np.argmin(m.W[:, 0])), int(np.argmax(m.W[:, 0]))}
    assert set(hull["index"].tolist()) == ends
    np.testing.assert_allclose(hull["dist"], 0.0, atol=1e-12)
    for v in range(2):
        assert np.abs(hull["phi"][v] - free[hull["index"][v]]).max() < 1e-10 * np.abs(free).max()


@pytest.mark.parametrize("r", [2, 3])
def test_smallest_rotation_beats_the_grid(r):
    m, Hbar = _model(r, seed=30 + r)
    _, _, _, Yt, _ = hull_frame(m, Hbar)
    best = tight_regular_simplex(Yt, rotation_matrix(r, smallest_rotation(Yt)))[1]
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.uniform(0, 2 * np.pi, 1) if r == 2 else rng.standard_normal(3)
        assert best <= tight_regular_simplex(Yt, rotation_matrix(r, a))[1] + 1e-12


def test_rank_two_rotation_has_period_120_degrees():
    m, Hbar = _model(2, seed=41)
    a = regular_hull(m, Hbar, angles=[0.3])
    b = regular_hull(m, Hbar, angles=[0.3 + 2 * np.pi / 3])
    order = [int(np.argmin(np.linalg.norm(b["Yv"] - y, axis=1))) for y in a["Yv"]]
    np.testing.assert_allclose(b["Yv"][order], a["Yv"], atol=1e-10)


@pytest.mark.parametrize("r", [2, 3])
def test_same_vertices_after_a_change_of_gauge(r):
    m, Hbar = _model(r, seed=50 + r)
    M = np.random.default_rng(1).standard_normal((r, r)) + 2 * np.eye(r)
    m2 = SymTP(W=m.W @ M, phi=np.linalg.solve(M, m.phi), R=m.R, N=m.N, p_grid=m.p_grid)
    a, b = regular_hull(m, Hbar), regular_hull(m2, Hbar)
    np.testing.assert_allclose(a["h"], b["h"], rtol=1e-6)
    scale = np.abs(a["phi"]).max()
    for v in range(r + 1):
        assert min(np.abs(b["phi"] - a["phi"][v]).max(axis=1)) < 1e-4 * scale
