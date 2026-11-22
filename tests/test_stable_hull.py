"""Stable-rotation rule for the regular hull (MATH §9.6), approved by Aditya 2026-11-22.

The kept rotation is the first, in order of simplex size, whose vertices all have spectral radius < 1; every smaller
candidate has an unstable vertex.  If no candidate is stable, the one with the smallest largest-vertex radius is kept and
flagged.  The vertex model still rebuilds the free fit exactly.
"""

import numpy as np
import pytest

from gdar_tp.graph import grid_graph
from gdar_tp.sym_tp import (SymTP, hull_candidates, hull_frame, regular_hull, spectral_radius, stable_regular_hull,
                            symmetric_basis, theta_of)

G = grid_graph(2, 3)
A = np.asarray(G.adjacency, bool)
N = G.n
K = 2
P = 12


def _model(r, scale, seed):
    rng = np.random.default_rng(seed)
    R = symmetric_basis(A, K)
    n_phi = R.shape[1]
    base = rng.standard_normal(n_phi) * 0.15
    phi = np.vstack([base] + [rng.standard_normal(n_phi) * 0.15 for _ in range(r - 1)])
    t = np.linspace(0, 2 * np.pi, P, endpoint=False)
    W = np.column_stack([np.ones(P)] + [scale * np.cos(t + k) for k in range(r - 1)])
    B = rng.standard_normal((n_phi, n_phi))
    Hbar = B @ B.T / n_phi + np.eye(n_phi)
    return SymTP(W=W, phi=phi, R=R, N=N, p_grid=np.arange(float(P))), Hbar, R


def _rhos(m, Hbar, R, a):
    hull = regular_hull(m, Hbar, angles=a)
    return np.array([spectral_radius(theta_of(ph, R, N), K) for ph in hull["phi"]])


def test_spectral_radius_matches_a_direct_companion():
    rng = np.random.default_rng(0)
    th = rng.standard_normal((N, K * N)) * 0.2
    Ac = np.zeros((K * N, K * N))
    Ac[:N] = th
    Ac[N:, :N] = np.eye(N)
    assert spectral_radius(th, K) == pytest.approx(np.abs(np.linalg.eigvals(Ac)).max())


@pytest.mark.parametrize("r", [2, 3])
@pytest.mark.parametrize("scale", [0.5, 2.0, 6.0, 20.0])
def test_kept_rotation_is_the_first_stable_one_or_the_least_unstable(r, scale):
    m, Hbar, R = _model(r, scale, seed=r)
    kw = dict(step_deg=2.0, n_random=60, seed=0)
    hull = stable_regular_hull(m, Hbar, R, N, K, **kw)
    _, _, _, Yt, _ = hull_frame(m, Hbar)
    cands, hs = hull_candidates(Yt, **kw)
    worst = np.array([_rhos(m, Hbar, R, a).max() for a in cands])
    if hull["stable"]:
        i = hull["n_checked"] - 1
        assert worst[i] < 1 and np.all(worst[:i] >= 1)
        assert hull["h"] == pytest.approx(hs[i])
        assert hull["rho"].max() < 1
    else:
        assert np.all(worst >= 1)
        assert hull["rho"].max() == pytest.approx(worst.min())
    free = m.W @ m.phi
    assert np.abs(hull["lam"] @ hull["phi"] - free).max() <= 1e-10 * np.abs(free).max()
    assert hull["lam"].min() >= -1e-12 and hull["h"] >= hull["h_smallest"] - 1e-12


def test_both_branches_are_exercised():
    seen = set()
    for scale in (0.5, 2.0, 6.0, 20.0):
        m, Hbar, R = _model(2, scale, seed=2)
        seen.add(stable_regular_hull(m, Hbar, R, N, K, step_deg=2.0)["stable"])
    assert seen == {True, False}
