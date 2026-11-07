"""Unit tests for gdar_tp/qlpv.py (grid, 2-D grid and windowed qLPV fits)."""

import numpy as np

from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import QLPVModel, _pad, _support_of, fit_grid, fit_grid_2d, fit_windowed
from gdar_tp.sim.linear import simulate_gdar

G = grid_graph(2, 2)
L, N = G.laplacian, G.n


def _A(a, b=0.08):
    return a * np.eye(N) - b * L


def _segment(a, T=3000, seed=0, B=None, U=None):
    X, _ = simulate_gdar([_A(a)], T, B=B, U=U, noise_std=1.0, rng=seed, burn_in=100)
    return X


def test_fit_grid_sorts_by_p_and_recovers_each_operator():
    p_values = [0.8, 0.2, 0.5]
    segments = [(_segment(0.3 + 0.4 * p, seed=i), None) for i, p in enumerate(p_values)]
    q = fit_grid(segments, p_values, K=1, laplacian=L, param="diffusion")
    assert isinstance(q, QLPVModel)
    assert np.allclose(q.p_grid[0], [0.2, 0.5, 0.8]) and q.param_shape == (3,)
    assert q.coeffs.shape == (3, N, N, 1) and q.B.shape == (3, N, 0)
    for g, p in enumerate(q.p_grid[0]):
        assert np.abs(q.coeffs[g, :, :, 0] - _A(0.3 + 0.4 * p)).max() < 0.03
    coeffs, B = q.coeff_tensor()
    assert coeffs is q.coeffs and B is q.B
    assert np.array_equal(q.support, support_mask(G.adjacency))


def test_support_of_mirrors_fit_gdar_defaults():
    S = np.eye(N, dtype=bool)
    assert _support_of(S, None, N) is S
    assert np.array_equal(_support_of(None, L, N), support_mask(G.adjacency))
    assert _support_of(None, None, N).all()


def test_pad_accepts_bare_arrays_and_tuples():
    X, U = np.zeros((5, 2)), np.ones((5, 1))
    assert _pad(X)[0] is X and _pad(X)[1] is None
    assert _pad((X, U)) == (X, U)


def test_fit_grid_2d_shape_and_values():
    p1, p2 = [0.0, 1.0], [0.0, 0.5, 1.0]
    grid = [[_segment(0.2 + 0.3 * a + 0.2 * b, seed=3 * i + j) for j, b in enumerate(p2)]
            for i, a in enumerate(p1)]
    q = fit_grid_2d(grid, p1, p2, K=1, laplacian=L, param="diffusion")
    assert q.coeffs.shape == (2, 3, N, N, 1) and q.param_shape == (2, 3)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            assert np.abs(q.coeffs[i, j, :, :, 0] - _A(0.2 + 0.3 * a + 0.2 * b)).max() < 0.03


def test_fit_windowed_recovers_two_regimes():
    X = np.vstack([_segment(0.3, T=2000, seed=10), _segment(0.7, T=2000, seed=11)])
    p = np.r_[np.zeros(2000), np.ones(2000)]
    q = fit_windowed(X, p, n_bins=2, win=200, hop=200, K=1, laplacian=L, param="diffusion")
    assert np.allclose(q.p_grid[0], [0.25, 0.75])
    assert np.abs(q.coeffs[0, :, :, 0] - _A(0.3)).max() < 0.05
    assert np.abs(q.coeffs[1, :, :, 0] - _A(0.7)).max() < 0.05


def test_fit_windowed_keeps_only_occupied_bins():
    X = np.vstack([_segment(0.3, T=1000, seed=12), _segment(0.7, T=1000, seed=13)])
    p = np.r_[np.zeros(1000), np.ones(1000)]
    q = fit_windowed(X, p, n_bins=4, win=200, hop=200, K=1, laplacian=L, param="diffusion")
    assert np.allclose(q.p_grid[0], [0.125, 0.875]) and q.coeffs.shape[0] == 2


def test_fit_windowed_with_input_recovers_B():
    rng = np.random.default_rng(14)
    B_true = rng.standard_normal((N, 1))
    U = rng.standard_normal((3100, 1))
    X, U_out = simulate_gdar([_A(0.5)], 3000, B=B_true, U=U, noise_std=0.5, rng=rng, burn_in=100)
    q = fit_windowed(X, np.zeros(3000), n_bins=1, win=300, hop=300, U=U_out, K=1,
                     laplacian=L, param="diffusion")
    assert q.B.shape == (1, N, 1)
    assert np.abs(q.B[0] - B_true).max() < 0.05
