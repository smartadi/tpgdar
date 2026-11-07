"""Unit tests for gdar_tp/sym_tp.py beyond tests/test_sym_tp.py."""

import numpy as np
import pytest

from gdar_tp.graph import grid_graph
from gdar_tp.sym_tp import (SymTP, _enclosing_triangle, _quad, _solve, fit_gls_pooled, fit_sym_tp, gls_quads,
                            polytope_vertices, symmetric_basis, theta_of)
from gdar_tp.tpgdar import resid_cov_sum, stack_stats, trial_stats

G = grid_graph(2, 2)
A = np.asarray(G.adjacency, bool)
N = G.n


def _stats(seed=0, P=3, K=1):
    rng = np.random.default_rng(seed)
    per = [[trial_stats(rng.standard_normal((80, N)), K=K) for _ in range(2)] for _ in range(P)]
    return stack_stats(per)


def _spd(seed=1):
    M = np.random.default_rng(seed).standard_normal((N, N))
    return M @ M.T + N * np.eye(N)


def test_theta_of_round_trips_through_the_basis():
    R = symmetric_basis(A, 2, node_inputs=True)
    phi = np.random.default_rng(2).standard_normal(R.shape[1])
    th = theta_of(phi, R, N)
    assert th.shape == (N, 2 * N + N)
    RtR = (R.T @ R).toarray()
    assert np.allclose(RtR, np.diag(np.diag(RtR)))         # columns of R are disjoint
    back = np.linalg.solve(RtR, R.T @ th.ravel(order="F"))
    assert np.allclose(back, phi)


def test_quad_matches_the_explicit_kronecker_form_and_the_loss():
    R = symmetric_basis(A, 1)
    st = _stats()
    Gzz, Gzy, Syy = st["Gzz"][0], st["Gzy"][0], st["Syy"][0]
    Sinv = np.linalg.inv(_spd())
    H, h = _quad(Gzz, Gzy, R, Sinv)
    Rd = R.toarray()
    assert np.allclose(H, Rd.T @ np.kron(Gzz, Sinv) @ Rd)
    assert np.allclose(h, Rd.T @ (Sinv @ Gzy.T).ravel(order="F"))
    phi = np.random.default_rng(3).standard_normal(R.shape[1]) * 0.1
    loss = np.trace(Sinv @ resid_cov_sum(theta_of(phi, R, N), Gzz, Gzy, Syy))
    assert np.isclose(phi @ H @ phi - 2 * phi @ h + np.trace(Sinv @ Syy), loss)


def test_solve_adds_a_jitter_relative_to_the_mean_diagonal():
    H = np.diag([1.0, 3.0])
    assert np.allclose(_solve(H, np.array([1.0, 1.0]), 0.5), [1 / 2.0, 1 / 4.0])


def test_gls_quads_stacks_the_per_condition_quads():
    R = symmetric_basis(A, 1)
    st, Sigma = _stats(), _spd()
    Hs, hs = gls_quads(st, R, Sigma)
    H1, h1 = _quad(st["Gzz"][1], st["Gzy"][1], R, np.linalg.pinv(Sigma))
    assert Hs.shape[0] == 3 and np.allclose(Hs[1], H1) and np.allclose(hs[1], h1)


def test_fit_gls_pooled_returns_the_ols_sigma_and_the_gls_optimum():
    R = symmetric_basis(A, 1)
    st = _stats()
    tot = {k: st[k].sum(0) for k in st}
    phi, Sigma = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    H0, h0 = _quad(tot["Gzz"], tot["Gzy"], R, np.eye(N))
    phi_ols = np.linalg.solve(H0, h0)
    assert np.allclose(Sigma, resid_cov_sum(theta_of(phi_ols, R, N), tot["Gzz"], tot["Gzy"], tot["Syy"]) / tot["n"])
    H, h = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sigma))
    assert np.allclose(H @ phi, h, atol=1e-8 * np.abs(h).max())


def _model(P=5, r=2, seed=4):
    R = symmetric_basis(A, 1)
    rng = np.random.default_rng(seed)
    return SymTP(W=rng.standard_normal((P, r)), phi=rng.standard_normal((r, R.shape[1])), R=R, N=N,
                 p_grid=np.linspace(0, 1, P))


def test_symtp_weights_grid_interpolation_and_parameter_count():
    m = _model()
    grid = m.theta_grid()
    assert grid.shape == (5, N, N)
    assert np.allclose(grid[3], theta_of(m.W[3] @ m.phi, m.R, N))
    assert np.allclose(m.theta_at(m.p_grid[2]), grid[2])
    assert np.allclose(m.theta_at(0.5 * (m.p_grid[1] + m.p_grid[2])), 0.5 * (grid[1] + grid[2]))
    assert m.n_params == 2 * m.phi.shape[1] + 5 * 2 - 4


def test_enclosing_triangle_of_a_triangle_is_that_triangle():
    tri = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    pts = np.vstack([tri, np.random.default_rng(5).dirichlet(np.ones(3), 10) @ tri])
    got = _enclosing_triangle(pts)
    assert sorted(map(tuple, np.round(got, 9))) == sorted(map(tuple, tri))


def test_enclosing_triangle_contains_every_point():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.2, 1.0], [0.0, 0.9], [0.5, 0.5]])
    tri = _enclosing_triangle(pts)
    M = np.vstack([tri.T, np.ones(3)])
    lam = np.linalg.solve(M, np.vstack([pts.T, np.ones(len(pts))]))
    assert np.all(lam >= -1e-9)


def test_polytope_vertices_rank_2_are_the_two_ends():
    P = 6
    m = _model(P=P, r=2)
    m.W = np.column_stack([np.ones(P), np.linspace(0, 1, P)])
    out = polytope_vertices(m, np.eye(m.phi.shape[1]))
    assert sorted(out["index"].tolist()) == [0, P - 1]
    assert np.all(out["lam"] >= 0) and np.allclose(out["lam"].sum(1), 1.0)
    assert out["outside"] < 1e-6 and out["affine_err"] < 1e-12
    assert np.allclose(out["lam"] @ out["phi"], m.W @ m.phi, atol=1e-6)


def test_polytope_vertices_rank_3_contains_the_trajectory():
    P = 8
    m = _model(P=P, r=3, seed=6)
    rng = np.random.default_rng(7)
    m.W = np.column_stack([np.ones(P), rng.standard_normal((P, 2))])
    out = polytope_vertices(m, np.eye(m.phi.shape[1]))
    assert out["phi"].shape == (3, m.phi.shape[1]) and out["index"].shape == (3,)
    assert np.all(out["lam"] >= 0) and out["outside"] < 1e-6
    assert np.allclose(out["lam"] @ out["phi"], m.W @ m.phi, atol=1e-6)


def test_fit_sym_tp_keeps_the_best_of_several_starts():
    R = symmetric_basis(A, 1)
    st, Sigma = _stats(P=4), _spd()
    both = fit_sym_tp(st, R, Sigma, 2, np.arange(4.0), starts=("pooled", "svd"))
    each = [fit_sym_tp(st, R, Sigma, 2, np.arange(4.0), starts=(s,)) for s in ("pooled", "svd")]
    assert both.objective[-1] == pytest.approx(min(e.objective[-1] for e in each))
