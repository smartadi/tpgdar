"""Regression tests for the 2026-10-26 code review (hub #65)."""

import warnings

import numpy as np
import pytest

from gdar_tp import metrics
from gdar_tp.gdar import GDARModel, fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.latent import constrained_m_step, fit_em, population_masks, project_A
from gdar_tp.qlpv import fit_grid, fit_grid_2d
from gdar_tp.sim.linear import simulate_gdar, stable_diffusion_coeffs
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams


def test_multi_step_nrmse_input_timing():
    # noiseless driven system scored with its own true model must be exact
    g = grid_graph(2, 2)
    A = stable_diffusion_coeffs(g.laplacian, a=[0.8], b=[0.05])
    B = np.eye(g.n)[:, :1]
    rng = np.random.default_rng(0)
    U = rng.standard_normal((600, 1))
    X, U = simulate_gdar(A, n_steps=600, B=B, U=U, x0=rng.standard_normal((1, g.n)))
    model = GDARModel(A=A, B=B, K=1, param="diffusion", support=support_mask(g.adjacency))
    assert metrics.one_step_nrmse(model, X, U) < 1e-10
    assert metrics.multi_step_nrmse(model, X, U, horizon=5) < 1e-10


def test_wc_input_drives_next_sample():
    g = grid_graph(2, 2)
    net = WCNetwork(g.adjacency, WCParams(g0=3.0))
    n, t0 = 50, 20
    U = np.zeros((n, 1)); U[t0] = 5.0
    B = np.ones((g.n, 1))
    x0 = np.full((2, g.n), 0.1)
    E0 = net.simulate(n, B=B, U=np.zeros_like(U), burn_in=0, x0=x0)
    E1 = net.simulate(n, B=B, U=U, burn_in=0, x0=x0)
    assert np.allclose(E0[:t0 + 1], E1[:t0 + 1])     # u(t0) does not touch x(t0)
    assert not np.allclose(E0[t0 + 1], E1[t0 + 1])   # ... it drives x(t0+1)


def test_fit_grid_without_support_or_laplacian():
    rng = np.random.default_rng(0)
    segs = [(rng.standard_normal((200, 3)), None) for _ in range(3)]
    q = fit_grid(segs, [0.0, 0.5, 1.0], K=1)
    assert q.support.shape == (3, 3) and q.support.all()
    q2 = fit_grid_2d([[segs[0][0], segs[1][0]], [segs[2][0], segs[0][0]]], [0, 1], [0, 1])
    assert q2.coeffs.shape == (2, 2, 3, 3, 1)
    assert np.allclose(q2.coeffs[0, 0], q2.coeffs[1, 1])   # same segment, same fit


def test_predict_warns_when_input_dropped():
    m = GDARModel(A=[0.5 * np.eye(2)], B=np.ones((2, 1)), K=1, param="edgewise",
                  support=np.ones((2, 2), bool))
    with pytest.warns(UserWarning, match="input term is dropped"):
        m.predict_one_step(np.zeros((5, 2)))
    m0 = GDARModel(A=[0.5 * np.eye(2)], B=np.zeros((2, 0)), K=1, param="edgewise",
                   support=np.ones((2, 2), bool))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m0.predict_one_step(np.zeros((5, 2)))


def test_constrained_m_step_is_optimal():
    g = grid_graph(2, 2)
    struct, sign, _ = population_masks(g.adjacency)
    d = struct.shape[0]
    rng = np.random.default_rng(4)
    Z = rng.standard_normal((500, d)) @ rng.standard_normal((d, d))   # correlated
    S00 = Z.T @ Z
    S10 = rng.standard_normal((d, d)) @ S00
    A_opt = constrained_m_step(S00, S10, struct, sign)
    assert np.all(A_opt[~struct] == 0)
    assert np.all(A_opt[sign > 0] >= -1e-12) and np.all(A_opt[sign < 0] <= 1e-12)

    def obj(A):
        return np.trace(A @ S00 @ A.T) - 2 * np.trace(A @ S10.T)

    A_proj = project_A(S10 @ np.linalg.inv(S00), struct, sign)
    assert obj(A_opt) <= obj(A_proj) + 1e-8
    # no constraints active -> reduces to the unconstrained solution
    full = np.ones_like(struct); free = np.zeros_like(sign)
    assert np.allclose(constrained_m_step(S00, S10, full, free),
                       S10 @ np.linalg.inv(S00), atol=1e-6)


def test_em_loglik_strictly_monotone():
    from tests.test_latent import _synth
    g, A, struct, sign, C, Y = _synth()
    _, info = fit_em(Y, struct, sign, C, n_iter=20, rng=1)
    assert np.all(np.diff(info["loglik"]) > -1e-6)
