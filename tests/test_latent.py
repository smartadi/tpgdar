import numpy as np
import pytest

from gdar_tp.graph import grid_graph
from gdar_tp.latent import (population_masks, project_A, fit_em, one_step_pred_nrmse)


def _synth(seed=0, T=2500):
    g = grid_graph(2, 2); d = 2 * g.n
    struct, sign, C = population_masks(g.adjacency)
    rng = np.random.default_rng(seed)
    A = 0.6 * np.eye(d) + 0.1 * rng.standard_normal((d, d))
    A = project_A(A, struct, sign)
    ev = np.max(np.abs(np.linalg.eigvals(A)))
    if ev >= 0.95:
        A *= 0.9 / ev
    Q = 0.02 * np.eye(d); R = 0.005 * np.eye(g.n)
    x = np.zeros(d); X = np.zeros((T, d))
    for t in range(T):
        x = A @ x + rng.multivariate_normal(np.zeros(d), Q); X[t] = x
    Y = X @ C.T + rng.multivariate_normal(np.zeros(g.n), R, size=T)
    Y = Y - Y.mean(0, keepdims=True)
    return g, A, struct, sign, C, Y


def test_masks_dale_and_structure():
    g = grid_graph(3, 3)
    struct, sign, C = population_masks(g.adjacency)
    d = 2 * g.n
    assert struct.shape == (d, d)
    # observation selects each node's E-slot
    assert np.allclose(C.sum(axis=1), 1.0)
    for i in range(g.n):
        assert C[i, 2 * i] == 1.0
    # between-node entries that exist are sign-constrained (Dale)
    assert np.all(sign[struct & (sign != 0)] != 0)


def test_project_A_respects_constraints():
    g = grid_graph(2, 2)
    struct, sign, C = population_masks(g.adjacency)
    rng = np.random.default_rng(1)
    A = project_A(rng.standard_normal(struct.shape), struct, sign)
    assert np.all(A[~struct] == 0.0)
    assert np.all(A[sign > 0] >= 0.0)
    assert np.all(A[sign < 0] <= 0.0)


def test_em_loglik_monotone_and_recovers():
    g, A, struct, sign, C, Y = _synth()
    lds, info = fit_em(Y, struct, sign, C, n_iter=20, rng=1)
    ll = info["loglik"]
    # log-likelihood should essentially never decrease
    assert np.all(np.diff(ll) > -1.0)
    # recovers the constrained dynamics on the support
    r = np.corrcoef(A[struct], lds.A[struct])[0, 1]
    assert r > 0.9
    # Dale respected in the fitted model
    assert np.all(lds.A[sign > 0] >= -1e-9)
    assert np.all(lds.A[sign < 0] <= 1e-9)


def test_one_step_pred_runs():
    g, A, struct, sign, C, Y = _synth(T=1500)
    lds, _ = fit_em(Y, struct, sign, C, n_iter=10, rng=2)
    err = one_step_pred_nrmse(lds, Y)
    assert 0.0 <= err < 2.0
