"""Unit tests for every function and method in gdar_tp/gdar.py."""

import warnings

import numpy as np
import pytest

from gdar_tp.gdar import (GDARModel, _build_design, _fit_diffusion, _fit_edgewise, coupling_matrix,
                          edge_flow, fit_gdar)
from gdar_tp.graph import grid_graph, support_mask


def _model(K=2, N=3, M=1, seed=0):
    rng = np.random.default_rng(seed)
    A = [0.3 * rng.standard_normal((N, N)) for _ in range(K)]
    B = rng.standard_normal((N, M))
    return GDARModel(A=A, B=B, K=K, param="edgewise", support=np.ones((N, N), bool))


def test_model_sizes():
    m = _model(K=2, N=4, M=3)
    assert m.n == 4 and m.m == 3


def test_build_design_alignment():
    rng = np.random.default_rng(1)
    T, N, K, M = 12, 3, 3, 2
    X, U = rng.standard_normal((T, N)), rng.standard_normal((T, M))
    Zlags, Ublk, Y = _build_design(X, U, K)
    assert len(Zlags) == K and Y.shape == (T - K, N) and Ublk.shape == (T - K, M)
    for idx, t in enumerate(range(K - 1, T - 1)):
        assert np.array_equal(Y[idx], X[t + 1])
        assert np.array_equal(Ublk[idx], U[t])            # u(t) drives x(t+1)
        for k in range(K):
            assert np.array_equal(Zlags[k][idx], X[t - k])


def test_build_design_without_input():
    _, Ublk, Y = _build_design(np.zeros((10, 2)), None, 1)
    assert Ublk.shape == (9, 0) and Y.shape == (9, 2)


def test_fit_edgewise_zero_ridge_is_masked_least_squares():
    rng = np.random.default_rng(2)
    T, N, K = 200, 4, 2
    X, U = rng.standard_normal((T, N)), rng.standard_normal((T, 1))
    support = rng.random((N, N)) < 0.5
    np.fill_diagonal(support, True)
    Zlags, Ublk, Y = _build_design(X, U, K)
    A, B = _fit_edgewise(Zlags, Ublk, Y, support, ridge=0.0)
    Zfull = np.hstack(Zlags + [Ublk])
    for i in range(N):
        js = np.where(support[i])[0]
        cols = [k * N + j for k in range(K) for j in js] + [K * N]
        beta = np.linalg.lstsq(Zfull[:, cols], Y[:, i], rcond=None)[0]
        got = [A[k][i, j] for k in range(K) for j in js] + [B[i, 0]]
        assert np.allclose(got, beta)
        for k in range(K):
            assert np.all(A[k][i, ~support[i]] == 0)


def test_fit_diffusion_is_exact_on_noise_free_regressors():
    g = grid_graph(2, 3)
    L, N, K, M = g.laplacian, g.n, 2, 2
    a, b = np.array([0.5, 0.2]), np.array([0.1, -0.05])
    A_true = [a[k] * np.eye(N) - b[k] * L for k in range(K)]
    rng = np.random.default_rng(3)
    Zlags = [rng.standard_normal((50, N)) for _ in range(K)]
    Ublk, B_true = rng.standard_normal((50, M)), rng.standard_normal((N, M))
    Y = sum(Z @ Ak.T for Z, Ak in zip(Zlags, A_true)) + Ublk @ B_true.T
    A, B, a_hat, b_hat = _fit_diffusion(Zlags, Ublk, Y, L, ridge=0.0)
    assert np.allclose(a_hat, a) and np.allclose(b_hat, b) and np.allclose(B, B_true)
    assert all(np.allclose(x, y) for x, y in zip(A, A_true))


def test_fit_gdar_rejects_bad_arguments():
    X = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError):
        fit_gdar(X, param="diffusion")                    # no Laplacian
    with pytest.raises(ValueError):
        fit_gdar(X, param="nope")


def test_fit_gdar_default_support_and_diffusion_support():
    rng = np.random.default_rng(4)
    g = grid_graph(2, 2)
    X = rng.standard_normal((100, g.n))
    assert fit_gdar(X, K=1).support.all()
    m = fit_gdar(X, K=1, laplacian=g.laplacian, param="diffusion")
    assert np.array_equal(m.support, support_mask(g.adjacency))
    assert set(m.diffusion) == {"a", "b"} and m.diffusion["a"].shape == (1,)


def test_predict_one_step_matches_recursion():
    rng = np.random.default_rng(5)
    m = _model(K=2, N=3, M=1)
    X, U = rng.standard_normal((15, 3)), rng.standard_normal((15, 1))
    pred, true = m.predict_one_step(X, U)
    assert pred.shape == true.shape == (13, 3)
    for idx, t in enumerate(range(1, 14)):
        want = m.A[0] @ X[t] + m.A[1] @ X[t - 1] + m.B @ U[t]
        assert np.allclose(pred[idx], want) and np.array_equal(true[idx], X[t + 1])


def test_rollout_matches_recursion_with_input():
    rng = np.random.default_rng(6)
    m = _model(K=2, N=3, M=1)
    hist, U = rng.standard_normal((2, 3)), rng.standard_normal((5, 1))
    out = m.rollout(hist, 5, U=U)
    buf = [hist[0], hist[1]]                               # newest last
    for s in range(5):
        nxt = m.A[0] @ buf[-1] + m.A[1] @ buf[-2] + m.B @ U[s]
        assert np.allclose(out[s], nxt)
        buf.append(nxt)


def test_rollout_warns_only_when_an_input_is_dropped():
    m = _model()
    with pytest.warns(UserWarning, match="input term is dropped"):
        m.rollout(np.zeros((2, 3)), 3)
    no_input = GDARModel(A=m.A, B=np.zeros((3, 0)), K=2, param="edgewise", support=m.support)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        no_input.rollout(np.zeros((2, 3)), 3)
        no_input.predict_one_step(np.zeros((6, 3)))


def test_coupling_matrix_is_lag_norm_with_zero_diagonal():
    m = _model(K=3, N=4)
    C = coupling_matrix(m)
    want = np.sqrt(sum(A ** 2 for A in m.A))
    np.fill_diagonal(want, 0.0)
    assert np.allclose(C, want)


def test_edge_flow_matches_definition_and_adds_up_to_prediction():
    rng = np.random.default_rng(7)
    N, K = 4, 2
    m = _model(K=K, N=N, M=1)
    X = rng.standard_normal((12, N))
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    flows, t_idx = edge_flow(m, X, edges)
    assert flows.shape == (12 - K, len(edges), 2)
    assert np.array_equal(t_idx, np.arange(K - 1, 11))
    for idx, t in enumerate(t_idx):
        for e, (i, j) in enumerate(edges):
            assert np.isclose(flows[idx, e, 0], sum(m.A[k][i, j] * X[t - k, j] for k in range(K)))
            assert np.isclose(flows[idx, e, 1], sum(m.A[k][j, i] * X[t - k, i] for k in range(K)))
    # inflow over all edges + self term = the model's prediction without input
    pred, _ = GDARModel(A=m.A, B=np.zeros((N, 0)), K=K, param="edgewise",
                        support=m.support).predict_one_step(X)
    inflow = np.zeros((len(t_idx), N))
    for e, (i, j) in enumerate(edges):
        inflow[:, i] += flows[:, e, 0]
        inflow[:, j] += flows[:, e, 1]
    self_term = np.stack([sum(np.diag(m.A[k]) * X[t - k] for k in range(K)) for t in t_idx])
    assert np.allclose(inflow + self_term, pred)
