"""Unit tests for gdar_tp/control.py and gdar_tp/latent.py beyond the existing files."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gdar_tp.control import (SynthesisResult, _lmi_residual, _vertices_to_companion, closed_loop_max_rho,
                             companion_form, is_schur_stable, synthesize_quadratic_feedback,
                             synthesize_slack_feedback)
from gdar_tp.latent import LDS, constrained_m_step, recovered_inhibitory

A0 = np.array([[1.1, 0.2], [0.0, 0.95]])
A1 = np.array([[0.9, -0.3], [0.4, 1.15]])


# ------------------------------------------------------------------------------ control
def test_companion_form_without_input_and_with_one_lag():
    Ac, Bc = companion_form([A0], np.zeros((2, 0)))
    assert np.array_equal(Ac, A0) and Bc.shape == (2, 0)


def test_vertices_to_companion_accepts_lag_lists_and_nnk_arrays():
    rng = np.random.default_rng(0)
    lags = [[rng.standard_normal((2, 2)) for _ in range(3)] for _ in range(2)]
    arr = np.stack([np.stack(v, axis=-1) for v in lags])  # (V, N, N, K)
    B = np.ones((2, 1))
    c1, B1 = _vertices_to_companion(lags, B)
    c2, B2 = _vertices_to_companion(arr, B)
    assert all(np.array_equal(a, b) for a, b in zip(c1, c2)) and np.array_equal(B1, B2)
    assert c1[0].shape == (6, 6) and np.array_equal(c1[1][:2], np.hstack(lags[1]))


def test_closed_loop_rho_and_schur_test():
    F = np.zeros((1, 2))
    Bc = np.zeros((2, 1))
    assert np.isclose(closed_loop_max_rho([np.diag([0.5, 0.2]), np.diag([0.1, -0.9])], Bc, F), 0.9)
    assert is_schur_stable(np.diag([0.5, -0.99])) and not is_schur_stable(np.diag([0.5, 1.0]))


def test_lmi_residual_is_negative_for_a_valid_certificate():
    res = _lmi_residual([np.diag([0.5, 0.3])], np.zeros((2, 1)), np.zeros((1, 2)), np.eye(2), 1.0)
    assert np.isclose(res, 0.25 - 1.0)
    assert _lmi_residual([np.diag([1.2, 0.3])], np.zeros((2, 1)), np.zeros((1, 2)), np.eye(2), 1.0) > 0


def test_synthesis_result_defaults():
    r = SynthesisResult(F=None, Q=None, feasible=False, method="x", max_rho=np.inf)
    assert np.isnan(r.lmi_residual) and r.solver == "" and r.status == ""


@pytest.fixture(scope="module")
def cvx():
    return pytest.importorskip("cvxpy")


def test_decay_rate_is_certified(cvx):
    res = synthesize_quadratic_feedback([A0[..., None], A1[..., None]], np.eye(2), decay=0.8)
    assert res.feasible and res.max_rho <= 0.8 + 1e-6 and res.lmi_residual <= 1e-6


def test_min_gain_gives_a_smaller_gain_than_min_trace(cvx):
    verts = [A0[..., None], A1[..., None]]
    rt = synthesize_quadratic_feedback(verts, np.eye(2))
    rg = synthesize_quadratic_feedback(verts, np.eye(2), min_gain=True)
    assert rg.feasible and np.linalg.norm(rg.F, 2) < np.linalg.norm(rt.F, 2)


def test_explicit_solver_is_used(cvx):
    res = synthesize_quadratic_feedback([A0[..., None]], np.eye(2), solver="SCS")
    assert res.feasible and res.status in ("optimal", "optimal_inaccurate")


def test_slack_returns_a_failure_object_when_infeasible(cvx):
    A = np.array([[1.3, 0.0], [0.0, 0.5]])
    B = np.array([[0.0], [1.0]])                            # the unstable mode is not actuated
    res = synthesize_slack_feedback([A[..., None]], B)
    assert not res.feasible and res.F is None and res.max_rho == np.inf and res.method == "slack"


# ------------------------------------------------------------------------------- latent
def _lds(seed=0):
    rng = np.random.default_rng(seed)
    d, n = 2, 1
    A = np.array([[0.8, 0.2], [-0.3, 0.6]])
    M = rng.standard_normal((d, d))
    return LDS(A=A, C=np.array([[1.0, 0.0]]), Q=0.1 * np.eye(d) + 0.02 * M @ M.T, R=0.05 * np.eye(n),
               mu0=np.array([0.3, -0.1]), P0=0.5 * np.eye(d))


def _joint(lds, T):
    """Mean and covariance of (x_0..x_{T-1}) and of (y_0..y_{T-1}) by brute force."""
    d = lds.A.shape[0]
    means, V = [lds.mu0], [lds.P0]
    for _ in range(T - 1):
        means.append(lds.A @ means[-1])
        V.append(lds.A @ V[-1] @ lds.A.T + lds.Q)
    Sxx = np.zeros((T * d, T * d))
    for s in range(T):
        for t in range(s, T):
            blk = np.linalg.matrix_power(lds.A, t - s) @ V[s]       # Cov(x_t, x_s)
            Sxx[t * d:(t + 1) * d, s * d:(s + 1) * d] = blk
            Sxx[s * d:(s + 1) * d, t * d:(t + 1) * d] = blk.T
    Cb = np.kron(np.eye(T), lds.C)
    Syy = Cb @ Sxx @ Cb.T + np.kron(np.eye(T), lds.R)
    return np.concatenate(means), Sxx, Cb, Syy


def test_filter_loglik_matches_the_joint_gaussian():
    lds, T = _lds(), 6
    Y = np.random.default_rng(1).standard_normal((T, 1))
    m, _, Cb, Syy = _joint(lds, T)
    want = multivariate_normal(Cb @ m, Syy).logpdf(Y.ravel())
    assert np.isclose(lds.filter(Y)["loglik"], want)


def test_smoother_matches_the_conditional_gaussian():
    lds, T = _lds(), 6
    d = 2
    Y = np.random.default_rng(2).standard_normal((T, 1))
    m, Sxx, Cb, Syy = _joint(lds, T)
    Sxy = Sxx @ Cb.T
    mean = m + Sxy @ np.linalg.solve(Syy, Y.ravel() - Cb @ m)
    cov = Sxx - Sxy @ np.linalg.solve(Syy, Sxy.T)
    sm = lds.smooth(Y)
    assert np.allclose(sm["xs"].ravel(), mean)
    for t in range(T):
        assert np.allclose(sm["Ps"][t], cov[t * d:(t + 1) * d, t * d:(t + 1) * d])
    for t in range(1, T):
        assert np.allclose(sm["Plag"][t], cov[t * d:(t + 1) * d, (t - 1) * d:t * d])


def test_constrained_m_step_without_signs_is_masked_least_squares():
    rng = np.random.default_rng(3)
    d = 4
    X = rng.standard_normal((200, d))
    S00, S10 = X[:-1].T @ X[:-1], X[1:].T @ X[:-1]
    struct = rng.random((d, d)) < 0.6
    np.fill_diagonal(struct, True)
    A = constrained_m_step(S00, S10, struct, np.zeros((d, d), int))
    for a in range(d):
        c = np.where(struct[a])[0]
        assert np.allclose(A[a, c], np.linalg.solve(S00[np.ix_(c, c)], S10[a, c]), atol=1e-6)
        assert np.all(A[a, ~struct[a]] == 0)


def test_recovered_inhibitory_takes_the_odd_latent_slots():
    xs = np.arange(12.0).reshape(2, 6)
    assert np.array_equal(recovered_inhibitory({"xs": xs}, 3), xs[:, 1::2])
