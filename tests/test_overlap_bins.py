"""Overlapping bins, MATH §2.4: the weights, and the weighted bowls against brute-force weighted least squares."""

import numpy as np
import pytest

from gdar_tp.bins import overlap_weights, weighted_bin_stats
from gdar_tp.sym_tp import _quad, symmetric_basis, theta_of

P = 16


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0, 3.0])
def test_weights_sum_to_one_and_are_nonnegative(delta):
    tau = np.random.default_rng(0).uniform(0, P, 2000)
    a = overlap_weights(tau, delta, P)
    assert (a >= 0).all()
    np.testing.assert_allclose(a.sum(1), 1.0, atol=1e-12)


def test_delta_zero_is_the_hard_bin():
    tau = np.array([0.0, 0.99, 4.2, 15.999, np.nan])
    a = overlap_weights(tau, 0, P)
    assert a[:4].argmax(1).tolist() == [0, 0, 4, 15]
    np.testing.assert_array_equal(a[:4].sum(1), 1.0)
    assert a[4].sum() == 0
    lab = np.array([0, 0, 4, 15, -1])
    np.testing.assert_array_equal(overlap_weights(tau, 0, P, label=lab), a)


def test_unused_samples_get_no_weight():
    a = overlap_weights(np.array([np.nan, 3.5]), 1.0, P)
    assert a[0].sum() == 0 and a[1, 3] == 1.0


def test_delta_one_is_linear_between_bin_centres():
    g = 6
    a = overlap_weights(np.array([g + 0.5, g + 1.0, g + 0.5 + 0.2, g + 0.5 - 0.3]), 1.0, P)
    assert a[0, g] == pytest.approx(1.0)                                       # bin centre: all in its own bin
    assert a[1, g] == pytest.approx(0.5) and a[1, g + 1] == pytest.approx(0.5)  # bin edge: half in each
    assert a[2, g] == pytest.approx(0.8) and a[2, g + 1] == pytest.approx(0.2)
    assert a[3, g] == pytest.approx(0.7) and a[3, g - 1] == pytest.approx(0.3)


def test_window_is_clipped_at_the_ends():
    a = overlap_weights(np.array([0.1, 15.9]), 1.0, P)
    assert a[0, 0] == pytest.approx(1.0)                 # [-0.4, 0.6] -> [0, 0.6], all in hold 1/4
    b = overlap_weights(np.array([15.9]), 2.0, P)[0]     # [14.9, 16.9] -> [14.9, 16]
    assert b[14] == pytest.approx(0.1 / 1.1) and b[15] == pytest.approx(1.0 / 1.1)


def test_delta_two_reaches_the_whole_neighbour_and_no_further():
    g = 6
    inside = np.linspace(g + 1.0, g + 2.0 - 1e-9, 50)    # every sample of bin g + 1
    assert (overlap_weights(inside, 2.0, P)[:, g] > 0).all()
    beyond = np.linspace(g + 2.0, g + 3.0 - 1e-9, 50)    # bin g + 2
    assert (overlap_weights(beyond, 2.0, P)[:, g] == 0).all()


def _data(seed=1, n=6, T=60, N=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, N))
    tau = np.sort(rng.uniform(0, 4, (n, T)), axis=1)
    tau[:, :5] = np.nan                                   # some unused samples
    return X, tau


def _hard_stats(X, lab, K, K0, fold, nfold, P):
    n, T, N = X.shape
    s = dict(Gzz=np.zeros((nfold, P, K * N, K * N)), Gzy=np.zeros((nfold, P, K * N, N)), Syy=np.zeros((nfold, P, N, N)),
             n=np.zeros((nfold, P)))
    for i in range(n):
        for t in range(K0 - 1, T - 1):
            g = lab[i, t + 1]
            if g < 0:
                continue
            z = np.concatenate([X[i, t - k] for k in range(K)])
            y = X[i, t + 1]
            s["Gzz"][fold[i], g] += np.outer(z, z)
            s["Gzy"][fold[i], g] += np.outer(z, y)
            s["Syy"][fold[i], g] += np.outer(y, y)
            s["n"][fold[i], g] += 1
    return s


def test_one_hot_weights_give_the_hard_statistics():
    X, tau = _data()
    K, K0, P4 = 2, 4, 4
    lab = np.where(np.isnan(tau), -1, np.floor(np.nan_to_num(tau, nan=0))).astype(int)
    fold = np.array([0, 1, 0, 1, 0, 1])
    soft = weighted_bin_stats(X, overlap_weights(tau, 0, P4, label=lab), K, K0, fold, 2)
    hard = _hard_stats(X, lab, K, K0, fold, 2, P4)
    for k in hard:
        np.testing.assert_allclose(soft[k], hard[k], rtol=1e-12, atol=1e-12)


def test_weighted_bowl_equals_brute_force_weighted_least_squares():
    X, tau = _data(seed=2, n=4, T=80)
    K, K0, P4, N = 2, 3, 4, X.shape[2]
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)     # a path graph
    R = symmetric_basis(A, K)
    w = overlap_weights(tau, 1.0, P4)
    s = weighted_bin_stats(X, w, K, K0, np.zeros(len(X), int), 1)
    M = np.random.default_rng(3).standard_normal((N, N))
    Sinv = M @ M.T + N * np.eye(N)
    g = 2
    H, h = _quad(s["Gzz"][0, g], s["Gzy"][0, g], R, Sinv)
    phi = np.linalg.solve(H, h)
    # brute force: rows sqrt(w) L' (z' kron I) R phi = sqrt(w) L' x(t+1), with L L' = Sinv
    L = np.linalg.cholesky(Sinv)
    Rd = R.toarray()
    rows, rhs = [], []
    for i in range(len(X)):
        for t in range(K0 - 1, X.shape[1] - 1):
            wt = w[i, t + 1, g]
            if wt == 0:
                continue
            z = np.concatenate([X[i, t - k] for k in range(K)])
            rows.append(np.sqrt(wt) * L.T @ np.kron(z[None], np.eye(N)) @ Rd)
            rhs.append(np.sqrt(wt) * L.T @ X[i, t + 1])
    phi_bf = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]
    np.testing.assert_allclose(phi, phi_bf, rtol=1e-8, atol=1e-10)
    # and the weighted count is the sum of the weights
    assert s["n"][0, g] == pytest.approx(w[:, K0:, g].sum())
    assert theta_of(phi, R, N).shape == (N, K * N)
