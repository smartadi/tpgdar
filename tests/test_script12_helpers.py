"""Unit tests for the helpers in scripts/12_tp_vs_gdar_ios.py.

Eleven other scripts import that file as a library (statistics, Sigma, held-out likelihood, folds, labels,
evoked kernels), so its helpers are tested here like library code.  Synthetic data only.
"""

import importlib
import os
import sys

import numpy as np
import pytest

from gdar_tp.control import companion_form
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.tpgdar import gaussian_ll, resid_cov_sum, stack_stats, trial_stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")


def _X(n=3, T=40, N=2, seed=0):
    return np.random.default_rng(seed).standard_normal((n, T, N))


def test_trial_cond_stats_one_condition_matches_trial_stats():
    X = _X()
    st = s12.trial_cond_stats(X, np.zeros(X.shape[:2], int), K=2, K0=2, P=1, per_trial=False)
    ref = stack_stats([[trial_stats(x, K=2) for x in X]])
    for k in ("Gzz", "Gzy", "Syy", "n"):
        assert np.allclose(st[k], ref[k])


def test_trial_cond_stats_splits_by_the_label_of_the_target():
    X = _X(n=2, T=30)
    lab = np.random.default_rng(1).integers(-1, 2, size=X.shape[:2])
    st = s12.trial_cond_stats(X, lab, K=1, K0=1, P=2)
    assert st["Gzz"].shape == (2, 2, 2, 2) and st["n"].shape == (2, 2)
    for i in range(2):
        for g in range(2):
            ts = np.arange(0, 29)
            keep = ts[lab[i, ts + 1] == g]
            Z, Y = X[i, keep], X[i, keep + 1]
            assert st["n"][i, g] == len(keep)
            assert np.allclose(st["Gzz"][i, g], Z.T @ Z) and np.allclose(st["Gzy"][i, g], Z.T @ Y)


def test_K0_fixes_the_sample_set_for_every_K():
    X = _X()
    lab = np.zeros(X.shape[:2], int)
    for K in (1, 2, 3):
        assert s12.trial_cond_stats(X, lab, K=K, K0=3, P=1, per_trial=False)["n"][0] == 3 * (40 - 3)


def test_sum_stats_and_pool():
    T = {"a": np.arange(12.0).reshape(3, 2, 2)}
    assert np.allclose(s12.sum_stats(T, np.array([True, False, True]))["a"], T["a"][0] + T["a"][2])
    assert np.allclose(s12.pool(T)["a"], T["a"].sum(0))


def test_sigma_of_is_the_pooled_residual_covariance_shrunk_to_its_diagonal():
    X = _X(N=3)
    st = s12.trial_cond_stats(X, np.random.default_rng(2).integers(0, 2, X.shape[:2]), K=1, K0=1, P=2,
                              per_trial=False)
    Th = 0.1 * np.random.default_rng(3).standard_normal((2, 3, 3))
    S = sum(resid_cov_sum(Th[g], st["Gzz"][g], st["Gzy"][g], st["Syy"][g]) for g in range(2)) / st["n"].sum()
    want = (1 - s12.SIG_SHRINK) * S + s12.SIG_SHRINK * np.diag(np.diag(S))
    assert np.allclose(s12.sigma_of(Th, st), want)


def test_ll_matrix_matches_gaussian_ll_per_trial_and_condition():
    X = _X(n=3, N=3)
    lab = np.random.default_rng(4).integers(-1, 2, X.shape[:2])
    lab[1] = np.where(lab[1] == 1, 0, lab[1])                 # trial 1 never visits condition 1
    T = s12.trial_cond_stats(X, lab, K=1, K0=1, P=2)
    Th = 0.1 * np.random.default_rng(5).standard_normal((2, 3, 3))
    Sigma = np.diag([1.0, 2.0, 0.5])
    sel = np.array([True, True, False])
    ll, sse = s12.ll_matrix(Th, T, sel, Sigma)
    assert ll.shape == (2, 2)
    assert ll[1, 1] == 0.0 and sse[1, 1] == 0.0
    for a, i in enumerate([0, 1]):
        for g in range(2):
            if T["n"][i, g]:
                args = (T["Gzz"][i, g], T["Gzy"][i, g], T["Syy"][i, g])
                assert np.isclose(ll[a, g], gaussian_ll(Th[g], *args, T["n"][i, g], Sigma))
                assert np.isclose(sse[a, g], np.trace(resid_cov_sum(Th[g], *args)))


def test_rho_is_the_companion_spectrum():
    rng = np.random.default_rng(6)
    N, K = 3, 2
    Theta = 0.3 * rng.standard_normal((N, K * N))
    Ac, _ = companion_form([Theta[:, k * N:(k + 1) * N] for k in range(K)], np.zeros((N, 0)))
    assert np.allclose(np.sort_complex(s12.rho(Theta, K)), np.sort_complex(np.linalg.eigvals(Ac)))


def test_folds_are_balanced_within_each_direction():
    ang = np.repeat([0, 90, 180], 10)
    f = s12.folds(ang)
    for a in (0, 90, 180):
        assert np.bincount(f[ang == a], minlength=s12.NFOLD).tolist() == [2] * s12.NFOLD
    assert np.array_equal(f, s12.folds(ang)) and not np.array_equal(f, s12.folds(ang, seed=1))


def _events():
    t = np.arange(-500, 1500, 3.0)
    go = np.array([300.0, 350, 420, 380, 300, 500])
    return dict(t=t, go=go, touch=go + 400, ang=np.array([0, 90, 0, 90, 0, 90]))


def test_labels_phase_and_time_bins():
    D = _events()
    lab = s12.labels(D)
    t, go, touch = D["t"], D["go"][0], D["touch"][0]
    ph = lab["phase"][0]
    assert np.all(ph[(t >= -400) & (t < 0)] == 0) and np.all(ph[(t >= 0) & (t < go)] == 1)
    assert np.all(ph[(t >= go) & (t < touch)] == 2) and np.all(ph[(t >= touch) & (t < touch + 500)] == 3)
    assert np.all(ph[t < -400] == -1) and np.all(ph[t >= touch + 500] == -1)
    tb = lab["time"][0]
    inside = (t - go >= s12.TBIN[0]) & (t - go < s12.TBIN[-1])
    assert np.all(tb[~inside] == -1)
    assert np.array_equal(tb[inside], ((t[inside] - go - s12.TBIN[0]) // 100).astype(int))


def test_remove_evoked_removes_a_planted_response():
    D = _events()
    Z = s12.evoked_design(D)
    n, T = len(D["go"]), len(D["t"])
    assert Z.shape[0] == n * T
    beta = np.random.default_rng(7).standard_normal((Z.shape[1], 2))
    X = np.asarray(Z @ beta).reshape(n, T, 2)
    assert np.linalg.norm(s12.remove_evoked(X, Z, np.ones(n, bool), lam=1e-8)) < 1e-5 * np.linalg.norm(X)


def test_h_step_vaf_is_one_for_the_true_noise_free_operator():
    th = 0.3
    A = 0.99 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X = np.stack([simulate_gdar([A], 120, x0=x0[None])[0] for x0 in np.eye(2)])
    lab = np.zeros(X.shape[:2], int)
    vaf = s12.h_step_vaf(A[None], X, lab, np.ones(2, bool), K=1, K0=1, horizons=(1, 5, 17))
    assert np.allclose(vaf, 1.0)
    worse = s12.h_step_vaf(0.5 * A[None], X, lab, np.ones(2, bool), K=1, K0=1, horizons=(1, 5))
    assert np.all(worse < 0.99)


@pytest.mark.parametrize("K", [1, 3])
def test_ar_spectrum_of_white_noise_is_flat(K):
    N = 2
    out = s12.ar_spectrum(np.zeros((N, K * N)), np.eye(N), K, np.array([5.0, 50.0]))
    assert np.allclose(out, 1.0)
