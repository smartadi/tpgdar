"""Unit tests for every function and method in gdar_tp/tpgdar.py."""

import numpy as np
from scipy.stats import multivariate_normal

from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.tpgdar import (TPGDAR, _normalize_gauge, _pooled_start, _ridge, _theta_step, _w_step, augment,
                            data_loss, fit_per_condition, fit_pooled, fit_static, gaussian_ll,
                            joint_objective, resid_cov_sum, row_columns, stack_stats, trial_stats)


def _regressors(X, U, K, Q):
    T = len(X)
    lag0 = max(K, Q) - 1
    ts = np.arange(lag0, T - 1)
    Z = np.concatenate([X[ts - k] for k in range(K)] + [U[ts - q] for q in range(Q)], axis=1)
    return Z, X[ts + 1]


def _small_problem(seed=0, P=4, T=300, ridge=1e-3):
    """2x2 grid, K=1, one input; Theta(p) affine in p, so rank 2."""
    rng = np.random.default_rng(seed)
    g = grid_graph(2, 2)
    N, mask = g.n, support_mask(g.adjacency)
    A0 = np.where(mask, 0.15 * rng.standard_normal((N, N)), 0) + 0.5 * np.eye(N)
    A1 = np.where(mask, 0.15 * rng.standard_normal((N, N)), 0)
    B0 = rng.standard_normal((N, 1))
    p_grid = np.linspace(0, 1, P)
    per = []
    for p in p_grid:
        trials = []
        for _ in range(2):
            U = rng.standard_normal((T, 1))
            X, _ = simulate_gdar([A0 + p * A1], T, B=(1 + p) * B0, U=U, noise_std=0.3, rng=rng)
            trials.append(trial_stats(X, U, K=1, Q=1))
        per.append(trials)
    stats = stack_stats(per)
    cols = row_columns(mask, 1, n_inputs=1)
    blocks = [np.arange(N), np.arange(N, N + 1)]
    return stats, cols, blocks, p_grid, ridge


# --------------------------------------------------------------------------- statistics
def test_trial_stats_matches_explicit_regressors():
    rng = np.random.default_rng(1)
    X, U = rng.standard_normal((40, 3)), rng.standard_normal((40, 2))
    s = trial_stats(X, U, K=2, Q=2)
    Z, Y = _regressors(X, U, 2, 2)
    assert np.allclose(s["Gzz"], Z.T @ Z) and np.allclose(s["Gzy"], Z.T @ Y)
    assert np.allclose(s["Syy"], Y.T @ Y) and s["n"] == 38


def test_trial_stats_input_lags_set_the_first_target():
    rng = np.random.default_rng(2)
    X, U = rng.standard_normal((40, 3)), rng.standard_normal((40, 1))
    assert trial_stats(X, U, K=1, Q=3)["n"] == 40 - 3
    assert trial_stats(X, K=3)["n"] == 40 - 3
    assert trial_stats(X, U.ravel(), K=1, Q=1)["Gzz"].shape == (4, 4)    # 1-D input is reshaped


def test_stack_stats_sums_trials_per_condition():
    rng = np.random.default_rng(3)
    s = [trial_stats(rng.standard_normal((30, 2)), K=1) for _ in range(3)]
    st = stack_stats([[s[0], s[1]], [s[2]]])
    assert st["Gzz"].shape == (2, 2, 2)
    assert np.allclose(st["Gzz"][0], s[0]["Gzz"] + s[1]["Gzz"]) and np.allclose(st["Gzy"][1], s[2]["Gzy"])
    assert st["n"].tolist() == [58.0, 29.0]


def test_row_columns_lists_supported_lags_then_inputs():
    support = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], bool)
    cols = row_columns(support, K=2, n_inputs=2)
    assert cols[0].tolist() == [0, 1, 3, 4, 6, 7]
    assert cols[1].tolist() == [1, 4, 6, 7]
    assert cols[2].tolist() == [0, 2, 3, 5, 6, 7]


def test_ridge_scales_with_each_regressors_energy():
    lam = _ridge(np.diag([1.0, 4.0, 9.0]), [np.array([0, 2]), np.array([1])], 0.1)
    assert np.allclose(lam[0], [0.1, 0.9]) and np.allclose(lam[1], [0.4])


# --------------------------------------------------------------------------------- loss
def test_resid_cov_sum_data_loss_and_gaussian_ll_match_residuals():
    rng = np.random.default_rng(4)
    X, U = rng.standard_normal((60, 3)), rng.standard_normal((60, 1))
    s = trial_stats(X, U, K=2, Q=1)
    Z, Y = _regressors(X, U, 2, 1)
    Theta = 0.2 * rng.standard_normal((3, Z.shape[1]))
    E = Y - Z @ Theta.T
    assert np.allclose(resid_cov_sum(Theta, s["Gzz"], s["Gzy"], s["Syy"]), E.T @ E)
    assert np.isclose(data_loss(Theta, s["Gzz"], s["Gzy"], s["Syy"]), np.sum(E ** 2))
    M = rng.standard_normal((3, 3))
    Sigma = M @ M.T + np.eye(3)
    want = multivariate_normal(np.zeros(3), Sigma).logpdf(E).sum()
    assert np.isclose(gaussian_ll(Theta, s["Gzz"], s["Gzy"], s["Syy"], s["n"], Sigma), want)


# ------------------------------------------------------------------------------ fitters
def test_fit_static_is_ridge_on_the_allowed_columns():
    rng = np.random.default_rng(5)
    X, U = rng.standard_normal((200, 3)), rng.standard_normal((200, 1))
    s = trial_stats(X, U, K=1, Q=1)
    Z, Y = _regressors(X, U, 1, 1)
    cols = row_columns(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], bool), 1, n_inputs=1)
    Theta = fit_static(s["Gzz"], s["Gzy"], cols, ridge=0.05)
    for i, c in enumerate(cols):
        Zc = Z[:, c]
        want = np.linalg.solve(Zc.T @ Zc + np.diag(0.05 * np.diag(Zc.T @ Zc)), Zc.T @ Y[:, i])
        assert np.allclose(Theta[i, c], want)
        assert np.all(Theta[i, np.setdiff1d(np.arange(4), c)] == 0)


def test_per_condition_and_pooled_are_fit_static_on_the_right_grams():
    stats, cols, *_ , ridge = _small_problem()
    per = fit_per_condition(stats, cols, ridge)
    assert per.shape == (4, 4, 5)
    assert np.allclose(per[2], fit_static(stats["Gzz"][2], stats["Gzy"][2], cols, ridge))
    assert np.allclose(fit_pooled(stats, cols, ridge),
                       fit_static(stats["Gzz"].sum(0), stats["Gzy"].sum(0), cols, ridge))


# ---------------------------------------------------------------------------- TP model
def _random_model(P=3, N=2, seed=6):
    rng = np.random.default_rng(seed)
    blocks = [np.arange(4), np.arange(4, 5)]               # D = 5: two lags of N = 2, one input
    return TPGDAR(blocks=blocks, W=[rng.standard_normal((P, 2)), rng.standard_normal((P, 1))],
                  Theta=[rng.standard_normal((2, N, 4)), rng.standard_normal((1, N, 1))],
                  p_grid=np.array([1.0, 2.0, 4.0]), D=5)


def test_tpgdar_ranks_theta_from_weights_and_grid():
    m = _random_model()
    assert m.ranks == [2, 1]
    w = [np.array([0.5, -1.0]), np.array([2.0])]
    th = m.theta_from_weights(w)
    assert np.allclose(th[:, :4], 0.5 * m.Theta[0][0] - m.Theta[0][1])
    assert np.allclose(th[:, 4:], 2.0 * m.Theta[1][0])
    grid = m.theta_grid()
    assert grid.shape == (3, 2, 5)
    assert np.allclose(grid[1], m.theta_from_weights([m.W[0][1], m.W[1][1]]))


def test_weights_at_interpolates_linearly_and_in_log_and_sorts_the_grid():
    m = _random_model()
    m.W = [np.array([[1.0], [2.0], [4.0]])]                # weight equal to p
    m.blocks, m.Theta = [np.arange(5)], [np.ones((1, 2, 5))]
    assert np.isclose(m.weights_at(3.0)[0][0], 3.0)
    assert np.isclose(m.weights_at(2 * np.sqrt(2), log=True)[0][0], 3.0)   # log-midpoint of 2 and 4
    assert np.allclose(m.theta_at(2.0), m.theta_grid()[1])
    m.p_grid, m.W = np.array([4.0, 1.0, 2.0]), [np.array([[4.0], [1.0], [2.0]])]
    assert np.isclose(m.weights_at(3.0)[0][0], 3.0)


def test_augment_adds_a_scaled_diagonal_per_grid_point():
    stats, *_ = _small_problem()
    aug = augment(stats, 0.1)
    for g in range(len(stats["n"])):
        G = stats["Gzz"][g]
        assert np.allclose(aug["Gzz"][g], G + 0.1 * np.diag(np.diag(G)))
    assert aug["Gzy"] is stats["Gzy"] and aug["Syy"] is stats["Syy"]


def test_joint_objective_is_the_sum_of_per_condition_losses():
    stats, cols, blocks, p, ridge = _small_problem()
    m = _pooled_start(stats, cols, blocks, [2, 1], p, ridge)
    aug = augment(stats, ridge)
    want = sum(data_loss(th, aug["Gzz"][g], aug["Gzy"][g], aug["Syy"][g]) for g, th in enumerate(m.theta_grid()))
    assert np.isclose(joint_objective(m, aug), want)


def test_pooled_start_rank_one_reproduces_the_pooled_fit():
    stats, cols, blocks, p, ridge = _small_problem()
    m = _pooled_start(stats, cols, blocks, [1, 1], p, ridge)
    pooled = fit_pooled(stats, cols, ridge)
    assert all(np.allclose(th, pooled) for th in m.theta_grid())
    m2 = _pooled_start(stats, cols, blocks, [2, 2], p, ridge)
    assert m2.ranks == [2, 2] and np.allclose(m2.W[0][:, 0], 1.0)


def test_normalize_gauge_keeps_operators_and_orthonormalises_W():
    stats, cols, blocks, p, ridge = _small_problem()
    m = _pooled_start(stats, cols, blocks, [2, 1], p, ridge)
    before = m.theta_grid()
    _normalize_gauge(m)
    assert np.allclose(m.theta_grid(), before)
    assert all(np.allclose(w.T @ w, np.eye(w.shape[1])) for w in m.W)


def _perturbed(model, which, idx, eps):
    m = TPGDAR(blocks=model.blocks, W=[w.copy() for w in model.W], Theta=[t.copy() for t in model.Theta],
               p_grid=model.p_grid, D=model.D)
    arr = m.Theta if which == "Theta" else m.W
    arr[idx[0]][idx[1:]] += eps
    return m


def test_theta_step_and_w_step_are_exact_block_minimisers():
    stats, cols, blocks, p, ridge = _small_problem()
    aug = augment(stats, ridge)
    m = _pooled_start(stats, cols, blocks, [2, 1], p, ridge)
    J0 = joint_objective(m, aug)
    _theta_step(m, aug, cols)
    J1 = joint_objective(m, aug)
    assert J1 <= J0 * (1 + 1e-12)
    c = int(cols[0][0])                                    # an allowed column of row 0, in block 0
    for eps in (1e-3, -1e-3):
        assert joint_objective(_perturbed(m, "Theta", (0, 1, 0, c), eps), aug) > J1
    _w_step(m, aug)
    J2 = joint_objective(m, aug)
    assert J2 <= J1 * (1 + 1e-12)
    for eps in (1e-3, -1e-3):
        assert joint_objective(_perturbed(m, "W", (0, 2, 1), eps), aug) > J2
