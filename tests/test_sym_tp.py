"""Symmetric TP-GDAR: same model class as Felix's GDAR, same ALS guarantees."""

import numpy as np

from gdar_tp.graph import grid_graph
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.sym_tp import fit_gls_pooled, fit_sym_tp, symmetric_basis, theta_of
from gdar_tp.tpgdar import stack_stats, trial_stats


def _setup(seed=0, P=4, K=2):
    g = grid_graph(3, 3)
    A = np.asarray(g.adjacency, bool)
    N = g.n
    R = symmetric_basis(A, K)
    rng = np.random.default_rng(seed)
    phi0, phi1 = rng.standard_normal((2, R.shape[1])) * 0.04
    phi0[:N] += 0.5
    grid = np.linspace(0, 1, P)
    per = []
    truth = []
    for p in grid:
        th = theta_of((1 - p) * phi0 + p * phi1, R, N)
        truth.append(th)
        trials = []
        for _ in range(4):
            X, _ = simulate_gdar([th[:, k * N:(k + 1) * N] for k in range(K)], 800, noise_std=0.3,
                                 rng=rng, burn_in=50)
            trials.append(trial_stats(X, K=K))
        per.append(trials)
    return stack_stats(per), R, N, K, grid, np.stack(truth), A


def test_basis_is_symmetric_on_the_graph():
    A = np.asarray(grid_graph(3, 3).adjacency, bool)
    R = symmetric_basis(A, 2)
    assert R.shape[1] == 2 * (9 + A.sum() // 2)
    th = theta_of(np.random.default_rng(1).standard_normal(R.shape[1]), R, 9)
    for k in range(2):
        block = th[:, k * 9:(k + 1) * 9]
        assert np.allclose(block, block.T)
        assert np.allclose(block[~(A | np.eye(9, dtype=bool))], 0)


def test_objective_never_increases_and_recovers_rank_2():
    stats, R, N, K, grid, truth, _ = _setup()
    phi, Sig = fit_gls_pooled(stats["Gzz"].sum(0), stats["Gzy"].sum(0), stats["Syy"].sum(0),
                              stats["n"].sum(), R)
    m = fit_sym_tp(stats, R, Sig, 2, grid)
    assert np.all(np.diff(m.objective) <= 1e-6 * np.abs(m.objective[:-1]))
    rel = np.linalg.norm(m.theta_grid() - truth) / np.linalg.norm(truth)
    pooled = np.stack([theta_of(phi, R, N)] * len(grid))
    assert rel < 0.5 * np.linalg.norm(pooled - truth) / np.linalg.norm(truth)


def test_full_rank_equals_per_condition():
    stats, R, N, K, grid, _, _ = _setup()
    _, Sig = fit_gls_pooled(stats["Gzz"].sum(0), stats["Gzy"].sum(0), stats["Syy"].sum(0),
                            stats["n"].sum(), R)
    m = fit_sym_tp(stats, R, Sig, len(grid), grid, n_iter=200, tol=1e-11)
    for g in range(len(grid)):
        single = {k: stats[k][[g]] for k in stats}
        one = fit_sym_tp(single, R, Sig, 1, grid[[g]], n_iter=1)
        assert np.linalg.norm(m.theta_grid()[g] - one.theta_grid()[0]) < 1e-3 * np.linalg.norm(one.theta_grid()[0])
