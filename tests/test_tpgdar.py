import numpy as np

from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.tpgdar import (data_loss, fit_joint, fit_per_condition, fit_pooled,
                            fit_two_stage, row_columns, stack_stats, trial_stats)


def _rank2_problem(seed=0, hetero=True, P=6, T=400, n_trials=4):
    """Truly rank-2 Theta(p) = (1-p) Theta_0 + p Theta_1 on a 3x3 grid, with input."""
    rng = np.random.default_rng(seed)
    g = grid_graph(3, 3); N = g.n
    mask = support_mask(g.adjacency)
    K, Q = 2, 1
    D = K * N + Q

    def rand_op(scale):
        A = [np.where(mask, rng.standard_normal((N, N)) * scale, 0.0) for _ in range(K)]
        A[0] += 0.5 * np.eye(N)
        return A

    A0, A1 = rand_op(0.12), rand_op(0.12)
    B0, B1 = rng.standard_normal((N, 1)), rng.standard_normal((N, 1))
    p_grid = np.linspace(0, 1, P)
    per_cond, truth = [], []
    for gi, p in enumerate(p_grid):
        A = [(1 - p) * a + p * b for a, b in zip(A0, A1)]
        B = (1 - p) * B0 + p * B1
        th = np.concatenate(A + [B], axis=1)
        truth.append(th)
        # heterogeneous: data length, input power and noise all vary by condition
        Tg = T if not hetero else int(T * (0.3 + 2.0 * (gi % 2)))
        upow = 1.0 if not hetero else (0.2 if gi % 3 == 0 else 2.0)
        trials = []
        for tr in range(n_trials):
            U = upow * rng.standard_normal((Tg, 1))
            X, _ = simulate_gdar(A, Tg, B=B, U=U, noise_std=0.3, rng=rng, burn_in=0)
            trials.append(trial_stats(X, U, K=K, Q=Q))
        per_cond.append(trials)
    stats = stack_stats(per_cond)
    cols = row_columns(mask, K, n_inputs=1)
    blocks = [np.arange(K * N), np.arange(K * N, D)]
    return stats, cols, blocks, p_grid, np.stack(truth)


def test_trial_stats_never_cross_boundaries():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((50, 3)), rng.standard_normal((50, 3))
    s = stack_stats([[trial_stats(a, K=2), trial_stats(b, K=2)]])
    s_cat = trial_stats(np.vstack([a, b]), K=2)
    assert s["n"][0] == 2 * 48 and s_cat["n"] == 98       # concatenation adds 2 rows
    assert not np.allclose(s["Gzz"][0], s_cat["Gzz"])


def test_joint_objective_monotone_and_beats_two_stage():
    stats, cols, blocks, p, truth = _rank2_problem()
    two = fit_two_stage(stats, cols, blocks, [2, 2], p)
    joint = fit_joint(stats, cols, blocks, [2, 2], p, n_iter=40)
    J = joint.objective
    assert np.all(np.diff(J) <= 1e-9 * abs(J[0]))           # never increases
    loss = lambda m: sum(data_loss(th, stats["Gzz"][g], stats["Gzy"][g], stats["Syy"][g])
                         for g, th in enumerate(m.theta_grid()))
    assert loss(joint) < loss(two)
    err = lambda m: np.linalg.norm(m.theta_grid() - truth) / np.linalg.norm(truth)
    assert err(joint) < err(two)


def test_full_rank_joint_equals_per_condition():
    # with r = P the TP constraint is vacuous: both learners give the per-condition fit
    stats, cols, blocks, p, _ = _rank2_problem(P=3)
    per = fit_per_condition(stats, cols, ridge=1e-12)
    joint = fit_joint(stats, cols, blocks, [3, 3], p, ridge=1e-12, n_iter=5)
    assert np.allclose(joint.theta_grid(), per, atol=1e-4 * np.abs(per).max())


def test_rank1_shared_A_is_pooled_when_B_rank1():
    stats, cols, blocks, p, _ = _rank2_problem(P=4)
    joint = fit_joint(stats, cols, blocks, [1, 1], p, n_iter=30)
    pooled = fit_pooled(stats, cols)
    # rank 1 in both blocks with free scale per condition; it can only match or beat pooled
    loss_j = sum(data_loss(th, stats["Gzz"][g], stats["Gzy"][g], stats["Syy"][g])
                 for g, th in enumerate(joint.theta_grid()))
    loss_p = sum(data_loss(pooled, stats["Gzz"][g], stats["Gzy"][g], stats["Syy"][g])
                 for g in range(len(p)))
    assert loss_j <= loss_p * (1 + 1e-9)


def test_interpolation_hits_grid_points():
    stats, cols, blocks, p, _ = _rank2_problem(P=4)
    m = fit_joint(stats, cols, blocks, [2, 2], p, n_iter=10)
    assert np.allclose(m.theta_at(p[2]), m.theta_grid()[2])


def test_joint_objective_bracketed_by_per_condition_and_pooled():
    # same (augmented-Gram) objective for all three: per-condition is its unconstrained
    # minimum, pooled is a feasible rank-1 point; a gauge leak would break the lower bound
    from gdar_tp.tpgdar import augment, joint_objective
    stats, cols, blocks, p, _ = _rank2_problem()
    for ridge in (1e-6, 1e-2):
        aug = augment(stats, ridge)
        J = lambda grid: sum(data_loss(grid[g], aug["Gzz"][g], aug["Gzy"][g], aug["Syy"][g])
                             for g in range(len(p)))
        per = J(fit_per_condition(stats, cols, ridge))
        pooled = J(np.stack([fit_pooled(stats, cols, ridge)] * len(p)))
        joint = fit_joint(stats, cols, blocks, [2, 2], p, ridge=ridge, n_iter=50)
        assert per <= joint_objective(joint, aug) * (1 + 1e-9)
        assert joint_objective(joint, aug) <= pooled * (1 + 1e-9)
