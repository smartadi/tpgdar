"""Symmetric TP-GDAR: same model class as Felix's GDAR, same ALS guarantees."""

import numpy as np

from gdar_tp.graph import grid_graph
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.sym_tp import (_quad, fit_gls_pooled, fit_sym_tp, polytope_vertices, symmetric_basis,
                            theta_of)
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


def test_node_inputs_add_a_diagonal_b():
    A = np.asarray(grid_graph(3, 3).adjacency, bool)
    R = symmetric_basis(A, 2, node_inputs=True)
    assert R.shape == (9 * (2 * 9 + 9), 2 * (9 + A.sum() // 2) + 9)
    th = theta_of(np.random.default_rng(2).standard_normal(R.shape[1]), R, 9)
    B = th[:, 18:]
    assert np.allclose(B, np.diag(np.diag(B))) and np.all(np.diag(B) != 0)
    for k in range(2):
        block = th[:, k * 9:(k + 1) * 9]
        assert np.allclose(block, block.T)


def test_objective_never_increases_and_recovers_rank_2():
    stats, R, N, K, grid, truth, _ = _setup()
    phi, Sig = fit_gls_pooled(stats["Gzz"].sum(0), stats["Gzy"].sum(0), stats["Syy"].sum(0),
                              stats["n"].sum(), R)
    m = fit_sym_tp(stats, R, Sig, 2, grid)
    assert np.all(np.diff(m.objective) <= 1e-6 * np.abs(m.objective[:-1]))
    rel = np.linalg.norm(m.theta_grid() - truth) / np.linalg.norm(truth)
    pooled = np.stack([theta_of(phi, R, N)] * len(grid))
    assert rel < 0.5 * np.linalg.norm(pooled - truth) / np.linalg.norm(truth)


def test_polytope_vertices_are_convex_and_reconstruct():
    stats, R, N, K, grid, truth, _ = _setup(P=5)
    tot = {k: stats[k].sum(0) for k in stats}
    _, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    m = fit_sym_tp(stats, R, Sig, 2, grid)
    H, _ = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sig))
    v = polytope_vertices(m, H)
    assert set(v["index"].tolist()) == {0, 4}
    assert np.all(v["lam"] >= 0) and np.allclose(v["lam"].sum(1), 1)
    assert np.all(np.diff(v["lam"][:, list(v["index"]).index(4)]) > 0)
    recon = np.stack([theta_of(l @ v["phi"], R, N) for l in v["lam"]])
    assert np.linalg.norm(recon - truth) < 0.1 * np.linalg.norm(truth)
    assert v["affine_err"] < 0.2 and v["outside"] < 0.05


def test_svd_start_handles_concentrated_energy():
    """Two unstable ends hold nearly all the energy; the pooled start alone stalls on these data."""
    A = np.asarray(grid_graph(3, 3).adjacency, bool)
    N, K, E = 9, 2, int(A.sum() // 2)
    R = symmetric_basis(A, K)
    rng = np.random.default_rng(4)

    def spectral_radius(ph):
        th = theta_of(ph, R, N)
        comp = np.zeros((K * N, K * N))
        comp[:N] = th
        comp[N:, :N] = np.eye(N)
        return np.abs(np.linalg.eigvals(comp)).max()

    def unstable():
        ph = np.zeros(R.shape[1])
        ph[:N] = rng.uniform(0.3, 0.8, N)
        ph[N:N + E] = rng.standard_normal(E) * 0.25
        ph[N + E:] = rng.standard_normal(N + E) * 0.06
        lo, hi = 0.0, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if spectral_radius(mid * ph) > 1.03 else (mid, hi)
        return lo * ph

    ph0, ph1 = unstable(), unstable()
    grid = np.linspace(0, 1, 5)
    truth = np.stack([theta_of((1 - p) * ph0 + p * ph1, R, N) for p in grid])
    per = []
    for th in truth:
        trials = []
        for _ in range(20):
            X, _ = simulate_gdar([th[:, k * N:(k + 1) * N] for k in range(K)], 200, noise_std=0.3, rng=rng, burn_in=0)
            trials.append(trial_stats(X, K=K))
        per.append(trials)
    st = stack_stats(per)
    tot = {k: st[k].sum(0) for k in st}
    _, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    both = fit_sym_tp(st, R, Sig, 2, grid, starts=("pooled", "svd"))
    pooled = fit_sym_tp(st, R, Sig, 2, grid)
    assert both.objective[-1] <= pooled.objective[-1]
    assert np.linalg.norm(both.theta_grid() - truth) / np.linalg.norm(truth) < 0.05


def test_full_rank_equals_per_condition():
    stats, R, N, K, grid, _, _ = _setup()
    _, Sig = fit_gls_pooled(stats["Gzz"].sum(0), stats["Gzy"].sum(0), stats["Syy"].sum(0),
                            stats["n"].sum(), R)
    m = fit_sym_tp(stats, R, Sig, len(grid), grid, n_iter=200, tol=1e-11)
    for g in range(len(grid)):
        single = {k: stats[k][[g]] for k in stats}
        one = fit_sym_tp(single, R, Sig, 1, grid[[g]], n_iter=1)
        assert np.linalg.norm(m.theta_grid()[g] - one.theta_grid()[0]) < 1e-3 * np.linalg.norm(one.theta_grid()[0])
