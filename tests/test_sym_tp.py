"""Symmetric TP-GDAR: same model class as Felix's GDAR, same ALS guarantees."""

import numpy as np

from gdar_tp.graph import grid_graph
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.sym_tp import (_quad, _solve, fit_gls_pooled, fit_sym_tp, gls_quads, polytope_vertices,
                            symmetric_basis, theta_of)
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


def _kronecker_lu_fit(stats, R, Sigma, r, jitter=1e-10, n_iter=60, tol=1e-7):
    """The fit as first written, Kronecker assembly and LU, pooled start: what fit_sym_tp must keep matching."""
    Sinv = np.linalg.pinv(Sigma)
    P = len(stats["n"])
    quads = [_quad(stats["Gzz"][g], stats["Gzy"][g], R, Sinv) for g in range(P)]
    Hs, hs = np.stack([q[0] for q in quads]), np.stack([q[1] for q in quads])
    n_phi = hs.shape[1]

    def lu(H, h):
        return np.linalg.solve(H + jitter * np.trace(H) / len(H) * np.eye(len(H)), h)

    def objective(W, phi):
        return sum((W[g] @ phi) @ Hs[g] @ (W[g] @ phi) - 2 * (W[g] @ phi) @ hs[g] for g in range(P))

    Hbar, hbar = Hs.sum(0), hs.sum(0)
    phi0 = lu(Hbar, hbar)
    phi = np.zeros((r, n_phi))
    phi[0] = phi0
    if r > 1:
        L = np.linalg.cholesky(Hbar + jitter * np.trace(Hbar) / n_phi * np.eye(n_phi))
        dev = np.stack([lu(Hs[g], hs[g]) - phi0 for g in range(P)])
        phi[1:] = np.linalg.solve(L.T, np.linalg.svd(dev @ L, full_matrices=False)[2][:r - 1].T).T
    W = np.zeros((P, r))
    W[:, 0] = 1.0
    obj = [objective(W, phi)]
    for _ in range(n_iter):
        for g in range(P):
            W[g] = np.linalg.lstsq(phi @ Hs[g] @ phi.T, phi @ hs[g], rcond=1e-12)[0]
        A = sum(np.kron(np.outer(W[g], W[g]), Hs[g]) for g in range(P))
        b = sum(np.kron(W[g], hs[g]) for g in range(P))
        phi = lu(A, b).reshape(r, n_phi)
        scale = np.linalg.norm(W, axis=0) / np.sqrt(P)
        W, phi = W / scale, phi * scale[:, None]
        obj.append(objective(W, phi))
        if abs(obj[-2] - obj[-1]) <= tol * abs(obj[-1]):
            break
    return np.stack([theta_of(W[g] @ phi, R, Sigma.shape[0]) for g in range(P)]), np.array(obj)


def _assert_matches_first_fit(stats, R, Sig, r, grid):
    want, obj = _kronecker_lu_fit(stats, R, Sig, r)
    got = fit_sym_tp(stats, R, Sig, r, grid)
    assert np.linalg.norm(got.theta_grid() - want) <= 1e-5 * np.linalg.norm(want)
    assert len(got.objective) == len(obj)
    assert np.allclose(got.objective, obj, rtol=1e-8, atol=0)


def test_block_cholesky_fit_matches_the_kronecker_lu_fit():
    stats, R, _, _, grid, _, _ = _setup()
    tot = {k: stats[k].sum(0) for k in stats}
    _, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    for r in (1, 2, 3):
        _assert_matches_first_fit(stats, R, Sig, r, grid)


def test_shared_quads_give_the_same_fit():
    stats, R, _, _, grid, _, _ = _setup()
    tot = {k: stats[k].sum(0) for k in stats}
    _, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    alone = fit_sym_tp(stats, R, Sig, 2, grid)
    shared = fit_sym_tp(stats, R, Sig, 2, grid, quads=gls_quads(stats, R, Sig))
    assert np.array_equal(alone.theta_grid(), shared.theta_grid())


def test_solve_falls_back_to_lu_when_not_positive_definite():
    H = np.array([[1.0, 2.0], [2.0, 1.0]])
    h = np.array([1.0, 0.0])
    assert np.allclose(_solve(H, h, 0.0), np.linalg.solve(H, h))


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


def test_inputs_at_chosen_nodes():
    A = np.asarray(grid_graph(3, 3).adjacency, bool)
    R = symmetric_basis(A, 2, node_inputs=[0, 4])
    assert R.shape == (9 * (2 * 9 + 2), 2 * (9 + A.sum() // 2) + 2)
    th = theta_of(np.random.default_rng(3).standard_normal(R.shape[1]), R, 9)
    B = th[:, 18:]
    assert B.shape == (9, 2) and B[0, 0] != 0 and B[4, 1] != 0
    assert np.count_nonzero(B) == 2
    assert R.shape == symmetric_basis(A, 2, node_inputs=np.array([0, 4])).shape


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
    _assert_matches_first_fit(st, R, Sig, 2, grid)  # the worst-conditioned case we have


def test_full_rank_equals_per_condition():
    stats, R, N, K, grid, _, _ = _setup()
    _, Sig = fit_gls_pooled(stats["Gzz"].sum(0), stats["Gzy"].sum(0), stats["Syy"].sum(0),
                            stats["n"].sum(), R)
    m = fit_sym_tp(stats, R, Sig, len(grid), grid, n_iter=200, tol=1e-11)
    for g in range(len(grid)):
        single = {k: stats[k][[g]] for k in stats}
        one = fit_sym_tp(single, R, Sig, 1, grid[[g]], n_iter=1)
        assert np.linalg.norm(m.theta_grid()[g] - one.theta_grid()[0]) < 1e-3 * np.linalg.norm(one.theta_grid()[0])
