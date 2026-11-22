"""TP-GDAR in vertex form (MATH §9.5): the checks Aditya approved on 2026-11-21, before any real-data number.

1. V = 1 reproduces GDAR.  2. Each block step is exact and the objective never increases.
3. At mu = 0 the training data loss sits between the free rank-V fit and the convex form of that fit.
Plus: the penalty pulls the vertices together, and a known two-vertex plant is recovered.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from gdar_tp.graph import grid_graph
from gdar_tp.sym_tp import (VertexTP, fit_gls_pooled, fit_sym_tp, fit_sym_tp_vertex, gls_quads, polytope_vertices,
                            simplex_qp, symmetric_basis, theta_of)
from gdar_tp.tpgdar import stack_stats, trial_stats

G = grid_graph(2, 3)
A = np.asarray(G.adjacency, bool)
N = G.n
K = 2


def _plant(seed=0):
    """Two stable symmetric operators on the graph; Theta(p) = (1 - p) Theta_0 + p Theta_1."""
    rng = np.random.default_rng(seed)
    R = symmetric_basis(A, K)
    out = []
    for _ in range(2):
        while True:
            ph = rng.standard_normal(R.shape[1]) * 0.25
            th = theta_of(ph, R, N)
            Ac = np.zeros((K * N, K * N))
            Ac[:N] = th
            Ac[N:, :-N] = np.eye((K - 1) * N)
            if np.abs(np.linalg.eigvals(Ac)).max() < 0.9:
                out.append(ph)
                break
    return R, out[0], out[1]


def _data(R, ph0, ph1, p_grid, seed=1, trials=20, T=150, noise=0.5):
    rng = np.random.default_rng(seed)
    per = []
    for p in p_grid:
        th = theta_of((1 - p) * ph0 + p * ph1, R, N)
        trial = []
        for _ in range(trials):
            x = np.zeros((T, N))
            for t in range(K - 1, T - 1):
                z = np.concatenate([x[t - k] for k in range(K)])
                x[t + 1] = th @ z + noise * rng.standard_normal(N)
            trial.append(trial_stats(x, K=K))
        per.append(trial)
    st = stack_stats(per)
    tot = {k: st[k].sum(0) for k in st}
    phi_pool, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    return st, phi_pool, Sig


P_GRID = np.linspace(0, 1, 6)


@pytest.fixture(scope="module")
def setup():
    R, ph0, ph1 = _plant()
    st, phi_pool, Sig = _data(R, ph0, ph1, P_GRID)
    return R, ph0, ph1, st, phi_pool, Sig


def _psd(V, rng, rank=None):
    M = rng.standard_normal((V, rank or V))
    return M @ M.T


@pytest.mark.parametrize("V", [2, 3, 4])
def test_simplex_qp_matches_slsqp(V):
    rng = np.random.default_rng(V)
    for case in range(200):
        Q = _psd(V, rng, rank=None if case % 3 else 1)          # every third Q is singular
        b = rng.standard_normal(V) * 2
        lam = simplex_qp(Q, b)
        assert lam.min() >= 0 and abs(lam.sum() - 1) < 1e-12

        def f(x):
            return x @ Q @ x - 2 * b @ x
        best = np.inf
        for x0 in np.eye(V).tolist() + [np.ones(V) / V]:
            r = minimize(f, x0, method="SLSQP", bounds=[(0, 1)] * V,
                         constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"ftol": 1e-12})
            best = min(best, f(np.clip(r.x, 0, None) / np.clip(r.x, 0, None).sum()))
        assert f(lam) <= best + 1e-8


def test_simplex_qp_matches_a_grid_for_three_vertices():
    rng = np.random.default_rng(7)
    g = np.linspace(0, 1, 201)
    a, b_ = np.meshgrid(g, g)
    keep = a + b_ <= 1
    pts = np.stack([a[keep], b_[keep], 1 - a[keep] - b_[keep]], 1)
    for _ in range(50):
        Q, b = _psd(3, rng), rng.standard_normal(3)
        vals = np.einsum("ni,ij,nj->n", pts, Q, pts) - 2 * pts @ b
        lam = simplex_qp(Q, b)
        assert lam @ Q @ lam - 2 * b @ lam <= vals.min() + 1e-12


def test_one_vertex_is_gdar(setup):
    R, _, _, st, phi_pool, Sig = setup
    m = fit_sym_tp_vertex(st, R, Sig, 1, P_GRID)
    assert isinstance(m, VertexTP)
    np.testing.assert_allclose(m.W, 1.0)
    # GDAR's phi from the same Sigma: fit_gls_pooled refits with the Sigma it estimates, which is Sig here
    np.testing.assert_allclose(m.phi[0], phi_pool, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("V, mu", [(2, 0.0), (2, 0.1), (3, 0.0), (3, 0.01)])
def test_objective_never_increases_and_weights_stay_convex(setup, V, mu):
    R, _, _, st, _, Sig = setup
    m = fit_sym_tp_vertex(st, R, Sig, V, P_GRID, mu=mu)
    assert np.all(np.diff(m.objective) <= 1e-9 * np.abs(m.objective[:-1]))
    assert m.W.min() >= 0
    np.testing.assert_allclose(m.W.sum(1), 1.0, atol=1e-12)
    assert m.n_params == V * R.shape[1] + len(P_GRID) * (V - 1)


@pytest.mark.parametrize("mu", [0.0, 0.05])
def test_phi_step_is_the_exact_minimiser(setup, mu):
    """At the end the vertices minimise the penalised objective for the final weights: its gradient is zero."""
    R, _, _, st, _, Sig = setup
    V = 2
    m = fit_sym_tp_vertex(st, R, Sig, V, P_GRID, mu=mu, n_iter=5, tol=0.0)
    Hs, hs = gls_quads(st, R, Sig)
    lam, phi = m.W, m.phi
    C = np.eye(V) - 1.0 / V
    grad = np.zeros_like(phi)
    for g in range(len(P_GRID)):
        f = lam[g] @ phi
        grad += np.outer(lam[g], 2 * (Hs[g] @ f - hs[g]))
    grad += 2 * mu * C @ phi @ Hs.sum(0)
    scale = np.abs(hs).sum()
    assert np.abs(grad).max() < 1e-7 * scale


def test_data_loss_between_free_fit_and_its_convex_form(setup):
    R, _, _, st, _, Sig = setup
    Hs, hs = gls_quads(st, R, Sig)

    def data_loss(lam, phi):
        return sum((lam[g] @ phi) @ Hs[g] @ (lam[g] @ phi) - 2 * (lam[g] @ phi) @ hs[g] for g in range(len(P_GRID)))
    for V in (2, 3):
        free = fit_sym_tp(st, R, Sig, V, P_GRID, quads=(Hs, hs))
        poly = polytope_vertices(free, Hs.sum(0) / st["n"].sum())
        m = fit_sym_tp_vertex(st, R, Sig, V, P_GRID, mu=0.0, start=(poly["lam"], poly["phi"]), quads=(Hs, hs))
        free_loss = data_loss(free.W, free.phi)
        convex_loss = data_loss(poly["lam"], poly["phi"])
        assert free_loss <= m.data_loss + 1e-9 * abs(free_loss)
        assert m.data_loss <= convex_loss + 1e-9 * abs(convex_loss)


def test_penalty_pulls_the_vertices_together(setup):
    R, _, _, st, _, Sig = setup
    Hsum = gls_quads(st, R, Sig)[0].sum(0)
    spread = []
    for mu in (0.0, 0.01, 0.1, 1.0):
        m = fit_sym_tp_vertex(st, R, Sig, 3, P_GRID, mu=mu)
        d = m.phi - m.phi.mean(0)
        spread.append(np.trace(d @ Hsum @ d.T))
    assert all(a >= b - 1e-9 * a for a, b in zip(spread, spread[1:]))


def test_recovers_a_two_vertex_plant(setup):
    """The grid visits both true vertices (p = 0 and 1), so the tightest segment is the truth."""
    R, ph0, ph1, st, _, Sig = setup
    m = fit_sym_tp_vertex(st, R, Sig, 2, P_GRID, mu=1e-4)
    truth = np.stack([ph0, ph1])
    err = min(np.linalg.norm(m.phi[list(o)] - truth) for o in ((0, 1), (1, 0))) / np.linalg.norm(truth)
    assert err < 0.1
    lam1 = m.W[:, 1] if np.linalg.norm(m.phi[1] - ph1) < np.linalg.norm(m.phi[0] - ph1) else m.W[:, 0]
    assert np.abs(lam1 - P_GRID).max() < 0.1
