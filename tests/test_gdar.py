import numpy as np

from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.gdar import fit_gdar, coupling_matrix, GDARModel
from gdar_tp.sim.linear import stable_diffusion_coeffs, simulate_gdar
from gdar_tp import metrics


def _make_data(rows=3, cols=3, T=15000, K=1, seed=0, noise=0.1, a=None, b=None):
    g = grid_graph(rows, cols)
    if a is None:
        a = [0.7] * K
    if b is None:
        b = [0.05] * K
    coeffs = stable_diffusion_coeffs(g.laplacian, a=a, b=b, k_lags=K)
    X, _ = simulate_gdar(coeffs, n_steps=T, noise_std=noise, rng=seed, burn_in=200)
    return g, coeffs, X


def test_edgewise_recovers_coeffs():
    g, coeffs, X = _make_data(K=1)
    mask = support_mask(g.adjacency)
    model = fit_gdar(X, K=1, support=mask, param="edgewise", ridge=1e-8)
    err = np.abs(model.A[0] - coeffs[0]).max()
    assert err < 2e-2, f"max coeff error {err}"


def test_support_constraint_respected():
    g, coeffs, X = _make_data(K=1)
    mask = support_mask(g.adjacency)
    model = fit_gdar(X, K=1, support=mask, param="edgewise", ridge=1e-6)
    # every entry off the support must be exactly zero
    assert np.all(model.A[0][~mask] == 0.0)


def test_diffusion_recovers_ab():
    g, coeffs, X = _make_data(K=1)
    model = fit_gdar(X, K=1, laplacian=g.laplacian, param="diffusion", ridge=1e-8)
    assert abs(model.diffusion["a"][0] - 0.7) < 2e-2
    assert abs(model.diffusion["b"][0] - 0.05) < 1e-2


def test_flow_correlation_beats_shuffle():
    g, coeffs, X = _make_data(K=1)
    mask = support_mask(g.adjacency)
    model = fit_gdar(X, K=1, support=mask, param="edgewise", ridge=1e-6)
    C = coupling_matrix(model)
    true_coupling = 0.05 * g.adjacency   # off-diagonal is b * (edge)
    # correlate over ALL off-diagonal entries: the edge/non-edge contrast is the
    # structure to recover (on-support entries alone are a constant, corr ill-defined)
    corr = metrics.flow_signal_correlation(C, true_coupling)
    assert corr > 0.8


def test_lag2_matches_oracle_prediction():
    # stable lag-2 diffusion coefficients (|a0|+|a1| < 1 per graph mode)
    g, coeffs, X = _make_data(K=2, noise=0.1, a=[0.4, 0.3], b=[0.03, 0.02])
    mask = support_mask(g.adjacency)
    model = fit_gdar(X, K=2, support=mask, param="edgewise", ridge=1e-6)
    # A noise-driven process is only weakly predictable; the right bar is that the
    # fit predicts as well as the true coefficients (the oracle) do.
    oracle = GDARModel(A=coeffs, B=np.zeros((g.n, 0)), K=2, param="edgewise",
                       support=mask)
    fit_err = metrics.one_step_nrmse(model, X)
    oracle_err = metrics.one_step_nrmse(oracle, X)
    assert fit_err <= oracle_err + 0.02


def test_input_matrix_recovery():
    g = grid_graph(3, 3)
    N = g.n
    coeffs = stable_diffusion_coeffs(g.laplacian, a=0.6, b=0.04, k_lags=1)
    B_true = np.zeros((N, 2)); B_true[0, 0] = 1.0; B_true[4, 1] = 0.8
    rng = np.random.default_rng(1)
    U = rng.standard_normal((6200, 2))
    X, Uout = simulate_gdar(coeffs, 6000, B=B_true, U=U, noise_std=0.02,
                            rng=2, burn_in=200)
    mask = support_mask(g.adjacency)
    model = fit_gdar(X, U=Uout, K=1, support=mask, param="edgewise", ridge=1e-8)
    assert np.abs(model.B - B_true).max() < 5e-2
