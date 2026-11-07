"""Unit tests for gdar_tp/graph.py and gdar_tp/metrics.py beyond tests/test_graph.py."""

import numpy as np

from gdar_tp import metrics
from gdar_tp.gdar import GDARModel
from gdar_tp.graph import grid_graph, normalized_laplacian, path_laplacian
from gdar_tp.sim.linear import simulate_gdar


# ------------------------------------------------------------------------------ graph
def test_coords_are_row_major():
    g = grid_graph(2, 3)
    c = g.coords()
    assert c.shape == (6, 2) and g.n == 6
    assert c[2].tolist() == [0, 2] and c[4].tolist() == [1, 1]


def test_diagonal_grid_edge_count_and_degrees():
    r, c = 3, 4
    g = grid_graph(r, c, diagonal=True)
    assert len(g.edges) == r * (c - 1) + c * (r - 1) + 2 * (r - 1) * (c - 1)
    deg = g.adjacency.sum(1)
    assert deg[0] == 3 and deg[1 * c + 1] == 8


def test_edges_sorted_unique_and_match_adjacency():
    g = grid_graph(3, 3)
    assert g.edges == sorted(set(g.edges)) and all(i < j for i, j in g.edges)
    assert all(g.adjacency[i, j] == 1 for i, j in g.edges)
    assert len(g.edges) == g.adjacency.sum() / 2


def test_normalized_laplacian_leaves_isolated_node_at_one():
    L = np.zeros((3, 3))
    L[:2, :2] = path_laplacian(2)
    nl = normalized_laplacian(L)
    assert nl[2, 2] == 1.0 and np.all(nl[2, :2] == 0) and np.all(nl[:2, 2] == 0)
    assert np.allclose(nl[:2, :2], [[1, -1], [-1, 1]])


# ---------------------------------------------------------------------------- metrics
def test_nrmse_zero_for_perfect_and_one_for_the_mean():
    true = np.random.default_rng(0).standard_normal((500, 3))
    assert metrics.nrmse(true, true) == 0.0
    assert np.isclose(metrics.nrmse(np.tile(true.mean(0), (500, 1)), true), 1.0)


def test_nrmse_is_a_per_channel_average():
    rng = np.random.default_rng(1)
    true = rng.standard_normal((1000, 2)) * [1.0, 100.0]
    err = rng.standard_normal((1000, 2))
    err = err / err.std(0) * 0.1 * true.std(0)             # 10% of each channel's own spread
    err -= err.mean(0)
    assert np.isclose(metrics.nrmse(true + err, true), 0.1, rtol=1e-3)


def test_one_step_nrmse_wraps_predict_one_step():
    class Fixed:
        def predict_one_step(self, X, U=None):
            return X[:-1], X[1:]
    X = np.random.default_rng(2).standard_normal((50, 2))
    assert metrics.one_step_nrmse(Fixed(), X) == metrics.nrmse(X[:-1], X[1:])


def test_multi_step_nrmse_is_zero_for_the_true_noise_free_model():
    th = 0.3
    A = 0.98 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X, _ = simulate_gdar([A], 200, x0=np.array([[1.0, 0.0]]))
    m = GDARModel(A=[A], B=np.zeros((2, 0)), K=1, param="edgewise", support=np.ones((2, 2), bool))
    for h in (1, 5, 20):
        assert metrics.multi_step_nrmse(m, X, horizon=h) < 1e-10


def test_flow_signal_correlation_mask_diagonal_and_degenerate():
    rng = np.random.default_rng(3)
    true = rng.random((5, 5))
    assert np.isclose(metrics.flow_signal_correlation(true, true), 1.0)
    est = true.copy()
    np.fill_diagonal(est, -100.0)                          # diagonal never counts
    assert np.isclose(metrics.flow_signal_correlation(est, true), 1.0)
    mask = np.zeros((5, 5), bool)
    mask[0, 1:] = True
    est2 = -true.copy()
    est2[mask] = true[mask]                                # right on the mask, anti-correlated elsewhere
    assert np.isclose(metrics.flow_signal_correlation(est2, true, mask=mask), 1.0)
    assert metrics.flow_signal_correlation(est2, true) < 0.5
    assert metrics.flow_signal_correlation(np.ones((5, 5)), true) == 0.0


def test_energy_spectrum_and_rank_at_energy():
    s = [4.0, 3.0]                                         # energies 16/25, 9/25
    assert np.allclose(metrics.energy_spectrum(s), [0.64, 1.0])
    assert metrics.rank_at_energy(s, 0.6) == 1
    assert metrics.rank_at_energy(s, 0.7) == 2
