import numpy as np

from gdar_tp.graph import (grid_graph, kron_sum_laplacian, path_laplacian,
                           support_mask, normalized_laplacian)


def test_laplacian_properties():
    g = grid_graph(4, 5)
    L = g.laplacian
    # rows sum to zero
    assert np.allclose(L.sum(axis=1), 0.0)
    # symmetric PSD with a zero eigenvalue (connected -> exactly one)
    assert np.allclose(L, L.T)
    w = np.linalg.eigvalsh(L)
    assert w[0] > -1e-9
    assert abs(w[0]) < 1e-9
    assert w[1] > 1e-9  # algebraic connectivity > 0 (connected)


def test_kron_sum_identity():
    for r, c in [(3, 4), (5, 5), (2, 7)]:
        g = grid_graph(r, c)
        assert np.allclose(g.laplacian, kron_sum_laplacian(r, c))


def test_path_laplacian_singleton():
    assert path_laplacian(1).shape == (1, 1)
    assert np.allclose(path_laplacian(1), 0.0)


def test_support_mask():
    g = grid_graph(3, 3)
    mask = support_mask(g.adjacency)
    assert mask.dtype == bool
    assert np.all(np.diag(mask))            # diagonal always allowed
    # corner (0,0) neighbours are (0,1)->idx1 and (1,0)->idx3
    assert mask[0, 1] and mask[0, 3]
    assert not mask[0, 2]                    # non-neighbour


def test_normalized_laplacian_spectrum():
    g = grid_graph(4, 4)
    Ln = normalized_laplacian(g.laplacian)
    w = np.linalg.eigvalsh(Ln)
    assert w.min() > -1e-9 and w.max() < 2.0 + 1e-9  # eigenvalues in [0, 2]


def test_edges_count_grid():
    # 4-neighbour grid r x c has r(c-1) + c(r-1) edges
    r, c = 4, 6
    g = grid_graph(r, c)
    assert len(g.edges) == r * (c - 1) + c * (r - 1)
