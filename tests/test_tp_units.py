"""Unit tests for the internals of gdar_tp/tp.py (HOSVD helpers and the hull builders)."""

import warnings

import numpy as np
import pytest

from gdar_tp.tp import (Polytope, TPModel, _barycentric, _box_vertices, _mode_mult, _spa_vertices,
                        _tight_hull_vertices, _unfold, convex_hull, hosvd, weight_functions)

RNG = np.random.default_rng(0)


def _square_cloud(n_inner=12, seed=1):
    """4 corners of the unit square (indices 0..3) plus interior points."""
    rng = np.random.default_rng(seed)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return np.vstack([corners, 0.1 + 0.8 * rng.random((n_inner, 2))])


def test_unfold_and_mode_product_identity():
    T = RNG.standard_normal((3, 4, 5))
    M = RNG.standard_normal((6, 4))
    out = _mode_mult(T, M, 1)
    assert out.shape == (3, 6, 5)
    assert np.allclose(_unfold(out, 1), M @ _unfold(T, 1))
    assert _unfold(T, 2).shape == (5, 12)


def test_tpmodel_reconstruct_vertices_and_count():
    tensor = RNG.standard_normal((4, 5, 2, 2))
    tp = hosvd(tensor, 2, energy=1.0)
    assert isinstance(tp, TPModel) and tp.ranks == [4, 5] and tp.n_vertices == 20
    assert tp.vertex_systems().shape == (20, 2, 2) and tp.sys_shape == (2, 2)
    assert np.allclose(tp.reconstruct(), tensor)


def test_hosvd_max_rank_caps_every_mode():
    tp = hosvd(RNG.standard_normal((4, 5, 3)), 2, max_rank=2)
    assert tp.ranks == [2, 2] and [f.shape for f in tp.factors] == [(4, 2), (5, 2)]
    assert all(np.allclose(f.T @ f, np.eye(2)) for f in tp.factors)


def test_weight_functions_are_copies():
    tp = hosvd(RNG.standard_normal((4, 3)), 1)
    w = weight_functions(tp)
    w[0][:] = 0.0
    assert not np.allclose(tp.factors[0], 0.0)


def test_barycentric_is_exact_inside_a_simplex():
    Yv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    lam_true = RNG.dirichlet(np.ones(3), size=20)
    pts = lam_true @ Yv
    lam = _barycentric(Yv, pts)
    assert np.all(lam >= 0) and np.allclose(lam.sum(1), 1.0)
    assert np.allclose(lam @ Yv, pts, atol=1e-6) and np.allclose(lam, lam_true, atol=1e-6)


def test_tight_hull_vertices_square_and_1d():
    Y = _square_cloud()
    assert sorted(int(i) for i in _tight_hull_vertices(Y)) == [0, 1, 2, 3]
    y1 = np.array([[0.3], [-2.0], [5.0], [1.0]])
    assert sorted(_tight_hull_vertices(y1)) == [1, 2]


def test_box_vertices_are_the_grid_points_nearest_the_corners():
    Y = _square_cloud()
    assert sorted(int(i) for i in _box_vertices(Y)) == [0, 1, 2, 3]


def test_spa_vertices_find_simplex_corners_from_mixtures():
    corners = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    Y = np.vstack([RNG.dirichlet(np.ones(3), size=30) @ corners, corners])
    assert sorted(_spa_vertices(Y, 3)) == [30, 31, 32]


def test_convex_hull_of_a_constant_tensor_is_one_vertex():
    poly = convex_hull(np.ones((5, 2, 2)), 1)
    assert isinstance(poly, Polytope) and poly.n_vertices == 1
    assert poly.affine_dim == 0 and np.all(poly.weights == 1.0) and poly.residual == 0.0


def test_convex_hull_fields_and_reconstruction():
    pts = _square_cloud()
    poly = convex_hull(pts, 1, method="tight")              # sys_shape (2,), 16 grid points
    assert poly.grid_shape == (16,) and poly.affine_dim == 2
    assert poly.vertex_grid_index.tolist() == [0, 1, 2, 3]
    assert np.array_equal(poly.vertex_systems, pts[:4])
    assert np.allclose(poly.weights.sum(1), 1.0) and np.all(poly.weights >= 0)
    assert poly.residual < 1e-6


def test_convex_hull_max_vertices_prunes_and_warns():
    with pytest.warns(UserWarning, match="do not contain"):
        poly = convex_hull(_square_cloud(), 1, method="tight", max_vertices=3)
    assert poly.n_vertices == 3


def test_convex_hull_rejects_bad_method_arguments():
    with pytest.raises(ValueError):
        convex_hull(_square_cloud(), 1, method="nope")
    with pytest.raises(ValueError):
        convex_hull(_square_cloud(), 1, method="spa")       # spa needs n_vertices


def test_convex_hull_spa_contains_a_simplex_trajectory():
    corners = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    pts = np.vstack([corners, RNG.dirichlet(np.ones(3), size=10) @ corners])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        poly = convex_hull(pts, 1, method="spa", n_vertices=3)
    assert poly.vertex_grid_index.tolist() == [0, 1, 2]
