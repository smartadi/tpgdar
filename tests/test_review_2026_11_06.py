"""Bugs found in the 2026-11-06 code review, pinned as strict xfails.

Each test states the behaviour the code should have.  They fail today; when a fix lands the test passes,
strict mode turns that into a failure, and the xfail marker is removed in the fixing commit.  Fixes to
gdar_tp/ go through the method gate, so none is applied here.
"""

import numpy as np
import pytest

from gdar_tp import metrics
from gdar_tp.control import _vertices_to_companion
from gdar_tp.gdar import GDARModel
from gdar_tp.qlpv import fit_grid
from gdar_tp.sim.wilson_cowan import WCNetwork
from gdar_tp.sym_tp import SymTP, _enclosing_triangle, polytope_vertices, symmetric_basis
from gdar_tp.tpgdar import stack_stats, trial_stats

xfail = pytest.mark.xfail(strict=True)


@xfail
def test_R1_vnn_vertex_array_is_not_silently_misread():
    """control: a (V, N, N) array (K = 1) is iterated as N 'lags' of length-N rows, giving an N^2 companion."""
    V = np.random.default_rng(0).standard_normal((2, 3, 3))
    try:
        comp, _ = _vertices_to_companion(V, np.ones((3, 1)))
    except ValueError:
        return
    assert comp[0].shape == (3, 3)


@xfail
def test_R2_symtp_theta_at_with_an_unsorted_grid():
    """sym_tp: SymTP.theta_at passes p_grid to np.interp unsorted (TPGDAR.weights_at sorts it)."""
    A = ~np.eye(3, dtype=bool)
    R = symmetric_basis(A, 1)
    phi = np.random.default_rng(1).standard_normal((1, R.shape[1]))
    m = SymTP(W=np.array([[2.0], [0.0], [1.0]]), phi=phi, R=R, N=3, p_grid=np.array([2.0, 0.0, 1.0]))
    assert np.allclose(m.theta_at(2.0), m.theta_grid()[0])


@xfail
def test_R3_enclosing_triangle_exists_for_a_parallelogram_hull():
    """sym_tp: with every side on a hull edge, a parallelogram has no bounded triangle; None reaches
    polytope_vertices (rank 3), which then fails."""
    tri = _enclosing_triangle(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    assert tri is not None and tri.shape == (3, 2)


@xfail
def test_R4_rank_one_polytope_is_finite():
    """sym_tp: rank 1 has zero spread, so 'outside' and 'dist' divide 0 by 0."""
    R = symmetric_basis(~np.eye(3, dtype=bool), 1)
    m = SymTP(W=np.ones((4, 1)), phi=np.ones((1, R.shape[1])), R=R, N=3, p_grid=np.arange(4.0))
    with np.errstate(invalid="ignore"):
        out = polytope_vertices(m, np.eye(R.shape[1]))
    assert np.isfinite(out["outside"])


@xfail
def test_R5_stack_stats_with_an_empty_condition():
    """tpgdar: a grid point with no trials sums to the scalar 0, and the stack fails."""
    s = trial_stats(np.random.default_rng(2).standard_normal((20, 3)), K=1)
    st = stack_stats([[s], []])
    assert st["n"].tolist() == [19.0, 0.0] and np.all(st["Gzz"][1] == 0)


@xfail
def test_R6_fit_grid_accepts_bare_arrays_like_fit_grid_2d():
    """qlpv: fit_grid unpacks each segment as (X, U); fit_grid_2d also takes a bare X."""
    X = np.random.default_rng(3).standard_normal((50, 3))
    q = fit_grid([X, X], [0.0, 1.0], K=1)
    assert q.coeffs.shape == (2, 3, 3, 1)


@xfail
def test_R7_wc_step_with_an_integer_seed_does_not_repeat_its_noise():
    """sim.wilson_cowan: step() re-seeds from an int on every call, so the 'noise' is the same each step."""
    net = WCNetwork(np.ones((3, 3)) - np.eye(3))
    z = np.zeros(3)
    try:
        a, _ = net.step(z, z, 1.0, z, noise_std=1.0, rng=7)
        b, _ = net.step(z, z, 1.0, z, noise_std=1.0, rng=7)
    except TypeError:
        return
    assert not np.allclose(a, b)


@xfail
def test_R8_multi_step_nrmse_uses_every_valid_start():
    """metrics: starts run to T - 2 - horizon, one short of the last start whose target is in the data."""
    calls = []

    class Spy(GDARModel):
        def rollout(self, x_hist, n_steps, U=None):
            calls.append(1)
            return np.zeros((n_steps, 2))

    m = Spy(A=[np.eye(2)], B=np.zeros((2, 0)), K=1, param="edgewise", support=np.ones((2, 2), bool))
    metrics.multi_step_nrmse(m, np.random.default_rng(4).standard_normal((20, 2)), horizon=3)
    assert len(calls) == 20 - 3
