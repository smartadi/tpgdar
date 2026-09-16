import numpy as np
import pytest

from gdar_tp.control import (companion_form, synthesize_quadratic_feedback,
                             synthesize_slack_feedback, closed_loop_max_rho,
                             is_schur_stable, _vertices_to_companion)

cp = pytest.importorskip("cvxpy")


def test_companion_form_shapes():
    N, K, M = 3, 2, 2
    A_list = [np.random.randn(N, N) for _ in range(K)]
    B = np.random.randn(N, M)
    Ac, Bc = companion_form(A_list, B)
    assert Ac.shape == (K * N, K * N)
    assert Bc.shape == (K * N, M)
    # subdiagonal identity present
    assert np.allclose(Ac[N:, :-N], np.eye((K - 1) * N))


def test_quadratic_stabilizes_toy_polytope():
    # two unstable 2x2 vertices, full actuation -> should be stabilizable
    A0 = np.array([[1.1, 0.2], [0.0, 0.95]])
    A1 = np.array([[0.9, -0.3], [0.4, 1.15]])
    B = np.eye(2)
    v0 = A0[..., None]  # (N, N, K=1)
    v1 = A1[..., None]
    res = synthesize_quadratic_feedback([v0, v1], B)
    assert res.feasible
    assert res.max_rho < 1.0


def test_slack_variant_also_stabilizes():
    A0 = np.array([[1.1, 0.2], [0.0, 0.95]])
    A1 = np.array([[0.9, -0.3], [0.4, 1.15]])
    B = np.eye(2)
    res = synthesize_slack_feedback([A0[..., None], A1[..., None]], B)
    assert res.feasible
    assert res.max_rho < 1.0


def test_uncontrollable_unstable_is_infeasible():
    # B drives only channel 0; channel 1 is open-loop unstable and decoupled
    A0 = np.array([[0.5, 0.0], [0.0, 1.2]])
    A1 = np.array([[0.4, 0.0], [0.0, 1.3]])
    B = np.array([[1.0], [0.0]])
    res = synthesize_quadratic_feedback([A0[..., None], A1[..., None]], B)
    assert not res.feasible


def test_closed_loop_rho_helper():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])   # nilpotent -> rho 0
    B = np.zeros((2, 1))
    comp, Bc = _vertices_to_companion([A[..., None]], B)
    F = np.zeros((1, 2))
    assert closed_loop_max_rho(comp, Bc, F) < 1e-9
    assert is_schur_stable(A)
