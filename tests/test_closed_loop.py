import numpy as np
import pytest

from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import fit_grid
from gdar_tp.tp import convex_hull
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams
from gdar_tp.sim.closed_loop import run_closed_loop

cp = pytest.importorskip("cvxpy")
from gdar_tp.control import synthesize_quadratic_feedback


def _segment(net, p, seed, n=8000):
    E = net.simulate(n, p=p, noise_std=0.03, rng=seed, burn_in=800)
    return E - E.mean(axis=0, keepdims=True)


@pytest.fixture(scope="module")
def controller():
    g = grid_graph(3, 3)
    net = WCNetwork(g.adjacency, WCParams(g0=3.0))
    mask = support_mask(g.adjacency)
    N = g.n
    p_train = np.linspace(0.0, 1.0, 5)
    segs = [(_segment(net, p, 10 + i), None) for i, p in enumerate(p_train)]
    q = fit_grid(segs, p_train, K=2, support=mask, param="edgewise", ridge=1e-3)
    poly = convex_hull(q.coeffs, n_param_modes=1, method="tight", energy=0.99)
    res = synthesize_quadratic_feedback(poly.vertex_systems, np.eye(N))
    op = net.simulate(12000, p=0.8, noise_std=0.03, rng=99, burn_in=800).mean(axis=0)
    return net, res, np.eye(N), op, poly


def test_polytope_two_vertices(controller):
    _, _, _, _, poly = controller
    assert poly.n_vertices == 2


def test_lmi_feasible_and_schur(controller):
    _, res, _, _, _ = controller
    assert res.feasible
    assert res.max_rho < 1.0


def test_closed_loop_reduces_fluctuation(controller):
    net, res, B, op, _ = controller
    common = dict(network=net, F=res.F, B=B, n_steps=1500, K=2,
                  operating_mean=op, p_schedule=0.8, u_limit=0.5,
                  proc_noise=0.03, obs_noise=0.005, rng=7)
    o = run_closed_loop(controller_on=False, **common)
    c = run_closed_loop(controller_on=True, **common)
    assert c.tracking_error < o.tracking_error   # controller helps on the true plant
    reduction = 1 - c.tracking_error / o.tracking_error
    assert reduction > 0.03


def test_actuator_limit_respected(controller):
    net, res, B, op, _ = controller
    c = run_closed_loop(network=net, F=res.F, B=B, n_steps=800, K=2,
                        operating_mean=op, p_schedule=0.8, u_limit=0.3,
                        proc_noise=0.03, rng=1, controller_on=True)
    assert np.abs(c.U).max() <= 0.3 + 1e-9
