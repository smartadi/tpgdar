"""Unit tests for gdar_tp/sim/*, gdar_tp/viz.py and gdar_tp/io.py."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gdar_tp import io, viz  # noqa: E402
from gdar_tp.sim.closed_loop import ClosedLoopResult, run_closed_loop  # noqa: E402
from gdar_tp.sim.linear import simulate_gdar, stable_diffusion_coeffs  # noqa: E402
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams, sigmoid  # noqa: E402

ADJ = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)


# ------------------------------------------------------------------------- linear plant
def test_stable_diffusion_coeffs():
    L = np.array([[1.0, -1.0], [-1.0, 1.0]])
    A = stable_diffusion_coeffs(L, [0.5, 0.2], [0.1, 0.0], k_lags=2)
    assert np.allclose(A[0], 0.5 * np.eye(2) - 0.1 * L) and np.allclose(A[1], 0.2 * np.eye(2))


def test_simulate_gdar_noise_free_recursion_and_input_timing():
    rng = np.random.default_rng(0)
    A = [0.3 * rng.standard_normal((2, 2)) for _ in range(2)]
    B, U = rng.standard_normal((2, 1)), rng.standard_normal((6, 1))
    x0 = rng.standard_normal((2, 2))
    X, U_out = simulate_gdar(A, 6, x0=x0, B=B, U=U)
    assert np.allclose(X[0], A[0] @ x0[-1] + A[1] @ x0[-2])            # no input before t = 0
    assert np.allclose(X[1], A[0] @ X[0] + A[1] @ x0[-1] + B @ U[0])
    for t in range(2, 6):
        assert np.allclose(X[t], A[0] @ X[t - 1] + A[1] @ X[t - 2] + B @ U[t - 1])
    assert np.array_equal(U_out, U)


def test_simulate_gdar_burn_in_and_validation():
    A = [0.5 * np.eye(2)]
    U = np.ones((15, 1))
    X, U_out = simulate_gdar(A, 10, B=np.ones((2, 1)), U=U, burn_in=5)
    assert X.shape == (10, 2) and np.array_equal(U_out, U[5:15])
    X0, U0 = simulate_gdar(A, 4)
    assert np.all(X0 == 0) and U0.shape == (4, 0)
    with pytest.raises(ValueError):
        simulate_gdar(A, 10, x0=np.zeros((2, 2)))            # K = 1 needs (1, N)
    with pytest.raises(ValueError):
        simulate_gdar(A, 10, B=np.ones((2, 1)), U=np.ones((5, 1)))


# ------------------------------------------------------------------------ Wilson-Cowan
def test_sigmoid():
    assert sigmoid(4.0, a=1.3, theta=4.0) == 0.5
    x = np.linspace(-5, 5, 11)
    assert np.all(np.diff(sigmoid(x)) > 0) and np.allclose(sigmoid(x) + sigmoid(-x), 1.0)


def test_coupling_is_row_stochastic_and_gain_is_affine():
    net = WCNetwork(np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]], float), WCParams(g0=2.0))
    assert np.allclose(net.W.sum(1), [1, 1, 0]) and net.n == 3
    assert np.allclose(net.coupling_gain([0.0, 1.0]), [1.0, 3.0])
    raw = WCNetwork(ADJ, normalize_coupling=False)
    assert np.array_equal(raw.W, ADJ)


def test_simulate_equals_repeated_steps_with_lagged_drive():
    rng = np.random.default_rng(1)
    net = WCNetwork(ADJ, WCParams(g0=3.0))
    x0 = (0.1 + 0.05 * rng.random(3), 0.1 + 0.05 * rng.random(3))
    stim = 0.5 * rng.standard_normal((8, 3))
    E_sim, I_sim = net.simulate(8, p=0.3, stim=stim, burn_in=0, x0=x0, return_full=True)
    E, I = x0
    for t in range(8):
        E, I = net.step(E, I, net.coupling_gain(0.3), stim[t - 1] if t else 0.0)
        assert np.allclose(E_sim[t], E) and np.allclose(I_sim[t], I)


def test_simulate_input_through_B_equals_direct_stim():
    rng = np.random.default_rng(2)
    net = WCNetwork(ADJ)
    B, U = rng.standard_normal((3, 2)), rng.standard_normal((30, 2))
    a = net.simulate(20, B=B, U=U, burn_in=10, rng=5)
    b = net.simulate(20, stim=U @ B.T, burn_in=10, rng=5)
    assert np.allclose(a, b)


def test_simulate_validation_and_reproducibility():
    net = WCNetwork(ADJ)
    with pytest.raises(ValueError):
        net.simulate(10, p=np.zeros(5), burn_in=10)
    a = net.simulate(30, noise_std=0.01, rng=7, burn_in=5)
    b = net.simulate(30, noise_std=0.01, rng=7, burn_in=5)
    c = net.simulate(30, noise_std=0.01, rng=8, burn_in=5)
    assert np.array_equal(a, b) and not np.allclose(a, c)


# ------------------------------------------------------------------------- closed loop
def _loop(**kw):
    net = WCNetwork(ADJ, WCParams(g0=3.0))
    args = dict(F=0.3 * np.array([[1.0, -0.5, 0.2]]), B=np.ones((3, 1)), n_steps=40, K=1,
                operating_mean=np.full(3, 0.2), warmup=20, rng=0)
    args.update(kw)
    return run_closed_loop(net, **args)


def test_open_loop_applies_no_input():
    res = _loop(controller_on=False)
    assert isinstance(res, ClosedLoopResult) and np.all(res.U == 0)
    assert res.meta == {"delay": 0, "controller_on": False}


def test_feedback_uses_the_state_before_each_step_and_the_delay():
    op = np.full(3, 0.2)
    for delay in (0, 2):
        res = _loop(delay=delay)
        F = 0.3 * np.array([[1.0, -0.5, 0.2]])
        for t in range(delay + 1, 40):
            assert np.allclose(res.U[t], F @ (res.E[t - 1 - delay] - op))


def test_tracking_error_and_actuator_clip():
    target = np.array([0.01, 0.0, -0.01])
    res = _loop(target=target, u_limit=0.05, F=5.0 * np.array([[1.0, -0.5, 0.2]]))
    assert np.isclose(res.tracking_error, np.sqrt(np.mean((res.centered - target) ** 2)))
    assert np.abs(res.U).max() <= 0.05 + 1e-12
    assert np.allclose(res.centered, res.E - 0.2)


# --------------------------------------------------------------------------- viz and io
def test_every_plot_helper_draws_on_an_axis():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5)
    assert viz.plot_array_map(np.arange(6), 2, 3, ax=axs[0], title="t") is axs[0]
    assert axs[0].images[0].get_array().shape == (2, 3)
    assert viz.plot_singular_spectrum([4.0, 2.0, 1.0], ax=axs[1]) is axs[1]
    assert np.allclose(axs[1].lines[0].get_ydata(), [1.0, 0.5, 0.25])
    assert len(viz.plot_weight_functions([0, 1, 2], np.ones((3, 2)), ax=axs[2]).lines) == 2
    assert viz.plot_coupling(np.eye(3), ax=axs[3]) is axs[3]
    assert len(viz.plot_traces({"a": [1, 2], "b": [3, 4]}, ax=axs[4]).lines) == 2
    assert viz.plot_traces({"a": [1, 2]}) is not None                 # makes its own axis
    plt.close("all")


def test_io_loader_is_still_a_placeholder():
    with pytest.raises(NotImplementedError):
        io.make_synthetic_recording()
