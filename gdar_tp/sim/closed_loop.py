"""Closed-loop simulation: polytopic state feedback on the WC plant.

Crucially the controller is run on the *nonlinear Wilson-Cowan simulator*, never
on the fitted GDAR model (risk row 4 in the plan).  The controller sees a
delayed, noisy observation, estimates the scheduling variable online (here it is
supplied), builds the stacked companion state from recent history, and applies

    u(t) = clip( F (xi(t) - xi_ref) , +/- u_limit )

mapped to channels through ``B``.  We regulate the *centered* activity (deviation
from an operating point) toward a target pattern -- e.g. zero, to suppress a
pathological mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClosedLoopResult:
    E: np.ndarray                 # (n_steps, N) observed activity
    U: np.ndarray                 # (n_steps, M) applied inputs
    centered: np.ndarray          # (n_steps, N) activity minus operating mean
    tracking_error: float         # RMS of (centered - target) over the run
    meta: dict = field(default_factory=dict)


def run_closed_loop(network, F, B, n_steps, K, operating_mean,
                    p_schedule=0.5, target=None, u_limit=None,
                    delay=0, obs_noise=0.0, proc_noise=0.0,
                    controller_on=True, warmup=200, rng=None, x0=None):
    """Run the WC plant under polytopic state feedback.

    Parameters
    ----------
    network : WCNetwork (the plant).
    F : (M, K*N) feedback gain from ``control.synthesize_*``.
    B : (N, M) input matrix (u -> channel drive).
    operating_mean : (N,) activity operating point to center about.
    p_schedule : float or (n_steps,) array, the true scheduling signal.
    target : (N,) desired *centered* pattern (default zeros = suppress).
    u_limit : float or None, per-channel actuator clip on ``u``.
    delay : int, observation/actuation loop delay in steps.
    obs_noise, proc_noise : float.
    controller_on : bool, if False runs open loop (u = 0) for comparison.
    warmup : int, uncontrolled steps to settle before logging.

    Returns
    -------
    ClosedLoopResult
    """
    rng = np.random.default_rng(rng)
    N = network.n
    M = B.shape[1]
    operating_mean = np.asarray(operating_mean, dtype=float)
    target = np.zeros(N) if target is None else np.asarray(target, dtype=float)
    p_arr = np.broadcast_to(np.asarray(p_schedule, float), (n_steps,)) \
        if np.ndim(p_schedule) == 0 else np.asarray(p_schedule, float)

    # init plant state
    if x0 is None:
        E = operating_mean + 0.01 * rng.standard_normal(N)
        I = 0.1 + 0.01 * rng.standard_normal(N)
    else:
        E, I = np.array(x0[0], float), np.array(x0[1], float)

    # warmup uncontrolled to reach the attractor
    for _ in range(warmup):
        g = network.coupling_gain(float(np.atleast_1d(p_arr)[0]))
        E, I = network.step(E, I, g, np.zeros(N), proc_noise, rng)

    hist = [E - operating_mean for _ in range(K)]   # centered history buffer
    obs_buf = []                                     # for observation delay
    E_log = np.zeros((n_steps, N))
    U_log = np.zeros((n_steps, M))
    cen_log = np.zeros((n_steps, N))

    xi_ref = np.tile(target, K)

    for t in range(n_steps):
        g = network.coupling_gain(p_arr[t])

        # delayed, noisy observation feeds the controller
        obs = E + (obs_noise * rng.standard_normal(N) if obs_noise else 0.0)
        obs_buf.append(obs)
        obs_used = obs_buf[max(0, len(obs_buf) - 1 - delay)]
        cen_obs = obs_used - operating_mean
        hist.append(cen_obs)
        hist.pop(0)

        if controller_on and F is not None:
            xi = np.concatenate(hist[::-1])          # [x(t), x(t-1), ...]
            u = F @ (xi - xi_ref)
            if u_limit is not None:
                u = np.clip(u, -u_limit, u_limit)
        else:
            u = np.zeros(M)

        drive = B @ u
        E, I = network.step(E, I, g, drive, proc_noise, rng)

        E_log[t] = E
        U_log[t] = u
        cen_log[t] = E - operating_mean

    err = np.sqrt(np.mean((cen_log - target) ** 2))
    return ClosedLoopResult(E=E_log, U=U_log, centered=cen_log,
                            tracking_error=float(err),
                            meta={"delay": delay, "controller_on": controller_on})
