"""Stage 5 -- in-silico control: polytopic LMI state feedback on the WC plant.

Fit a parameter-varying GDAR model, reduce it to a 2-vertex polytope (Stage 2),
synthesize a single quadratically-stabilizing state-feedback gain via LMIs, then
close the loop on the *nonlinear Wilson-Cowan simulator* (never on the fitted
model).  The controller's job here: suppress the ongoing activity fluctuation
(regulate centered activity toward zero).  Reports open- vs closed-loop error and
a robustness sweep over loop delay.

Run:  python scripts/05_control_insilico.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdar_tp import viz
from gdar_tp.control import synthesize_quadratic_feedback, synthesize_slack_feedback
from gdar_tp.gdar import fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import fit_grid
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams
from gdar_tp.sim.closed_loop import run_closed_loop
from gdar_tp.tp import convex_hull

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)
K = 2
G0 = 3.0


def simulate_segment(net, p, seed, n=20000):
    E = net.simulate(n_steps=n, p=p, noise_std=0.03, rng=seed, burn_in=800)
    return E - E.mean(axis=0, keepdims=True)


def main():
    rows, cols = 4, 4
    g = grid_graph(rows, cols)
    net = WCNetwork(g.adjacency, WCParams(g0=G0))
    mask = support_mask(g.adjacency)
    N = g.n

    # --- Stage 2 in brief: qLPV fit -> 2-vertex polytope ---
    p_train = np.linspace(0.0, 1.0, 7)
    segments = [(simulate_segment(net, p, 10 + i), None) for i, p in enumerate(p_train)]
    qmodel = fit_grid(segments, p_train, K=K, support=mask, param="edgewise", ridge=1e-3)
    poly = convex_hull(qmodel.coeffs, n_param_modes=1, method="tight", energy=0.99)
    print("=== Stage 5: in-silico polytopic LMI control ===")
    print(f"grid {rows}x{cols}={N} ch, K={K} -> polytope with {poly.n_vertices} vertices")

    # --- actuator model: full-array stimulation (B = I) ---
    B_act = np.eye(N)

    # --- LMI synthesis over the polytope ---
    res = synthesize_quadratic_feedback(poly.vertex_systems, B_act)
    print(f"quadratic LMI: feasible={res.feasible}, worst closed-loop rho={res.max_rho:.4f} "
          f"({res.status})")
    if not res.feasible:
        res = synthesize_slack_feedback(poly.vertex_systems, B_act)
        print(f"slack LMI:     feasible={res.feasible}, worst rho={res.max_rho:.4f}")
    F = res.F

    # --- operating point at the control condition ---
    p_ctrl = 0.8
    E_op = net.simulate(n_steps=20000, p=p_ctrl, noise_std=0.03, rng=99, burn_in=800)
    op_mean = E_op.mean(axis=0)
    u_limit = 0.5

    # --- open vs closed loop on the true nonlinear plant ---
    common = dict(network=net, F=F, B=B_act, n_steps=4000, K=K,
                  operating_mean=op_mean, p_schedule=p_ctrl, u_limit=u_limit,
                  proc_noise=0.03, obs_noise=0.005, rng=7)
    open_run = run_closed_loop(controller_on=False, **common)
    closed_run = run_closed_loop(controller_on=True, **common)
    reduction = 100 * (1 - closed_run.tracking_error / open_run.tracking_error)
    print(f"\nfluctuation RMS (centered activity):")
    print(f"  open loop   : {open_run.tracking_error:.4f}")
    print(f"  closed loop : {closed_run.tracking_error:.4f}   ({reduction:+.1f}%)")
    print(f"  max |u| used : {np.abs(closed_run.U).max():.3f} (limit {u_limit})")

    # --- robustness sweep: loop delay ---
    print(f"\nrobustness to loop delay:")
    print(f"{'delay (steps)':>14}{'closed RMS':>13}{'reduction %':>13}")
    delays, reds = [], []
    for d in [0, 2, 5, 10, 20]:
        r = run_closed_loop(controller_on=True, delay=d,
                            **{k: v for k, v in common.items()})
        red = 100 * (1 - r.tracking_error / open_run.tracking_error)
        delays.append(d); reds.append(red)
        print(f"{d:>14}{r.tracking_error:>13.4f}{red:>13.1f}")

    _figures(open_run, closed_run, delays, reds, rows, cols)
    print(f"\nfigures -> {FIGDIR}")


def _figures(open_run, closed_run, delays, reds, rows, cols):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.linalg.norm(open_run.centered, axis=1)[:800], label="open loop", lw=1)
    axes[0].plot(np.linalg.norm(closed_run.centered, axis=1)[:800], label="closed loop", lw=1)
    axes[0].set_title("fluctuation magnitude ||x - x*||")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("norm"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(delays, reds, "o-")
    axes[1].set_title("disturbance reduction vs loop delay")
    axes[1].set_xlabel("loop delay (steps)"); axes[1].set_ylabel("RMS reduction (%)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "05_closed_loop.png"), dpi=130)


if __name__ == "__main__":
    main()
