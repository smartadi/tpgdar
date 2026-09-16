"""Stage 2 -- parameter-varying fit + TP (HOSVD) decomposition on synthetic data.

A Wilson-Cowan network whose global coupling varies with a 1-D scheduling
variable ``p``.  We fit a GDAR model at each ``p`` on a grid, stack the
coefficients into a tensor, run HOSVD over the parameter mode (interpretable
vertex systems + weighting functions), build the convex-hull polytope, and check
that the low-dimensional TP model predicts held-out ``p`` values.

Run:  python scripts/02_synthetic_tp.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdar_tp import metrics, viz
from gdar_tp.gdar import GDARModel, fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import fit_grid
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams
from gdar_tp.tp import convex_hull, hosvd

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)
K = 2
# g0=3.0 puts the network coupling on the same footing as the local E-I circuit,
# so the scheduling variable genuinely restructures the effective dynamics (a
# clean rank-2 parameter dependence) rather than being a ~6% perturbation.
G0 = 3.0
SEG_LEN = 20000


def simulate_segment(net, p, seed, n=SEG_LEN):
    """Simulate a WC segment at fixed p, return mean-centered activity."""
    E = net.simulate(n_steps=n, p=p, noise_std=0.03, rng=seed, burn_in=800)
    return E - E.mean(axis=0, keepdims=True)


def fit_at(net, p, mask, seed):
    X = simulate_segment(net, p, seed)
    return fit_gdar(X, K=K, support=mask, param="edgewise", ridge=1e-3)


def main():
    rows, cols = 5, 5
    g = grid_graph(rows, cols)
    net = WCNetwork(g.adjacency, WCParams(g0=G0))
    mask = support_mask(g.adjacency)

    # --- training grid of p ---
    p_train = np.linspace(0.0, 1.0, 7)
    segments = [(simulate_segment(net, p, seed=10 + i), None) for i, p in enumerate(p_train)]
    qmodel = fit_grid(segments, p_train, K=K, support=mask, param="edgewise", ridge=1e-3)

    # --- HOSVD over the parameter mode ---
    tp = hosvd(qmodel.coeffs, n_param_modes=1, energy=0.99)
    print("=== Stage 2: parameter-varying fit + TP decomposition ===")
    print(f"grid {rows}x{cols}={g.n} ch, K={K}, p-grid={np.round(p_train,2)}, seg_len={SEG_LEN}")
    s = tp.singular_values[0]
    print(f"HOSVD singular values (param mode): {np.round(s[:6] / s[0], 4)}")
    print(f"  -> dominant mode + one p-mode at {s[1]/s[0]:.1%}, then a noise floor "
          f"({s[2]/s[0]:.1%}): the parameter dependence is rank ~2")

    # --- vertex-system reconstruction error vs. #modes (elbow = true mode count) ---
    print("\nvertex-system reconstruction error vs. retained rank:")
    print(f"{'rank':>6}{'rel. recon error':>20}")
    full_norm = np.linalg.norm(qmodel.coeffs)
    for r in range(1, 6):
        tpr = hosvd(qmodel.coeffs, n_param_modes=1, max_rank=r, energy=1.0)
        err = np.linalg.norm(tpr.reconstruct() - qmodel.coeffs) / full_norm
        print(f"{r:>6}{err:>20.4f}")

    # --- convex hull / polytope (for the control handoff, Stage 5) ---
    poly = convex_hull(qmodel.coeffs, n_param_modes=1, method="tight", energy=0.99)
    per_point_norm = np.sqrt((qmodel.coeffs ** 2).sum(axis=(1, 2, 3))).max()
    rel_resid = poly.residual / per_point_norm
    print(f"\nconvex hull (energy 0.99): {poly.n_vertices} vertices "
          f"(affine dim {poly.affine_dim}), rel. trajectory residual {rel_resid:.2e}")
    print(f"convex weights sum to one: {np.allclose(poly.weights.sum(1), 1.0)}; "
          f"min weight {poly.weights.min():.3f}")

    # --- held-out p generalization ---
    p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    print(f"\nheld-out p generalization (predict fresh segment at each p):")
    print(f"{'p':>6}{'TP-interp NRMSE':>18}{'oracle NRMSE':>15}")
    tp_errs, or_errs = [], []
    for pt in p_test:
        Xtest = simulate_segment(net, pt, seed=500 + int(pt * 100))
        A_interp = interp_coeffs(qmodel.p_grid[0], qmodel.coeffs, pt)  # (N,N,K)
        m_interp = GDARModel(A=[A_interp[..., k] for k in range(K)],
                             B=np.zeros((g.n, 0)), K=K, param="edgewise", support=mask)
        m_oracle = fit_gdar(Xtest, K=K, support=mask, param="edgewise", ridge=1e-3)
        e_tp = metrics.one_step_nrmse(m_interp, Xtest)
        e_or = metrics.one_step_nrmse(m_oracle, Xtest)
        tp_errs.append(e_tp); or_errs.append(e_or)
        print(f"{pt:>6.1f}{e_tp:>18.4f}{e_or:>15.4f}")
    print(f"mean gap (TP - oracle): {np.mean(tp_errs) - np.mean(or_errs):+.4f}")

    _figures(tp, poly, qmodel, rows, cols)
    print(f"\nfigures -> {FIGDIR}")


def interp_coeffs(p_grid, coeffs, p_star):
    """Elementwise linear interpolation of the coefficient tensor at p_star."""
    P = coeffs.shape[0]
    flat = coeffs.reshape(P, -1)
    out = np.array([np.interp(p_star, p_grid, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape(coeffs.shape[1:])


def _figures(tp, poly, qmodel, rows, cols):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    viz.plot_singular_spectrum(tp.singular_values[0], ax=axes[0],
                               title="HOSVD spectrum (parameter mode)")
    viz.plot_weight_functions(qmodel.p_grid[0], poly.weights, ax=axes[1],
                              title="convex weight functions w_v(p)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "02_tp_spectrum_weights.png"), dpi=130)

    # vertex-system diffusion maps: row-sum of |off-diagonal A_0| per channel
    V = poly.vertex_systems  # (nv, N, N, K)
    nv = V.shape[0]
    fig2, axes2 = plt.subplots(1, nv, figsize=(4 * nv, 3.6))
    axes2 = np.atleast_1d(axes2)
    for v in range(nv):
        A0 = V[v, :, :, 0].copy()
        np.fill_diagonal(A0, 0.0)
        strength = np.abs(A0).sum(axis=1)
        viz.plot_array_map(strength, rows, cols, ax=axes2[v],
                           title=f"vertex {v}: coupling strength")
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGDIR, "02_vertex_mode_maps.png"), dpi=130)


if __name__ == "__main__":
    main()
