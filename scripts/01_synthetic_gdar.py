"""Stage 1 -- baseline GDAR on synthetic data.

Generates a linear graph-diffusion process on a grid, fits the structural GDAR
model (edge-wise and strict-diffusion), compares to an unconstrained VAR, and
reports prediction error, flow-signal recovery, and residual whiteness.

Run:  python scripts/01_synthetic_gdar.py
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
from gdar_tp.gdar import coupling_matrix, fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.sim.linear import simulate_gdar, stable_diffusion_coeffs

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def ljung_box(resid, n_lags=20):
    """Ljung-Box Q statistic per channel (larger => more autocorrelation)."""
    T = resid.shape[0]
    Q = []
    for c in range(resid.shape[1]):
        x = resid[:, c] - resid[:, c].mean()
        denom = np.sum(x * x)
        q = 0.0
        for k in range(1, n_lags + 1):
            rk = np.sum(x[k:] * x[:-k]) / denom
            q += rk * rk / (T - k)
        Q.append(T * (T + 2) * q)
    return np.array(Q)


def main():
    rng = 0
    rows, cols = 6, 6
    g = grid_graph(rows, cols)
    N = g.n
    K = 2
    coeffs = stable_diffusion_coeffs(g.laplacian, a=[0.5, 0.25], b=[0.04, 0.02], k_lags=K)
    X, _ = simulate_gdar(coeffs, n_steps=20000, noise_std=0.1, rng=rng, burn_in=500)
    X_train, X_test = X[:16000], X[16000:]
    mask = support_mask(g.adjacency)

    # --- fits ---
    m_edge = fit_gdar(X_train, K=K, support=mask, param="edgewise", ridge=1e-6)
    m_diff = fit_gdar(X_train, K=K, laplacian=g.laplacian, param="diffusion", ridge=1e-6)
    m_var = fit_gdar(X_train, K=K, support=np.ones((N, N), bool), param="edgewise", ridge=1e-6)

    print("=== Stage 1: baseline GDAR on synthetic data ===")
    print(f"grid {rows}x{cols} = {N} channels, K={K}, T_train=16000")
    print(f"{'model':<14}{'1-step NRMSE':>14}{'10-step NRMSE':>15}{'flow corr':>12}{'#params':>10}")
    true_coupling = coupling_matrix_from_coeffs(coeffs)
    for name, m in [("edgewise", m_edge), ("diffusion", m_diff), ("unconstr.VAR", m_var)]:
        n1 = metrics.one_step_nrmse(m, X_test)
        n10 = metrics.multi_step_nrmse(m, X_test, horizon=10)
        C = coupling_matrix(m)
        corr = metrics.flow_signal_correlation(C, true_coupling)
        nparams = int(sum((np.abs(A) > 0).sum() for A in m.A))
        if m.param == "diffusion":
            nparams = 2 * K
        print(f"{name:<14}{n1:>14.4f}{n10:>15.4f}{corr:>12.3f}{nparams:>10d}")

    # --- residual whiteness ---
    pred, true = m_edge.predict_one_step(X_train)
    resid = true - pred
    Q = ljung_box(resid, n_lags=20)
    print(f"\nresidual Ljung-Box Q (edgewise): median={np.median(Q):.1f} "
          f"(chi2_0.95(20)=31.4; below => white)")

    # --- figures ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    viz.plot_coupling(true_coupling, ax=axes[0], title="true coupling")
    viz.plot_coupling(coupling_matrix(m_edge), ax=axes[1], title="GDAR edgewise")
    viz.plot_coupling(coupling_matrix(m_var), ax=axes[2], title="unconstrained VAR")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "01_coupling_recovery.png"), dpi=130)

    fig2, ax = plt.subplots(figsize=(9, 3.5))
    viz.plot_traces({"true x0": X_test[:200, 0],
                     "GDAR 1-step": np.r_[[np.nan] * (K - 1), m_edge.predict_one_step(X_test[:202])[0][:200, 0]]},
                    ax=ax, title="one-step prediction, channel 0")
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGDIR, "01_prediction_trace.png"), dpi=130)
    print(f"\nfigures -> {FIGDIR}")


def coupling_matrix_from_coeffs(coeffs):
    stack = np.stack(coeffs, axis=0)
    C = np.sqrt((stack ** 2).sum(axis=0))
    np.fill_diagonal(C, 0.0)
    return C


if __name__ == "__main__":
    main()
