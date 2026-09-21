"""Diagnostic -- does the TP (HOSVD) decomposition actually buy anything?

One scientific question: on a parameter-varying GDAR model, does the low-rank
tensor-product decomposition of the coefficient tensor beat the "dynamics are
stationary" null (a single pooled GDAR fit) when we predict FRESH held-out
segments at HELD-OUT scheduling values ``p``?

We simulate a Wilson-Cowan network on a grid graph whose coupling varies with a
1-D scheduling variable ``p``, fit on a training grid of ``p``, then evaluate
one-step prediction on fresh segments at held-out ``p``.  Four models compete:

  1. Pooled GDAR            -- one fit on all training segments concatenated
                               (ignores p; the "TP is unnecessary" null).
  2. Per-condition (nearest)-- independent GDAR per training p; at held-out p
                               use the fit at the NEAREST trained p (no interp).
  3. TP low-rank interp     -- rank-2 HOSVD reconstruction of the coeff tensor,
                               elementwise-linearly interpolated to held-out p.
  4. Oracle                 -- GDAR fit on the held-out segment itself
                               (lower bound on achievable one-step error).

TP earns its keep iff (3) clearly beats (1) and approaches (4).  We also run a
CV rank-stability check: refit qLPV over 5 independent noise realizations, run
HOSVD each time, and report the retained rank at energy 0.99 (should be 2) plus
the sv2/sv1 spread.

Run:  python scripts/06_tp_diagnostic.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdar_tp import metrics
from gdar_tp.gdar import GDARModel, fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import fit_grid
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams
from gdar_tp.tp import hosvd

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)
K = 2
# g0=3.0 puts the network coupling on the same footing as the local E-I circuit,
# so p genuinely restructures the dynamics (clean rank-2 parameter dependence)
# rather than being a ~6% perturbation that leaves HOSVD looking rank-1.
G0 = 3.0
SEG_LEN = 20000
TP_RANK = 2  # the rank the TP story claims; also what the CV check should recover


def simulate_segment(net, p, seed, n=SEG_LEN):
    """Simulate a WC segment at fixed p, return mean-centered activity."""
    E = net.simulate(n_steps=n, p=p, noise_std=0.03, rng=seed, burn_in=800)
    return E - E.mean(axis=0, keepdims=True)


def interp_coeffs(p_grid, coeffs, p_star):
    """Elementwise linear interpolation of the coefficient tensor at p_star."""
    P = coeffs.shape[0]
    flat = coeffs.reshape(P, -1)
    out = np.array([np.interp(p_star, p_grid, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape(coeffs.shape[1:])   # (N, N, K)


def make_model(A_tensor, n, mask):
    """Build a GDARModel from an (N, N, K) coefficient tensor (no input)."""
    return GDARModel(A=[A_tensor[..., k] for k in range(K)],
                     B=np.zeros((n, 0)), K=K, param="edgewise", support=mask)


def main():
    rows, cols = 5, 5
    g = grid_graph(rows, cols)
    net = WCNetwork(g.adjacency, WCParams(g0=G0))
    mask = support_mask(g.adjacency)
    N = g.n

    p_train = np.linspace(0.0, 1.0, 7)
    p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    print("=== TP diagnostic: does HOSVD buy anything over plain GDAR? ===")
    print(f"grid {rows}x{cols}={N} ch, K={K}, g0={G0}, seg_len={SEG_LEN}")
    print(f"train p = {np.round(p_train, 2)}")
    print(f"held-out p = {np.round(p_test, 2)}")

    # ---------------------------------------------------------------
    # Fit the four contenders on the TRAINING grid (seeds 10..16).
    # ---------------------------------------------------------------
    train_segs = [simulate_segment(net, p, seed=10 + i) for i, p in enumerate(p_train)]

    # 1. Pooled GDAR: one fit on all training segments concatenated.
    X_pooled = np.vstack(train_segs)
    m_pooled = fit_gdar(X_pooled, K=K, support=mask, param="edgewise", ridge=1e-3)

    # 2. Per-condition GDAR: independent fit at each trained p (for nearest lookup).
    per_cond = [fit_gdar(X, K=K, support=mask, param="edgewise", ridge=1e-3)
                for X in train_segs]

    # 3. TP low-rank interp: fit qLPV, take rank-2 HOSVD reconstruction.
    #    fit_grid SORTS by p, so align interpolation to qmodel.p_grid[0].
    qmodel = fit_grid([(X, None) for X in train_segs], p_train,
                      K=K, support=mask, param="edgewise", ridge=1e-3)
    p_sorted = qmodel.p_grid[0]
    tp = hosvd(qmodel.coeffs, n_param_modes=1, max_rank=TP_RANK, energy=1.0)
    tp_coeffs = tp.reconstruct()  # (P, N, N, K), rank-TP_RANK in the p mode

    # ---------------------------------------------------------------
    # Evaluate one-step NRMSE on FRESH held-out segments (seeds 500+).
    # ---------------------------------------------------------------
    results = {"pooled": [], "nearest": [], "tp": [], "oracle": []}
    for pt in p_test:
        Xtest = simulate_segment(net, pt, seed=500 + int(round(pt * 100)))

        # 1. pooled
        results["pooled"].append(metrics.one_step_nrmse(m_pooled, Xtest))

        # 2. per-condition, nearest trained p (no interpolation)
        j = int(np.argmin(np.abs(p_train - pt)))
        results["nearest"].append(metrics.one_step_nrmse(per_cond[j], Xtest))

        # 3. TP low-rank interpolation to held-out p
        A_interp = interp_coeffs(p_sorted, tp_coeffs, pt)
        results["tp"].append(metrics.one_step_nrmse(make_model(A_interp, N, mask), Xtest))

        # 4. oracle: fit on the held-out segment itself
        m_or = fit_gdar(Xtest, K=K, support=mask, param="edgewise", ridge=1e-3)
        results["oracle"].append(metrics.one_step_nrmse(m_or, Xtest))

    for k in results:
        results[k] = np.array(results[k])

    # ---------------------------------------------------------------
    # Results table.
    # ---------------------------------------------------------------
    print("\nheld-out one-step NRMSE (fresh segments at held-out p):")
    print(f"{'p':>6}{'pooled':>12}{'per-cond':>12}{'TP-interp':>12}{'oracle':>12}")
    for i, pt in enumerate(p_test):
        print(f"{pt:>6.1f}{results['pooled'][i]:>12.4f}{results['nearest'][i]:>12.4f}"
              f"{results['tp'][i]:>12.4f}{results['oracle'][i]:>12.4f}")
    print(f"{'mean':>6}{results['pooled'].mean():>12.4f}{results['nearest'].mean():>12.4f}"
          f"{results['tp'].mean():>12.4f}{results['oracle'].mean():>12.4f}")

    tp_beats_pooled = results["tp"].mean() < results["pooled"].mean()
    gap_vs_pooled = results["pooled"].mean() - results["tp"].mean()
    gap_vs_oracle = results["tp"].mean() - results["oracle"].mean()
    print(f"\nTP vs pooled : {gap_vs_pooled:+.4f} NRMSE  "
          f"({'TP wins' if tp_beats_pooled else 'TP does NOT beat pooled'})")
    print(f"TP vs oracle : {gap_vs_oracle:+.4f} NRMSE  (gap above the achievable floor)")
    verdict = ("YES -- TP beats the stationary/pooled null and approaches the oracle."
               if tp_beats_pooled and gap_vs_pooled > gap_vs_oracle else
               "NO -- TP does not clearly beat the pooled null here.")
    print(f"Does TP buy anything? {verdict}")

    # ---------------------------------------------------------------
    # CV rank-stability: refit qLPV on 5 independent noise realizations.
    # ---------------------------------------------------------------
    print("\nCV rank-stability (5 independent noise realizations, HOSVD @ energy 0.99):")
    cv_ranks, cv_ratios = [], []
    for r_i in range(5):
        segs = [simulate_segment(net, p, seed=9000 + 100 * r_i + i)
                for i, p in enumerate(p_train)]
        qm = fit_grid([(X, None) for X in segs], p_train,
                      K=K, support=mask, param="edgewise", ridge=1e-3)
        tp_cv = hosvd(qm.coeffs, n_param_modes=1, energy=0.99)
        s = tp_cv.singular_values[0]
        cv_ranks.append(int(tp_cv.ranks[0]))
        cv_ratios.append(float(s[1] / s[0]))
        print(f"  realization {r_i}: rank={tp_cv.ranks[0]}, "
              f"sv2/sv1={s[1] / s[0]:.4f}, sv3/sv1={s[2] / s[0]:.4f}")
    cv_ranks = np.array(cv_ranks)
    cv_ratios = np.array(cv_ratios)
    print(f"retained rank  -> min {cv_ranks.min()}, median {int(np.median(cv_ranks))}, "
          f"max {cv_ranks.max()}")
    print(f"sv2/sv1 spread -> min {cv_ratios.min():.4f}, median {np.median(cv_ratios):.4f}, "
          f"max {cv_ratios.max():.4f}")
    print(f"rank estimate stable at {TP_RANK}? "
          f"{'YES' if np.all(cv_ranks == TP_RANK) else 'NO'}")

    # ---------------------------------------------------------------
    # Figures.
    # ---------------------------------------------------------------
    _figures(p_test, results, tp, cv_ranks, cv_ratios)
    print(f"\nfigure -> {os.path.join(FIGDIR, '06_tp_diagnostic.png')}")


def _figures(p_test, results, tp, cv_ranks, cv_ratios):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) held-out NRMSE per model per p
    ax = axes[0]
    for key, label, style in [("pooled", "pooled (null)", "o-"),
                              ("nearest", "per-cond (nearest)", "s-"),
                              ("tp", "TP interp (rank 2)", "^-"),
                              ("oracle", "oracle", "d--")]:
        ax.plot(p_test, results[key], style, label=label)
    ax.set_xlabel("held-out p"); ax.set_ylabel("one-step NRMSE")
    ax.set_title("held-out prediction error"); ax.legend(fontsize=8)

    # (b) mean NRMSE bar chart
    ax = axes[1]
    keys = ["pooled", "nearest", "tp", "oracle"]
    means = [results[k].mean() for k in keys]
    ax.bar(range(len(keys)), means,
           color=["#c44", "#e9a", "#48c", "#4a4"])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["pooled", "per-cond", "TP", "oracle"], fontsize=9)
    ax.set_ylabel("mean held-out NRMSE"); ax.set_title("mean over held-out p")
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # (c) HOSVD spectrum + CV rank stability
    ax = axes[2]
    s = tp.singular_values[0]
    ax.plot(range(1, len(s) + 1), s / s[0], "o-", label="HOSVD (train)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("component"); ax.set_ylabel("singular value / sv1")
    ax.set_title(f"param-mode spectrum\nCV rank {cv_ranks.min()}-{cv_ranks.max()}, "
                 f"sv2/sv1 {cv_ratios.min():.2f}-{cv_ratios.max():.2f}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "06_tp_diagnostic.png"), dpi=130)


if __name__ == "__main__":
    main()
