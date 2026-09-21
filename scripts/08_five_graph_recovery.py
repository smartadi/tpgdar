"""Stage 2b -- ground-truth recovery: can TP-GDAR find the 5 graphs?

Simulation designed to mimic the OS-pulse Utah-array setting: an 8x8 electrode
grid with **8-nearest-neighbour** (king-move) edges as the anatomical support.
The ground-truth dynamics are a convex blend of **5 distinct graph operators**
sharing that one support -- an isotropic diffusion, rightward and upward
advection, a focal hotspot, and a global synchrony mode -- mixed by a scheduling
variable p via a partition-of-unity so that each operator is the *sole* active
one at one of five knots (making them the true convex-hull vertices, hence
identifiable).

TP-GDAR's job: from data alone, recover (i) that there are 5 modes, (ii) the 5
operators, and (iii) the mixing weights.  This is the end-to-end validation of
the whole method on realistic array geometry.

Run:  python scripts/08_five_graph_recovery.py
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
from gdar_tp.gdar import fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import fit_grid
from gdar_tp.sim.linear import simulate_gdar
from gdar_tp.tp import convex_hull, hosvd

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)
N_GRAPHS = 5
RHO = 0.8   # target spectral radius for stability


def _scale(A, rho=RHO):
    ev = np.max(np.abs(np.linalg.eigvals(A)))
    return A * (rho / ev) if ev > 1e-9 else A


def gap_rank(s, kmax=10):
    """Rank = the elbow: largest multiplicative drop in the spectrum *tail*.

    The mode-1 (DC) drop is always the largest, so we skip it and find the
    biggest drop among the remaining components -- the noise-floor onset.
    """
    r = s[:kmax] / s[0]
    ratios = r[1:] / r[:-1]              # ratios[j] = drop from mode j+1 -> j+2
    k = int(np.argmin(ratios[1:])) + 1  # full-array index of the tail elbow
    return k + 1


def build_operators(g):
    """Five distinct graph operators on the shared 8-neighbour support."""
    N = g.n; rows, cols = g.rows, g.cols
    L = g.laplacian
    idx = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    I = np.eye(N)

    # 1. isotropic diffusion (symmetric smoothing)
    A1 = 0.7 * I - 0.05 * L

    # 2. rightward advection: pull from the left neighbour, leak to the right
    A2 = 0.7 * I.copy()
    for (r, c), i in idx.items():
        if (r, c - 1) in idx:
            A2[i, idx[(r, c - 1)]] += 0.30
        if (r, c + 1) in idx:
            A2[i, idx[(r, c + 1)]] -= 0.10

    # 3. upward advection
    A3 = 0.7 * I.copy()
    for (r, c), i in idx.items():
        if (r + 1, c) in idx:
            A3[i, idx[(r + 1, c)]] += 0.30
        if (r - 1, c) in idx:
            A3[i, idx[(r - 1, c)]] -= 0.10

    # 4. focal hotspot: strong local recurrence in the top-left quadrant only
    A4 = 0.5 * I.copy()
    adj = g.adjacency
    focal = np.zeros(N)
    for (r, c), i in idx.items():
        if r < rows // 2 and c < cols // 2:
            focal[i] = 1.0
    A4 += 0.25 * (focal[:, None] * adj * focal[None, :])

    # 5. global synchrony: strong row-normalized neighbour averaging
    deg = adj.sum(1, keepdims=True); deg[deg == 0] = 1
    W = adj / deg
    A5 = 0.45 * I + 0.45 * W

    return [_scale(A) for A in (A1, A2, A3, A4, A5)]


def hat_weights(p, knots):
    """Partition-of-unity hat functions on `knots` (1-D linear interpolation)."""
    p = np.atleast_1d(p)
    W = np.zeros((len(p), len(knots)))
    for m, pv in enumerate(p):
        # linear interpolation weights between the two bracketing knots
        if pv <= knots[0]:
            W[m, 0] = 1.0
        elif pv >= knots[-1]:
            W[m, -1] = 1.0
        else:
            j = np.searchsorted(knots, pv) - 1
            t = (pv - knots[j]) / (knots[j + 1] - knots[j])
            W[m, j] = 1 - t; W[m, j + 1] = t
    return W


def main():
    rng = 0
    g = grid_graph(8, 8, diagonal=True)      # 8-neighbour Utah-style array
    N = g.n
    mask = support_mask(g.adjacency)
    ops = build_operators(g)
    knots = np.linspace(0, 1, N_GRAPHS)

    print("=== Stage 2b: can TP-GDAR recover 5 graphs? ===")
    print(f"array 8x8={N} ch, 8-neighbour support ({int(mask.sum()-N)} directed edges), "
          f"{N_GRAPHS} ground-truth operators, K=1")
    # check blends stay stable
    worst = 0.0
    for pv in np.linspace(0, 1, 41):
        w = hat_weights(pv, knots)[0]
        A = sum(w[j] * ops[j] for j in range(N_GRAPHS))
        worst = max(worst, np.max(np.abs(np.linalg.eigvals(A))))
    print(f"max spectral radius over the p-path: {worst:.3f} (stable if <1)")

    # --- simulate one noise-driven segment per grid p, fit GDAR, stack tensor ---
    p_grid = np.linspace(0, 1, 21)
    segments = []
    for i, pv in enumerate(p_grid):
        w = hat_weights(pv, knots)[0]
        A = sum(w[j] * ops[j] for j in range(N_GRAPHS))
        X, _ = simulate_gdar([A], n_steps=6000, noise_std=0.1, rng=100 + i, burn_in=300)
        segments.append((X, None))
    qmodel = fit_grid(segments, p_grid, K=1, support=mask, param="edgewise", ridge=1e-4)

    # --- HOSVD: how many modes? detect by the largest spectral gap ---
    tp = hosvd(qmodel.coeffs, n_param_modes=1, energy=1.0)
    s = tp.singular_values[0]
    r_gap = gap_rank(s)
    print(f"\nHOSVD singular values (normalized): {np.round(s[:8] / s[0], 4)}")
    print(f"gap-detected rank: {r_gap} (true = {N_GRAPHS}); "
          f"sv5/sv1={s[4]/s[0]:.3f} -> sv6/sv1={s[5]/s[0]:.3f} (a {s[4]/s[5]:.1f}x drop)")

    # --- denoise to the detected rank, then take the convex hull ---
    tp_r = hosvd(qmodel.coeffs, n_param_modes=1, max_rank=r_gap, energy=1.0)
    denoised = tp_r.reconstruct()
    per_pt = np.sqrt((denoised ** 2).sum(axis=(1, 2, 3))).max()
    poly = convex_hull(denoised, n_param_modes=1, method="spa", energy=0.9999,
                       n_vertices=r_gap)
    print(f"convex hull of the rank-{r_gap} trajectory: {poly.n_vertices} vertices, "
          f"affine dim {poly.affine_dim} (true = {N_GRAPHS} vertices, {N_GRAPHS-1} dims); "
          f"these {poly.n_vertices} reconstruct A(p) to rel. residual {poly.residual/per_pt:.2e}")
    print("  (identifiability condition: the schedule visits each regime -- every"
          " graph is the sole-active operator somewhere in the recording.)")

    # --- match recovered vertices to the true operators ---
    recovered = poly.vertex_systems[:, :, :, 0]        # (V, N, N)
    corr = _match(ops, recovered, mask)
    print(f"\nrecovered-vs-true operator match (|corr| on support, best assignment):")
    order, per = corr
    for j, (v, c) in enumerate(zip(order, per)):
        names = ["isotropic", "rightward", "upward", "focal", "synchrony"]
        print(f"  graph {j} ({names[j]:>9}) -> recovered vertex {v}: |corr|={c:.3f}")
    print(f"  mean |corr| = {np.mean(per):.3f}")

    _figures(g, ops, recovered, order, tp, poly, p_grid, knots)
    print(f"\nfigures -> {FIGDIR}")


def _match(true_ops, recovered, mask):
    """Best one-to-one assignment of recovered vertices to true operators."""
    from scipy.optimize import linear_sum_assignment
    nt = len(true_ops); nr = recovered.shape[0]
    Cmat = np.zeros((nt, nr))
    m = mask
    for i in range(nt):
        a = true_ops[i][m]
        a = a - a.mean()
        for j in range(nr):
            b = recovered[j][m]; b = b - b.mean()
            s = a.std() * b.std()
            Cmat[i, j] = abs(np.mean(a * b) / s) if s > 1e-12 else 0.0
    ri, ci = linear_sum_assignment(-Cmat)
    order = ci[np.argsort(ri)]
    per = [Cmat[i, order[i]] for i in range(nt)]
    return order, per


def _figures(g, ops, recovered, order, tp, poly, p_grid, knots):
    rows, cols = g.rows, g.cols
    names = ["isotropic", "rightward", "upward", "focal", "synchrony"]

    # spectrum + weights
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    viz.plot_singular_spectrum(tp.singular_values[0], ax=ax[0],
                               title="HOSVD spectrum: 5 modes then floor")
    ax[0].axvline(5.5, color="C3", ls=":", label="true # graphs = 5"); ax[0].legend(fontsize=8)
    for v in range(poly.weights.shape[1]):
        ax[1].plot(p_grid, poly.weights[:, v], "-o", ms=3)
    ax[1].set_title("recovered convex weights w_v(p)")
    ax[1].set_xlabel("scheduling p"); ax[1].set_ylabel("weight"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "08_recovery_spectrum.png"), dpi=130)

    # true vs recovered operator coupling maps (per-channel out-strength)
    fig2, axes = plt.subplots(2, len(ops), figsize=(3.1 * len(ops), 6.2))
    for j in range(len(ops)):
        At = ops[j].copy(); np.fill_diagonal(At, 0)
        Ar = recovered[order[j]].copy(); np.fill_diagonal(Ar, 0)
        viz.plot_array_map(np.abs(At).sum(1), rows, cols, ax=axes[0, j],
                           title=f"true: {names[j]}")
        viz.plot_array_map(np.abs(Ar).sum(1), rows, cols, ax=axes[1, j],
                           title="recovered")
    axes[0, 0].set_ylabel("TRUE", fontsize=11)
    axes[1, 0].set_ylabel("RECOVERED", fontsize=11)
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGDIR, "08_recovery_maps.png"), dpi=130)


if __name__ == "__main__":
    main()
