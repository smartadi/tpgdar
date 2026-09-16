"""Stage 4 -- interpretability of the TP vertex systems.

Two things the vertex decomposition buys us:

1. **Mode maps** -- each vertex LTI system is a graph-diffusion operator; we map
   its per-channel coupling strength onto the array layout, so a "vertex" reads
   as a spatial communication pattern.

2. **Weight tracking** -- when the scheduling variable ``p`` changes over time,
   the convex weight ``w_v(t)`` of each vertex should track it.  We drive the WC
   network with a time-varying ``p(t)`` (a low->high->low protocol), fit GDAR in
   sliding windows, project each window onto the 2-vertex polytope, and show the
   weight of the high-coupling vertex follows ``p(t)``.

Run:  python scripts/04_interpret_modes.py
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
from gdar_tp.gdar import fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.qlpv import fit_grid
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams
from gdar_tp.tp import convex_hull

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)
K = 2
G0 = 3.0


def simulate_segment(net, p, seed, n=20000):
    E = net.simulate(n_steps=n, p=p, noise_std=0.03, rng=seed, burn_in=800)
    return E - E.mean(axis=0, keepdims=True)


def project_to_polytope(A_vec, V0, V1):
    """Convex weight of vertex-1 for a coefficient vector on the V0-V1 segment."""
    d = V1 - V0
    lam = float(np.dot(A_vec - V0, d) / np.dot(d, d))
    return np.clip(lam, 0.0, 1.0)


def main():
    rows, cols = 5, 5
    g = grid_graph(rows, cols)
    net = WCNetwork(g.adjacency, WCParams(g0=G0))
    mask = support_mask(g.adjacency)

    # --- build the 2-vertex polytope on a training grid (Stage 2) ---
    p_train = np.linspace(0.0, 1.0, 7)
    segments = [(simulate_segment(net, p, 10 + i), None) for i, p in enumerate(p_train)]
    qmodel = fit_grid(segments, p_train, K=K, support=mask, param="edgewise", ridge=1e-3)
    poly = convex_hull(qmodel.coeffs, n_param_modes=1, method="tight", energy=0.99)
    V = poly.vertex_systems.reshape(poly.n_vertices, -1)
    # identify vertices by the p-value of their grid index: V1 = the high-p vertex
    vert_p = qmodel.p_grid[0][poly.vertex_grid_index]
    order = np.argsort(vert_p)
    V0, V1 = V[order[0]], V[order[-1]]

    def couple_strength(vec):
        A0 = vec.reshape(g.n, g.n, K)[..., 0].copy()
        np.fill_diagonal(A0, 0.0)
        return np.abs(A0).sum()

    print("=== Stage 4: interpretability of TP vertex systems ===")
    print(f"grid {rows}x{cols}={g.n} ch, {poly.n_vertices} vertices at p={np.round(np.sort(vert_p),2)}")
    print(f"coupling strength (sum |off-diag A0|): low-p vertex {couple_strength(V0):.3f}, "
          f"high-p vertex {couple_strength(V1):.3f}")

    # --- weight tracking under a time-varying p(t) ---
    T, burn = 60000, 800
    p_t = np.concatenate([np.full(T // 3, 0.15),
                          np.full(T // 3, 0.9),
                          np.full(T - 2 * (T // 3), 0.15)])
    p_full = np.concatenate([np.full(burn, p_t[0]), p_t])   # cover burn-in too
    E = net.simulate(n_steps=T, p=p_full, noise_std=0.03, rng=321, burn_in=burn)
    Xc = E - E.mean(axis=0, keepdims=True)

    win, hop = 3000, 1000
    centers, w1, p_win = [], [], []
    for start in range(0, T - win + 1, hop):
        Xw = Xc[start:start + win]
        m = fit_gdar(Xw, K=K, support=mask, param="edgewise", ridge=1e-3)
        Avec = np.stack(m.A, axis=-1).ravel()
        w1.append(project_to_polytope(Avec, V0, V1))
        centers.append(start + win // 2)
        p_win.append(p_t[start:start + win].mean())
    w1 = np.array(w1); p_win = np.array(p_win); centers = np.array(centers)
    corr = np.corrcoef(w1, p_win)[0, 1]
    print(f"\nweight-vs-schedule tracking: corr(w_high-p-vertex(t), p(t)) = {corr:.3f}")
    print(f"  mean high-p-vertex weight when p=0.15: {w1[p_win < 0.5].mean():.3f}; "
          f"when p=0.9: {w1[p_win > 0.5].mean():.3f}")

    _figures(poly, rows, cols, K, g.n, centers, w1, p_win)
    print(f"\nfigures -> {FIGDIR}")


def _figures(poly, rows, cols, K, N, centers, w1, p_win):
    V = poly.vertex_systems
    nv = V.shape[0]
    fig, axes = plt.subplots(1, nv + 1, figsize=(4 * (nv + 1), 3.6))
    for v in range(nv):
        A0 = V[v, :, :, 0].copy(); np.fill_diagonal(A0, 0.0)
        viz.plot_array_map(np.abs(A0).sum(1), rows, cols, ax=axes[v],
                           title=f"vertex {v} coupling map")
    ax = axes[-1]
    ax.plot(centers, w1, "-o", ms=3, label="w_high-p-vertex(t)")
    ax.plot(centers, p_win, "--", label="p(t) (schedule)")
    ax.set_xlabel("step"); ax.set_ylabel("weight / p"); ax.legend(fontsize=8)
    ax.set_title("vertex weight tracks the schedule"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "04_interpret_modes.png"), dpi=130)


if __name__ == "__main__":
    main()
