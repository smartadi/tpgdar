r"""Rank-3 figures for TP-GDAR_IOS_Reach.pptx.  No model is fitted on data here: those fits come from the caches of scripts 12 and 13.

  tp_mix_r3            vertices and weights, evoked left in
  vertices_r3_induced  the same with the evoked response removed
  tp_vs_gdar_r3        held-out gain of rank 3 against separate fits, and on task times left out
  convex_r3            the TP paper's Transformations 1 and 2 on a known rank-3 S(p), plus the smallest triangle

The last is a worked example, mine and not the paper's, approved by Aditya on 2026-11-05.  S(p) = S_0 + p S_1 + p^2 S_2 on
[1, 2], with orthonormal S_j, is sampled on 41 points.  The steps follow Method 4; Transformation 2 uses
Z = (b I + 11')/(n + b), since the paper's Z as printed doesn't normalise.  Numbers are printed and go in the speaker notes.

Run:  .venv\Scripts\python scripts/make_deck_figures_r3.py [names]
"""

import importlib
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
real = importlib.import_module("make_deck_figures_real")
s12, s13 = real.s12, real.s13
from gdar_tp.sym_tp import _enclosing_triangle  # noqa: E402

OUT = real.OUT
COLORS = real.VERTEX
BLUE, ORANGE, GREEN, GREY = real.BLUE, real.ORANGE, real.GREEN, real.GREY


def mix_r3(D, res, name, tags):
    """Three vertex maps with their weights across the trial."""
    import matplotlib.pyplot as plt
    f = res["full"][3]
    N = len(D["pos"])
    S = [real.summed(ph, res, N) for ph in f["poly"]["phi"]]
    _, iu, ju = real.edges_of(D)
    vm = max(np.abs(m[iu, ju]).max() for m in S)
    panels = []
    fig, axs = plt.subplots(1, 4, figsize=real.FIGSIZE, constrained_layout=True, width_ratios=[1, 1, 1, 1.6])
    for v, ax in enumerate(axs[:3]):
        panels.append(real.edge_map(ax, S[v], D, vm, [0, v, 1, 1], f"Vertex {v + 1}")[1])
        real.colour_key(ax, COLORS[v])
    ax = axs[3]
    series = real.weights(ax, f, 3)
    real.phase_axis(ax, top=1.17)
    texts = []
    for v, (x, y) in enumerate(tags):
        real.tag(ax, x, y, f"Vertex {v + 1}", texts, shift=0.5, color=COLORS[v])
        texts[-1]["color"] = COLORS[v]      # the slide draws a swatch of this colour beside the label
    ax.set_title("Weights")
    panels.append({"kind": "line", "cell": [0, 3, 1, 1], "title": "Weights", "cat_groups": real.PHASE_GROUPS, "ylabel": "",
                   "ylim": [-0.05, 1.17], "legend": False, "series": series, "texts": texts})
    print(name, "weights by bin (rows: hold, plan, move, post; 4 bins each):")
    print(np.round(f["poly"]["lam"], 2))
    print("  edge scale", round(float(vm), 3), "vertices nearest", f["poly"]["index"], "dist", np.round(f["poly"]["dist"], 2),
          "sum-to-one error", round(f["poly"]["affine_err"], 3), "numbers", f["model"].n_params)
    real.save(fig, name, [1, 4], panels, col_widths=[1, 1, 1, 1.6])


def gains_r3(R):
    """Gain over one GDAR: rank 3 against separate fits, on both grids, and on left-out task times."""
    import matplotlib.pyplot as plt
    variants = (("felix", BLUE, "evoked left in"), ("induced", ORANGE, "evoked removed"))
    ylim = [-0.15, 0.33]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=real.FIGSIZE, constrained_layout=True, sharey=True)
    sa, ta = [], []
    cats = ["GDAR", "rank 3", "separate"]
    for j, (v, col, lab) in enumerate(variants):
        g = R[v]["phase"]
        m = [(0.0, 0.0)] + [s12._gain(g, n) for n in ("TP r3", "per-condition")]
        y, e = [q for q, _ in m], [q for _, q in m]
        a.errorbar(np.arange(3) + (j - 0.5) * 0.14, y, e, fmt="-o", ms=9, color=col, elinewidth=2)
        sa.append({"name": lab, "y": y, "err": e, "color": col, "line": "solid", "marker": True, "width": 2})
        print(v, "phase grid:", [f"{q:+.3f}" for q in y], "SE", [f"{q:.3f}" for q in e])
    a.set_xticks(range(3), cats)
    a.set_xlim(-0.45, 2.6)
    a.set_ylabel("gain (nats/sample)")
    a.set_title("4 phases")
    real.tag(a, 0.95, 0.315, "evoked removed", ta, color=ORANGE)
    real.tag(a, 1.5, 0.11, "evoked left in", ta, color=BLUE)
    sb = []
    for v, col, lab in variants:
        g = R[v]["time"]
        m = [(0.0, 0.0)] + [s12._gain(g, n) for n in ("TP r3", "per-condition")]
        y, e = [q for q, _ in m], [q for _, q in m]
        b.errorbar(range(3), y, e, fmt="-o", ms=9, color=col, elinewidth=2)
        sb.append({"name": lab, "y": y, "err": e, "color": col, "line": "solid", "marker": True, "width": 2})
        print(v, "time grid:", [f"{q:+.3f}" for q in y], "SE", [f"{q:.3f}" for q in e],
              "numbers", g["params"]["GDAR"], g["params"]["TP r3"], g["params"]["per-condition"])
    b.set_xticks(range(3), cats)
    b.set_xlim(-0.45, 2.6)
    b.set_title("12 time bins")
    g = R["felix"]["time"]
    t = g["p_grid"][1:-1]
    nb = g["n"][:, 1:-1].sum(0)
    c.axvline(0, color=s12.MUTED, lw=1, ls=":")
    sc = [{"name": "go", "x": [0, 0], "y": ylim, "color": s12.MUTED, "line": "dash", "marker": False, "width": 1}]
    tc = []
    for name, lab, col in (("nearest bin", "fit on neighbour", GREY), ("oracle", "fit on that bin", GREEN), ("TP r3", "TP rank 3", BLUE)):
        y = (g["lobo"][name] - g["lobo"]["GDAR"])[:, 1:-1].sum(0) / nb
        c.plot(t, y, "-o", ms=6, lw=2.4, color=col)
        sc.append({"name": lab, "x": t, "y": y, "color": col, "line": "solid", "marker": True, "width": 2.4})
    c.set_xlabel("ms from go")
    c.set_title("Task time left out")
    # a key in the empty lower right, in the curves' order (rank 3 > that bin > neighbour in every bin since the Sigma fix)
    for y, lab, col in ((-0.035, "TP rank 3", BLUE), (-0.075, "fit on that bin", GREEN), (-0.115, "fit on neighbour", GREY)):
        real.tag(c, 420, y, lab, tc, color=col)
        tc[-1]["color"] = col       # the slide draws a swatch of this colour beside the label
    for ax in (a, b, c):
        s12._zero(ax)
    a.set_ylim(*ylim)
    a.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
    panels = [{"kind": "line", "cell": [0, 0, 1, 1], "title": "4 phases", "categories": cats, "xlabel": "",
               "ylabel": "gain over GDAR (nats/sample)", "ylim": ylim, "legend": False, "series": sa, "texts": ta},
              {"kind": "line", "cell": [0, 1, 1, 1], "title": "12 time bins", "categories": cats, "xlabel": "",
               "ylabel": "", "ylim": ylim, "legend": False, "series": sb, "texts": []},
              {"kind": "xy", "cell": [0, 2, 1, 1], "title": "Task time left out", "xlabel": "ms from go", "ylabel": "",
               "xlog": False, "ylog": False, "xlim": c.get_xlim(), "ylim": ylim, "legend": False, "series": sc,
               "hlines": [real.ZERO_LINE], "texts": tc}]
    real.save(fig, "tp_vs_gdar_r3", [1, 3], panels)


def convex_example():
    """(p, weights at each step).  T1 and T2 as in Method 4, then the smallest triangle."""
    p = np.linspace(1, 2, 41)
    basis = np.eye(3)                                   # S_0, S_1, S_2, orthonormal
    T = np.outer(np.ones_like(p), basis[0]) + np.outer(p, basis[1]) + np.outer(p ** 2, basis[2])
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    print("singular values / first:", np.round(s / s[0], 5))
    r = int((s > 1e-10 * s[0]).sum())
    assert r == 3
    Ur, core = U[:, :r], s[:r, None] * Vt[:r]
    one = np.ones(len(p))
    c = Ur.T @ one
    assert np.linalg.norm(Ur @ c - one) < 1e-10, "the constant is in the span here, so T1 adds no column"
    assert np.abs(c).min() > 1e-6
    U1 = Ur @ np.diag(c)
    C1 = np.linalg.solve(np.diag(c), core)
    a = U1.min()
    b = 1.0 if a >= -1 else 1 / abs(a)
    Z = (b * np.eye(r) + np.ones((r, r))) / (r + b)
    U2 = U1 @ Z
    C2 = np.linalg.solve(Z, C1)
    assert np.allclose(U1.sum(1), 1) and np.allclose(U2.sum(1), 1)
    assert U2.min() >= -1e-12 and np.abs(U2 @ C2 - T).max() < 1e-9
    # smallest triangle in the plane of the sum-to-one weights
    m = U2.mean(0)
    _, _, W = np.linalg.svd(U2 - m, full_matrices=False)
    tri = _enclosing_triangle((U2 - m) @ W[:2].T) @ W[:2] + m
    U3 = np.linalg.solve(tri.T, U2.T).T
    C3 = tri @ C2
    assert U3.min() > -1e-9 and np.allclose(U3.sum(1), 1) and np.abs(U3 @ C3 - T).max() < 1e-8

    def area(v):
        return 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
    print("c", np.round(c, 3), "min after T1", round(float(a), 3), "b", round(float(b), 3), "shrink", round(float(b / (r + b)), 3))
    print("weight ranges: svd", np.round([Ur.min(), Ur.max()], 3), "T1", np.round([U1.min(), U1.max()], 3),
          "T2", np.round([U2.min(), U2.max()], 3), "tight", np.round([U3.min(), U3.max()], 3))
    print("largest weight reached per vertex: T2", np.round(U2.max(0), 2), "tight", np.round(U3.max(0), 2))
    print("vertices T2\n", np.round(C2, 2), "\nvertices tight\n", np.round(C3, 2))
    print("largest vertex norm: T2", round(float(np.linalg.norm(C2, axis=1).max()), 2), "tight", round(float(np.linalg.norm(C3, axis=1).max()), 2),
          " largest true S(p) norm", round(float(np.linalg.norm(T, axis=1).max()), 2))
    print("triangle area in vertex space: T2", round(float(area(C2)), 3), "tight", round(float(area(C3)), 3))
    return p, [Ur, U1, U2, U3]


def convex_r3():
    p, steps = convex_example()
    titles = ["SVD weights", "1: sum to one", "2: nonnegative", "Tightest"]
    panels = []
    for i, (W, title) in enumerate(zip(steps, titles)):
        lo, hi = float(np.floor(W.min() * 5) / 5), float(np.ceil(W.max() * 5) / 5)
        series = [{"name": f"weight {j + 1}", "x": p.tolist(), "y": W[:, j].tolist(), "color": COLORS[j], "line": "solid", "marker": False, "width": 2.4}
                  for j in range(3)]
        panels.append({"kind": "xy", "cell": [0, i, 1, 1], "title": title, "xlabel": "p", "ylabel": "weight" if i == 0 else "", "xlog": False, "ylog": False,
                       "xlim": [1, 2], "ylim": [lo, hi], "legend": False, "series": series, "hlines": [real.ZERO_LINE], "texts": []})
    spec = {"grid": [1, 4], "panels": panels}
    with open(os.path.join(OUT, "convex_r3.json"), "w", encoding="utf-8") as fh:
        json.dump(real.clean(spec), fh, allow_nan=False)
    print("wrote convex_r3")


def main():
    real.style()
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or ["tp_mix_r3", "vertices_r3_induced", "tp_vs_gdar_r3", "convex_r3"]
    if "convex_r3" in names:
        convex_r3()
    rest = [n for n in names if n != "convex_r3"]
    if rest:
        D = s12.load()
        with open(s12.CACHE, "rb") as fh:
            R = pickle.load(fh)
        with open(s13.CACHE, "rb") as fh:
            c13 = pickle.load(fh)
        jobs = {"tp_mix_r3": lambda: mix_r3(D, c13["felix"], "tp_mix_r3", ((1.6, 1.08), (2.3, 0.6), (10.5, 1.08))),
                "vertices_r3_induced": lambda: mix_r3(D, c13["induced"], "vertices_r3_induced", ((1.4, 1.08), (6.9, 1.08), (12.4, 1.08))),
                "tp_vs_gdar_r3": lambda: gains_r3(R)}
        for n in rest:
            jobs[n]()


if __name__ == "__main__":
    main()
