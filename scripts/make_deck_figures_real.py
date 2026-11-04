r"""Clean, low-text talk figures from existing caches.  Nothing is re-fitted.

Reads cache/12_results_H_2021-11-12.pkl, cache/13_vertices_H_2021-11-12.pkl and cache/14_days.pkl,
writes figures/deck/<name>.png, the reference look, and figures/deck/<name>.json, the same plotted
numbers for drawing the figure as native PowerPoint charts and shapes.  Each panel dict is built from the
arrays handed to matplotlib, so the two cannot drift apart.

Beyond the panel schema, the JSON adds these optional keys:
  series "err"     symmetric half-width per point (SE whiskers or the jackknife band in the PNG)
  "node_gamma"     the PNG colours nodes by (value / node_vmax) ** gamma
  "line" panel     "xlabel" and "texts" where the PNG has them; "x" in "texts" is the category index
All "texts" positions are the centre of the text.

Run:  .venv\Scripts\python scripts/make_deck_figures_real.py [name ...]
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm, TwoSlopeNorm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s13 = importlib.import_module("13_tp_vertices_ios")
s14 = importlib.import_module("14_tp_vertices_days")
from gdar_tp.sym_tp import theta_of  # noqa: E402

OUT = os.path.join(s12.FIG, "deck")
FIGSIZE = (12.1, 5.6)
DPI = 170
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8f98"
VERTEX = (BLUE, ORANGE, GREEN)
BG = "#fcfcfb"
ZERO = "#f0efec"
EDGE_CMAP = LinearSegmentedColormap.from_list("bwr_", [s12.C_FELIX, ZERO, "#e34948"])  # as s13.fig_variant
HEAT_CMAP = LinearSegmentedColormap.from_list("heat_", ["#f6f1ea", ORANGE, "#7d2406"])
GAMMA = 0.45
K = s13.K
NSUB = s13.NSUB
PHASE_GROUPS = [[ph, NSUB] for ph in s12.PHASES]
ZERO_LINE = {"y": 0, "color": s12.MUTED, "dash": False}

# what scripts/14 logs for the move-minus-rest edge correlation with 11-12
EXPECT_R = {"felix": (0.15, 0.12, 0.21), "induced": (0.47, 0.15, 0.13)}


def style():
    s12._style()
    matplotlib.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                                "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})


def clean(o):
    """Plain JSON values only, NaN as null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def save(fig, name, grid, panels, row_heights=None, col_widths=None):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI)
    plt.close(fig)
    spec = {"grid": grid, "row_heights": row_heights, "col_widths": col_widths, "panels": panels}
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(clean(spec), fh, allow_nan=False)
    print("wrote", name, ".png .json", flush=True)


# ---------------------------------------------------------------------------- pieces
def edges_of(D):
    pos = D["pos"]
    iu, ju = np.where(np.triu(D["A"], 1))
    return [(pos[i], pos[j]) for i, j in zip(iu, ju)], iu, ju


def summed(phi, res, N):
    """Edge weights summed over lags."""
    th = theta_of(phi, res["R"], N)
    return sum(th[:, k * N:(k + 1) * N] for k in range(K))


def network_panel(D, cell, title, edge_values=None, vmax=1.0, node_values=None, node_vmax=1.0, **extra):
    _, iu, ju = edges_of(D)
    return {"kind": "network", "cell": cell, "title": title, "pos": D["pos"], "edges": np.c_[iu, ju],
            "edge_values": edge_values, "vmax": vmax, "node_values": node_values, "node_vmax": node_vmax,
            "node_cmap": "seq", **extra}


def edge_map(ax, M, D, vm, cell, title, lw=5.0):
    """Draws the map and returns (artist, its panel dict)."""
    segs, iu, ju = edges_of(D)
    vals = M[iu, ju]
    lc = LineCollection(segs, array=vals, cmap=EDGE_CMAP, norm=TwoSlopeNorm(0, -vm, vm), linewidths=lw,
                        capstyle="round")
    ax.add_collection(lc)
    ax.scatter(*D["pos"].T, s=48, color=BG, edgecolor=s12.MUTED, lw=1.1, zorder=3)
    ax.set_aspect("equal")
    ax.margins(0.05)
    ax.axis("off")
    ax.set_title(title)
    return lc, network_panel(D, cell, title, edge_values=vals, vmax=vm)


def matrix_axes(ax, title):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GREY)
    ax.set_title(title)


def colour_key(ax, colour):
    """Coloured rule under a map's title: ties the map to its line in the weights panel."""
    ax.plot([0, 1], [1.0, 1.0], transform=ax.transAxes, color=colour, lw=4, clip_on=False, solid_capstyle="butt")


def phase_axis(ax, top=1.05):
    for b in range(1, 4):
        ax.axvline(b * NSUB, color=s12.MUTED, lw=1, ls=":")
    ax.set_xticks((np.arange(4) + 0.5) * NSUB, s12.PHASES)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.05, top)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])


def weights(ax, f, r):
    """Draws the weight lines with their jackknife bands, returns the JSON series."""
    lam, se = f["poly"]["lam"], f["lam_se"]
    x = np.arange(lam.shape[0]) + 0.5
    series = []
    for v in range(r):
        ax.fill_between(x, lam[:, v] - se[:, v], lam[:, v] + se[:, v], color=VERTEX[v], alpha=0.18, lw=0)
        ax.plot(x, lam[:, v], "-o", ms=5, lw=2.4, color=VERTEX[v])
        series.append({"name": f"Vertex {v + 1}", "y": lam[:, v], "err": se[:, v], "color": VERTEX[v], "line": "solid",
                       "marker": True, "width": 2.4})
    return series


def tag(ax, x, y, s, texts=None, shift=0.0, **kw):
    """Direct label centred on (x, y).  The JSON records x - shift (bin i sits at i + 0.5 on the PNG's phase axis)."""
    ax.text(x, y, s, ha="center", va="center", **{"color": s12.INK, **kw})
    if texts is not None:
        texts.append({"x": x - shift, "y": y, "text": s})


# --------------------------------------------------------------------------- figures
def graph_array(D):
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.55, 5.0, 5.6, 0.95])
    a, b = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    segs, _, _ = edges_of(D)
    a.add_collection(LineCollection(segs, colors=GREY, linewidths=2))
    a.scatter(*D["pos"].T, s=190, color=BLUE, edgecolor=BG, lw=2, zorder=3)
    a.set_aspect("equal")
    a.margins(0.06)
    a.axis("off")
    a.set_title("Electrodes and neighbours")
    A = D["A"].astype(int)
    b.imshow(A, cmap=ListedColormap([ZERO, BLUE]), vmin=0, vmax=1, interpolation="nearest")
    matrix_axes(b, "Adjacency A")
    panels = [network_panel(D, [0, 0, 1, 1], "Electrodes and neighbours", node_values=np.ones(len(A))),  # uniform blue
              {"kind": "matrix", "cell": [0, 1, 1, 1], "title": "Adjacency A", "values": A, "vmax": 1, "binary": True}]
    save(fig, "graph_array", [1, 2], panels, col_widths=[0.75, 1])


def diffusion(D):
    A = D["A"].astype(float)
    L = np.diag(A.sum(1)) - A
    pos = D["pos"]
    src = int(np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1)))
    step = np.eye(len(A)) - 0.12 * L
    x0 = np.zeros(len(A))
    x0[src] = 1.0
    norm = PowerNorm(GAMMA, vmin=0, vmax=1)     # one scale for all panels; sqrt-like so t = 8 stays visible
    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, constrained_layout=True)
    segs, _, _ = edges_of(D)
    panels = []
    for col, (ax, t) in enumerate(zip(axs, (0, 1, 3, 8))):
        x = np.linalg.matrix_power(step, t) @ x0
        ax.add_collection(LineCollection(segs, colors="#d4d3cf", linewidths=1.6))
        ax.scatter(*pos.T, c=x, s=170, cmap=HEAT_CMAP, norm=norm, edgecolor=s12.MUTED, lw=0.9, zorder=3)
        ax.set_aspect("equal")
        ax.margins(0.06)
        ax.axis("off")
        ax.set_title(f"t = {t}")
        panels.append(network_panel(D, [0, col, 1, 1], f"t = {t}", node_values=x, node_vmax=1.0, node_gamma=GAMMA))
    save(fig, "diffusion", [1, 4], panels)


def gdar_fit(D, c13):
    res = c13["felix"]
    N = len(D["pos"])
    th = theta_of(res["phi0"], res["R"], N)
    A1 = th[:, :N]
    S = summed(res["phi0"], res, N)
    _, iu, ju = edges_of(D)
    off = A1[iu, ju]
    print(f"gdar_fit: A1 diagonal {np.diag(A1).min():.2f}..{np.diag(A1).max():.2f}, on-graph off-diagonal "
          f"{off.min():.3f}..{off.max():.3f}; edges over all lags {S[iu, ju].min():.3f}..{S[iu, ju].max():.3f}", flush=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 5.4, 3.8, 0.8])
    a, b = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    vm = np.abs(off).max()                      # diagonal saturates, else the graph entries vanish
    im = a.imshow(A1, cmap=EDGE_CMAP, norm=TwoSlopeNorm(0, -vm, vm), interpolation="nearest")
    matrix_axes(a, "A₁")
    _, pb = edge_map(b, S, D, np.abs(S[iu, ju]).max(), [0, 1, 1, 1], "Edges, all lags")
    # the two panels have different ranges (about 10x apart), so the bar gives sign only
    cb = fig.colorbar(im, ax=[a, b], fraction=0.02, pad=0.01, shrink=0.45, ticks=[-vm, 0, vm], panchor=False)
    cb.ax.set_yticklabels(["−", "0", "+"])
    cb.outline.set_visible(False)
    cb.set_label("weight")
    panels = [{"kind": "matrix", "cell": [0, 0, 1, 1], "title": "A₁", "values": A1, "vmax": vm, "binary": False}, pb]
    save(fig, "gdar_fit", [1, 2], panels, col_widths=[1, 0.75])


def tp_mix(D, c13):
    res = c13["felix"]
    f = res["full"][2]
    N = len(D["pos"])
    S = [summed(ph, res, N) for ph in f["poly"]["phi"]]
    _, iu, ju = edges_of(D)
    vm = max(np.abs(m[iu, ju]).max() for m in S)
    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True, width_ratios=[1, 1, 1.5])
    panels = []
    for v, ax in enumerate(axs[:2]):
        panels.append(edge_map(ax, S[v], D, vm, [0, v, 1, 1], f"Vertex {v + 1}")[1])
        colour_key(ax, VERTEX[v])
    ax = axs[2]
    series = weights(ax, f, 2)
    phase_axis(ax, top=1.17)
    texts = []
    tag(ax, 10.5, 1.075, "Vertex 1", texts, shift=0.5)
    tag(ax, 2.0, 1.055, "Vertex 2", texts, shift=0.5)
    ax.set_title("Weights")
    panels.append({"kind": "line", "cell": [0, 2, 1, 1], "title": "Weights", "cat_groups": PHASE_GROUPS, "ylabel": "",
                   "ylim": [-0.05, 1.17], "legend": False, "series": series, "texts": texts})
    save(fig, "tp_mix", [1, 3], panels, col_widths=[1, 1, 1.5])


def tp_vs_gdar(R):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True, sharey=True)
    ylim = [-0.15, 0.33]
    variants = (("felix", BLUE, "evoked left in"), ("induced", ORANGE, "evoked removed"))
    # gain vs rank, phase grid
    names = ["TP r1", "TP r2", "TP r3", "per-condition"]
    cats = ["1", "2", "3", "separate"]
    sa = []
    for j, (v, col, lab) in enumerate(variants):
        g = R[v]["phase"]
        m = [s12._gain(g, n) for n in names]
        y, e = [q for q, _ in m], [q for _, q in m]
        a.errorbar(np.arange(4) + (j - 0.5) * 0.14, y, e, fmt="o", ms=9, color=col, elinewidth=2)
        sa.append({"name": lab, "y": y, "err": e, "color": col, "line": "none", "marker": True, "width": 2})
    a.set_xticks(range(4), cats)
    a.set_xlim(-0.45, 3.6)
    a.set_xlabel("rank")
    a.set_ylabel("gain (nats/sample)")
    a.set_title("Gain vs rank")
    ta = []
    tag(a, 0.5, 0.262, "evoked removed", ta)
    tag(a, 2.45, 0.11, "evoked left in", ta)
    tag(a, -0.085, -0.022, "GDAR", ta, color=s12.MUTED)
    sa.append({"name": "", "y": [0] * len(cats), "color": s12.MUTED, "line": "solid", "marker": False, "width": 1})
    # low rank vs separate fits, time grid
    sb, tb_ = [], []
    for v, col, lab in variants:
        g = R[v]["time"]
        ns = [f"TP r{r}" for r in s12.RANKS_TIME] + ["per-condition"]
        x, y = [g["params"][n] for n in ns], [s12._gain(g, n)[0] for n in ns]
        b.plot(x, y, "-o", ms=7, lw=2.4, color=col)
        sb.append({"name": lab, "x": x, "y": y, "color": col, "line": "solid", "marker": True, "width": 2.4})
    g = R["felix"]["time"]
    tag(b, g["params"]["TP r3"], 0.113, "rank 3", tb_)
    tag(b, 8900, 0.07, "12 separate", tb_)
    b.set_xscale("log")
    b.set_xticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
    b.minorticks_off()
    b.set_xlabel("free parameters")
    b.set_title("Low rank vs separate")
    # leave-one-bin-out, evoked left in
    tb = g["p_grid"][1:-1]
    nb = g["n"][:, 1:-1].sum(0)
    c.axvline(0, color=s12.MUTED, lw=1, ls=":")
    sc = [{"name": "go", "x": [0, 0], "y": ylim, "color": s12.MUTED, "line": "dash", "marker": False, "width": 1}]
    tc = []
    for name, lab, col in (("nearest bin", "fit on neighbour", GREY), ("oracle", "fit on that bin", GREEN),
                           ("TP r3", "TP rank 3", BLUE)):
        y = (g["lobo"][name] - g["lobo"]["GDAR"])[:, 1:-1].sum(0) / nb
        c.plot(tb, y, "-o", ms=6, lw=2.4, color=col)
        sc.append({"name": lab, "x": tb, "y": y, "color": col, "line": "solid", "marker": True, "width": 2.4})
    c.set_xlabel("ms from go")
    c.set_title("Unseen task time")
    tag(c, 330, 0.195, "TP rank 3", tc)
    tag(c, 400, 0.04, "fit on that bin", tc)
    tag(c, 330, -0.075, "fit on neighbour", tc)
    for ax in (a, b, c):
        s12._zero(ax)
    a.set_ylim(*ylim)
    a.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
    panels = [{"kind": "line", "cell": [0, 0, 1, 1], "title": "Gain vs rank", "categories": cats, "xlabel": "rank",
               "ylabel": "gain (nats/sample)", "ylim": ylim, "legend": False, "series": sa, "texts": ta},
              {"kind": "xy", "cell": [0, 1, 1, 1], "title": "Low rank vs separate", "xlabel": "free parameters", "ylabel": "",
               "xlog": True, "ylog": False, "xlim": b.get_xlim(), "ylim": ylim, "legend": False, "series": sb,
               "hlines": [ZERO_LINE], "texts": tb_},
              {"kind": "xy", "cell": [0, 2, 1, 1], "title": "Unseen task time", "xlabel": "ms from go", "ylabel": "",
               "xlog": False, "ylog": False, "xlim": c.get_xlim(), "ylim": ylim, "legend": False, "series": sc,
               "hlines": [ZERO_LINE], "texts": tc}]
    save(fig, "tp_vs_gdar", [1, 3], panels)


def vertices(D, res, r, tags, name, pad):
    f = res["full"][r]
    N = len(D["pos"])
    S = [summed(ph, res, N) for ph in f["poly"]["phi"]]
    _, iu, ju = edges_of(D)
    vm = max(np.abs(m[iu, ju]).max() for m in S)
    heights = [1, 1.75]
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    top, bot = fig.subfigures(2, 1, height_ratios=heights)
    ax = top.subplots()
    series = weights(ax, f, r)
    phase_axis(ax, top=1.25)
    texts = []
    for x, y, s in tags:
        tag(ax, x, y, s, texts, shift=0.5)
    ax.set_title("Weights")
    panels = [{"kind": "line", "cell": [0, 0, 1, r], "title": "Weights", "cat_groups": PHASE_GROUPS, "ylabel": "",
               "ylim": [-0.05, 1.25], "legend": False, "series": series, "texts": texts}]
    gs = bot.add_gridspec(1, r + 2, width_ratios=[pad] + [1] * r + [pad])
    axs = [bot.add_subplot(gs[0, v + 1]) for v in range(r)]
    for v, a in enumerate(axs):
        lc, pv = edge_map(a, S[v], D, vm, [1, v, 1, 1], f"Vertex {v + 1}")
        panels.append(pv)
        colour_key(a, VERTEX[v])
    t = np.floor(vm * 100) / 100
    cb = bot.colorbar(lc, ax=axs, fraction=0.02, pad=0.01, shrink=0.8, ticks=[-t, 0, t], panchor=False)
    cb.ax.set_yticklabels([f"−{t:g}", "0", f"{t:g}"])
    cb.outline.set_visible(False)
    cb.set_label("weight")
    print(f"{name}: shared edge scale +/-{vm:.4f}", flush=True)
    save(fig, name, [2, r], panels, row_heights=heights)


def other_days(D, c13):
    with open(s14.CACHE, "rb") as fh:
        out = pickle.load(fh)
    runs = {"11-12": dict(ch=D["ch"], A=D["A"], felix=c13["felix"], induced=c13["induced"]),
            "08-23": out["08-23 no stim"], "11-11 no light": out["11-11 no light"], "11-11 light": out["11-11 light"]}
    others = list(runs)[1:]
    fig, (a, b) = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True, width_ratios=[1.25, 1])
    # blue and orange mean the evoked variants on the right, so no day here wears orange
    x = np.arange(16) + 0.5
    sa = []
    for (rn, r), col in zip(runs.items(), (BLUE, GREEN, PURPLE, GREY)):
        lam = s14.rank2(r["felix"], len(r["ch"]))[0]
        w = 3.2 if rn == "11-12" else 2.2
        a.plot(x, lam, lw=w, color=col, label=rn, zorder=3 if rn == "11-12" else 2)
        sa.append({"name": rn, "y": lam, "color": col, "line": "solid", "marker": False, "width": w})
    phase_axis(a)
    a.set_ylabel("movement vertex")
    a.set_title("Weights across the trial")
    a.legend(loc="upper left", bbox_to_anchor=(0.01, 1.0), borderaxespad=0, handlelength=1.4, labelspacing=0.3,
             frameon=True, facecolor=BG, edgecolor="none", framealpha=1)
    # correlation of move-minus-rest edges with 11-12, on the edges both days share
    y = np.arange(len(others))
    hgt = 0.36
    sb = []
    for j, (v, col, lab) in enumerate((("felix", BLUE, "evoked left in"), ("induced", ORANGE, "evoked removed"))):
        d0 = s14.rank2(runs["11-12"][v], len(runs["11-12"]["ch"]))[1]
        iu, ju = np.where(np.triu(runs["11-12"]["A"], 1))
        ref = {(int(runs["11-12"]["ch"][p]), int(runs["11-12"]["ch"][q])): d0[p, q] for p, q in zip(iu, ju)}
        raw, nedge = [], []
        for rn in others:
            r = runs[rn]
            dd = s14.rank2(r[v], len(r["ch"]))[1]
            iu, ju = np.where(np.triu(r["A"], 1))
            ed = {(int(r["ch"][p]), int(r["ch"][q])): dd[p, q] for p, q in zip(iu, ju)}
            common = sorted(set(ref) & set(ed))
            raw.append(np.corrcoef([ref[e] for e in common], [ed[e] for e in common])[0, 1])
            nedge.append(len(common))
        vals = [round(float(q), 2) for q in raw]    # bars and printed values are the same two-decimal numbers
        pos = y + (j - 0.5) * (hgt + 0.03)
        b.barh(pos, vals, hgt, color=col)
        for yy, cc in zip(pos, vals):
            b.text(cc + 0.015, yy, f"{cc:.2f}", va="center", color=s12.INK)
        b.text(vals[0] + 0.13, pos[0], lab, va="center", color=s12.INK)
        sb.append({"name": lab, "values": vals, "color": col})
        flag = "" if tuple(vals) == EXPECT_R[v] else f"  MISMATCH, scripts/14 logs {EXPECT_R[v]}"
        print(f"other_days: {lab} r = {[f'{q:.3f}' for q in raw]} over {nedge} edges{flag}", flush=True)
    b.set_yticks(y, others)
    b.invert_yaxis()
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 1)
    b.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    b.set_xlabel("correlation")
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color=s12.GRID, lw=0.7)
    b.set_title("Edge maps vs 11-12")
    panels = [{"kind": "line", "cell": [0, 0, 1, 1], "title": "Weights across the trial", "cat_groups": PHASE_GROUPS,
               "ylabel": "movement vertex", "ylim": [-0.05, 1.05], "legend": True, "series": sa},
              {"kind": "bar", "cell": [0, 1, 1, 1], "title": "Edge maps vs 11-12", "categories": others, "horizontal": True,
               "vlabel": "correlation", "vlim": [0, 1], "legend": True, "show_values": True, "series": sb}]
    save(fig, "other_days", [1, 2], panels, col_widths=[1.25, 1])


def main():
    style()
    os.makedirs(OUT, exist_ok=True)
    D = s12.load()
    with open(s12.CACHE, "rb") as fh:
        R = pickle.load(fh)
    with open(s13.CACHE, "rb") as fh:
        c13 = pickle.load(fh)
    jobs = {
        "graph_array": lambda: graph_array(D),
        "diffusion": lambda: diffusion(D),
        "gdar_fit": lambda: gdar_fit(D, c13),
        "tp_mix": lambda: tp_mix(D, c13),
        "tp_vs_gdar": lambda: tp_vs_gdar(R),
        "vertices_r3_felix": lambda: vertices(D, c13["felix"], 3, ((1.6, 1.12, "Vertex 1"), (14.5, 0.64, "Vertex 2"),
                                                                    (10.5, 1.05, "Vertex 3")), "vertices_r3_felix", 0.12),
        "vertices_r2_induced": lambda: vertices(D, c13["induced"], 2, ((2.5, 1.12, "Vertex 1"), (10.5, 1.12, "Vertex 2")),
                                                "vertices_r2_induced", 0.7),
        "other_days": lambda: other_days(D, c13),
    }
    for name in sys.argv[1:] or jobs:
        jobs[name]()


if __name__ == "__main__":
    main()
