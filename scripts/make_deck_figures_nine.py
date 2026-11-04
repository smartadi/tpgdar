r"""Low-text talk figures for the synthetic 9-node control example, written to figures/deck/.

Plots the cache of scripts/15_nine_node_control.py and noise-free runs of its true plants.  Nothing is
re-fitted.  Each figure is built as a panel spec; the PNG is drawn from that spec and the same spec is
dumped next to it as <name>.json for native PowerPoint charts, so the numbers match by construction.
Keys that start with "_" are PNG-only styling and are left out of the JSON.  "texts" are centre-anchored.

Run:  .venv\Scripts\python scripts/make_deck_figures_nine.py
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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s15 = importlib.import_module("15_nine_node_control")
s12 = s15.s12
from gdar_tp.graph import grid_graph  # noqa: E402
from gdar_tp.sym_tp import symmetric_basis, theta_of  # noqa: E402

OUT = os.path.join(ROOT, "figures", "deck")
K, N, EXAMPLE = s15.K, s15.N, s15.EXAMPLE
C_NONE, C_LQR, C_POLY = "#8a8f98", "#2a78d6", "#eb6834"
CTRL = (("open loop", "no control", C_NONE), ("pooled LQR", "pooled LQR", C_LQR), ("polytopic", "polytopic", C_POLY))
INK, MUTED, PAPER = s12.INK, s12.MUTED, "#fcfcfb"
GAIN_CMAP = LinearSegmentedColormap.from_list("gain", ["#f6f6f4", "#3a3a3a"])
GAIN_VMAX = 1.5
MAX_POINTS = 300
START_SEED = 0
V_STOP = 1e6
HOLD_P = 1.0
plain = FuncFormatter(lambda v, _: f"{v:g}")
power = FuncFormatter(lambda v, _: "1" if v == 1 else f"$10^{{{int(round(np.log10(v)))}}}$")


# ------------------------------------------------------------------------------ spec helpers
def _nums(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


def _thin(x, y, n=MAX_POINTS):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) <= n:
        return x, y
    i = np.unique(np.linspace(0, len(x) - 1, n).round().astype(int))
    return x[i], y[i]


def _series(name, x, y, color, line="solid", marker=False, width=2.0, **extra):
    x, y = _thin(x, y)
    return dict(name=name, x=_nums(x), y=_nums(y), color=color, line=line, marker=marker, width=width, **extra)


def _xy(cell, title, xlabel, ylabel, series, xlim, ylim, xlog=False, ylog=False, hlines=(), texts=(), **extra):
    return dict(kind="xy", cell=list(cell), title=title, xlabel=xlabel, ylabel=ylabel, xlog=xlog, ylog=ylog,
                xlim=list(xlim), ylim=list(ylim), legend=False, series=list(series), hlines=list(hlines),
                texts=list(texts), **extra)


def _text(x, y, text, color=None, **extra):
    """``color`` marks a line label; the PNG draws a swatch of that colour beside it."""
    t = dict(x=float(x), y=float(y), text=text, **extra)
    if color is not None:
        t["color"] = color
    return t


def _note(x, y, text):
    return _text(x, y, text, _muted=True)


def _p_panel(cell, p_t, xmax, title):
    return _xy(cell, title, "", "p", [_series("p", np.arange(len(p_t)), p_t, MUTED, width=1.3)], [0, xmax], [-0.12, 1.12],
               _yticks=[0, 1], _strip=True)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------------- drawing
def _tag(ax, t):
    kids = []
    if "color" in t:
        swatch = DrawingArea(24, 10, 0, 0)
        swatch.add_artist(Line2D([0, 24], [5, 5], color=t["color"], lw=2.6, ls=t.get("_ls", "-"), solid_capstyle="round"))
        kids.append(swatch)
    kids.append(TextArea(t["text"], textprops=dict(color=MUTED if t.get("_muted") else INK, fontsize=14)))
    tag = AnnotationBbox(HPacker(children=kids, align="center", pad=0, sep=6), (t["x"], t["y"]), frameon=False, pad=0,
                         annotation_clip=False, zorder=6, box_alignment=(0.5, 0.5))
    tag.set_in_layout(False)
    ax.add_artist(tag)


def _draw_xy(ax, pn):
    if "_shade_to" in pn:
        ax.axhspan(1, pn["_shade_to"], color=C_NONE, alpha=0.16, lw=0, zorder=0)
    for h in pn["hlines"]:
        ax.axhline(h["y"], color=h["color"], lw=1, ls="--" if h["dash"] else "-")
    for s in pn["series"]:
        ax.plot(s["x"], s["y"], color=s["color"], lw=s["width"], alpha=s.get("_alpha", 1), ls={"solid": "-", "dash": "--", "none": "None"}[s["line"]],
                marker="o" if s["marker"] else None, ms=s.get("marker_size", 8), mec=PAPER, mew=1.3, zorder=5 if s["marker"] else 2)
    if pn["xlog"]:
        ax.set_xscale("log")
    if pn["ylog"]:
        ax.set_yscale("log")
    ax.set_xlim(pn["xlim"])
    ax.set_ylim(pn["ylim"])
    ax.set_xlabel(pn["xlabel"])
    ax.set_ylabel(pn["ylabel"])
    ax.set_title(pn["title"])
    if "_xticks" in pn:
        ax.set_xticks(pn["_xticks"])
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(NullFormatter())
    if "_yticks" in pn:
        ax.set_yticks(pn["_yticks"])
        ax.yaxis.set_major_formatter(power if pn.get("_power") else plain)
        ax.yaxis.set_minor_locator(NullLocator())
    if pn.get("_no_ylabels"):
        ax.tick_params(labelleft=False)
    if pn.get("_strip"):
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_ylabel(pn["ylabel"], rotation=0, va="center", labelpad=8)
    for t in pn["texts"]:
        _tag(ax, t)


def _draw_network(ax, pn):
    """Edge colour = sign, width = size.  Node fill = value, 0 to node_vmax."""
    xy = np.array(pn["pos"])
    for (i, j), v in zip(pn["edges"], pn["edge_values"]):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=C_LQR if v > 0 else C_POLY, lw=0.6 + 7 * abs(v) / pn["vmax"],
                solid_capstyle="round", zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=430, c=pn["node_values"], cmap=GAIN_CMAP, norm=Normalize(0, pn["node_vmax"]),
               edgecolors="#333333", linewidths=0.9, zorder=2)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-2.4, 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(pn["title"])


def _draw_strip(ax, pn):
    """One dot per plant, a bar at the median."""
    rng = np.random.default_rng(0)
    for i, v in enumerate(pn["values"]):
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, s=34, color=pn["color"], alpha=0.6, lw=0)
        ax.plot([i - 0.24, i + 0.24], [np.median(v)] * 2, color=INK, lw=2.2, solid_capstyle="round")
    ax.set_xticks(range(len(pn["categories"])), pn["categories"])
    ax.set_xlim(-0.55, len(pn["categories"]) - 0.45)
    if pn["ylog"]:
        ax.set_yscale("log")
    ax.set_ylim(pn["ylim"])
    ax.set_yticks(pn["_yticks"])
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(pn["ylabel"])
    ax.set_title(pn["title"])


def _graph_key(ax):
    """PNG key for the graphs.  The JSON carries the same facts in its top-level "key"."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.12, 0.18], [0.5, 0.5], color=C_LQR, lw=3.5, solid_capstyle="round")
    ax.text(0.195, 0.5, "+", va="center")
    ax.plot([0.27, 0.33], [0.5, 0.5], color=C_POLY, lw=3.5, solid_capstyle="round")
    ax.text(0.345, 0.5, "−", va="center")
    for i, b in enumerate((0.5, 1.0, 1.5)):
        ax.scatter(0.47 + 0.05 * i, 0.5, s=170, color=GAIN_CMAP(Normalize(0, GAIN_VMAX)(b)), edgecolors="#333333", linewidths=0.8)
    ax.text(0.595, 0.5, "input gain", va="center")


def render(spec, name, key_cell=None):
    nrow, ncol = spec["grid"]
    fig = plt.figure(figsize=(12.1, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(nrow, ncol, height_ratios=spec["row_heights"], width_ratios=spec["col_widths"])
    draw = {"xy": _draw_xy, "network": _draw_network, "strip": _draw_strip}
    for pn in spec["panels"]:
        r, c, rs, cs = pn["cell"]
        draw[pn["kind"]](fig.add_subplot(gs[r:r + rs, c:c + cs]), pn)
    if key_cell is not None:
        r, c, rs, cs = key_cell
        _graph_key(fig.add_subplot(gs[r:r + rs, c:c + cs]))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=170)
    plt.close(fig)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(_clean(spec), f, allow_nan=False)
    s12.log("wrote", name + ".png", name + ".json")


# ----------------------------------------------------------------------------- 1, learning
def spec_learning(plants, A):
    g = grid_graph(3, 3)
    xy = np.column_stack([g.coords()[:, 1], -g.coords()[:, 0]]).astype(float)
    R = symmetric_basis(A, K, node_inputs=True)
    r = plants[EXAMPLE]
    order = np.argsort(-r["poly"]["lam"][0])
    th = [theta_of(r["ph0"], R, N), theta_of(r["ph1"], R, N),
          theta_of(r["poly"]["phi"][order[0]], R, N), theta_of(r["poly"]["phi"][order[1]], R, N)]
    summed = [sum(t[:, k * N:(k + 1) * N] for k in range(K)) for t in th]
    vmax = float(max(np.abs(S[~np.eye(N, dtype=bool)]).max() for S in summed))
    iu, ju = np.where(np.triu(A, 1))
    panels = []
    for cell, t, S, title in zip(((0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)), th, summed,
                                 ("True, p = 0", "True, p = 1", "Learned 1", "Learned 2")):
        panels.append(dict(kind="network", cell=list(cell), title=title, pos=xy.tolist(),
                           edges=[[int(i), int(j)] for i, j in zip(iu, ju)], edge_values=_nums(S[iu, ju]), vmax=vmax,
                           node_values=_nums(np.diag(t[:, K * N:])), node_vmax=GAIN_VMAX, node_cmap="seq"))
    lines = []
    for i, q in enumerate(plants):
        o = np.argsort(-q["poly"]["lam"][0])
        lines.append(_series(f"learned, plant {i}", s15.P_GRID, q["poly"]["lam"][:, o[1]], C_POLY, width=1.2, _alpha=0.55))
    lines.append(_series("true", [0, 1], [0, 1], INK, line="dash", width=1.3))
    panels.append(_xy((0, 2, 1, 1), "Weights", "p", "weight", lines, [-0.03, 1.03], [-0.05, 1.05], _xticks=[0, 0.5, 1], _yticks=[0, 0.5, 1],
                      texts=[_text(0.2, 0.82, "true", INK, _ls="--"), _text(0.72, 0.18, "learned", C_POLY)]))
    panels.append(dict(kind="strip", cell=[1, 2, 2, 1], title="Error", categories=["pooled", "rank 2"],
                       values=[_nums([q["err"][k] for q in plants]) for k in ("pooled", "rank 2")], ylabel="relative error",
                       ylog=True, ylim=[0.002, 0.5], color=MUTED, median=True, _yticks=[0.01, 0.1]))
    return dict(grid=[3, 3], row_heights=[1, 1, 0.1], col_widths=[1, 1, 1.5], panels=panels,
                key=dict(edge_positive=C_LQR, edge_negative=C_POLY, positive_label="+", negative_label="−", node_label="input gain"))


# ------------------------------------------------------------------------------ 2, control
def spec_control(plants):
    left = [_series(label, s15.P_DENSE, np.median(np.stack([q["rho"][key] for q in plants]), 0), c, width=2.6) for key, label, c in CTRL]
    r = plants[EXAMPLE]
    right = []
    for key, label, c in CTRL:
        x = r["sims"][key][0]
        trace = np.sqrt(np.convolve(np.mean(x ** 2, 1), np.ones(25) / 25, mode="valid"))
        right.append(_series(label, np.arange(len(trace)) + 12, trace, c, width=2.0))
    one = [dict(y=1, color=MUTED, dash=False)]
    T = len(r["p_t"])
    panels = [
        _xy((0, 0, 2, 1), "Across p", "p", "growth per step", left, [0, 1], [0, 1.25], hlines=one, _shade_to=1.25, _xticks=[0, 0.5, 1], _yticks=[0, 0.5, 1],
            texts=[_note(0.5, 1.125, "unstable"), _text(0.5, 0.79, "no control", C_NONE), _text(0.5, 0.615, "pooled LQR", C_LQR), _text(0.5, 0.29, "polytopic", C_POLY)]),
        _p_panel((0, 1, 1, 1), r["p_t"], T, "As p moves"),
        _xy((1, 1, 1, 1), "", "time step", "activity", right, [0, T], [0.1, 200], ylog=True, _yticks=[0.1, 1, 10, 100],
            texts=[_text(620, 110, "no control", C_NONE), _text(350, 0.165, "pooled LQR", C_LQR), _text(1000, 0.165, "polytopic", C_POLY)]),
    ]
    return dict(grid=[2, 2], row_heights=[1, 6.5], col_widths=[1, 1.1], panels=panels)


# ------------------------------------------------------------------------------- 3, effort
def spec_effort(scenarios):
    one = [dict(y=1, color=MUTED, dash=False)]
    panels = []
    for col, (name, title) in enumerate((("one unstable", "One unstable end"), ("two unstable", "Two unstable ends"))):
        plants = scenarios[name]
        series = []
        for key, c in (("pooled LQR", C_LQR), ("polytopic", C_POLY)):
            n_set = len(plants[0]["sweep"][key])
            wr = np.stack([[q["sweep"][key][i]["worst_rho"] for i in range(n_set)] for q in plants])
            ru = np.stack([[q["sweep"][key][i]["rms_u"] for i in range(n_set)] for q in plants])
            series.append(_series(key, np.nanmedian(ru, 0), np.median(wr, 0), c, marker=True, width=2.4))
        panels.append(_xy((0, col, 1, 1), title, "input used", "growth per step" if col == 0 else "", series, [0.012, 0.3], [0.48, 1.06], xlog=True, hlines=one,
                          _shade_to=1.06, _xticks=[0.02, 0.05, 0.1, 0.2], _yticks=[0.5, 0.75, 1], _no_ylabels=col == 1,
                          texts=[_note(0.06, 1.03, "unstable"), _text(0.045, 0.575, "polytopic", C_POLY), _text(0.165, 0.635, "pooled LQR", C_LQR)] if col == 0 else []))
    return dict(grid=[1, 2], row_heights=None, col_widths=[1, 1], panels=panels)


# ------------------------------------------------------------------- 4, polytopic explainer
def energy_ratio(r, R, P, F):
    """Largest eigenvalue of P^(-1/2) A_cl(p)' P A_cl(p) P^(-1/2), the worst one-step energy ratio."""
    w, U = np.linalg.eigh(P)
    Pmh = (U / np.sqrt(w)) @ U.T
    out = []
    for p in s15.P_DENSE:
        Ac, Bc = s15.comp(s15.truth_at(r["ph0"], r["ph1"], R, p))
        Acl = Ac + (Bc @ F if F is not None else 0)
        M = Pmh @ Acl.T @ P @ Acl @ Pmh
        out.append(np.linalg.eigvalsh((M + M.T) / 2).max())
    return np.array(out)


def energy_run(r, R, P, F, p_t, xi0, stop=None):
    """V = xi' P xi on the noise-free true plant, optionally stopped once V passes ``stop``."""
    xi, V = xi0.copy(), [xi0 @ P @ xi0]
    for t in range(len(p_t) - 1):
        Ac, Bc = s15.comp(s15.truth_at(r["ph0"], r["ph1"], R, p_t[t]))
        xi = (Ac + (Bc @ F if F is not None else 0)) @ xi
        V.append(xi @ P @ xi)
        if stop is not None and V[-1] > stop:
            break
    return np.array(V)


def spec_explainer(plants, A, p_t, ylim, yticks, tags):
    """Left: worst one-step energy ratio against p.  Right: one noise-free run under ``p_t``."""
    R = symmetric_basis(A, K, node_inputs=True)
    r = plants[EXAMPLE]
    P, F = np.linalg.inv(r["Q"]), r["F_poly"]
    p = s15.P_DENSE
    with_gain, without = energy_ratio(r, R, P, F), energy_ratio(r, R, P, None)
    gap = (1 - p) * with_gain[0] + p * with_gain[-1] - with_gain
    s12.log(f"with gain: ends {with_gain[0]:.4f} {with_gain[-1]:.4f}, next in {with_gain[1]:.4f} {with_gain[-2]:.4f}, min {with_gain.min():.4f}; "
            f"chord minus curve: min {gap.min():.2e}, max {gap.max():.4f}")
    s12.log(f"no control: ends {without[0]:.3f} {without[-1]:.3f}, min {without.min():.3f}, max {without.max():.3f}, "
            f"chord minus curve: max {((1 - p) * without[0] + p * without[-1] - without).max():.3f}")
    left = [_series("no control", p, without, C_NONE, width=2.6),
            _series("no control, chord", [0, 1], [without[0], without[-1]], C_NONE, line="dash", width=1.0),
            _series("with the gain", p, with_gain, C_POLY, width=2.6),
            _series("with the gain, chord", [0, 1], [with_gain[0], with_gain[-1]], INK, line="dash", width=1.0),
            _series("vertex", [0, 1], [with_gain[0], with_gain[-1]], C_POLY, line="none", marker=True, width=0.0, marker_size=14)]

    xi0 = np.random.default_rng(START_SEED).standard_normal(2 * N)
    xi0 /= np.linalg.norm(xi0)
    V_gain = energy_run(r, R, P, F, p_t, xi0)
    V_none = energy_run(r, R, P, None, p_t, xi0, stop=V_STOP)
    s12.log(f"no-control run: {len(V_none)} steps, V start {V_none[0]:.3g}, peak {V_none.max():.3g}, end {V_none[-1]:.3g}; passed {V_STOP:g}: {bool(V_none.max() > V_STOP)}")
    below = np.flatnonzero(V_gain < ylim[0])
    end = below[0] if len(below) else len(V_gain)
    s12.log(f"gain run: falls every step until the axis floor {ylim[0]:g}: {bool(np.all(np.diff(V_gain[:end]) < 0))}, floor reached at step {end}")
    right = [_series("no control", np.arange(len(V_none)), V_none, C_NONE, width=2.2),
             _series("with the gain", np.arange(end), V_gain[:end], C_POLY, width=2.2)]
    xmax = len(p_t)
    panels = [
        _xy((0, 0, 2, 1), "Energy change per step", "p", "worst energy ratio", left, [-0.03, 1.03], [0.5, 2], hlines=[dict(y=1, color=MUTED, dash=False)],
            _xticks=[0, 0.5, 1], _yticks=[0.5, 1, 1.5, 2],
            texts=[_note(0.5, 1.09, "energy grows"), _note(0.5, 0.91, "energy shrinks"), _text(0.07, 0.68, "vertex"), _text(0.93, 0.68, "vertex"),
                   _text(0.5, 1.40, "no control", C_NONE), _text(0.5, 0.68, "with the gain", C_POLY)]),
        _p_panel((0, 1, 1, 1), p_t, xmax, "Energy over time"),
        _xy((1, 1, 1, 1), "", "time step", "energy", right, [0, xmax], ylim, ylog=True, _yticks=yticks, _power=True,
            texts=[_text(*tags["none"], "no control", C_NONE), _text(*tags["gain"], "with the gain", C_POLY)]),
    ]
    return dict(grid=[2, 2], row_heights=[1, 6.5], col_widths=[1, 1.1], panels=panels)


def main():
    os.makedirs(OUT, exist_ok=True)
    s12._style()
    matplotlib.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                                "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
    with open(s15.CACHE, "rb") as f:
        out = pickle.load(f)
    sc, A = out["scenarios"], out["A"]
    render(spec_learning(sc["one unstable"], A), "nine_learning", key_cell=(2, 0, 1, 2))
    render(spec_control(sc["two unstable"]), "nine_control")
    render(spec_effort(sc), "nine_effort")
    render(spec_explainer(sc["two unstable"], A, s15.schedule(), [1e-36, 1e3], [1e-30, 1e-20, 1e-10, 1],
                          dict(none=(760, 1e-5), gain=(300, 1e-31))), "polytopic_explainer")
    # Same left panel, p parked at one unstable end.  Under the schedule above the no-control run shrinks overall.
    render(spec_explainer(sc["two unstable"], A, np.full(300, HOLD_P), [1e-14, 1e7], [1e-10, 1e-5, 1, 1e5],
                          dict(none=(190, 10 ** 0.5), gain=(85, 1e-11))), "polytopic_explainer_hold")


if __name__ == "__main__":
    main()
