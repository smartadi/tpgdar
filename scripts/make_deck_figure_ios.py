r"""IOS Reach slide figure: the events of a trial, and one real trial above the trial average.

Hydrogen 2021-11-12, no stim, through s12.load(): decimated x3, per-trial demeaned.  Nothing is picked by eye.

  trial     closest to the median go-cue-to-touch time, earliest trial on a tie.  The segment file has no
            separate reaction time: move_ms runs from the go cue to touch, so it includes the reaction.
  channels  the quarter, half and three-quarter positions of the kept channel list.
  window    -600..+800 ms re go cue, which fits every trial, so the average uses all of them.

The timeline places touch at median plan + median move.  The hold line is dashed and carries no duration:
the file starts 500 ms before the target and does not say how long the hold lasted.

Writes figures/deck/ios_reach.json, then draws figures/deck/ios_reach.png from that JSON, so the numbers in
the two are the same.  The JSON feeds native PowerPoint charts, at most 300 points per series, so the traces
are every 2nd sample (6 ms).  Trace y values are uV plus a channel offset; texts are centred on (x, y).

Run:  .venv\Scripts\python scripts/make_deck_figure_ios.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")

PNG = os.path.join(s12.FIG, "deck", "ios_reach.png")
JSN = os.path.join(s12.FIG, "deck", "ios_reach.json")
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8f98"
WIN = (-600.0, 800.0)       # ms re go cue, the traces
XLIM = (-700.0, 800.0)      # the extra room on the left holds the channel names
GAP = 150.0                 # uV between stacked channels
BAR = 50.0                  # uV, scale bar
STEP = 2                    # keep every 2nd sample in the series
YLIM = (-0.75 * GAP, 3.3 * GAP)
LABEL_Y = 3.1 * GAP         # event names
EVENT_TOP = 2.97 * GAP      # event lines stop under their names
NMAX = 300


def channel_name(c):
    """0-indexed 0..31 -> A1..A16, B1..B16 (iosreach/loader.py)."""
    return f"A{c + 1}" if c < 16 else f"B{c - 15}"


def go_aligned(D):
    """(trials, T, channels) on a common grid around the go cue, plus that grid."""
    X, t, go = D["X"], D["t"], D["go"]
    tg = np.arange(WIN[0], WIN[1] + 1, 3.0)
    assert (go + WIN[0] >= t[0]).all() and (go + WIN[1] <= t[-1]).all()
    Y = np.stack([np.stack([np.interp(tg, t - go[k], X[k, :, c]) for c in range(X.shape[2])], 1)
                  for k in range(len(go))])
    return Y, tg


def nums(a):
    return np.round(np.asarray(a, float), 3).tolist()


def timeline_panel(plan, move):
    touch = plan + move
    series = [dict(name="hold", x=[-500.0, 0.0], y=[0.0, 0.0], color=GREY, line="dash", marker=True, width=2.5),
              dict(name="plan and move", x=nums([0.0, plan, touch]), y=[0.0, 0.0, 0.0], color=GREY, line="solid",
                   marker=True, width=2.5)]
    texts = [dict(x=x, y=0.2, text=name) for x, name in
             ((-500.0, "hold"), (0.0, "target on"), (plan, "go cue"), (touch, "touch"))]
    texts += [dict(x=x, y=-0.2, text=f"{d:.0f} ms") for x, d in ((plan / 2, plan), (plan + move / 2, move))]
    return dict(kind="xy", cell=[0, 0, 2, 1], title="Trial timeline", xlabel="", ylabel="", xlog=False, ylog=False,
                xlim=nums([-720.0, touch + 170.0]), ylim=[-1.0, 1.0], legend=False, series=series, hlines=[], texts=texts)


def trace_panel(cell, title, xlabel, tt, Y, names, color, events):
    """Y (T, 3), stacked with the first channel on top, on one fixed scale."""
    off = GAP * np.arange(len(names))[::-1]
    series = [dict(name=name, x=nums(tt), y=nums(Y[:, j] + off[j]), color=color, line="solid", marker=False, width=1.4)
              for j, name in enumerate(names)]
    series += [dict(name=name, x=nums([x, x]), y=nums([YLIM[0], EVENT_TOP]), color=GREY, line="dash", marker=False, width=1.5)
               for x, name in events]
    x0, y0 = WIN[1] - 18.0, -0.6 * GAP
    series.append(dict(name=f"{BAR:.0f} uV scale", x=[x0, x0], y=nums([y0, y0 + BAR]), color=s12.INK, line="solid",
                       marker=False, width=2.5))
    texts = [dict(x=x, y=LABEL_Y, text=name) for x, name in events]
    texts += [dict(x=-655.0, y=o, text=name) for o, name in zip(off.tolist(), names)]
    texts.append(dict(x=x0 - 82.0, y=y0 + BAR / 2, text=f"{BAR:.0f} µV"))
    return dict(kind="xy", cell=cell, title=title, xlabel=xlabel, ylabel="", xlog=False, ylog=False, xlim=list(XLIM),
                ylim=list(YLIM), legend=False, series=series, hlines=[], texts=texts)


def draw(ax, p):
    dash = {"solid": "-", "dash": (0, (3, 2)), "none": "none"}
    for s in p["series"]:
        ax.plot(s["x"], s["y"], color=s["color"], lw=s["width"], ls=dash[s["line"]], solid_capstyle="butt",
                marker="|" if s["marker"] else None, ms=26, mew=2.5, mec=s12.INK)
    for tx in p["texts"]:
        ax.text(tx["x"], tx["y"], tx["text"], ha="center", va="center")
    ax.set_xlim(*p["xlim"])
    ax.set_ylim(*p["ylim"])
    ax.set_title(p["title"])
    ax.set_xlabel(p["xlabel"])
    ax.grid(False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = s12.load()
    go, touch, ang, ch, X = D["go"], D["touch"], D["ang"], D["ch"], D["X"]
    n, _, N = X.shape
    move = touch - go
    plan_med, move_med = float(np.median(go)), float(np.median(move))
    k = int(np.argmin(np.abs(move - move_med)))
    pick = [N // 4, N // 2, 3 * N // 4]
    names = [channel_name(int(ch[c])) for c in pick]

    Y, tg = go_aligned(D)
    mean = Y[:, :, pick].mean(0)
    tk = D["t"] - go[k]
    w = (tk >= WIN[0]) & (tk <= WIN[1])
    single = X[k][w][:, pick]

    spec = dict(grid=[2, 2], row_heights=[1, 1], col_widths=[1, 1.45], panels=[
        timeline_panel(plan_med, move_med),
        trace_panel([0, 1, 1, 1], "Single trial", "", tk[w][::STEP], single[::STEP], names, BLUE,
                    ((float(-go[k]), "target on"), (0.0, "go cue"), (float(move[k]), "touch"))),
        trace_panel([1, 1, 1, 1], "Trial average", "ms from go cue", tg[::STEP], mean[::STEP], names, ORANGE,
                    ((-plan_med, "target on"), (0.0, "go cue"), (move_med, "touch"))),
    ])
    assert all(len(s["x"]) <= NMAX and len(s["x"]) == len(s["y"]) for p in spec["panels"] for s in p["series"])
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    with open(JSN, "w", encoding="utf-8") as f:
        json.dump(spec, f, allow_nan=False)
    with open(JSN, encoding="utf-8") as f:
        spec = json.load(f)

    s12._style()
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
    fig = plt.figure(figsize=(12.1, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(*spec["grid"], height_ratios=spec["row_heights"], width_ratios=spec["col_widths"])
    axs = []
    for p in spec["panels"]:
        r, c, rs, cs = p["cell"]
        ax = fig.add_subplot(gs[r:r + rs, c:c + cs])
        draw(ax, p)
        axs.append(ax)
    axs[0].axis("off")
    for ax in axs[1:]:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(np.arange(WIN[0], WIN[1] + 1, 200))
    axs[1].tick_params(axis="x", labelbottom=False)
    fig.savefig(PNG, dpi=170)

    s12.log(f"{n} trials, {N} channels, median plan {plan_med} ms, median move {move_med} ms, "
            f"median touch {float(np.median(touch))} ms, sum of medians {plan_med + move_med} ms")
    s12.log(f"trial index {k} of the kept trials: plan {go[k]:.0f} ms, go-to-touch {move[k]:.0f} ms, angle {ang[k]} deg")
    s12.log(f"channels {names} (0-indexed {[int(ch[c]) for c in pick]})")
    s12.log(f"single trial range {single.min():.1f}..{single.max():.1f} uV, average range {mean.min():.1f}..{mean.max():.1f} uV")
    s12.log("points per trace series", len(spec["panels"][1]["series"][0]["x"]), len(spec["panels"][2]["series"][0]["x"]))
    s12.log("wrote", JSN)
    s12.log("wrote", PNG)


if __name__ == "__main__":
    main()
