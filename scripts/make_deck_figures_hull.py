r"""Figures for TP-GDAR_stable_vertices.pptx (the 2026-11-23 update).  Nothing is fitted or recomputed here.

Reads cache/13_vertices_H_2021-11-12.pkl and cache/18_single_trial_H_2021-11-12.pkl (written by scripts 13 and 18 on
2026-11-22, with the MATH §9.6 hull and its rotation rule) and writes figures/deck/<name>.png, plus <name>.json for the
panels the slides draw as shapes (scripts/deck_native.py):

  hull_r2_geometry   rank 2, evoked left in: the 16 bins in the pooled data metric (cached Y), the kept triangle (cached
                     Yv) and the same triangle at tau = 1.5, which is Yv scaled by 1.5 about its centre (MATH §9.6) (PNG only)
  hull_rotation_r2   rank 2, both variants: size h and the largest vertex spectral radius over rotations at tau = 1,
                     from script 13's sweep
  hull_weights_r3    rank 3 vertex weights across the trial, evoked left in and removed
  hull_maps_r3       rank 3, evoked left in: each vertex's edge map and its model spectrum re GDAR (as script 13's figure)
  single_trial_r3    one held-out trial's weights read off 60 ms of data, and the move vertex by go-time thirds
                     (as script 18's figure)
  single_trial_gain  held-out gain of the clock weights and of weights read off 30, 60, 120 ms of past data

Numbers that go on slides are printed.

Run:  .venv\Scripts\python scripts/make_deck_figures_hull.py
"""

import importlib
import os
import pickle
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
real = importlib.import_module("make_deck_figures_real")
s12, s13 = real.s12, real.s13
s18 = importlib.import_module("18_single_trial_states")
from gdar_tp.sym_tp import theta_of  # noqa: E402

plt = real.plt
OUT = real.OUT
COL = tuple(s13.COL)                    # the four vertex colours of scripts 13 and 18
BLUE, ORANGE, GREY = real.BLUE, real.ORANGE, real.GREY
VARIANTS = (("felix", "Evoked left in"), ("induced", "Evoked removed"))
W60 = 20                                # 20 samples at 3 ms = 60 ms, script 18's middle window


def geometry(c13):
    p = c13["felix"]["full"][2]["poly"]
    Y, kept = p["Y"], p["Yv"]
    centre = kept.mean(0)
    loose = centre + 1.5 * (kept - centre)
    print(f"geometry: kept angle {np.rad2deg(p['angles'][0]):.1f} deg, h {p['h']:.3f}, tau {p['tau']}, "
          f"rho {np.round(p['rho'], 3)}, distances {np.round(p['dist'], 2)}")
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    close = lambda V: np.vstack([V, V[:1]])  # noqa: E731
    ax.plot(*close(loose).T, color=GREY, lw=1.8, ls="--")
    ax.plot(*close(kept).T, color=s12.INK, lw=2.4)
    ax.plot(*Y.T, "-", color="#c9c6bf", lw=1.2, zorder=2)
    for ph, shade in enumerate(("#d5d2cb", "#a19d94", "#5f5b54", "#1c1b19")):     # light to dark through the trial
        ax.scatter(*Y[4 * ph:4 * ph + 4].T, s=60, color=shade, edgecolor="white", lw=0.8, zorder=3, label=s12.PHASES[ph])
    ax.legend(loc="upper right", fontsize=14, frameon=False, handletextpad=0.2)
    for v, xy in enumerate(kept):   # vertex number inside its dot
        ax.scatter(*xy, s=420, color=COL[v], zorder=4, edgecolor="white", lw=1.5)
        ax.text(*xy, str(v + 1), ha="center", va="center", fontsize=14, color="white", fontweight="bold", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    lo, hi = loose.min(0), loose.max(0)
    pad = 0.08 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    fig.savefig(os.path.join(OUT, "hull_r2_geometry.png"), dpi=220, transparent=True)
    plt.close(fig)
    print("wrote hull_r2_geometry .png")


def rotation(c13):
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.0, 5.6), constrained_layout=True, sharex=True)
    sa, sb = [], []
    for (v, lab), col in zip(VARIANTS, (BLUE, ORANGE)):
        f = c13[v]["full"][2]
        sw = f["sweep"]
        deg = np.rad2deg(sw["angles"][:, 0])
        h, top = sw["h"][:, 0], sw["rho"][:, 0].max(1)
        a.plot(deg, h, color=col, lw=2.2)
        b.plot(deg, top, color=col, lw=2.2)
        sa.append({"name": lab, "x": deg, "y": h, "color": col, "line": "solid", "marker": False, "width": 2.2})
        sb.append({"name": lab, "x": deg, "y": top, "color": col, "line": "solid", "marker": False, "width": 2.2})
        k = float(np.rad2deg(f["poly"]["angles"][0]) % 120)
        for ax, ss, yv in ((a, sa, f["poly"]["h"]), (b, sb, max(f["poly"]["rho"]))):
            ax.plot([k], [yv], "o", color=col, ms=9)
            ss.append({"name": None, "x": [k], "y": [yv], "color": col, "line": "none", "marker": True, "width": 2.2})
        print(f"rotation {v}: all vertices stable at {np.mean(top < 1):.0%} of 120 rotations; kept {k:.1f} deg, "
              f"h {f['poly']['h']:.3f}, largest rho {max(f['poly']['rho']):.3f}")
    b.axhline(1, color=s12.MUTED, lw=1, ls="--")
    a.set_ylabel("size h")
    b.set_ylabel("largest vertex ρ")
    b.set_xlabel("rotation (degrees)")
    real.save(fig, "hull_rotation_r2", [2, 1], [
        {"kind": "xy", "cell": [0, 0, 1, 1], "title": "Size of the tight triangle", "xlabel": "", "ylabel": "size h",
         "xlim": [0, 120], "legend": True, "series": sa, "texts": []},
        {"kind": "xy", "cell": [1, 0, 1, 1], "title": "Largest vertex spectral radius", "xlabel": "rotation (degrees)",
         "ylabel": "ρ", "xlim": [0, 120], "legend": False, "series": sb, "texts": [],
         "hlines": [{"y": 1, "color": s12.MUTED, "dash": True}]}])


def weights(c13):
    fig, axs = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True, sharey=True)
    panels = []
    for i, ((v, lab), ax) in enumerate(zip(VARIANTS, axs)):
        f = c13[v]["full"][3]
        lam, se = f["poly"]["lam"], f["lam_se"]
        x = np.arange(16) + 0.5
        series = []
        for k in range(4):
            ax.fill_between(x, lam[:, k] - se[:, k], lam[:, k] + se[:, k], color=COL[k], alpha=0.18, lw=0)
            ax.plot(x, lam[:, k], "-o", ms=5, lw=2.4, color=COL[k], label=f"Vertex {k + 1}")
            series.append({"name": f"Vertex {k + 1}", "y": lam[:, k], "err": se[:, k], "color": COL[k], "line": "solid",
                           "marker": True, "width": 2.4})
        real.phase_axis(ax)
        ax.set_title(lab)
        panels.append({"kind": "line", "cell": [0, i, 1, 1], "title": lab, "cat_groups": real.PHASE_GROUPS,
                       "ylabel": "weight" if i == 0 else "", "ylim": [-0.05, 1.05], "legend": i == 0, "series": series,
                       "texts": []})
        print(f"weights {v} rank 3 (rows: bins hold 1/4 .. post 4/4; columns: vertices):")
        print(np.round(lam, 2))
    axs[0].legend(ncol=4, loc="upper center")
    real.save(fig, "hull_weights_r3", [1, 2], panels)


def maps(D, c13):
    res = c13["felix"]
    p = res["full"][3]["poly"]
    R, N, Sig = res["R"], len(D["pos"]), res["Sigma"]
    S = [real.summed(ph, res, N) for ph in p["phi"]]
    _, iu, ju = real.edges_of(D)
    vm = max(np.abs(m[iu, ju]).max() for m in S)
    rhos = [np.abs(s12.rho(theta_of(ph, R, N), real.K)).max() for ph in p["phi"]]
    fig = plt.figure(figsize=(12.1, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    panels = []
    for v in range(4):
        ax = fig.add_subplot(gs[0, v])
        title = f"Vertex {v + 1}, ρ = {rhos[v]:.3f}"
        panels.append(real.edge_map(ax, S[v], D, vm, [0, v, 1, 1], title)[1])
        real.colour_key(ax, COL[v])
    ax = fig.add_subplot(gs[1, :])
    freqs = np.arange(2.0, 101.0, 1.0)
    base = s12.ar_spectrum(theta_of(res["phi0"], R, N), Sig, real.K, freqs)
    series = []
    for v in range(4):
        db = 10 * np.log10(s12.ar_spectrum(theta_of(p["phi"][v], R, N), Sig, real.K, freqs) / base)
        ax.plot(freqs, db, color=COL[v], lw=2.2, label=f"Vertex {v + 1}")
        series.append({"name": f"Vertex {v + 1}", "x": freqs, "y": db, "color": COL[v], "line": "solid", "marker": False,
                       "width": 2.2})
        print(f"maps: vertex {v + 1} rho {rhos[v]:.3f}, spectrum re GDAR max {db.max():+.1f} dB at {freqs[db.argmax()]:.0f} Hz, "
              f"min {db.min():+.1f} dB at {freqs[db.argmin()]:.0f} Hz")
    s12._zero(ax)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB re GDAR")
    ax.legend(ncol=4)
    panels.append({"kind": "xy", "cell": [1, 0, 1, 4], "title": "Model spectrum of each vertex, relative to GDAR",
                   "xlabel": "frequency (Hz)", "ylabel": "dB", "xlim": [0, 100], "legend": True, "series": series,
                   "hlines": [real.ZERO_LINE], "texts": []})
    print(f"maps: edge scale {vm:.3f}")
    real.save(fig, "hull_maps_r3", [2, 4], panels, row_heights=[1.15, 1])


def single_trial(C):
    go, touch = C["go"], C["touch"]
    F = C["felix"]
    te = F["te"]
    hat = np.concatenate([r["hat"][W60] for r in F["folds"]])
    clock = np.concatenate([r["clock"] for r in F["folds"]])
    tt = F["folds"][0]["tt"]
    move_v = int(np.argmax(F["ref"][2 * s13.NSUB:3 * s13.NSUB].mean(0)))
    k = np.argsort(go[te])[int(0.5 * (len(te) - 1))]
    i = te[k]
    fig, (a, b) = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True, width_ratios=[1.5, 1])
    sa, ta = [], []
    for v in range(4):
        a.plot(tt, hat[k][:, v], color=COL[v], lw=1.6)
        a.plot(tt, clock[k][:, v], color=COL[v], lw=1.8, ls="--")
        sa.append({"name": f"Vertex {v + 1}", "x": tt, "y": hat[k][:, v], "color": COL[v], "line": "solid", "marker": False,
                   "width": 1.4})
        sa.append({"name": None, "x": tt, "y": clock[k][:, v], "color": COL[v], "line": "dash", "marker": False,
                   "width": 1.8})
    for x, lab in ((0.0, "target"), (float(go[i]), "go"), (float(touch[i]), "touch")):
        a.axvline(x, color=s12.MUTED, lw=1, ls=":")
        sa.append({"name": None, "x": [x, x], "y": [-0.02, 1.0], "color": s12.MUTED, "line": "dash", "marker": False,
                   "width": 1.0})
        ta.append({"x": x, "y": 1.08, "text": lab})
    a.set_xlim(-400, 1400)
    a.set_xlabel("ms from target on")
    a.set_title(f"Trial {i}: solid from 60 ms of data, dashed the clock")
    sb = []
    terc = np.quantile(go[te], [1 / 3, 2 / 3])
    for which, (lo, hi), shade in zip(("early", "middle", "late"), ((-np.inf, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)),
                                      (0.35, 0.65, 1.0)):
        sel = (go[te] >= lo) & (go[te] < hi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(hat[sel][:, :, move_v], 0)
        c = np.array(plt.matplotlib.colors.to_rgb(COL[move_v])) * shade + (1 - shade)
        hexc = "#" + "".join(f"{int(round(255 * q)):02x}" for q in c)
        g = float(go[te][sel].mean())
        b.plot(tt, m, color=hexc, lw=2.0, label=f"{which} go, mean {g:.0f} ms")
        b.axvline(g, color=hexc, lw=1.2, ls=":")
        sb.append({"name": f"{which} go", "x": tt, "y": m, "color": hexc, "line": "solid",
                   "marker": False, "width": 2.0})
        sb.append({"name": None, "x": [g, g], "y": [0, 0.75], "color": hexc, "line": "dash", "marker": False, "width": 1.2})
        print(f"single trial: {which} third n {sel.sum()}, mean go {g:.0f} ms")
    b.set_xlim(-400, 1400)
    b.set_xlabel("ms from target on")
    b.set_title(f"Vertex {move_v + 1} (move), mean over trials")
    b.legend(loc="upper right", fontsize=12)
    print(f"single trial: example trial {i}, go {go[i]:.0f} ms, touch {touch[i]:.0f} ms; move vertex {move_v + 1}")
    real.save(fig, "single_trial_r3", [1, 2], [
        {"kind": "xy", "cell": [0, 0, 1, 1], "title": f"Trial {i}: solid, from 60 ms of data; dashed, the clock",
         "xlabel": "ms from target on", "ylabel": "weight", "xlim": [-400, 1400], "ylim": [-0.02, 1.12], "legend": False,
         "series": sa, "texts": ta},
        {"kind": "xy", "cell": [0, 1, 1, 1], "title": f"Vertex {move_v + 1} (move), mean by go time",
         "xlabel": "ms from target on", "ylabel": "", "xlim": [-400, 1400], "ylim": [0, 0.8], "legend": True,
         "series": sb, "texts": []}], col_widths=[1.5, 1])


def gains(C):
    cats = ["clock", "past 30 ms", "past 60 ms", "past 120 ms"]
    keys = ["clock", 10, 20, 40]
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    series = []
    for j, ((v, lab), col) in enumerate(zip(VARIANTS, (BLUE, ORANGE))):
        g = [float(C[v]["gain"][k]) for k in keys]
        se = [float(C[v]["gain_se"][k]) for k in keys]
        ax.bar(np.arange(4) + (j - 0.5) * 0.38, g, 0.38, yerr=se, color=col, label=lab)
        series.append({"name": lab, "values": [round(q, 3) for q in g], "color": col})
        print(f"gains {v}: " + ", ".join(f"{c} {q:+.4f} ({e:.4f})" for c, q, e in zip(cats, g, se)))
    ax.set_xticks(range(4), cats)
    ax.set_ylabel("gain over GDAR (nats/sample)")
    ax.legend()
    real.save(fig, "single_trial_gain", [1, 1], [
        {"kind": "bar", "cell": [0, 0, 1, 1], "title": "", "categories": cats, "series": series, "vlabel": "nats/sample",
         "ylabel": "gain over GDAR (nats/sample)", "show_values": True, "legend": True, "vlim": [0, 0.3]}])


def main():
    real.style()
    os.makedirs(OUT, exist_ok=True)
    with open(s13.CACHE, "rb") as fh:
        c13 = pickle.load(fh)
    with open(s18.CACHE, "rb") as fh:
        c18 = pickle.load(fh)
    D = s12.load()
    geometry(c13)
    rotation(c13)
    weights(c13)
    maps(D, c13)
    single_trial(c18)
    gains(c18)


if __name__ == "__main__":
    main()
