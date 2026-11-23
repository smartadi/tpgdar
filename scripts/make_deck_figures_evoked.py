r"""Figures for TP-GDAR_evoked_overlap.pptx (2026-11-23): the evoked response, the noart steps, overlapping bins (MATH §2.4),
the step mask and the null test (script 20).

Nothing new is fitted.  The bowls figure draws the separate GLS fit of three bins (MATH §6.1, §7.1) on all trials, with and
without overlap; everything else reads caches:
  cache/13_vertices_H_2021-11-12<suffix>.pkl   suffix "", _overlap0.5, _overlap1, _overlap2, _mask, _overlap1_mask, _null,
                                               _nullnowave, _mask_null, _overlap1_mask_null
  cache/18_single_trial_H_2021-11-12<suffix>.pkl  suffix "", _overlap0.5, _overlap1, _overlap2, _mask, _overlap1_mask
  cache/20_null_H_2021-11-12.npz
and the segment file for the steps.  Writes figures/deck/ev_*.png and .json (scripts/deck_native.py draws the .json as
shapes).  Numbers that go on slides are printed.

  ev_wave      the trial-mean LFP of three channels with the fitted evoked wave, and the size of the trial mean over time for
               the recordings, the recordings with evoked removed, and the null data
  ev_steps     median sample-to-sample change at 1 kHz around target onset and 900 ms, noart against raw lfp, with the span
               of targets the mask drops
  ev_weights_a overlap weights a_g(tau) of hold 4/4, plan 1/4 and plan 2/4 for windows of 0, 1 and 2 bins
  ev_bowls     one coefficient's bowl in three bins, without and with a 1-bin overlap
  ev_gain      held-out gain over GDAR against the overlap, with and without the mask
  ev_weights   rank 3 weights, no overlap and 1-bin overlap, without and with the mask (evoked left in)
  ev_null_gain held-out gain on the recordings and on the null data
  ev_null_weights rank 3 weights on the recordings and on the null data

Run:  .venv\Scripts\python scripts/make_deck_figures_evoked.py
"""

import importlib
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
real = importlib.import_module("make_deck_figures_real")
s12, s13 = real.s12, real.s13
s12b = importlib.import_module("12b_tp_vs_gdar_robust")
from gdar_tp.bins import overlap_weights, weighted_bin_stats  # noqa: E402
from gdar_tp.sym_tp import _solve, fit_gls_pooled, gls_quads, symmetric_basis  # noqa: E402

plt = real.plt
BLUE, ORANGE, GREEN, PURPLE, GREY = real.BLUE, real.ORANGE, real.GREEN, real.PURPLE, real.GREY
INK, MUTED = s12.INK, s12.MUTED
ROLE_NAMES = ("peaks in hold", "peaks in move", "peaks in post", "fourth vertex")
ROLE_COL = (GREEN, PURPLE, ORANGE, BLUE)
NAMES = [f"{ph} {q}/4" for ph in s12.PHASES for q in range(1, 5)]
LAGS = s13.K                               # the 16 lags of scripts 13 and 18


def c13(suffix):
    with open(os.path.join(ROOT, "cache", f"13_vertices_H_2021-11-12{suffix}.pkl"), "rb") as fh:
        return pickle.load(fh)


def c18(suffix):
    with open(os.path.join(ROOT, "cache", f"18_single_trial_H_2021-11-12{suffix}.pkl"), "rb") as fh:
        return pickle.load(fh)


def gain(c, v, r):
    o = c[v]
    g = o["full"][r]["gain"] if r else o["per"]
    fold = g.sum(1) / o["cnt"].sum(1)
    return float(g.sum() / o["cnt"].sum()), float(fold.std(ddof=1) / np.sqrt(len(fold)))


def roles(lam):
    """Vertex order: the one with most weight at hold 1/4, then most mean weight over move, then most at post 3/4, then the last."""
    left = list(range(lam.shape[1]))
    hold = max(left, key=lambda k: lam[0, k])
    left.remove(hold)
    move = max(left, key=lambda k: lam[8:12, k].mean())
    left.remove(move)
    post = max(left, key=lambda k: lam[14, k])
    left.remove(post)
    return [hold, move, post, left[0]]


def weight_panel(ax, f, cell, title, legend, ylabel):
    lam, se = f["poly"]["lam"], f["lam_se"]
    x = np.arange(16) + 0.5
    series = []
    for role, k in enumerate(roles(lam)):
        ax.fill_between(x, lam[:, k] - se[:, k], lam[:, k] + se[:, k], color=ROLE_COL[role], alpha=0.18, lw=0)
        ax.plot(x, lam[:, k], "-o", ms=4, lw=2.2, color=ROLE_COL[role], label=ROLE_NAMES[role])
        series.append({"name": ROLE_NAMES[role], "y": lam[:, k], "err": se[:, k], "color": ROLE_COL[role], "line": "solid",
                       "marker": True, "width": 2.2})
    real.phase_axis(ax)
    ax.set_title(title)
    if legend:
        ax.legend(ncol=2, loc="upper center", fontsize=11)
    return {"kind": "line", "cell": cell, "title": title, "cat_groups": real.PHASE_GROUPS, "ylabel": ylabel,
            "ylim": [-0.05, 1.05], "legend": legend, "series": series, "texts": []}


def describe(tag, f):
    p = f["poly"]
    lam = p["lam"]
    order = roles(lam)
    print(f"{tag}: rank 3, size h {p['h']:.3f}, stable {p['stable']}")
    for role, k in enumerate(order):
        print(f"   {ROLE_NAMES[role]:14s} rho {p['rho'][k]:.3f}, {p['dist'][k]:.2f} spreads from {NAMES[p['index'][k]]}; "
              f"hold 1/4 {lam[0, k]:.2f}, plan 1/4 {lam[4, k]:.2f}, move mean {lam[8:12, k].mean():.2f}, "
              f"post 3/4 {lam[14, k]:.2f}, max {lam[:, k].max():.2f} at {NAMES[int(lam[:, k].argmax())]}")


# ------------------------------------------------------------------------------ the evoked wave
def raw_lfp(D):
    """The raw lfp field, prepared as s12.load prepares noart: same channels and trials, decimated by 3, trial means removed."""
    from scipy.signal import decimate
    z = np.load(s12.SEGS, allow_pickle=True)
    keep = ~np.isin(z["channels"], s12.BAD)
    ok = z["move_ms"] < 3000
    X = decimate(z["lfp"][keep][:, :, ok].astype(float), s12.DEC, axis=1, zero_phase=True)
    X = (X - X.mean(1, keepdims=True)).transpose(2, 1, 0)
    return X[:, :D["X"].shape[1]]


def wave(D):
    X, t = D["X"], D["t"]
    Z = s12.evoked_design(D)
    Xr = s12.remove_evoked(X, Z, np.ones(len(X), bool))
    m = X - Xr
    xn = np.load(s13.NULL_DATA)["X_wave"]
    sel = (t >= -400) & (t <= 1400)
    tt = t[sel]
    mx, mm = X.mean(0)[sel], m.mean(0)[sel]
    top = np.argsort(np.sqrt((mx ** 2).mean(0)))[::-1][:3]
    go_med, touch_med = float(np.median(D["go"])), float(np.median(D["touch"]))
    fig, axs = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True)
    sa = []
    for c, col in zip(top, (BLUE, ORANGE, GREEN)):
        axs[0].plot(tt, mx[:, c], color=col, lw=1.8, label=f"channel {D['ch'][c]}")
        axs[0].plot(tt, mm[:, c], color=col, lw=1.4, ls="--")
        sa.append({"name": f"channel {D['ch'][c]}", "x": tt, "y": mx[:, c], "color": col, "line": "solid", "marker": False,
                   "width": 1.8})
        sa.append({"name": None, "x": tt, "y": mm[:, c], "color": col, "line": "dash", "marker": False, "width": 1.4})
    rms = {name: np.sqrt((A.mean(0)[sel] ** 2).mean(1)) for name, A in
           (("noart", X), ("raw LFP", raw_lfp(D)), ("evoked removed", Xr), ("null", xn))}
    sb = []
    for (name, r_), col, dash in zip(rms.items(), (INK, GREY, ORANGE, PURPLE), (False, False, False, True)):
        axs[1].plot(tt, r_, color=col, lw=1.8, ls="--" if dash else "-", label=name)
        sb.append({"name": name, "x": tt, "y": r_, "color": col, "line": "dash" if dash else "solid", "marker": False, "width": 1.8})
    texts = []
    for ax, series, top_y in ((axs[0], sa, None), (axs[1], sb, None)):
        lo = min(min(s["y"]) for s in series)
        hi = max(max(s["y"]) for s in series)
        for x0, lab in ((0.0, "target"), (go_med, "median go"), (touch_med, "median touch")):
            ax.axvline(x0, color=MUTED, lw=1, ls=":")
            series.append({"name": None, "x": [x0, x0], "y": [lo, hi], "color": MUTED, "line": "dash", "marker": False, "width": 1.0})
        ax.set_xlabel("ms from target on")
    for x0, lab in ((0.0, "target"), (go_med, "median go"), (touch_med, "median touch")):
        texts.append({"x": x0, "y": max(max(s["y"]) for s in sb) * 1.06, "text": lab})
    axs[0].set_title("Trial mean, 3 channels; dashed: fitted wave")
    axs[1].set_title("Size of the trial mean, RMS over channels")
    axs[0].legend(fontsize=11)
    axs[1].legend(fontsize=11)
    real.save(fig, "ev_wave", [1, 2], [
        {"kind": "xy", "cell": [0, 0, 1, 1], "title": "Trial mean, 3 channels; dashed: fitted wave",
         "xlabel": "ms from target on", "ylabel": "µV", "xlim": [-400, 1400], "legend": False, "series": sa, "texts": []},
        {"kind": "xy", "cell": [0, 1, 1, 1], "title": "Size of the trial mean, RMS over channels", "xlabel": "ms from target on",
         "ylabel": "µV", "xlim": [-400, 1400], "legend": True, "series": sb, "texts": []}])
    for name, r_ in rms.items():
        print(f"wave: RMS of the trial mean, {name}: largest {r_.max():.2f} uV at {tt[int(r_.argmax())]:.0f} ms, "
              f"median {np.median(r_):.2f} uV; peaks " +
              ", ".join(f"{lo}-{hi} ms {r_[(tt >= lo) & (tt <= hi)].max():.2f} at {tt[(tt >= lo) & (tt <= hi)][r_[(tt >= lo) & (tt <= hi)].argmax()]:.0f}"
                        for lo, hi in ((0, 60), (100, 250), (880, 960))))
    print(f"wave: median go {go_med:.0f} ms, median touch {touch_med:.0f} ms; channels drawn {[int(D['ch'][c]) for c in top]}")


# ------------------------------------------------------------------------------ the noart steps
def steps(D):
    z = np.load(s12.SEGS, allow_pickle=True)
    keep = ~np.isin(z["channels"], s12.BAD)
    ok = z["move_ms"] < 3000
    t1 = np.arange(2000) - 500.0
    med = {}
    for field in ("noart", "lfp"):
        A = z[field][keep][:, :, ok].astype(float)
        med[field] = np.median(np.abs(np.diff(A, axis=1)), axis=(0, 2))      # change from t1[i] to t1[i + 1]
    bad = s13.step_mask(D["t"])
    dropped = D["t"][bad]
    spans = []
    edge = np.where(np.diff(bad.astype(int)) != 0)[0]
    starts = [i + 1 for i in edge if not bad[i]]
    ends = [i for i in edge if bad[i]]
    for a, b in zip(starts, ends):
        spans.append((float(D["t"][a]), float(D["t"][b])))
    fig, axs = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True, sharey=True)
    panels = []
    for i, (lo, hi) in enumerate(((-40.0, 100.0), (860.0, 1000.0))):
        m = (t1[:-1] >= lo) & (t1[:-1] <= hi)
        series = []
        for field, col, name in (("noart", BLUE, "noart (used)"), ("lfp", GREY, "raw lfp")):
            axs[i].plot(t1[:-1][m], med[field][m], color=col, lw=1.8, label=name)
            series.append({"name": name, "x": t1[:-1][m], "y": med[field][m], "color": col, "line": "solid", "marker": False,
                           "width": 1.8})
        top = float(max(med["noart"][m].max(), med["lfp"][m].max()))
        for a, b in spans:
            if lo <= a <= hi:
                for x0 in (a, b):
                    axs[i].axvline(x0, color=ORANGE, lw=1.2, ls="--")
                    series.append({"name": None, "x": [x0, x0], "y": [0, top], "color": ORANGE, "line": "dash", "marker": False,
                                   "width": 1.2})
        title = "Around target onset" if i == 0 else "Around 900 ms"
        axs[i].set_title(title)
        axs[i].set_xlabel("ms from target on")
        panels.append({"kind": "xy", "cell": [0, i, 1, 1], "title": title, "xlabel": "ms from target on",
                       "ylabel": "median |change| per ms (µV)" if i == 0 else "", "xlim": [lo, hi], "ylim": [0, None],
                       "legend": i == 0, "series": series,
                       "texts": [{"x": (a + b) / 2, "y": top * 1.02, "text": "targets dropped"} for a, b in spans if lo <= a <= hi]})
    axs[0].legend()
    real.save(fig, "ev_steps", [1, 2], panels)
    typ = float(np.median(med["noart"]))
    for field in ("noart", "lfp"):
        big = np.argsort(med[field])[::-1][:4]
        print(f"steps: {field} typical {np.median(med[field]):.2f} uV; largest " +
              ", ".join(f"{t1[j]:.0f}->{t1[j] + 1:.0f} ms {med[field][j]:.1f}" for j in sorted(big)))
    print(f"steps: noart typical {typ:.2f}; mask drops {int(bad.sum())} of {len(bad)} target times, spans {spans} ms "
          f"(first {dropped.min():.0f}, last {dropped.max():.0f})")


# ------------------------------------------------------------------------------ overlap weights and bowls
def weight_shapes():
    tau = np.linspace(2.5, 6.5, 801)
    fig, axs = plt.subplots(1, 3, figsize=real.FIGSIZE, constrained_layout=True, sharey=True)
    panels = []
    for i, d in enumerate((0.0, 1.0, 2.0)):
        a = overlap_weights(tau, d, 16)
        series = []
        for g, col in ((3, GREEN), (4, ORANGE), (5, BLUE)):
            axs[i].plot(tau, a[:, g], color=col, lw=2.2, label=NAMES[g])
            series.append({"name": NAMES[g], "x": tau, "y": a[:, g], "color": col, "line": "solid", "marker": False, "width": 2.2})
        series.append({"name": None, "x": [4, 4], "y": [0, 1.05], "color": MUTED, "line": "dash", "marker": False, "width": 1.0})
        title = "No overlap (today)" if d == 0 else f"Window of {d:g} bin{'s' if d > 1 else ''}"
        axs[i].set_title(title)
        axs[i].axvline(4, color=MUTED, ls=":")
        axs[i].set_xlabel("warped time (bins)")
        panels.append({"kind": "xy", "cell": [0, i, 1, 1], "title": title, "xlabel": "warped time (bins)",
                       "ylabel": "weight in the bin" if i == 0 else "", "xlim": [2.5, 6.5], "ylim": [0, 1.1],
                       "legend": i == 0, "series": series, "texts": []})
    axs[0].legend(fontsize=11)
    real.save(fig, "ev_weights_a", [1, 3], panels)


def bowls(D):
    X = D["X"]
    N = X.shape[2]
    lab = s13.warped_labels(D)
    fo = s12.folds(D["ang"])
    P = 16
    R = symmetric_basis(D["A"], LAGS)
    hard = s12b.fold_stats(X, lab, LAGS, fo, P)
    soft = weighted_bin_stats(X, overlap_weights(s13.warped_time(D, lab), 1.0, P), LAGS, s12b.K0, fo, s12.NFOLD)
    st = {k: hard[k].sum(0) for k in hard}
    so = {k: soft[k].sum(0) for k in soft}
    tot = {k: st[k].sum(0) for k in st}
    _, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    out = {}
    for name, S in (("today", st), ("overlap", so)):
        Hs, hs = gls_quads(S, R, Sig)
        out[name] = (Hs, np.stack([_solve(Hs[g], hs[g], 1e-10) for g in range(P)]), S["n"])
    Hs, ph, n = out["today"]
    gs = (3, 4, 5)
    sep = [abs(ph[4, i] - ph[3, i]) * np.sqrt(Hs[4, i, i] / n[4]) for i in range(N)]   # lag-1 self terms come first
    i = int(np.argmax(sep))
    fig, axs = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True, sharey=True)
    centres = [out[k][1][g, i] for k in out for g in gs]
    widths = [1 / np.sqrt(out[k][0][g, i, i] / out[k][2][g]) for k in out for g in gs]
    a = np.linspace(min(centres) - 2.5 * max(widths), max(centres) + 2.5 * max(widths), 400)
    panels = []
    for j, (k, title) in enumerate((("today", "Today's bins"), ("overlap", "Window of 1 bin"))):
        Hk, pk, nk = out[k]
        series = []
        for g, col in zip(gs, (GREEN, ORANGE, BLUE)):
            y = 0.5 * (a - pk[g, i]) ** 2 * Hk[g, i, i] / nk[g]
            axs[j].plot(a, y, color=col, lw=2.2, label=NAMES[g])
            series.append({"name": NAMES[g], "x": a, "y": y, "color": col, "line": "solid", "marker": False, "width": 2.2})
            print(f"bowls {k}: {NAMES[g]} bottom {pk[g, i]:+.4f}, curvature per sample {Hk[g, i, i] / nk[g]:.4f}, "
                  f"samples {nk[g]:.0f}")
        axs[j].set_title(title)
        axs[j].set_xlabel(f"lag-1 self coefficient, channel {int(D['ch'][i])}")
        panels.append({"kind": "xy", "cell": [0, j, 1, 1], "title": title,
                       "xlabel": f"lag-1 self coefficient, channel {int(D['ch'][i])}",
                       "ylabel": "added loss per sample (nats)" if j == 0 else "", "xlim": [float(a[0]), float(a[-1])],
                       "ylim": [0, None], "legend": j == 0, "series": series, "texts": []})
    axs[0].set_ylim(0, None)
    axs[0].legend(fontsize=11)
    real.save(fig, "ev_bowls", [1, 2], panels)


# ------------------------------------------------------------------------------ results
def gains_vs_overlap():
    ds = (0, 0.5, 1, 2)
    fig, axs = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True)
    panels = []
    for i, (v, lab) in enumerate((("felix", "Evoked left in"), ("induced", "Evoked removed"))):
        series = []
        for r, col, name in ((2, BLUE, "rank 2"), (3, ORANGE, "rank 3"), (None, GREY, "separate fit per bin")):
            y = [gain(c13("" if d == 0 else f"_overlap{d:g}"), v, r)[0] for d in ds]
            axs[i].plot(ds, y, "-o", color=col, lw=2.2, label=name)
            series.append({"name": name, "x": list(ds), "y": y, "color": col, "line": "solid", "marker": True, "width": 2.2})
            ym = [gain(c13(s), v, r)[0] for s in ("_mask", "_overlap1_mask")]
            axs[i].plot([0, 1], ym, "--o", color=col, lw=1.6)
            series.append({"name": name + ", mask" if i == 0 and r == 2 else None, "x": [0, 1], "y": ym, "color": col,
                           "line": "dash", "marker": True, "width": 1.6})
        axs[i].set_title(lab)
        axs[i].set_xlabel("overlap window (bins)")
        panels.append({"kind": "xy", "cell": [0, i, 1, 1], "title": lab, "xlabel": "overlap window (bins)",
                       "ylabel": "gain over GDAR (nats/sample)" if i == 0 else "", "xlim": [-0.1, 2.1], "legend": True,
                       "series": series, "texts": []})
    real.save(fig, "ev_gain", [1, 2], panels)


def weights_grid():
    runs = ((0, 0, "", "No overlap, no mask"), (0, 1, "_overlap1", "1-bin overlap, no mask"),
            (1, 0, "_mask", "No overlap, steps masked"), (1, 1, "_overlap1_mask", "1-bin overlap, steps masked"))
    fig, axs = plt.subplots(2, 2, figsize=(12.1, 8.4), constrained_layout=True, sharey=True)
    panels = []
    for r_, c_, suf, title in runs:
        f = c13(suf)["felix"]["full"][3]
        panels.append(weight_panel(axs[r_, c_], f, [r_, c_, 1, 1], title, r_ == 0 and c_ == 0, "weight" if c_ == 0 else ""))
        describe(f"weights {title}, evoked left in", f)
        describe(f"weights {title}, evoked removed", c13(suf)["induced"]["full"][3])
    real.save(fig, "ev_weights", [2, 2], panels)


def null_gain():
    sets = ((("", "recordings"), ("_null", "fixed network + wave"), ("_nullnowave", "fixed network only")),
            (("_overlap1_mask", "recordings"), ("_overlap1_mask_null", "fixed network + wave")))
    titles = ("No overlap, no mask", "1-bin overlap, steps masked")
    fig, axs = plt.subplots(1, 2, figsize=real.FIGSIZE, constrained_layout=True, gridspec_kw={"width_ratios": [3, 2]})
    panels = []
    for i, (runs, title) in enumerate(zip(sets, titles)):
        series = []
        for v, lab, col in (("felix", "rank 3, evoked left in", BLUE), ("induced", "rank 3, evoked removed", ORANGE)):
            vals = [gain(c13(s), v, 3)[0] for s, _ in runs]
            series.append({"name": lab, "values": [round(q, 3) for q in vals], "color": col})
            axs[i].bar(np.arange(len(runs)) + (0 if v == "felix" else 0.38) - 0.19, vals, 0.38, color=col, label=lab)
        axs[i].set_xticks(range(len(runs)), [n for _, n in runs])
        axs[i].set_title(title)
        panels.append({"kind": "bar", "cell": [0, i, 1, 1], "title": title, "categories": [n for _, n in runs], "series": series,
                       "ylabel": "gain over GDAR (nats/sample)" if i == 0 else "", "show_values": True, "legend": i == 0,
                       "vlim": [min(0.0, min(min(s["values"]) for s in series)), 0.35]})
        for s, n in runs:
            for v in ("felix", "induced"):
                g2, e2 = gain(c13(s), v, 2)
                g3, e3 = gain(c13(s), v, 3)
                gp, _ = gain(c13(s), v, None)
                print(f"null gain [{title}] {n:22s} {v:8s} rank 2 {g2:+.4f} ({e2:.4f}), rank 3 {g3:+.4f} ({e3:.4f}), per bin {gp:+.4f}")
    axs[0].legend()
    real.save(fig, "ev_null_gain", [1, 2], panels, col_widths=[3, 2])


def null_weights():
    runs = ((0, 0, "", "Recordings, no overlap, no mask"), (0, 1, "_null", "Null data, no overlap, no mask"),
            (1, 0, "_overlap1_mask", "Recordings, 1-bin overlap, masked"), (1, 1, "_overlap1_mask_null", "Null data, 1-bin overlap, masked"))
    fig, axs = plt.subplots(2, 2, figsize=(12.1, 8.4), constrained_layout=True, sharey=True)
    panels = []
    for r_, c_, suf, title in runs:
        f = c13(suf)["felix"]["full"][3]
        panels.append(weight_panel(axs[r_, c_], f, [r_, c_, 1, 1], title, r_ == 0 and c_ == 0, "weight" if c_ == 0 else ""))
        describe(f"null weights {title}, evoked left in", f)
    real.save(fig, "ev_null_weights", [2, 2], panels)


def single_trials():
    for suf, name in (("", "no overlap"), ("_overlap0.5", "0.5-bin overlap"), ("_overlap1", "1-bin overlap"),
                      ("_overlap2", "2-bin overlap"), ("_mask", "no overlap, masked"), ("_overlap1_mask", "1-bin overlap, masked")):
        c = c18(suf)
        for v in ("felix", "induced"):
            S = c[v]
            row = ", ".join(f"{'clock' if k == 'clock' else f'{int(k) * 3} ms'} {S['gain'][k]:+.4f} ({S['gain_se'][k]:.4f})"
                            for k in ("clock", 10, 20, 40))
            extra = ""
            if "timing" in S:
                extra = ("; go r " + " ".join(f"{S['timing'][W]['go']['r']:+.2f}" for W in (10, 20, 40))
                         + "; touch r " + " ".join(f"{S['timing'][W]['touch']['r']:+.2f}" for W in (10, 20, 40)))
            print(f"single trial [{name}] {v}: {row}; fold match " + " ".join(f"{m:.2f}" for m in S["match"]) + extra)


def main():
    real.style()
    os.makedirs(real.OUT, exist_ok=True)
    D = s12.load()
    wave(D)
    steps(D)
    weight_shapes()
    bowls(D)
    gains_vs_overlap()
    weights_grid()
    null_gain()
    null_weights()
    single_trials()


if __name__ == "__main__":
    main()
