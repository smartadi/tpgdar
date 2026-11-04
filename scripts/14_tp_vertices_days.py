r"""Does the TP-GDAR switch repeat?  Script 13's fit on other IOS Reach days, compared with H 2021-11-12.

  H 2021-08-23            no stim
  H 2021-11-11 no light   stim day, unlit trials only
  H 2021-11-11 light      stim day, lit trials only

The model has no input term, so lit and unlit trials are fitted separately rather than mixed;
on lit trials A(p) can absorb the light's effect, so those vertices are not intrinsic dynamics.
On the stim day a sample is dropped if it or any of its K lags falls in the light artefact
windows (-20..60 and 860..960 ms re target, as in iosreach lti_phase), on both trial types.

Run:  .venv\Scripts\python scripts/14_tp_vertices_days.py [--plot]
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys

import numpy as np
from scipy.signal import decimate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s13 = importlib.import_module("13_tp_vertices_ios")
from gdar_tp.sym_tp import theta_of  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "14_days.pkl")
RUNS = {"08-23 no stim": ("2021-08-23", None), "11-11 no light": ("2021-11-11", 0), "11-11 light": ("2021-11-11", 1)}
ART = ((-20.0, 60.0), (860.0, 960.0))
K = s13.K
log = s12.log


def load(date, stimmed):
    z = np.load(s12.SEGS.replace("2021-11-12", date), allow_pickle=True)
    ch = z["channels"]
    keep = ~np.isin(ch, s12.BAD)
    ok = z["move_ms"] < 3000
    if stimmed is not None:
        ok &= z["stimmed"] == stimmed
    else:
        assert int(z["stimmed"].sum()) == 0
    X = z["noart"][keep][:, :, ok].astype(float)
    X = decimate(X, s12.DEC, axis=1, zero_phase=True)
    X = (X - X.mean(1, keepdims=True)).transpose(2, 1, 0)
    t = (np.arange(2000) - 500)[::s12.DEC].astype(float)[:X.shape[1]]
    with open(s12.POS, "rb") as f:
        p = pickle.load(f)
    pos = np.array([p[int(c)] for c in ch[keep]], float)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    go = z["plan_ms"][ok].astype(float)
    return dict(X=X, t=t, pos=pos, A=(d <= s12.GRAPH_TH + 1e-9) & (d > 0), go=go, touch=go + z["move_ms"][ok],
                ang=z["angle"][ok].astype(int) % 360, ch=ch[keep])


def mask_artefact(lab, t):
    """Drop targets whose own sample or any of their K lags lies in an artefact window."""
    art = np.zeros(len(t), bool)
    for a, b in ART:
        art |= (t >= a) & (t < b)
    bad = np.convolve(art, np.ones(K + 1), mode="full")[:len(t)] > 0
    out = lab.copy()
    out[:, bad] = -1
    return out


def main():
    out = {}
    for name, (date, stimmed) in RUNS.items():
        log(f"\n===== {name} =====")
        D = load(date, stimmed)
        lab = s13.warped_labels(D)
        if stimmed is not None:
            lab = mask_artefact(lab, D["t"])
        log(f"{D['X'].shape[0]} trials, {D['X'].shape[2]} channels, {int(D['A'].sum()/2)} edges; samples per bin:",
            [int((lab == g).sum()) for g in range(16)])
        fo = s12.folds(D["ang"])
        Z = s12.evoked_design(D)
        out[name] = dict(ch=D["ch"], A=D["A"], n_trials=D["X"].shape[0])
        for v in ("felix", "induced"):
            out[name][v] = s13.run_variant(D, lab, fo, Z, v)
            with open(CACHE, "wb") as f:
                pickle.dump(out, f)
    log("done")


def rank2(res, N):
    """(lam of the movement vertex over bins, move-minus-rest edges summed over lags)."""
    p = res["full"][2]["poly"]
    move = int(np.argmax(p["lam"][8:12].mean(0)))
    summed = [sum(theta_of(ph, res["R"], N)[:, k * N:(k + 1) * N] for k in range(K)) for ph in p["phi"]]
    return p["lam"][:, move], summed[move] - summed[1 - move], res["full"][2]["lam_se"][:, move]


def plot():
    import matplotlib.pyplot as plt
    with open(CACHE, "rb") as f:
        out = pickle.load(f)
    with open(s13.CACHE, "rb") as f:
        ref = pickle.load(f)
    s12._style()
    D0 = s12.load()
    runs = {"11-12 no stim": dict(ch=D0["ch"], A=D0["A"], n_trials=D0["X"].shape[0], felix=ref["felix"], induced=ref["induced"]), **out}
    cols = dict(zip(runs, (s12.C_FELIX, s12.C_IND, s12.C_3, "#6250d6")))
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True, width_ratios=[1.5, 1, 1])
    fig.suptitle("Does the TP-GDAR switch repeat? Hydrogen, K = 16, rank 2 unless noted, held-out trials",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    for row, (v, name) in enumerate((("felix", "evoked left in"), ("induced", "evoked removed"))):
        x = np.arange(16) + 0.5
        refd = None
        for i, (rn, r) in enumerate(runs.items()):
            N = len(r["ch"])
            lam, diff, se = rank2(r[v], N)
            axs[row, 0].plot(x, lam, "-o", ms=4, color=cols[rn], label=f"{rn} ({r['n_trials']} trials)")
            axs[row, 0].fill_between(x, lam - se, lam + se, color=cols[rn], alpha=0.15, lw=0)
            gain = [r[v]["full"][q]["gain"].sum() / r[v]["cnt"].sum() for q in (2, 3)]
            sep = r[v]["per"].sum() / r[v]["cnt"].sum()
            axs[row, 1].plot([0, 1, 2], gain + [sep], "o", ms=7, color=cols[rn])
            iu, ju = np.where(np.triu(r["A"], 1))
            edges = {(int(r["ch"][a]), int(r["ch"][b])): diff[a, b] for a, b in zip(iu, ju)}
            if refd is None:
                refd = edges
            else:
                common = sorted(set(refd) & set(edges))
                cc = np.corrcoef([refd[e] for e in common], [edges[e] for e in common])[0, 1]
                axs[row, 2].barh(rn, cc, color=cols[rn], height=0.5)
                axs[row, 2].text(max(cc, 0) + 0.02, rn, f"{cc:.2f}, {len(common)} edges", va="center", fontsize=8)
                log(f"{name} {rn}: gains r2 {gain[0]:+.3f} r3 {gain[1]:+.3f} separate {sep:+.3f}; "
                    f"move-minus-rest edge r with 11-12 = {cc:.3f} over {len(common)} edges; "
                    f"move vertex lam: " + " ".join(f"{a:.2f}" for a in lam))
        ax = axs[row, 0]
        for b in range(1, 4):
            ax.axvline(b * s13.NSUB, color=s12.MUTED, lw=0.8, ls=":")
        ax.set_xticks((np.arange(4) + 0.5) * s13.NSUB, ["hold", "plan", "move", "post"])
        ax.set_xlim(0, 16)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("weight of the movement vertex")
        ax.set_title(f"{'AD'[row]}  Weights, {name}")
        ax = axs[row, 1]
        ax.set_xticks([0, 1, 2], ["rank 2", "rank 3", "separate\nfits"])
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(0, None)
        ax.set_ylabel("gain over GDAR (nats/sample)")
        ax.set_title(f"{'BE'[row]}  Held-out gain")
        ax = axs[row, 2]
        ax.set_xlim(min(0, ax.get_xlim()[0]), 1.0)
        ax.axvline(0, color=s12.MUTED, lw=1)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=s12.GRID, lw=0.7)
        ax.set_xlabel("correlation with 11-12")
        ax.set_title(f"{'CF'[row]}  Move-minus-rest edges")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(os.path.join(s12.FIG, "14_days_comparison.png"), dpi=170)
    log("wrote comparison")
    for rn, r in out.items():
        D = load(*RUNS[rn])
        for v, name in (("felix", "evoked left in"), ("induced", "evoked removed")):
            f = s13.fig_variant(r[v], D, 2, f"TP-GDAR rank 2, H {rn}, {name}: K = {K}, held-out gain "
                                            f"{r[v]['full'][2]['gain'].sum()/r[v]['cnt'].sum():+.3f} nats/sample")
            f.savefig(os.path.join(s12.FIG, f"14_vertices_{rn.replace(' ', '_')}_{v}_r2.png"), dpi=170)
            plt.close(f)


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else main()
