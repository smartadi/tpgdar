r"""Vertex graphs of TP-GDAR and their weights across the trial, H 2021-11-12, no stim.

Each trial is warped onto one axis, p = trial phase: hold, plan, move and post, each cut
into 4 equal fractions of its own duration on that trial (16 grid points).  The symmetric
TP-GDAR is fitted at K = 16 and put in convex form,
Theta(p) = sum_v lam_v(p) Theta_v, lam >= 0, sum 1 (`gdar_tp.sym_tp.polytope_vertices`).

Rank 2 vertices are grid points on the trajectory; rank 3 vertices are the corners of an
enclosing triangle, generally not visited by the data.  Either way they are one choice of
coordinates for the fitted rank-r family; only the family itself is unique.  Uncertainty: the same fit
on the 5 training splits, expressed in the full-data vertices (jackknife SE).  The held-out
gain over GDAR comes from those same splits.

Run:  .venv\Scripts\python scripts/13_tp_vertices_ios.py [--plot]
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s12b = importlib.import_module("12b_tp_vs_gdar_robust")
from gdar_tp.sym_tp import (_quad, fit_gls_pooled, fit_sym_tp, polytope_vertices,  # noqa: E402
                            symmetric_basis, theta_of)
from gdar_tp.tp import _barycentric  # noqa: E402

K = 16
NSUB = 4
RANKS = (2, 3)
CACHE = os.path.join(ROOT, "cache", "13_vertices_H_2021-11-12.pkl")
log = s12.log


def warped_labels(D):
    t, go, touch = D["t"], D["go"], D["touch"]
    lab = np.full((len(go), len(t)), -1)
    for i in range(len(go)):
        for ph, (a, b) in enumerate(((-400.0, 0.0), (0.0, go[i]), (go[i], touch[i]), (touch[i], touch[i] + 500.0))):
            m = (t >= a) & (t < b) & (t <= 1400)
            lab[i, m] = ph * NSUB + np.minimum(((t[m] - a) / (b - a) * NSUB).astype(int), NSUB - 1)
    return lab


def bin_name(g):
    return f"{s12.PHASES[g // NSUB]} {g % NSUB + 1}/{NSUB}"


def in_vertices(phi_grid, verts, Hbar):
    """Weights of a fitted family on given vertices, in the pooled data metric."""
    L = np.linalg.cholesky(Hbar + 1e-10 * np.trace(Hbar) / len(Hbar) * np.eye(len(Hbar)))
    return _barycentric(verts @ L, phi_grid @ L)


def run_variant(D, lab, fo, Z, variant):
    t0 = time.time()
    X0 = D["X"]
    N = X0.shape[2]
    P = lab.max() + 1
    R = symmetric_basis(D["A"], K)
    grid = np.arange(float(P))
    out = {"variant": variant}

    X = X0 if variant == "felix" else s12.remove_evoked(X0, Z, np.ones(len(X0), bool))
    s = s12b.fold_stats(X, lab, K, fo, P)
    st = {k: s[k].sum(0) for k in s}
    tot = {k: st[k].sum(0) for k in st}
    phi0, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    Hbar = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sig))[0] / tot["n"]
    out.update(phi0=phi0, Sigma=Sig, R=R, n=st["n"])
    full = {}
    for r in RANKS:
        m = fit_sym_tp(st, R, Sig, r, grid, n_iter=40)
        full[r] = dict(model=m, poly=polytope_vertices(m, Hbar), J=m.objective)
        p = full[r]["poly"]
        log(f"  [{variant}] rank {r}: {len(m.objective)-1} sweeps, vertices {[bin_name(i) for i in p['index']]}, "
            f"sum-to-one error {p['affine_err']:.3f}, outside simplex {p['outside']:.3f}, {time.time()-t0:.0f}s")

    # the 5 training splits: jackknife weights and held-out gain
    lams = {r: [] for r in RANKS}
    gain = {r: np.zeros((s12.NFOLD, P)) for r in RANKS}
    per = np.zeros((s12.NFOLD, P))
    cnt = np.zeros((s12.NFOLD, P))
    for f in range(s12.NFOLD):
        Xf = X0 if variant == "felix" else s12.remove_evoked(X0, Z, fo != f)
        sf = s12b.fold_stats(Xf, lab, K, fo, P)
        tr, te = s12b.sel(sf, f), s12b.sel(sf, f, False)
        a = {k: tr[k].sum(0) for k in tr}
        phi, Sf = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
        Base = np.stack([theta_of(phi, R, N)] * P)
        base = s12b.ll(Base, te, [s12.sigma_of(Base, tr)] * P)
        cnt[f] = te["n"]
        Per = np.stack([fit_sym_tp({k: tr[k][[g]] for k in tr}, R, Sf, 1, grid[[g]], n_iter=1).theta_grid()[0]
                        for g in range(P)])
        per[f] = s12b.ll(Per, te, [s12.sigma_of(Per, tr)] * P) - base
        for r in RANKS:
            m = fit_sym_tp(tr, R, Sf, r, grid, n_iter=25)
            Th = m.theta_grid()
            gain[r][f] = s12b.ll(Th, te, [s12.sigma_of(Th, tr)] * P) - base
            lams[r].append(in_vertices(m.W @ m.phi, full[r]["poly"]["phi"], Hbar))
        log(f"  [{variant}] fold {f}, {time.time()-t0:.0f}s")
    for r in RANKS:
        L = np.stack(lams[r])
        full[r]["lam_se"] = np.sqrt((s12.NFOLD - 1) / s12.NFOLD * ((L - L.mean(0)) ** 2).sum(0))
        full[r]["lam_folds"] = L
        full[r]["gain"] = gain[r]
        g = gain[r].sum(1) / cnt.sum(1)
        log(f"  [{variant}] rank {r} held-out gain {gain[r].sum()/cnt.sum():+.4f} (fold SE {g.std(ddof=1)/np.sqrt(s12.NFOLD):.4f}), "
            f"jackknife SE of lam, median {np.median(full[r]['lam_se']):.3f}")
        log("     lam: " + " | ".join(bin_name(g) + " " + " ".join(f"{x:.2f}" for x in full[r]['poly']['lam'][g]) for g in range(P)))
        for v, phi_v in enumerate(full[r]["poly"]["phi"]):
            rho = np.abs(s12.rho(theta_of(phi_v, R, N), K)).max()
            log(f"     vertex {v+1} (nearest {bin_name(full[r]['poly']['index'][v])}, {full[r]['poly']['dist'][v]:.2f} of spread away): spectral radius {rho:.3f}")
    log(f"  [{variant}] separate fit per bin, held-out gain {per.sum()/cnt.sum():+.4f}")
    out.update(full=full, per=per, cnt=cnt)
    return out


COL = (s12.C_FELIX, s12.C_IND, s12.C_3)


def fig_variant(res, D, r, title):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("bwr_", [s12.C_FELIX, "#f0efec", "#e34948"])
    f = res["full"][r]
    p = f["poly"]
    R, N, Sig = res["R"], D["X"].shape[2], res["Sigma"]
    P = p["lam"].shape[0]
    pos, A = D["pos"], D["A"]
    iu, ju = np.where(np.triu(A, 1))

    def summed(phi):
        th = theta_of(phi, R, N)
        return sum(th[:, k * N:(k + 1) * N] for k in range(K))

    S = [summed(ph) for ph in p["phi"]]
    S0 = summed(res["phi0"])
    fig = plt.figure(figsize=(13.5, 9.6), constrained_layout=True)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[1.0, 1.05, 1.05])
    a_w, a_s = top.subplots(1, 2, width_ratios=[1.9, 1])

    x = np.arange(P) + 0.5
    for v in range(r):
        a_w.plot(x, p["lam"][:, v], "-o", ms=5, color=COL[v], label=f"vertex {v+1}")
        a_w.fill_between(x, p["lam"][:, v] - f["lam_se"][:, v], p["lam"][:, v] + f["lam_se"][:, v],
                         color=COL[v], alpha=0.18, lw=0)
    for b in range(1, 4):
        a_w.axvline(b * NSUB, color=s12.MUTED, lw=0.8, ls=":")
    a_w.set_xticks((np.arange(4) + 0.5) * NSUB, ["hold", "plan\n(from target on)", "move\n(from go cue)", "post\n(from touch)"])
    a_w.set_xlim(0, P)
    a_w.set_ylim(-0.05, 1.05)
    a_w.set_ylabel("vertex weight λ (sums to 1)")
    a_w.set_title("A  Weights across the trial (band: jackknife SE over 5 training splits)")
    a_w.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=r, borderaxespad=0)

    freqs = np.arange(2.0, 101.0, 1.0)
    base = s12.ar_spectrum(theta_of(res["phi0"], R, N), Sig, K, freqs)
    for v in range(r):
        a_s.plot(freqs, 10 * np.log10(s12.ar_spectrum(theta_of(p["phi"][v], R, N), Sig, K, freqs) / base),
                 color=COL[v], label=f"vertex {v+1}")
    s12._zero(a_s)
    a_s.set_xlabel("frequency (Hz)")
    a_s.set_ylabel("model spectrum re GDAR (dB)")
    a_s.set_title("B  What each vertex does to the spectrum")
    a_s.legend(loc="upper right")

    def maps(sub, mats, titles, tag):
        vm = max(np.abs(m[iu, ju]).max() for m in mats)
        axs = np.atleast_1d(sub.subplots(1, len(mats)))
        for ax, M, ttl in zip(axs, mats, titles):
            lc = LineCollection([(pos[i], pos[j]) for i, j in zip(iu, ju)], array=M[iu, ju], cmap=cmap,
                                norm=TwoSlopeNorm(0, -vm, vm), linewidths=3.2)
            ax.add_collection(lc)
            ax.scatter(*pos.T, s=20, color="#fcfcfb", edgecolor=s12.MUTED, lw=0.8, zorder=3)
            ax.set_aspect("equal")
            ax.margins(0.08)
            ax.axis("off")
            ax.set_title(ttl, fontsize=9)
        cb = sub.colorbar(lc, ax=list(axs), orientation="vertical", fraction=0.02, pad=0.01, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
        cb.set_label(tag, fontsize=8)

    rhos = [np.abs(s12.rho(theta_of(ph, R, N), K)).max() for ph in p["phi"]]
    maps(mid, S, [f"C  Vertex {v+1}: {'near ' if p['dist'][v] > 1e-9 else ''}{bin_name(p['index'][v])}, ρ = {rhos[v]:.3f}" for v in range(r)],
         "edge weight, summed over lags")
    maps(bot, [m - S0 for m in S] + [S0], [f"D  Vertex {v+1} minus GDAR" for v in range(r)] + ["GDAR, for reference"],
         "edge weight, summed over lags")
    return fig


def plot():
    with open(CACHE, "rb") as fh:
        out = pickle.load(fh)
    s12._style()
    D = s12.load()
    for v, name in (("felix", "evoked left in"), ("induced", "evoked removed")):
        for r in RANKS:
            fig = fig_variant(out[v], D, r,
                              f"TP-GDAR rank {r} vertex graphs and weights across the trial, {name}: "
                              f"H 2021-11-12, no stim, K = {K}, held-out gain {out[v]['full'][r]['gain'].sum()/out[v]['cnt'].sum():+.3f} nats/sample")
            path = os.path.join(s12.FIG, f"13_vertices_{v}_r{r}.png")
            fig.savefig(path, dpi=170)
            log("wrote", path)


def main():
    D = s12.load()
    lab = warped_labels(D)
    log("samples per bin:", [int((lab == g).sum()) for g in range(lab.max() + 1)])
    fo = s12.folds(D["ang"])
    Z = s12.evoked_design(D)
    out = {}
    for v in ("felix", "induced"):
        out[v] = run_variant(D, lab, fo, Z, v)
        with open(CACHE, "wb") as fh:
            pickle.dump(out, fh)
    log("done")


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else main()
