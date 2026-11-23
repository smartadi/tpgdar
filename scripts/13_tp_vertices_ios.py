r"""Vertex graphs of TP-GDAR and their weights across the trial, H 2021-11-12, no stim.

Each trial is warped onto one axis, p = trial phase: hold, plan, move and post, each cut
into 4 equal fractions of its own duration on that trial (16 grid points).  The symmetric
TP-GDAR is fitted at K = 16 and written exactly in convex form on r + 1 vertices,
Theta(p) = sum_v lam_v(p) Theta_v, lam >= 0, sum 1 (`gdar_tp.sym_tp.regular_hull`, MATH §9.6, approved
by Aditya 2026-11-22).  The vertices form a regular simplex in the pooled data metric; its rotation and
tightness tau are free parameters.  Tables and figures use tau = 1 and the smallest rotation whose
vertices are all stable (spectral radius < 1; `stable_regular_hull`, rule approved 2026-11-22), or,
where no rotation is, the least unstable one, flagged in the log.  A sweep over rotations (every degree at rank 2, 100 random rotations at rank 3) and
tau in TAUS reports each vertex's distance from the data, its spectral radius and its largest weight.
The model is the free fit, so the held-out gain doesn't depend on the vertices.

The vertices are one choice of coordinates for the fitted rank-r family; only the family itself is
unique.  Uncertainty: the same fit on the 5 training splits, expressed in the full-data vertices
(jackknife SE).  The held-out gain over GDAR comes from those same splits.  Before 2026-11-22 the
vertices came from `polytope_vertices` (r vertices after a sum-to-one projection); the log still
lists them for comparison.

Overlapping bins (MATH §2.4, approved by Aditya 2026-11-23): with --overlap D (D > 0, in bins) each bin's bowl also takes part of
its neighbours' samples, across target, go and touch, with each sample's weight split over a window of D bins.  Held-out samples
are still scored with their own bin's operator, so D = 0 is the run above.  Caches and figures carry the D.

Run:  .venv\Scripts\python scripts/13_tp_vertices_ios.py [--overlap D] [--plot]
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
from gdar_tp.sym_tp import (_quad, _rotation_grid, fit_gls_pooled, fit_sym_tp, gls_quads,  # noqa: E402
                            polytope_vertices, regular_hull, stable_regular_hull, symmetric_basis, theta_of)
from gdar_tp.tp import _barycentric  # noqa: E402
from gdar_tp.bins import overlap_weights, weighted_bin_stats  # noqa: E402

K = 16
NSUB = 4
RANKS = (2, 3)
TAUS = (1.0, 1.25, 1.5, 2.0)
N_ROT3 = 100
OVERLAP = float(sys.argv[sys.argv.index("--overlap") + 1]) if "--overlap" in sys.argv else 0.0   # MATH §2.4, in bins
SUFFIX = f"_overlap{OVERLAP:g}" if OVERLAP else ""
CACHE = os.path.join(ROOT, "cache", f"13_vertices_H_2021-11-12{SUFFIX}.pkl")
log = s12.log


def hull_sweep(m, Hbar, R, N):
    """Each vertex's distance from the data, spectral radius and largest weight, over rotations and tau (MATH §9.6)."""
    r = m.W.shape[1]
    rots = _rotation_grid(r) if r != 3 else _rotation_grid(3, n_random=N_ROT3, seed=1)
    shape = (len(rots), len(TAUS), r + 1)
    out = dict(angles=np.array(rots), taus=np.array(TAUS), h=np.zeros(shape[:2]), dist=np.zeros(shape),
               rho=np.zeros(shape), lam_max=np.zeros(shape))
    for i, a in enumerate(rots):
        for j, tau in enumerate(TAUS):
            hu = regular_hull(m, Hbar, angles=a, tau=tau)
            out["h"][i, j] = hu["h"]
            out["dist"][i, j] = hu["dist"]
            out["lam_max"][i, j] = hu["lam"].max(0)
            out["rho"][i, j] = [np.abs(s12.rho(theta_of(ph, R, N), K)).max() for ph in hu["phi"]]
    return out


def log_sweep(tag, sw):
    n = len(sw["angles"])
    for j, tau in enumerate(sw["taus"]):
        rmax = sw["rho"][:, j].max(1)
        log(f"     {tag} sweep tau {tau:.2f} over {n} rotations: size h {sw['h'][:, j].min():.3f} to {sw['h'][:, j].max():.3f}; "
            f"vertex distance from nearest bin median {np.median(sw['dist'][:, j]):.2f}, max {sw['dist'][:, j].max():.2f}; "
            f"largest vertex spectral radius {rmax.min():.3f} to {rmax.max():.3f} (median {np.median(rmax):.3f}), "
            f"all vertices stable at {np.mean(rmax < 1):.0%} of rotations; largest weight per vertex, median {np.median(sw['lam_max'][:, j]):.2f}")


def warped_labels(D):
    t, go, touch = D["t"], D["go"], D["touch"]
    lab = np.full((len(go), len(t)), -1)
    for i in range(len(go)):
        for ph, (a, b) in enumerate(((-400.0, 0.0), (0.0, go[i]), (go[i], touch[i]), (touch[i], touch[i] + 500.0))):
            m = (t >= a) & (t < b) & (t <= 1400)
            lab[i, m] = ph * NSUB + np.minimum(((t[m] - a) / (b - a) * NSUB).astype(int), NSUB - 1)
    return lab


def warped_time(D, lab):
    """Warped time tau in bins (MATH §2.4), NaN where the sample is unused; its floor (capped per phase) is the §2.3 label."""
    t, go, touch = D["t"], D["go"], D["touch"]
    tau = np.full((len(go), len(t)), np.nan)
    for i in range(len(go)):
        for ph, (a, b) in enumerate(((-400.0, 0.0), (0.0, go[i]), (go[i], touch[i]), (touch[i], touch[i] + 500.0))):
            m = (t >= a) & (t < b) & (t <= 1400)
            tau[i, m] = ph * NSUB + NSUB * (t[m] - a) / (b - a)
    ok = lab >= 0
    assert np.array_equal(ok, ~np.isnan(tau))
    assert np.array_equal(np.minimum(np.floor(tau[ok]), lab[ok] // NSUB * NSUB + NSUB - 1).astype(int), lab[ok])
    return tau


def bin_weights(D, lab, overlap=OVERLAP):
    """Each target sample's weight per bin (MATH §2.4); None at overlap 0, where the hard statistics are used."""
    if not overlap:
        return None
    return overlap_weights(warped_time(D, lab), overlap, lab.max() + 1)


def fit_stats(X, lab, wts, fo, P, hard=None):
    """Statistics the bins are fitted on: the hard ones at overlap 0, else weighted by ``wts``."""
    if wts is None:
        return hard if hard is not None else s12b.fold_stats(X, lab, K, fo, P)
    return weighted_bin_stats(X, wts, K, s12b.K0, fo, s12.NFOLD)


def bin_name(g):
    return f"{s12.PHASES[g // NSUB]} {g % NSUB + 1}/{NSUB}"


def in_vertices(phi_grid, verts, Hbar):
    """Weights of a fitted family on given vertices, in the pooled data metric."""
    L = np.linalg.cholesky(Hbar + 1e-10 * np.trace(Hbar) / len(Hbar) * np.eye(len(Hbar)))
    return _barycentric(verts @ L, phi_grid @ L)


def fold_fit(job):
    """Fits and held-out scores of one training split from its Grams, in a worker process.

    ``sfit`` are the statistics the bins are fitted on (weighted with overlapping bins, MATH §2.4); ``sf`` the hard ones, by
    which every held-out sample is scored with its own bin's operator and each model's scoring Sigma is computed (§8).
    """
    sf, sfit, f, R, N, P = job
    tr, te = s12b.sel(sf, f), s12b.sel(sf, f, False)
    trf = s12b.sel(sfit, f)
    a = {k: tr[k].sum(0) for k in tr}
    phi, Sf = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
    Base = np.stack([theta_of(phi, R, N)] * P)
    base = s12b.ll(Base, te, [s12.sigma_of(Base, tr)] * P)
    grid = np.arange(float(P))
    Hs, hs = gls_quads(trf, R, Sf)
    Per = np.stack([fit_sym_tp({k: trf[k][[g]] for k in trf}, R, Sf, 1, grid[[g]], n_iter=1, quads=(Hs[[g]], hs[[g]])).theta_grid()[0]
                    for g in range(P)])
    gain, family = {}, {}
    for r in RANKS:
        m = fit_sym_tp(trf, R, Sf, r, grid, n_iter=25, quads=(Hs, hs))
        Th = m.theta_grid()
        gain[r] = s12b.ll(Th, te, [s12.sigma_of(Th, tr)] * P) - base
        family[r] = m.W @ m.phi
    return dict(cnt=te["n"], per=s12b.ll(Per, te, [s12.sigma_of(Per, tr)] * P) - base, gain=gain, family=family)


def run_variant(D, lab, fo, Z, variant, wts=None):
    t0 = time.time()
    X0 = D["X"]
    N = X0.shape[2]
    P = lab.max() + 1
    R = symmetric_basis(D["A"], K)
    grid = np.arange(float(P))
    out = {"variant": variant}

    X = X0 if variant == "felix" else s12.remove_evoked(X0, Z, np.ones(len(X0), bool))
    s = s12b.fold_stats(X, lab, K, fo, P)
    sfit = fit_stats(X, lab, wts, fo, P, hard=s)
    st = {k: s[k].sum(0) for k in s}
    stf = {k: sfit[k].sum(0) for k in sfit}
    tot = {k: st[k].sum(0) for k in st}     # pooled: the same with or without overlap, since the weights sum to 1
    phi0, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    Hbar = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sig))[0] / tot["n"]
    out.update(phi0=phi0, Sigma=Sig, R=R, n=st["n"], n_fit=stf["n"], overlap=OVERLAP if wts is not None else 0.0)
    full = {}
    quads = gls_quads(stf, R, Sig)
    for r in RANKS:
        m = fit_sym_tp(stf, R, Sig, r, grid, n_iter=40, quads=quads)
        old = polytope_vertices(m, Hbar)
        smallest = regular_hull(m, Hbar)
        full[r] = dict(model=m, poly=stable_regular_hull(m, Hbar, R, N, K), smallest=smallest, old=old, J=m.objective)
        p = full[r]["poly"]
        ang = (f"{np.rad2deg(p['angles'][0]):.1f} deg" if r == 2 else
               "rotation vector " + " ".join(f"{a:+.3f}" for a in p["angles"]))
        what = (f"smallest regular simplex with every vertex stable (candidate {p['n_checked']})" if p["stable"] else
                f"NO STABLE ROTATION among {p['n_checked']} candidates; the least unstable kept")
        log(f"  [{variant}] rank {r}: {len(m.objective)-1} sweeps; {r + 1} vertices, {what}, at {ang}, tau 1, "
            f"size h {p['h']:.3f} (smallest without the stability rule {p['h_smallest']:.3f}), rebuilt-bin miss "
            f"{p['outside']:.1e}; nearest bins {[bin_name(i) for i in p['index']]}, distance "
            f"{' '.join(f'{d:.2f}' for d in p['dist'])} of spread, spectral radius {' '.join(f'{x:.3f}' for x in p['rho'])}, "
            f"{time.time()-t0:.0f}s")
        rs = [np.abs(s12.rho(theta_of(ph, R, N), K)).max() for ph in smallest["phi"]]
        log(f"     smallest without the stability rule (default 2026-11-22 morning): distance "
            f"{' '.join(f'{d:.2f}' for d in smallest['dist'])}, spectral radius {' '.join(f'{x:.3f}' for x in rs)}")
        log(f"     before 2026-11-22 ({r} vertices after the sum-to-one projection): nearest bins "
            f"{[bin_name(i) for i in old['index']]}, distance {' '.join(f'{d:.2f}' for d in old['dist'])}, "
            f"sum-to-one error {old['affine_err']:.3f}")
        full[r]["sweep"] = hull_sweep(m, Hbar, R, N)
        log_sweep(f"[{variant}] rank {r}", full[r]["sweep"])

    # the 5 training splits: jackknife weights and held-out gain
    lams = {r: [] for r in RANKS}
    gain = {r: np.zeros((s12.NFOLD, P)) for r in RANKS}
    per = np.zeros((s12.NFOLD, P))
    cnt = np.zeros((s12.NFOLD, P))
    jobs = []
    for f in range(s12.NFOLD):
        if variant == "felix":
            jobs.append((s, sfit, f, R, N, P))
        else:
            Xf = s12.remove_evoked(X0, Z, fo != f)
            hf = s12b.fold_stats(Xf, lab, K, fo, P)
            jobs.append((hf, fit_stats(Xf, lab, wts, fo, P, hard=hf), f, R, N, P))
    for f, done in enumerate(s12.fold_map(fold_fit, jobs)):
        cnt[f] = done["cnt"]
        per[f] = done["per"]
        for r in RANKS:
            gain[r][f] = done["gain"][r]
            lams[r].append(in_vertices(done["family"][r], full[r]["poly"]["phi"], Hbar))
    log(f"  [{variant}] folds done, {time.time()-t0:.0f}s")
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


COL = (s12.C_FELIX, s12.C_IND, s12.C_3, "#6250d6")


def fig_variant(res, D, r, title):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("bwr_", [s12.C_FELIX, "#f0efec", "#e34948"])
    f = res["full"][r]
    p = f["poly"]
    R, N, Sig = res["R"], D["X"].shape[2], res["Sigma"]
    P, V = p["lam"].shape
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
    for v in range(V):
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
    a_w.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=V, borderaxespad=0)

    freqs = np.arange(2.0, 101.0, 1.0)
    base = s12.ar_spectrum(theta_of(res["phi0"], R, N), Sig, K, freqs)
    for v in range(V):
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
    maps(mid, S, [f"C  Vertex {v+1}: {'near ' if p['dist'][v] > 1e-9 else ''}{bin_name(p['index'][v])}, ρ = {rhos[v]:.3f}" for v in range(V)],
         "edge weight, summed over lags")
    maps(bot, [m - S0 for m in S] + [S0], [f"D  Vertex {v+1} minus GDAR" for v in range(V)] + ["GDAR, for reference"],
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
                              f"TP-GDAR rank {r}, {r + 1} vertices (smallest stable regular simplex), {name}: "
                              f"H 2021-11-12, no stim, K = {K}, held-out gain {out[v]['full'][r]['gain'].sum()/out[v]['cnt'].sum():+.3f} nats/sample")
            path = os.path.join(s12.FIG, f"13_vertices_{v}_r{r}{SUFFIX}.png")
            fig.savefig(path, dpi=170)
            log("wrote", path)


def main():
    D = s12.load()
    lab = warped_labels(D)
    log("samples per bin:", [int((lab == g).sum()) for g in range(lab.max() + 1)])
    wts = bin_weights(D, lab)
    if wts is not None:
        log(f"overlapping bins, window {OVERLAP:g} bins (MATH §2.4); weighted samples per bin:",
            [round(float(x)) for x in wts.sum((0, 1))])
    fo = s12.folds(D["ang"])
    Z = s12.evoked_design(D)
    out = {}
    for v in ("felix", "induced"):
        out[v] = run_variant(D, lab, fo, Z, v, wts)
        with open(CACHE, "wb") as fh:
            pickle.dump(out, fh)
    log("done")


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else main()
