r"""Can the rank 3 states be found on a single held-out trial, without the task clock?  H 2021-11-12, no stim.

Method approved by Aditya on 2026-11-05.  The fit is script 13's fold fit, unchanged: the joint rank-3 TP-GDAR at K = 16
on the 16 warped bins (hold, plan, move, post, 4 each), fitted on 4 of the 5 folds, put in convex form on the smallest
enclosing triangle (weights >= 0, sum 1).  Then, on every trial and at every time t, the weights are estimated from that
trial's own data, with no go or touch time and no bin label:

    lam_hat(t) = argmin_{lam >= 0, sum lam = 1}  sum_{s in window}  || x(s+1) - sum_v lam_v Theta_v z(s) ||^2_{Sigma^-1}

This is the joint fit's own weight step on one trial's window, restricted to the simplex, and solved exactly.  Sigma is
the pooled training Sigma the fit uses.  Windows are 10, 20 and 40 samples (30, 60, 120 ms), and none is tuned.

  1. Timing (evoked left in only).  Per held-out trial, the go time is the shift that best matches its centred lam_hat
     to the training trials' go-aligned mean (-200 to +300 ms about the event), searched over 150..700 ms after target
     on.  Touch is found the same way over 250..1135 ms, on the trials whose touch lies there (the template has to
     end inside the data).  Scored by the correlation with the true time, with a permutation p, and the median error
     against guessing the training median.  Movement time (touch less go) too, since touch alone could follow go.
  2. Agreement with the clock.  Per trial, the correlation of centred lam_hat with the clock weights from its own go and
     touch: all three weights, each less its trial mean, so vertex order doesn't matter.  Reference: the same against a
     clock built from the training median go and touch, i.e. the clock without this trial's events.
  3. Without the clock.  Held-out one-step gain over GDAR with lam_hat from the window before t only, against the clock
     weights.  Every row is scored with GDAR's Sigma, as the leave-one-bin-out rows, on the same samples.
Evoked removed subtracts kernels placed at each trial's own go and touch, so timing leaks in: tests 2 and 3 only.

Vertices of each fold are put in the order of script 13's full-data fit (the permutation whose 16-bin weights match best).

Run:  .venv\Scripts\python scripts/18_single_trial_states.py [--plot]
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import warnings
from itertools import permutations

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s12b = importlib.import_module("12b_tp_vs_gdar_robust")
s13 = importlib.import_module("13_tp_vertices_ios")
from gdar_tp.sym_tp import _quad, fit_gls_pooled, fit_sym_tp, polytope_vertices, symmetric_basis, theta_of  # noqa: E402

K = s13.K
P = 4 * s13.NSUB
RANK = 3
WINDOWS = (10, 20, 40)                  # samples of 3 ms
DT = 1000.0 / s12.FS
TAU = np.arange(-200.0, 300.0, DT)      # template, ms about the event
GO_GRID = np.arange(150.0, 701.0, DT)
TOUCH_GRID = np.arange(250.0, 1136.0, DT)
NPERM = 10000
CACHE = os.path.join(ROOT, "cache", "18_single_trial_H_2021-11-12.pkl")
FIGURE = os.path.join(s12.FIG, "18_single_trial_states.png")
log = s12.log


# ------------------------------------------------------------------------------ the weight step on one window
def simplex_qp(Q, b):
    """``argmin lam' Q lam - 2 b' lam`` over the simplex, for stacks ``Q (..., 3, 3)``, ``b (..., 3)``.

    Exact for three vertices: the optimum is the best of the three edges (each a 1-D problem, clipped, which covers
    the corners) and the interior KKT point where that point is feasible."""
    def obj(lam):
        return np.einsum("...i,...ij,...j->...", lam, Q, lam) - 2 * np.einsum("...i,...i->...", b, lam)

    best, best_obj = None, None
    for v, w in ((0, 1), (0, 2), (1, 2)):
        den = Q[..., v, v] - 2 * Q[..., v, w] + Q[..., w, w]
        num = Q[..., w, w] - Q[..., v, w] + b[..., v] - b[..., w]
        x = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num > 0, 1.0, 0.0))
        x = np.clip(x, 0.0, 1.0)
        lam = np.zeros(b.shape)
        lam[..., v], lam[..., w] = x, 1 - x
        o = obj(lam)
        if best is None:
            best, best_obj = lam, o
        else:
            m = o < best_obj
            best[m], best_obj[m] = lam[m], o[m]
    M = np.zeros(Q.shape[:-2] + (4, 4))
    M[..., :3, :3] = Q
    M[..., :3, 3] = 1.0
    M[..., 3, :3] = 1.0
    rhs = np.concatenate([b, np.ones(b.shape[:-1] + (1,))], -1)[..., None]
    try:
        sol = np.linalg.solve(M, rhs)[..., 0]
    except np.linalg.LinAlgError:
        sol = (np.linalg.pinv(M) @ rhs)[..., 0]
    lam = sol[..., :3]
    o = obj(lam)
    m = np.all(lam >= 0, -1) & (o < best_obj)
    best[m] = lam[m]
    return best


def window_sums(A, W, causal):
    """Sums of ``A (trials, samples, ...)`` over a window of W samples: centred on each sample, or the W before it."""
    ns = A.shape[1]
    C = np.concatenate([np.zeros_like(A[:, :1]), np.cumsum(A, axis=1)], axis=1)
    j = np.arange(ns)
    lo = j - W if causal else j - W // 2
    hi = j if causal else lo + W
    ok = (lo >= 0) & (hi <= ns)
    return C[:, np.clip(hi, 0, ns)] - C[:, np.clip(lo, 0, ns)], ok


def lam_hat(Q, b, W, causal):
    Qw, ok = window_sums(Q, W, causal)
    bw, _ = window_sums(b, W, causal)
    Qw[:, ~ok] = np.eye(3)
    bw[:, ~ok] = 0.0
    lam = simplex_qp(Qw, bw)
    lam[:, ~ok] = np.nan
    return lam, ok


def whitener(S):
    """Row vectors r @ whitener(S) have identity covariance when r ~ N(0, S)."""
    return np.linalg.inv(np.linalg.cholesky(S)).T


def align(lam, ref):
    """Vertex order of a fold fit that best matches the full-data fit's 16-bin weights, and the mean correlation."""
    best = max(permutations(range(RANK)),
               key=lambda p: sum(np.corrcoef(lam[:, p[v]], ref[:, v])[0, 1] for v in range(RANK)))
    return list(best), float(np.mean([np.corrcoef(lam[:, best[v]], ref[:, v])[0, 1] for v in range(RANK)]))


def at(h, times, t0):
    """Weights ``h (samples, 3)`` at the given times (ms, any shape), NaN outside."""
    idx = np.rint((times - t0) / DT).astype(int)
    ok = (idx >= 0) & (idx < len(h))
    v = h[np.clip(idx, 0, len(h) - 1)]
    v[~ok] = np.nan
    return v


def locate(h, template, grid, t0):
    """Event time on ``grid`` whose template window matches ``h`` best (mean squared distance of the weights)."""
    vals = at(h, grid[:, None] + TAU[None], t0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmean(((vals - template) ** 2).sum(-1), 1)
    return grid[np.nanargmin(d)]


def mean_aligned(H, events, t0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([at(h, e + TAU, t0) for h, e in zip(H, events)]), 0)


def trial_corr(a, c, m):
    """Per trial, correlation of the two weight series over mask m, each weight less its trial mean."""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        x, y = a[i][m[i]], c[i][m[i]]
        x, y = x - x.mean(0), y - y.mean(0)
        out[i] = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
    return out


# ------------------------------------------------------------------------------ one fold
def fold_job(job):
    variant, f, X, lab, fo, t, go, touch, A, ref = job
    n, T, N = X.shape
    te, trn = np.where(fo == f)[0], fo != f
    R = symmetric_basis(A, K)
    s = s12b.fold_stats(X, lab, K, fo, P)
    tr = s12b.sel(s, f)
    a = {k: tr[k].sum(0) for k in tr}
    phi0, Sf = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
    Hbar = _quad(a["Gzz"], a["Gzy"], R, np.linalg.pinv(Sf))[0] / a["n"]
    model = fit_sym_tp(tr, R, Sf, RANK, np.arange(float(P)), n_iter=25)
    poly = polytope_vertices(model, Hbar)
    perm, match = align(poly["lam"], ref)
    lam_bins = poly["lam"][:, perm]
    Th0 = theta_of(phi0, R, N)
    Th = [theta_of(poly["phi"][v], R, N) for v in perm]
    Ss = s12.sigma_of(np.stack([Th0] * P), tr)

    ts = np.arange(s12b.K0 - 1, T - 1)
    tt = t[ts + 1]                                      # time of each target x(s+1)
    Y = X[:, ts + 1]
    U = np.zeros((n, len(ts), RANK + 1, N))             # GDAR, then the vertices
    for k in range(K):
        Xk = X[:, ts - k]
        blk = slice(k * N, (k + 1) * N)
        U[:, :, 0] += Xk @ Th0[:, blk].T
        for v in range(RANK):
            U[:, :, v + 1] += Xk @ Th[v][:, blk].T
    Wf, Ws = whitener(Sf), whitener(Ss)
    Uf, Yf = U[:, :, 1:] @ Wf, Y @ Wf                   # metric of the fit, for the weights
    Qf, bf = np.einsum("isvn,iswn->isvw", Uf, Uf), np.einsum("isvn,isn->isv", Uf, Yf)
    del Uf, Yf
    Us, Ys = U[te] @ Ws, Y[te] @ Ws                     # GDAR's scoring Sigma, test trials only
    Qs, bs = np.einsum("isvn,iswn->isvw", Us[:, :, 1:], Us[:, :, 1:]), np.einsum("isvn,isn->isv", Us[:, :, 1:], Ys)
    cs = (Ys ** 2).sum(-1)
    rn0 = ((Ys - Us[:, :, 0]) ** 2).sum(-1)
    del U, Us, Ys

    def rn(lam):
        return cs - 2 * np.einsum("isv,isv->is", bs, lam) + np.einsum("isv,isvw,isw->is", lam, Qs, lam)

    lt = lab[:, ts + 1]
    clock = np.where(lt[..., None] >= 0, lam_bins[np.maximum(lt, 0)], np.nan)
    go_med, touch_med = float(np.median(go[trn])), float(np.median(touch[trn]))
    lab_med = s13.warped_labels(dict(t=t, go=np.full(n, go_med), touch=np.full(n, touch_med)))[:, ts + 1]
    clock_med = np.where(lab_med[..., None] >= 0, lam_bins[np.maximum(lab_med, 0)], np.nan)

    Wmax = max(WINDOWS)
    j = np.arange(len(ts))
    centred_ok = (j - Wmax // 2 >= 0) & (j - Wmax // 2 + Wmax <= len(ts))
    m2 = (lt[te] >= 0) & (lab_med[te] >= 0) & centred_ok
    m3 = (lt[te] >= 0) & (j >= Wmax)
    phase = np.maximum(lt[te], 0) // s13.NSUB
    gsum = {"clock": np.array([(-0.5 * (rn(np.nan_to_num(clock[te])) - rn0))[m3 & (phase == q)].sum() for q in range(4)])}
    cnt = np.array([(m3 & (phase == q)).sum() for q in range(4)])

    out = dict(f=f, te=te, tt=tt, match=match, lam_bins=lam_bins, go_med=go_med, touch_med=touch_med,
               clock=clock[te].astype(np.float32), hat={}, r_own={}, r_med={}, gsum=gsum, cnt=cnt,
               go_hat={}, touch_hat={}, tmpl_go={}, tmpl_touch={})
    for W in WINDOWS:
        hc, _ = lam_hat(Qf, bf, W, causal=False)
        hp, _ = lam_hat(Qf[te], bf[te], W, causal=True)
        out["hat"][W] = hc[te].astype(np.float32)
        out["r_own"][W] = trial_corr(hc[te], clock[te], m2)
        out["r_med"][W] = trial_corr(hc[te], clock_med[te], m2)
        g = -0.5 * (rn(np.nan_to_num(hp)) - rn0)
        gsum[W] = np.array([g[m3 & (phase == q)].sum() for q in range(4)])
        if variant == "felix":
            rows = np.where(trn)[0]
            out["tmpl_go"][W] = Tg = mean_aligned(hc[rows], go[rows], tt[0])
            rows_t = rows[touch[rows] <= TOUCH_GRID[-1]]
            out["tmpl_touch"][W] = Tt = mean_aligned(hc[rows_t], touch[rows_t], tt[0])
            out["go_hat"][W] = np.array([locate(hc[i], Tg, GO_GRID, tt[0]) for i in te])
            out["touch_hat"][W] = np.array([locate(hc[i], Tt, TOUCH_GRID, tt[0]) for i in te])
    return out


# ------------------------------------------------------------------------------ pooled over folds
def timing_stats(true, est, guess, rng):
    r = np.corrcoef(true, est)[0, 1]
    perm = np.array([np.corrcoef(rng.permutation(true), est)[0, 1] for _ in range(NPERM)])
    err, err0 = np.abs(est - true), np.abs(guess - true)
    return dict(r=r, p=(1 + (perm >= r).sum()) / (1 + NPERM), mae=float(np.median(err)), mae_guess=float(np.median(err0)),
                better=float(np.mean(err < err0)), n=len(true))


def summarize(variant, res, D):
    from scipy.stats import wilcoxon
    go, touch = D["go"], D["touch"]
    te = np.concatenate([r["te"] for r in res])
    S = dict(te=te, folds=res, match=[r["match"] for r in res])
    log(f"\n=== {variant}: vertex match to the full-data fit (mean r of the 16-bin weights), by fold: "
        + " ".join(f"{m:.2f}" for m in S["match"]))

    cnt = sum(r["cnt"] for r in res)
    S["gain"], S["gain_se"], S["gain_phase"] = {}, {}, {}
    log(f"  test 3, one-step gain over GDAR, GDAR's Sigma, {int(cnt.sum())} samples (nats/sample, fold SE):")
    for name in ["clock"] + list(WINDOWS):
        tot = sum(r["gsum"][name] for r in res)
        per_fold = np.array([r["gsum"][name].sum() / r["cnt"].sum() for r in res])
        S["gain"][name], S["gain_se"][name] = tot.sum() / cnt.sum(), per_fold.std(ddof=1) / np.sqrt(len(res))
        S["gain_phase"][name] = tot / cnt
        label = "clock weights (own go, touch)" if name == "clock" else f"weights from the {name * DT:.0f} ms before"
        log(f"    {label:32s} {S['gain'][name]:+.4f} ({S['gain_se'][name]:.4f})   by phase "
            + " ".join(f"{q} {x:+.3f}" for q, x in zip(s12.PHASES, S["gain_phase"][name])))

    S["r_own"], S["r_med"] = {}, {}
    log("  test 2, per-trial correlation of the window weights with the clock weights (median, IQR):")
    for W in WINDOWS:
        a = np.concatenate([r["r_own"][W] for r in res])
        b = np.concatenate([r["r_med"][W] for r in res])
        S["r_own"][W], S["r_med"][W] = a, b
        log(f"    {W * DT:4.0f} ms  own clock {np.median(a):.2f} ({np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f})   "
            f"median-timed clock {np.median(b):.2f}   own higher on {np.mean(a > b):.0%} of trials, Wilcoxon p {wilcoxon(a, b).pvalue:.1e}")

    if variant != "felix":
        return S
    rng = np.random.default_rng(0)
    gm = np.concatenate([np.full(len(r["te"]), r["go_med"]) for r in res])
    tm = np.concatenate([np.full(len(r["te"]), r["touch_med"]) for r in res])
    ok = touch[te] <= TOUCH_GRID[-1]
    S["timing"] = {}
    log(f"  test 1, event times read off the weights ({len(te)} trials; touch on the {ok.sum()} with touch <= {TOUCH_GRID[-1]:.0f} ms):")
    for W in WINDOWS:
        gh = np.concatenate([r["go_hat"][W] for r in res])
        th = np.concatenate([r["touch_hat"][W] for r in res])
        T = dict(go=timing_stats(go[te], gh, gm, rng), touch=timing_stats(touch[te][ok], th[ok], tm[ok], rng),
                 move=timing_stats((touch - go)[te][ok], (th - gh)[ok], (tm - gm)[ok], rng), go_hat=gh, touch_hat=th)
        S["timing"][W] = T
        for ev in ("go", "touch", "move"):
            x = T[ev]
            log(f"    {W * DT:4.0f} ms  {ev:5s} r {x['r']:+.2f} (perm p {x['p']:.4f})   median error {x['mae']:.0f} ms, "
                f"guessing the median {x['mae_guess']:.0f} ms, closer on {x['better']:.0%}")
    S["go_med"], S["touch_med"] = gm, tm
    return S


def main():
    D = s12.load()
    lab = s13.warped_labels(D)
    fo = s12.folds(D["ang"])
    Z = s12.evoked_design(D)
    with open(s13.CACHE, "rb") as fh:
        c13 = pickle.load(fh)
    out = dict(go=D["go"], touch=D["touch"], t=D["t"], ang=D["ang"])
    for variant in ("felix", "induced"):
        ref = c13[variant]["full"][RANK]["poly"]["lam"]
        jobs = [(variant, f, D["X"] if variant == "felix" else s12.remove_evoked(D["X"], Z, fo != f),
                 lab, fo, D["t"], D["go"], D["touch"], D["A"], ref) for f in range(s12.NFOLD)]
        out[variant] = summarize(variant, s12.fold_map(fold_job, jobs), D)
        out[variant]["ref"] = ref
        with open(CACHE, "wb") as fh:
            pickle.dump(out, fh)
    log("done")


# ------------------------------------------------------------------------------ figure
def plot():
    import matplotlib.pyplot as plt
    s12._style()
    with open(CACHE, "rb") as fh:
        C = pickle.load(fh)
    go, touch = C["go"], C["touch"]
    F, I = C["felix"], C["induced"]
    W = 20
    col = s13.COL
    te = F["te"]
    hat = np.concatenate([r["hat"][W] for r in F["folds"]])
    clock = np.concatenate([r["clock"] for r in F["folds"]])
    tt = F["folds"][0]["tt"]
    move_v = int(np.argmax(F["ref"][2 * s13.NSUB:3 * s13.NSUB].mean(0)))

    fig = plt.figure(figsize=(14, 11.5), constrained_layout=True)
    fig.suptitle(f"Rank 3 weights on single held-out trials, read off {W * DT:.0f} ms of data with no task clock "
                 "(H 2021-11-12, evoked left in unless marked)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    rows = fig.subfigures(3, 1, height_ratios=[1, 1, 1])
    ex = rows[0].subplots(1, 3, sharey=True)
    order = np.argsort(go[te])
    for ax, q, name in zip(ex, (0.1, 0.5, 0.9), ("early", "median", "late")):
        k = order[int(q * (len(te) - 1))]
        i = te[k]
        for v in range(RANK):
            ax.plot(tt, hat[k][:, v], color=col[v], lw=1.3, label=f"vertex {v + 1}")
            ax.plot(tt, clock[k][:, v], color=col[v], lw=1.6, ls="--")
        for x, lab in ((0, "target"), (go[i], "go"), (touch[i], "touch")):
            ax.axvline(x, color=s12.MUTED, lw=0.9, ls=":")
            ax.text(x, 1.04, lab, ha="center", va="bottom", fontsize=8, color=s12.MUTED)
        ax.set_xlim(-400, 1400)
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlabel("ms from target on")
        ax.set_title(f"A  Trial {i}: go {go[i]:.0f} ms", loc="left")
    ex[0].set_ylabel("weight (solid: from data; dashed: clock)")
    ex[2].legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8, frameon=False, borderaxespad=0.2,
                 handlelength=1.2, columnspacing=0.8)

    mid = rows[1].subplots(1, 3)
    T = F["timing"][W]
    for ax, ev, true, grid, ttl in ((mid[0], "go", go[te], GO_GRID, "B  Go cue read off the weights"),
                                    (mid[1], "touch", touch[te], TOUCH_GRID, "C  Touch read off the weights")):
        est = T[f"{ev}_hat"]
        ok = np.ones(len(te), bool) if ev == "go" else touch[te] <= TOUCH_GRID[-1]
        ax.scatter(true[ok], est[ok], s=10, color=s12.C_FELIX, alpha=0.6, lw=0)
        lo, hi = grid[0], grid[-1]
        ax.plot([lo, hi], [lo, hi], color=s12.MUTED, lw=0.9, ls=":")
        x = T[ev]
        ax.set_title(ttl, loc="left")
        ax.text(0.03, 0.97, f"r = {x['r']:+.2f}, perm p {x['p']:.4f}\nmedian error {x['mae']:.0f} ms "
                f"(median guess {x['mae_guess']:.0f})\n{x['n']} trials", transform=ax.transAxes, va="top", fontsize=8.5)
        ax.set_xlabel(f"true {ev} (ms from target on)")
        ax.set_ylabel(f"estimated {ev}")
        ax.set_xlim(lo - 20, hi + 20)
        ax.set_ylim(lo - 20, hi + 20)
    ax = mid[2]
    terc = np.quantile(go[te], [1 / 3, 2 / 3])
    for which, (a, b), shade in zip(("early", "middle", "late"), ((-np.inf, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)),
                                    (0.35, 0.65, 1.0)):
        sel = (go[te] >= a) & (go[te] < b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(hat[sel][:, :, move_v], 0)
        c = np.array(plt.matplotlib.colors.to_rgb(col[move_v])) * shade + (1 - shade)
        ax.plot(tt, m, color=c, lw=1.8, label=f"{which} third, go {go[te][sel].mean():.0f} ms")
        ax.axvline(go[te][sel].mean(), color=c, lw=1, ls=":")
    ax.set_xlim(-400, 1400)
    ax.set_xlabel("ms from target on")
    ax.set_ylabel(f"vertex {move_v + 1} weight, trial mean")
    ax.set_title("D  The move vertex by go time (dotted: mean go)", loc="left")
    ax.legend(fontsize=8, loc="center left")

    bot = rows[2].subplots(1, 2, width_ratios=[1, 1.4])
    ax = bot[0]
    for k, (S, c, name) in enumerate(((F, s12.C_FELIX, "evoked left in"), (I, s12.C_IND, "evoked removed"))):
        x = np.arange(len(WINDOWS)) + (k - 0.5) * 0.2
        ax.plot(x, [np.median(S["r_own"][w]) for w in WINDOWS], "o-", color=c, label=f"{name}: own clock")
        ax.plot(x, [np.median(S["r_med"][w]) for w in WINDOWS], "o--", color=c, mfc="white", label=f"{name}: median-timed clock")
    ax.set_xticks(range(len(WINDOWS)), [f"{w * DT:.0f} ms" for w in WINDOWS])
    ax.set_ylabel("per-trial r with clock weights (median)")
    ax.set_title("E  Agreement with the clock", loc="left")
    ax.legend(fontsize=8)
    ax = bot[1]
    names = ["clock"] + list(WINDOWS)
    for k, (S, c, name) in enumerate(((F, s12.C_FELIX, "evoked left in"), (I, s12.C_IND, "evoked removed"))):
        x = np.arange(len(names)) + (k - 0.5) * 0.3
        ax.bar(x, [S["gain"][n] for n in names], 0.3, yerr=[S["gain_se"][n] for n in names], color=c, label=name)
    ax.set_xticks(range(len(names)), ["clock\n(own go, touch)"] + [f"past {w * DT:.0f} ms\nof data" for w in WINDOWS])
    s12._zero(ax)
    ax.set_ylabel("one-step gain over GDAR (nats/sample)")
    ax.set_title("F  Held-out prediction: clock weights vs weights from the past window", loc="left")
    ax.legend(fontsize=8)
    fig.savefig(FIGURE, dpi=130)
    log("wrote", FIGURE)


if __name__ == "__main__":
    plot() if "--plot" in sys.argv else main()
