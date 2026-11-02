"""TP-GDAR vs Felix Schwock's GDAR, no input, on a no-stim IOS Reach day.

Hydrogen 2021-11-12: 0 lit trials, center-out reach.  What TP-GDAR adds over one
state-blind operator, tested on held-out trials.

Baseline ("GDAR"): Felix's model and estimator, symmetric VAR on the proximity graph,
GLS (`gdar_tp.sym_tp`, checked against `CVARModel.fit_gdar` to 1e-11), his 1.5 unit
graph threshold on his 32ch_mmad positions, his preprocessing (decimate x3, per-trial
demean), his no-penalty fit.  It is pooled over trials without crossing trial boundaries.

Scheduling variable p, two grids:
  phase  hold / plan / move / post-touch.  Categorical, so p cannot be interpolated.
  time   12 bins of 100 ms around the go cue.  Ordered, so leave-one-bin-out asks
         whether TP predicts a part of the task it never saw.

Two data variants, because the task-evoked response is an unmodelled input that a
state-dependent A can absorb:
  felix    per-trial demean only
  induced  also removes per-direction target-, go- and touch-locked kernels, fitted on
           the training trials of each fold

Score: held-out one-step Gaussian log-likelihood (each model has its own full Sigma from
its training residuals), 5-fold CV over trials, as Delta-ell vs GDAR, nats per sample.
Order K is chosen by CV of the baseline only, then fixed for every model.

Run:  .venv\\Scripts\\python scripts/12_tp_vs_gdar_ios.py [--plot]
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import pickle
import sys
import time

import numpy as np
from scipy.signal import decimate
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gdar_tp.sym_tp import fit_gls_pooled, fit_sym_tp, symmetric_basis, theta_of  # noqa: E402
from gdar_tp.tpgdar import (fit_joint, fit_per_condition, fit_pooled,  # noqa: E402
                            resid_cov_sum, row_columns)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

YL = os.path.join(os.path.dirname(ROOT), "yazdanlab")
SEGS = os.path.join(YL, "iosreach", "cache", "segs", "H_2021-11-12.npz")
POS = os.path.join(YL, "repos", "network_analysis_software", "recording_arrays",
                   "32ch_mmad_electrode_positions.pkl")
CACHE = os.path.join(ROOT, "cache", "12_results_H_2021-11-12.pkl")
FIG = os.path.join(ROOT, "figures")

DEC = 3
FS = 1000.0 / DEC
BAD = [29]                  # B14 on Hydrogen, as in iosreach/scripts/lti_phase.py
GRAPH_TH = 1.5
NFOLD = 5
KS = (1, 2, 3, 5, 8, 12, 16)
PHASES = ["hold", "plan", "move", "post"]
TBIN = np.arange(-400, 801, 100)
RANKS_PHASE = (1, 2, 3)
RANKS_TIME = (1, 2, 3, 4, 6)
RANKS_LOBO = (1, 2, 3)
RIDGES = (1e-6, 1e-8, 1e-10, 1e-12)
SIG_SHRINK = 0.01
HORIZONS = (1, 3, 6, 10, 17)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- data
def load():
    z = np.load(SEGS, allow_pickle=True)
    ch = z["channels"]
    assert int(z["stimmed"].sum()) == 0
    keep = ~np.isin(ch, BAD)
    ok = z["move_ms"] < 3000
    X = z["noart"][keep][:, :, ok].astype(float)            # (N, 2000, trials), uV
    X = decimate(X, DEC, axis=1, zero_phase=True)
    X = (X - X.mean(1, keepdims=True)).transpose(2, 1, 0)   # (trials, T, N)
    t = (np.arange(2000) - 500)[::DEC].astype(float)
    t = t[:X.shape[1]]
    with open(POS, "rb") as f:
        p = pickle.load(f)
    pos = np.array([p[int(c)] for c in ch[keep]], float)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    A = (d <= GRAPH_TH + 1e-9) & (d > 0)
    go = z["plan_ms"][ok].astype(float)
    touch = go + z["move_ms"][ok]
    ang = z["angle"][ok].astype(int) % 360
    return dict(X=X, t=t, pos=pos, A=A, go=go, touch=touch, ang=ang, ch=ch[keep])


def labels(D):
    """Per sample condition labels of the target x(t+1), -1 = unused. (trials, T)."""
    t, go, touch = D["t"], D["go"], D["touch"]
    n = len(go)
    ph = np.full((n, len(t)), -1)
    tb = np.full((n, len(t)), -1)
    for k in range(n):
        ph[k, (t >= -400) & (t < 0)] = 0
        ph[k, (t >= 0) & (t < go[k])] = 1
        ph[k, (t >= go[k]) & (t < touch[k])] = 2
        ph[k, (t >= touch[k]) & (t < touch[k] + 500) & (t <= 1400)] = 3
        b = np.searchsorted(TBIN, t - go[k], side="right") - 1
        tb[k] = np.where((b >= 0) & (b < len(TBIN) - 1) & (t >= -400) & (t <= 1400), b, -1)
    return {"phase": ph, "time": tb}


def folds(ang, seed=0):
    rng = np.random.default_rng(seed)
    f = np.zeros(len(ang), int)
    for a in np.unique(ang):
        i = rng.permutation(np.where(ang == a)[0])
        f[i] = np.arange(len(i)) % NFOLD
    return f


def evoked_design(D):
    """Sparse per-direction target/go/touch kernels, 9 ms bins, as in iosreach lti_stim."""
    import scipy.sparse as sp
    t, go, touch, ang = D["t"], D["go"], D["touch"], D["ang"]
    n, T = len(go), len(t)
    dirs = np.unique(ang)
    rows, cols, off = [], [], 0
    for ev, lo, hi in ((np.zeros(n), -500, 1500), (go, -300, 800), (touch, -400, 600)):
        nb = int((hi - lo) // 9)
        for k in range(n):
            b = np.floor((t - ev[k] - lo) / 9).astype(int)
            s = np.where((b >= 0) & (b < nb))[0]
            rows.append(k * T + s)
            cols.append(off + np.searchsorted(dirs, ang[k]) * nb + b[s])
        off += nb * len(dirs)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * T, off))


def remove_evoked(X, Z, train, lam=1e-2):
    """Kernels fitted on the training trials, subtracted from every trial."""
    n, T, N = X.shape
    Y = X.reshape(n * T, N)
    r = np.repeat(train, T)
    Zr = Z[r]
    G = (Zr.T @ Zr).toarray()
    d = np.median(np.diag(G)[np.diag(G) > 0])
    beta = np.linalg.solve(G + lam * d * np.eye(len(G)), Zr.T @ Y[r])
    return (Y - Z @ beta).reshape(n, T, N)


# ------------------------------------------------------------------------------ stats
def trial_cond_stats(X, lab, K, K0):
    """Per (trial, condition) Grams.  Every K uses the samples allowed by K0 >= K."""
    n, T, N = X.shape
    D_ = K * N
    P = lab.max() + 1
    Gzz = np.zeros((n, P, D_, D_))
    Gzy = np.zeros((n, P, D_, N))
    Syy = np.zeros((n, P, N, N))
    cnt = np.zeros((n, P))
    ts = np.arange(K0 - 1, T - 1)
    for i in range(n):
        Z = np.concatenate([X[i, ts - k] for k in range(K)], 1)
        Y = X[i, ts + 1]
        l = lab[i, ts + 1]
        for g in range(P):
            m = l == g
            if m.any():
                Zg, Yg = Z[m], Y[m]
                Gzz[i, g] = Zg.T @ Zg
                Gzy[i, g] = Zg.T @ Yg
                Syy[i, g] = Yg.T @ Yg
                cnt[i, g] = m.sum()
    return dict(Gzz=Gzz, Gzy=Gzy, Syy=Syy, n=cnt)


def sum_stats(T, sel):
    return {k: T[k][sel].sum(0) for k in T}


def pool(stats):
    return {k: stats[k].sum(0) for k in stats}


def sigma_of(Thetas, stats):
    S = sum(resid_cov_sum(Thetas[g], stats["Gzz"][g], stats["Gzy"][g], stats["Syy"][g])
            for g in range(len(Thetas)))
    S = S / stats["n"].sum()
    return (1 - SIG_SHRINK) * S + SIG_SHRINK * np.diag(np.diag(S))


def ll_matrix(Thetas, T, sel, Sigma):
    """Log-likelihood sum per (test trial, condition)."""
    Sinv = np.linalg.inv(Sigma)
    _, ld = np.linalg.slogdet(Sigma)
    N = Sigma.shape[0]
    idx = np.where(sel)[0]
    out = np.zeros((len(idx), Thetas.shape[0]))
    sse = np.zeros_like(out)
    for a, i in enumerate(idx):
        for g in range(Thetas.shape[0]):
            if T["n"][i, g] == 0:
                continue
            S = resid_cov_sum(Thetas[g], T["Gzz"][i, g], T["Gzy"][i, g], T["Syy"][i, g])
            out[a, g] = -0.5 * (np.sum(Sinv * S) + T["n"][i, g] * (N * np.log(2 * np.pi) + ld))
            sse[a, g] = np.trace(S)
    return out, sse


def rho(Theta, K):
    N = Theta.shape[0]
    Ac = np.zeros((K * N, K * N))
    Ac[:N] = Theta
    if K > 1:
        Ac[N:, :-N] = np.eye((K - 1) * N)
    return np.linalg.eigvals(Ac)


def h_step_vaf(Thetas, X, lab, sel, K, K0, horizons=HORIZONS, stride=3):
    """h-step-ahead VAF on held-out trials; the operator follows the label of each target."""
    N = X.shape[2]
    T = X.shape[1]
    hmax = max(horizons)
    sse = {h: 0.0 for h in horizons}
    sst = {h: 0.0 for h in horizons}
    for x, l in zip(X[sel], lab[sel]):
        t0 = np.arange(K0 - 1, T - 1 - hmax, stride)
        t0 = t0[l[t0 + 1] >= 0]
        if not len(t0):
            continue
        hist = np.stack([x[t0 - k] for k in range(K)], 1)       # (m, K, N), newest first
        for h in range(1, hmax + 1):
            z = hist.reshape(len(t0), K * N)
            g = np.clip(l[t0 + h], 0, None)
            pred = np.einsum("mij,mj->mi", Thetas[g], z)
            hist = np.concatenate([pred[:, None], hist[:, :-1]], 1)
            if h in sse:
                use = l[t0 + h] >= 0
                sse[h] += np.sum((x[t0 + h][use] - pred[use]) ** 2)
                sst[h] += np.sum(x[t0 + h][use] ** 2)
    return np.array([1 - sse[h] / sst[h] for h in horizons])


# ------------------------------------------------------------------------------- run
def select_K(D, labs, fo):
    """CV of the baseline only, felix variant, same sample set for every K."""
    log("order selection (GDAR baseline only)")
    A = D["A"]
    out = {}
    N = D["X"].shape[2]
    K0 = max(KS)
    lab = (labs["phase"] >= 0).astype(int) - 1
    T = trial_cond_stats(D["X"], lab, K0, K0)
    for K in KS:
        R = symmetric_basis(A, K)
        c = K * N
        sub = {"Gzz": T["Gzz"][:, :, :c, :c], "Gzy": T["Gzy"][:, :, :c], "Syy": T["Syy"], "n": T["n"]}
        vals = []
        for f in range(NFOLD):
            tr = pool(sum_stats(sub, fo != f))
            phi, _ = fit_gls_pooled(tr["Gzz"], tr["Gzy"], tr["Syy"], tr["n"], R)
            Th = theta_of(phi, R, N)[None]
            st = sum_stats(sub, fo != f)
            Sig = sigma_of(Th, st)
            ll, _ = ll_matrix(Th, sub, fo == f, Sig)
            vals.append(ll.sum() / T["n"][fo == f].sum())
        out[K] = float(np.mean(vals))
        log(f"  K={K:<3d} held-out ell {out[K]:.4f}")
    ks = list(out)
    best = max(ks, key=lambda k: out[k])
    return best, out


def fit_grid(st, R, N, K, ranks, p_grid, mask, ridge):
    """All models on one grid.  name -> (P, N, D) operators, plus params."""
    P = len(p_grid)
    tot = pool(st)
    phi, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    n_phi = R.shape[1]
    models = {"GDAR": np.stack([theta_of(phi, R, N)] * P)}
    params = {"GDAR": n_phi}
    tps = {}
    for r in ranks:
        m = fit_sym_tp(st, R, Sig, r, p_grid)
        models[f"TP r{r}"] = m.theta_grid()
        params[f"TP r{r}"] = m.n_params
        tps[r] = m
    one = {k: None for k in ()}
    per = []
    for g in range(P):
        single = {k: st[k][[g]] for k in st}
        per.append(fit_sym_tp(single, R, Sig, 1, p_grid[[g]], n_iter=1).theta_grid()[0])
    models["per-condition"] = np.stack(per)
    params["per-condition"] = P * n_phi
    # the repo's edge-wise TP-GDAR (not symmetric), same Grams
    cols = row_columns(mask, K)
    D_ = K * N
    if ridge is not None:
        pm = fit_pooled(st, cols, ridge)
        models["edge-wise pooled"] = np.stack([pm] * P)
        params["edge-wise pooled"] = sum(len(c) for c in cols)
        for r in ranks:
            if r <= 3:
                mj = fit_joint(st, cols, [np.arange(D_)], [r], p_grid, ridge, n_iter=40)
                models[f"edge-wise TP r{r}"] = mj.theta_grid()
                params[f"edge-wise TP r{r}"] = r * sum(len(c) for c in cols) + P * r - r * r
    return models, params, tps, Sig


def pick_ridge(T_all, fo, cols):
    """Edge-wise ridge by validation inside the training folds of fold 0."""
    inner = (fo != 0) & (fo != 1)
    val = fo == 1
    best = None
    tr = sum_stats(T_all, inner)
    for rho_ in RIDGES:
        Th = np.stack([fit_pooled(tr, cols, rho_)] * tr["n"].shape[0])
        ll, _ = ll_matrix(Th, T_all, val, sigma_of(Th, tr))
        v = ll.sum() / T_all["n"][val].sum()
        if best is None or v > best[1]:
            best = (rho_, v)
    return best[0]


def run_variant(D, labs, fo, K, variant, Z):
    t0 = time.time()
    X0 = D["X"]
    n, _, N = X0.shape
    R = symmetric_basis(D["A"], K)
    mask = D["A"] | np.eye(N, dtype=bool)
    cols = row_columns(mask, K)
    res = {"K": K, "variant": variant}
    grids = {"phase": (np.arange(4.0), RANKS_PHASE), "time": ((TBIN[:-1] + 50.0), RANKS_TIME)}
    ridge = None
    K0 = max(KS)
    for grid, (p_grid, ranks) in grids.items():
        lab = labs[grid]
        P = len(p_grid)
        store = {}
        counts = np.zeros((n, P))
        params = None
        vaf = {}
        lobo = {}
        for f in range(NFOLD):
            X = X0 if variant == "felix" else remove_evoked(X0, Z, fo != f)
            T = trial_cond_stats(X, lab, K, K0)
            counts += T["n"] * (fo == f)[:, None]
            tr = sum_stats(T, fo != f)
            if ridge is None:
                ridge = pick_ridge(T, fo, cols)
                log(f"  edge-wise ridge {ridge:.0e}")
            models, params, _, _ = fit_grid(tr, R, N, K, ranks, p_grid, mask,
                                            ridge if grid == "phase" else None)
            for name, Th in models.items():
                Sig = sigma_of(Th, tr)
                ll, sse = ll_matrix(Th, T, fo == f, Sig)
                s = store.setdefault(name, dict(ll=np.zeros((n, P)), sse=np.zeros((n, P))))
                s["ll"][fo == f] = ll
                s["sse"][fo == f] = sse
                if grid == "phase" and name in ("GDAR", "TP r2", "TP r3", "per-condition"):
                    vaf.setdefault(name, []).append(h_step_vaf(Th, X, lab, fo == f, K, K0))
            if grid == "time":
                for h in range(1, P - 1):           # leave-one-bin-out, interior bins
                    keep = [g for g in range(P) if g != h]
                    sub = {k: tr[k][keep] for k in tr}
                    tot = pool(sub)
                    phi, Sig0 = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
                    per_full = fit_sym_tp({k: tr[k][[h]] for k in tr}, R, Sig0, 1, p_grid[[h]], n_iter=1)
                    near = fit_sym_tp({k: tr[k][[h - 1]] for k in tr}, R, Sig0, 1, p_grid[[h - 1]], n_iter=1)
                    cand = {"GDAR": (np.stack([theta_of(phi, R, N)] * len(keep)), theta_of(phi, R, N)),
                            "nearest bin": (np.stack([near.theta_grid()[0]] * len(keep)), near.theta_grid()[0]),
                            "oracle": (np.stack([per_full.theta_grid()[0]] * len(keep)), per_full.theta_grid()[0])}
                    for r in RANKS_LOBO:
                        m = fit_sym_tp(sub, R, Sig0, r, p_grid[keep])
                        cand[f"TP r{r}"] = (m.theta_grid(), m.theta_at(p_grid[h]))
                    for name, (Thtr, Thh) in cand.items():
                        Sig = sigma_of(Thtr, sub)
                        ll, _ = ll_matrix(np.stack([Thh] * P), {k: T[k] for k in T}, fo == f, Sig)
                        arr = lobo.setdefault(name, np.zeros((n, P)))
                        arr[fo == f, h] = ll[:, h]
            log(f"  [{variant}/{grid}] fold {f} done, {time.time()-t0:.0f}s")
        res[grid] = dict(store=store, n=counts, params=params, p_grid=p_grid,
                         vaf={k: np.mean(v, 0) for k, v in vaf.items()}, lobo=lobo)

    # full-data operators for the interpretation figure, plus disjoint split-half reliability
    X = X0 if variant == "felix" else remove_evoked(X0, Z, np.ones(n, bool))
    T = trial_cond_stats(X, labs["phase"], K, K0)
    st = sum_stats(T, np.ones(n, bool))
    mods, _, tps, Sig = fit_grid(st, R, N, K, (1, 2, 3), np.arange(4.0), mask, None)
    res["full"] = dict(models={k: v for k, v in mods.items()}, W2=tps[2].W, W3=tps[3].W, Sigma=Sig,
                       J={r: tps[r].objective for r in tps},
                       rho={k: [np.abs(rho(th, K)).max() for th in v] for k, v in mods.items()},
                       eig={k: [rho(th, K) for th in v] for k, v in mods.items()})
    rng = np.random.default_rng(1)
    rel = {}
    for name in ("TP r2", "TP r3", "per-condition"):
        rel[name] = []
    for rep in range(5):
        half = rng.permutation(n) < n // 2
        a, _, _, _ = fit_grid(sum_stats(T, half), R, N, K, (2, 3), np.arange(4.0), mask, None)
        b, _, _, _ = fit_grid(sum_stats(T, ~half), R, N, K, (2, 3), np.arange(4.0), mask, None)
        for name in rel:
            da = (a[name] - a["GDAR"]).ravel()
            db = (b[name] - b["GDAR"]).ravel()
            rel[name].append(np.corrcoef(da, db)[0, 1])
    res["reliability"] = {k: np.array(v) for k, v in rel.items()}
    return res


def summarize(res):
    for grid in ("phase", "time"):
        g = res[grid]
        n = g["n"]
        base = g["store"]["GDAR"]["ll"]
        log(f"\n[{res['variant']}] grid={grid}: Delta-ell vs GDAR, nats/sample, held-out trials")
        for name, s in g["store"].items():
            d = (s["ll"] - base).sum(1) / n.sum(1)
            w = wilcoxon(d[np.abs(d) > 0]).pvalue if np.any(np.abs(d) > 0) else np.nan
            pc = (s["ll"] - base).sum(0) / n.sum(0)
            vaf = 1 - s["sse"].sum() / g["store"]["GDAR"]["sse"].sum()
            log(f"  {name:<20} {(s['ll']-base).sum()/n.sum():+.4f}  trial mean {d.mean():+.4f} +/- {d.std()/np.sqrt(len(d)):.4f}"
                f"  p={w:.2g}  params {g['params'].get(name, 0):>6}  SSE gain {100*vaf:+.2f}%  by cond "
                + " ".join(f"{x:+.3f}" for x in pc))
        if g["lobo"]:
            log("  leave-one-bin-out, interior bins")
            b = g["lobo"]["GDAR"]
            nn = n[:, 1:-1].sum()
            for name, a in g["lobo"].items():
                d = (a - b)[:, 1:-1]
                log(f"    {name:<14} {d.sum()/nn:+.4f}   by bin " + " ".join(f"{x:+.3f}" for x in d.sum(0) / n[:, 1:-1].sum(0)))
        if g["vaf"]:
            for name, v in g["vaf"].items():
                log(f"  h-step VAF {name:<14} " + " ".join(f"{x:.4f}" for x in v))
    log(f"  spectral radius: " + "; ".join(f"{k} {np.round(v, 3).tolist()}" for k, v in res["full"]["rho"].items()))
    log("  split-half reliability of Theta(p)-pooled: " + "; ".join(f"{k} {v.mean():.3f}" for k, v in res["reliability"].items()))


# ------------------------------------------------------------------------------- plots
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_FELIX, C_IND, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
VAR = {"felix": ("evoked left in (Felix)", C_FELIX), "induced": ("evoked removed", C_IND)}


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
                         "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "text.color": INK, "pdf.fonttype": 42})


def _gain(g, name, base="GDAR"):
    """Pooled Delta-ell and its across-trial SEM, nats per sample."""
    d = g["store"][name]["ll"] - g["store"][base]["ll"]
    per = d.sum(1) / g["n"].sum(1)
    return d.sum() / g["n"].sum(), per.std() / np.sqrt(len(per))


def _zero(ax):
    ax.axhline(0, color=MUTED, lw=1)


def fig_prediction(R):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    fig.suptitle(f"TP-GDAR vs Felix's GDAR, no input: Hydrogen 2021-11-12, no stim, {R['n_trials']} reach trials, "
                 f"held-out trials (5-fold), K = {R['K']}", fontsize=11, fontweight="bold", x=0.01, ha="left")
    # A: phase grid, by rank
    ax = axs[0, 0]
    names = ["TP r1", "TP r2", "TP r3", "per-condition"]
    for j, v in enumerate(VAR):
        lab, c = VAR[v]
        g = R[v]["phase"]
        x = np.arange(4) + (j - 0.5) * 0.16
        m = [_gain(g, n) for n in names]
        ax.errorbar(x, [a for a, _ in m], [b for _, b in m], fmt="o", ms=7, color=c, elinewidth=1.5, label=lab)
    _zero(ax)
    ax.set_xticks(range(4), ["rank 1", "rank 2", "rank 3", "separate fit\nper phase"])
    ax.set_ylabel("held-out gain over GDAR (nats/sample)")
    ax.set_title("A  Task-phase TP beats one state-blind GDAR")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    # B: by phase, rank 2
    ax = axs[0, 1]
    for j, v in enumerate(VAR):
        lab, c = VAR[v]
        g = R[v]["phase"]
        d = (g["store"]["TP r2"]["ll"] - g["store"]["GDAR"]["ll"]).sum(0) / g["n"].sum(0)
        ax.bar(np.arange(4) + (j - 0.5) * 0.4, d, 0.38, color=c, label=lab)
    _zero(ax)
    ax.set_xticks(range(4), PHASES)
    ax.set_ylabel("rank-2 gain over GDAR (nats/sample)")
    ax.set_title("B  Rank 2 gains least during planning")
    ax.legend(loc="upper right")
    # C: parameters, time grid
    ax = axs[0, 2]
    for v in VAR:
        lab, c = VAR[v]
        g = R[v]["time"]
        ns = [f"TP r{r}" for r in RANKS_TIME] + ["per-condition"]
        ax.plot([g["params"][n] for n in ns], [_gain(g, n)[0] for n in ns], "-o", ms=6, color=c, label=lab)
        if v == "felix":
            ax.annotate("12 separate\nfits", (g["params"]["per-condition"], _gain(g, "per-condition")[0]),
                        textcoords="offset points", xytext=(-4, -24), ha="right", fontsize=8, color=MUTED)
            ax.annotate("rank 3", (g["params"]["TP r3"], _gain(g, "TP r3")[0]),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel(f"free parameters (GDAR = {R['felix']['time']['params']['GDAR']})")
    ax.set_ylabel("held-out gain over GDAR (nats/sample)")
    ax.set_ylim(0, None)
    ax.set_title("C  12 time bins: low rank vs separate fits")
    ax.set_xticks([1000, 2000, 5000, 10000], ["1k", "2k", "5k", "10k"])
    ax.minorticks_off()
    ax.legend(loc="lower right")
    # D, E: leave-one-bin-out
    for ax, v, ttl in ((axs[1, 0], "felix", "D  Unseen task time, evoked left in"),
                       (axs[1, 1], "induced", "E  Unseen task time, evoked removed")):
        g = R[v]["time"]
        tb = g["p_grid"][1:-1]
        nb = g["n"][:, 1:-1].sum(0)
        for name, c, lab in (("TP r3", C_FELIX, "TP-GDAR rank 3, weights interpolated"),
                             ("oracle", C_3, "separate fit on that bin (seen)"),
                             ("nearest bin", C_IND, "separate fit on the neighbouring bin")):
            d = (g["lobo"][name] - g["lobo"]["GDAR"])[:, 1:-1].sum(0) / nb
            ax.plot(tb, d, "-o", ms=5, color=c, label=lab)
        _zero(ax)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("time of the held-out bin re go cue (ms)")
        ax.set_ylabel("gain over GDAR (nats/sample)")
        ax.set_ylim(-0.2, 0.3)
        ax.set_title(ttl)
    axs[1, 0].legend(loc="lower right", fontsize=7.5)
    # F: h-step
    ax = axs[1, 2]
    hms = np.array(HORIZONS) * 1000 / FS
    for v in VAR:
        lab, c = VAR[v]
        vf = R[v]["phase"]["vaf"]
        ax.plot(hms, 100 * (vf["TP r2"] - vf["GDAR"]), "-o", ms=6, color=c, label=lab)
    _zero(ax)
    ax.set_xlabel("prediction horizon (ms)")
    ax.set_ylabel("rank-2 VAF minus GDAR VAF (points)")
    ax.set_title("F  Forecasts gain most at about 20 ms ahead")
    ax.legend(loc="upper right")
    return fig


def ar_spectrum(Theta, Sigma, K, freqs):
    N = Theta.shape[0]
    out = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        z = np.exp(-2j * np.pi * f / FS)
        Af = np.eye(N, dtype=complex)
        for k in range(K):
            Af -= Theta[:, k * N:(k + 1) * N] * z ** (k + 1)
        H = np.linalg.inv(Af)
        out[i] = np.real(np.trace(H @ Sigma @ H.conj().T)) / N
    return out


def fig_interpret(R, Rb):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("bwr_", [C_FELIX, "#f0efec", "#e34948"])
    fig = plt.figure(figsize=(13.5, 7.8), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])
    axs_top = top.subplots(1, 3, width_ratios=[1, 1.1, 1])
    axs_map = bottom.subplots(1, 5)
    fig.suptitle("Is the gain real, and what changes? Robustness, and rank-2 operators with evoked left in (edges summed over lags), H 2021-11-12",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    # A: order
    ax = axs_top[0]
    for v in VAR:
        lab, c = VAR[v]
        rows = Rb[v, "order"]
        Ks = sorted(rows)
        g = [(rows[k]["tot"]["TP r2"]["shared"] - rows[k]["tot"]["GDAR"]["shared"]).sum() / rows[k]["n"].sum() for k in Ks]
        ax.plot(Ks, g, "-o", ms=6, color=c, label=lab)
    ax.set_ylim(0, None)
    ax.set_xlabel(f"GDAR order K (lags, at {1000/FS:.0f} ms each)")
    ax.set_ylabel("rank-2 gain over same-K GDAR (nats/sample)")
    ax.set_title("A  The gain does not shrink with more lags")
    ax.legend(loc="lower right")
    # B: checks
    ax = axs_top[1]

    def val(v, key, field):
        r = Rb[v, key][16]
        return (r["tot"]["TP r2"][field] - r["tot"]["GDAR"][field]).sum() / r["n"].sum()
    checks = [("K = 16, as in the main fit", "order", "shared"), ("one noise covariance per phase", "order", "per"),
              ("hold trimmed to -300..-20 ms", "hold2", "shared"), ("hold dropped", "nohold", "shared")]
    y = np.arange(len(checks))[::-1]
    for j, v in enumerate(VAR):
        lab, c = VAR[v]
        ax.plot([val(v, k, f) for _, k, f in checks], y + (0.5 - j) * 0.22, "o", ms=7, color=c, label=lab)
    ax.set_yticks(y, [c[0] for c in checks])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_xlim(0, None)
    ax.set_xlabel("rank-2 gain over GDAR (nats/sample)")
    ax.set_title("B  Survives per-phase noise; hold is a large share")
    ax.legend(loc="lower right")
    # C: spectra
    ax = axs_top[2]
    full = R["felix"]["full"]
    K = R["K"]
    freqs = np.arange(2.0, 101.0, 1.0)
    Sig = full["Sigma"]
    Th = full["models"]["TP r2"]
    base = ar_spectrum(full["models"]["GDAR"][0], Sig, K, freqs)
    for g, (nm, c) in enumerate(zip(PHASES, ("#6250d6", C_3, C_IND, "#e87ba4"))):
        ax.plot(freqs, 10 * np.log10(ar_spectrum(Th[g], Sig, K, freqs) / base), color=c, label=nm)
    _zero(ax)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("model spectrum re GDAR (dB)")
    ax.set_title("C  Near 15 Hz: up in hold, down in move, post")
    ax.legend(ncol=2, loc="upper right")
    # maps, evoked left in
    pos = R["pos"]
    A = R["A"]
    iu, ju = np.where(np.triu(A, 1))
    N = len(pos)

    def summed(th):
        return sum(th[:, k * N:(k + 1) * N] for k in range(K))
    G = summed(full["models"]["GDAR"][0])
    dmax = max(np.abs(summed(Th[g]) - G)[iu, ju].max() for g in range(4))
    gmax = np.abs(G[iu, ju]).max()
    for c0, (ttl, M, vm) in enumerate([("D  GDAR edge weights", G, gmax)] +
                                      [(f"{nm} minus GDAR", summed(Th[g]) - G, dmax) for g, nm in enumerate(PHASES)]):
        ax = axs_map[c0]
        segs = [(pos[i], pos[j]) for i, j in zip(iu, ju)]
        lc = LineCollection(segs, array=M[iu, ju], cmap=cmap, norm=TwoSlopeNorm(0, -vm, vm), linewidths=3.2)
        ax.add_collection(lc)
        ax.scatter(*pos.T, s=20, color="#fcfcfb", edgecolor=MUTED, lw=0.8, zorder=3)
        ax.set_aspect("equal")
        ax.margins(0.08)
        ax.axis("off")
        ax.set_title(ttl, fontsize=9)
        cb = bottom.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    return fig


def plot():
    with open(CACHE, "rb") as f:
        R = pickle.load(f)
    with open(CACHE.replace("12_results", "12b_robust"), "rb") as f:
        Rb = pickle.load(f)
    _style()
    os.makedirs(FIG, exist_ok=True)
    for name, fig in (("12_tp_vs_gdar_prediction", fig_prediction(R)), ("12_tp_vs_gdar_operators", fig_interpret(R, Rb))):
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=170)
        log("wrote", os.path.join(FIG, name + ".png"))


def main():
    t0 = time.time()
    D = load()
    n, T, N = D["X"].shape
    log(f"H 2021-11-12: {n} trials, {T} samples at {FS:.0f} Hz, {N} channels, {int(D['A'].sum()/2)} edges")
    labs = labels(D)
    log("samples per phase:", [(PHASES[g], int((labs['phase'] == g).sum())) for g in range(4)])
    log("samples per time bin:", [int((labs['time'] == g).sum()) for g in range(len(TBIN) - 1)])
    fo = folds(D["ang"])
    K, Kcv = select_K(D, labs, fo)
    log(f"-> K = {K}  ({1000*K/FS:.0f} ms)")
    Z = evoked_design(D)
    out = {"K": K, "Kcv": Kcv, "pos": D["pos"], "A": D["A"], "ch": D["ch"], "n_trials": n}
    for variant in ("felix", "induced"):
        log(f"\n=== variant {variant} ===")
        out[variant] = run_variant(D, labs, fo, K, variant, Z)
        summarize(out[variant])
        with open(CACHE, "wb") as f:
            pickle.dump(out, f)
    log(f"\ndone in {time.time()-t0:.0f}s, wrote {CACHE}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    plot() if "--plot" in sys.argv else main()
