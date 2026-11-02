r"""Robustness of script 12's result (H 2021-11-12, phase grid, 5-fold CV over trials).

  1. Order.  K = 16 is the edge of script 12's grid: does the TP gain survive larger K?
  2. Noise.  One Sigma per model across phases, or one per phase?  If phases differ only
     in power, a per-phase Sigma should remove the TP gain.
  3. Hold.   Hold carries the largest gain: shrink it away from the segment edge and the
     target, or drop it.
  4. Metric. The edge-wise fit is row-wise OLS; refit it by nothing else, score it on a
     per-channel (diagonal Sigma) likelihood too, to see why SSE and full-Sigma ell disagree.

Run:  .venvScriptspython scripts/12b_tp_vs_gdar_robust.py
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
from gdar_tp.sym_tp import fit_gls_pooled, fit_sym_tp, symmetric_basis, theta_of  # noqa: E402
from gdar_tp.tpgdar import fit_pooled, resid_cov_sum, row_columns  # noqa: E402

K0 = 33     # 33 x 3 ms lags: hold (-400 ms on) still has them all inside the -500 ms segment
OUT = os.path.join(ROOT, "cache", "12b_robust_H_2021-11-12.pkl")
log = s12.log


def fold_stats(X, lab, K, fo, P):
    """Per (fold, condition) Grams; per-trial would not fit in memory at K = 32."""
    n, T, N = X.shape
    D_ = K * N
    s = dict(Gzz=np.zeros((s12.NFOLD, P, D_, D_)), Gzy=np.zeros((s12.NFOLD, P, D_, N)),
             Syy=np.zeros((s12.NFOLD, P, N, N)), n=np.zeros((s12.NFOLD, P)))
    ts = np.arange(K0 - 1, T - 1)
    for i in range(n):
        Z = np.concatenate([X[i, ts - k] for k in range(K)], 1)
        Y = X[i, ts + 1]
        l = lab[i, ts + 1]
        f = fo[i]
        for g in range(P):
            m = l == g
            if m.any():
                s["Gzz"][f, g] += Z[m].T @ Z[m]
                s["Gzy"][f, g] += Z[m].T @ Y[m]
                s["Syy"][f, g] += Y[m].T @ Y[m]
                s["n"][f, g] += m.sum()
    return s


def sel(s, f, train=True):
    keep = np.arange(s12.NFOLD) != f if train else np.arange(s12.NFOLD) == f
    return {k: s[k][keep].sum(0) for k in s}


def sigma_per(Th, st):
    out = []
    for g in range(len(Th)):
        S = resid_cov_sum(Th[g], st["Gzz"][g], st["Gzy"][g], st["Syy"][g]) / st["n"][g]
        out.append((1 - s12.SIG_SHRINK) * S + s12.SIG_SHRINK * np.diag(np.diag(S)))
    return out


def ll(Th, st, Sigs):
    """Total held-out ell per condition, one Sigma per condition."""
    N = Th.shape[1]
    out = np.zeros(len(Th))
    for g in range(len(Th)):
        S = resid_cov_sum(Th[g], st["Gzz"][g], st["Gzy"][g], st["Syy"][g])
        _, ld = np.linalg.slogdet(Sigs[g])
        out[g] = -0.5 * (np.sum(np.linalg.inv(Sigs[g]) * S) + st["n"][g] * (N * np.log(2 * np.pi) + ld))
    return out


def run(X_of, lab, P, fo, A, Ks, ranks=(2,), edge=False, label=""):
    N = A.shape[0]
    rows = {}
    for K in Ks:
        t0 = time.time()
        R = symmetric_basis(A, K)
        tot = {}
        ncond = None
        for f in range(s12.NFOLD):
            s = fold_stats(X_of(f), lab, K, fo, P)
            tr, te = sel(s, f), sel(s, f, False)
            ncond = te["n"] if ncond is None else ncond + te["n"]
            all_tr = {k: tr[k].sum(0) for k in tr}
            phi, Sig0 = fit_gls_pooled(all_tr["Gzz"], all_tr["Gzy"], all_tr["Syy"], all_tr["n"], R)
            models = {"GDAR": np.stack([theta_of(phi, R, N)] * P)}
            for r in ranks:
                models[f"TP r{r}"] = fit_sym_tp(tr, R, Sig0, r, np.arange(float(P)), n_iter=25).theta_grid()
            models["per-phase"] = np.stack([
                fit_sym_tp({k: tr[k][[g]] for k in tr}, R, Sig0, 1, np.arange(1.0), n_iter=1).theta_grid()[0]
                for g in range(P)])
            if edge:
                mask = A | np.eye(N, dtype=bool)
                models["edge-wise OLS"] = np.stack([fit_pooled(tr, row_columns(mask, K), 1e-8)] * P)
            for name, Th in models.items():
                shared = s12.sigma_of(Th, tr)
                d = tot.setdefault(name, dict(shared=np.zeros(P), per=np.zeros(P), diag=np.zeros(P), sse=np.zeros(P)))
                d["shared"] += ll(Th, te, [shared] * P)
                d["per"] += ll(Th, te, sigma_per(Th, tr))
                d["diag"] += ll(Th, te, [np.diag(np.diag(shared))] * P)
                d["sse"] += np.array([np.trace(resid_cov_sum(Th[g], te["Gzz"][g], te["Gzy"][g], te["Syy"][g])) for g in range(P)])
        nc = ncond.sum()
        rows[K] = dict(tot=tot, n=ncond)
        log(f"{label} K={K}  GDAR ell {tot['GDAR']['shared'].sum()/nc:.3f}  ({time.time()-t0:.0f}s)")
        for name, d in tot.items():
            b = tot["GDAR"]
            log(f"   {name:<14} shared-Sigma {(d['shared']-b['shared']).sum()/nc:+.4f} "
                f"[{' '.join(f'{x:+.3f}' for x in (d['shared']-b['shared'])/ncond)}]  "
                f"per-phase-Sigma {(d['per']-b['per']).sum()/nc:+.4f} [{' '.join(f'{x:+.3f}' for x in (d['per']-b['per'])/ncond)}]  "
                f"diag {(d['diag']-b['diag']).sum()/nc:+.4f}  SSE {100*(1-d['sse'].sum()/b['sse'].sum()):+.2f}%")
    return rows


def main():
    D = s12.load()
    labs = s12.labels(D)
    fo = s12.folds(D["ang"])
    A = D["A"]
    Z = s12.evoked_design(D)
    t = D["t"]
    ph = labs["phase"]
    # power per phase, and whether "noart" differs from raw on this no-stim day
    z = np.load(s12.SEGS, allow_pickle=True)
    log("lfp vs noart identical on this day:", np.allclose(z["lfp"], z["noart"]))
    X0 = D["X"]
    log("rms per phase (uV):", [round(float(np.sqrt((X0[ph == g] ** 2).mean())), 1) for g in range(4)])
    ev = {}

    def x_of(variant):
        def f(fold):
            if variant == "felix":
                return X0
            if fold not in ev:
                ev[fold] = s12.remove_evoked(X0, Z, fo != fold)
            return ev[fold]
        return f

    out = {}
    res = x_of("induced")(0)
    log("induced rms per phase (uV):", [round(float(np.sqrt((res[ph == g] ** 2).mean())), 1) for g in range(4)])
    for variant in ("felix", "induced"):
        log(f"\n=== {variant}: order ===")
        out[variant, "order"] = run(x_of(variant), ph, 4, fo, A, (5, 16, 24, 32), edge=False, label=variant)
    hold2 = ph.copy()
    hold2[(ph == 0) & ((t[None] < -300) | (t[None] >= -20))] = -1
    nohold = ph - 1
    nohold[ph == 0] = -1
    for variant in ("felix", "induced"):
        log(f"\n=== {variant}: hold -300..-20 ms, K=16 ===")
        out[variant, "hold2"] = run(x_of(variant), hold2, 4, fo, A, (16,), label=variant)
        log(f"\n=== {variant}: no hold, K=16 ===")
        out[variant, "nohold"] = run(x_of(variant), nohold, 3, fo, A, (16,), label=variant)
        log(f"\n=== {variant}: edge-wise OLS vs GLS by metric, K=16 ===")
        out[variant, "metric"] = run(x_of(variant), ph, 4, fo, A, (16,), ranks=(), edge=True, label=variant)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    log("done")


if __name__ == "__main__":
    main()
