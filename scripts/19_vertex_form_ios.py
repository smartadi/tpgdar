r"""TP-GDAR fitted in vertex form (MATH §9.5, approved by Aditya 2026-11-21), against the free fit and its convex form.

The free fit (`fit_sym_tp`) leaves the weights free and is put in convex form afterwards (`polytope_vertices`), which
changes the model.  The vertex form (`fit_sym_tp_vertex`) fits the convex weights directly, with a penalty
mu n sum_v ||phi_v - phibar||^2_Hbar that pulls the vertices together as far as the data allow.

  --synthetic   Script 15's plants, unchanged: 9 nodes, K = 2, Theta(p) = (1 - p) Theta_0 + p Theta_1 with p on 7 points
                including 0 and 1, an input at each node, 20 plants per case.  Vertex error and weight error against the
                truth, for the convex form of the free rank 2 fit and for the vertex form at each mu in SYN_MUS.
  (default)     H 2021-11-12 in script 13's setting: 16 warped bins, K = 16, 5 folds over trials, evoked left in and
                removed, V = 2 and 3.  Per outer fold: GDAR; the free fit (25 sweeps, as script 13); its convex form; the
                vertex form at each mu in MUS, each started from that convex form.  mu is chosen inside the training
                folds: fit on 3 of the 4, validate on the 4th (fold f + 1 mod 5) by held-out likelihood; for evoked removed
                the kernels are refitted on those 3 folds.  Every model is scored as in script 13: held-out one-step
                likelihood over GDAR, nats/sample, each model with its own Sigma.  Then a full-data fit at the median
                chosen mu, for the weights across the trial.

Run:  .venv\Scripts\python scripts/19_vertex_form_ios.py [--synthetic]      Log: cache/19_run.log, cache/19_synthetic.log
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
s13 = importlib.import_module("13_tp_vertices_ios")
from gdar_tp.sym_tp import (fit_gls_pooled, fit_sym_tp, fit_sym_tp_vertex, gls_quads, polytope_vertices,  # noqa: E402
                            symmetric_basis, theta_of)

K = s13.K
P = 4 * s13.NSUB
VS = (2, 3)
MUS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
SYN_MUS = (0.0, 1e-3, 1e-2)
N_ITER = 100
CACHE = os.path.join(ROOT, "cache", "19_vertex_form_H_2021-11-12.pkl")
LOG = os.path.join(ROOT, "cache", "19_run.log")
SYN_LOG = os.path.join(ROOT, "cache", "19_synthetic.log")
_lines = []


def log(*a):
    s12.log(*a)
    _lines.append(" ".join(str(x) for x in a))


def write_log(path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")


def sel_mask(s, keep):
    return {k: s[k][keep].sum(0) for k in s}


def geometry(lam, phi, Hbar, R, N):
    """Each vertex's distance from the nearest bin (fraction of the bins' spread, data metric) and spectral radius."""
    L = np.linalg.cholesky(Hbar + 1e-10 * np.trace(Hbar) / len(Hbar) * np.eye(len(Hbar)))
    bins = (lam @ phi) @ L
    verts = phi @ L
    spread = np.max(np.linalg.norm(bins - bins.mean(0), axis=1))
    dist = np.linalg.norm(verts[:, None] - bins[None], axis=2).min(1) / spread
    rho = np.array([np.abs(s12.rho(theta_of(p, R, N), K)).max() for p in phi])
    return dist, rho


def fits(tr, R, N, grid):
    """GDAR, then per V the free fit, its convex form and the vertex form at each mu.  name -> (P, N, KN) operators."""
    a = {k: tr[k].sum(0) for k in tr}
    phi0, Sf = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
    Hs, hs = gls_quads(tr, R, Sf)
    Hbar = Hs.sum(0) / a["n"]
    models = {"GDAR": np.stack([theta_of(phi0, R, N)] * P)}
    extra = {}
    for V in VS:
        free = fit_sym_tp(tr, R, Sf, V, grid, n_iter=25, quads=(Hs, hs))
        models[f"free r{V}"] = free.theta_grid()
        poly = polytope_vertices(free, Hbar)
        models[f"convex r{V}"] = np.stack([theta_of(l @ poly["phi"], R, N) for l in poly["lam"]])
        extra[f"convex r{V}"] = geometry(poly["lam"], poly["phi"], Hbar, R, N) + (0,)
        for mu in MUS:
            m = fit_sym_tp_vertex(tr, R, Sf, V, grid, mu=mu, start=(poly["lam"], poly["phi"]), n_iter=N_ITER,
                                  quads=(Hs, hs))
            name = f"vertex r{V} mu={mu:g}"
            models[name] = m.theta_grid()
            extra[name] = geometry(m.W, m.phi, Hbar, R, N) + (len(m.objective) - 1,)
    return models, extra


def score(models, tr, te):
    base = s12b.ll(models["GDAR"], te, [s12.sigma_of(models["GDAR"], tr)] * P)
    return {k: s12b.ll(Th, te, [s12.sigma_of(Th, tr)] * P) - base for k, Th in models.items()}


def fold_job(job):
    variant, f = job
    t0 = time.time()
    D = s12.load()
    lab = s13.warped_labels(D)
    fo = s12.folds(D["ang"])
    N = D["X"].shape[2]
    R = symmetric_basis(D["A"], K)
    grid = np.arange(float(P))
    Z = s12.evoked_design(D) if variant == "induced" else None
    # inner split: train on the 3 folds other than f and f_val, validate on f_val
    f_val = (f + 1) % s12.NFOLD
    inner = (fo != f) & (fo != f_val)
    Xi = D["X"] if Z is None else s12.remove_evoked(D["X"], Z, inner)
    si = s12b.fold_stats(Xi, lab, K, fo, P)
    folds_in = np.array([g not in (f, f_val) for g in range(s12.NFOLD)])
    mi, _ = fits(sel_mask(si, folds_in), R, N, grid)
    val = score(mi, sel_mask(si, folds_in), sel_mask(si, np.arange(s12.NFOLD) == f_val))
    # outer: train on the 4 folds other than f, test on f
    Xo = D["X"] if Z is None else s12.remove_evoked(D["X"], Z, fo != f)
    so = s12b.fold_stats(Xo, lab, K, fo, P)
    tr, te = s12b.sel(so, f), s12b.sel(so, f, False)
    mo, extra = fits(tr, R, N, grid)
    gain = score(mo, tr, te)
    return dict(variant=variant, f=f, cnt=te["n"], val={k: v.sum() for k, v in val.items()},
                val_cnt=si["n"][f_val].sum(), gain=gain, extra=extra, secs=time.time() - t0)


def full_fit(variant, mu):
    """All trials, one mu per V: the weights across the trial."""
    D = s12.load()
    lab = s13.warped_labels(D)
    N = D["X"].shape[2]
    R = symmetric_basis(D["A"], K)
    X = D["X"] if variant == "felix" else s12.remove_evoked(D["X"], s12.evoked_design(D), np.ones(len(D["X"]), bool))
    fo = s12.folds(D["ang"])
    s = s12b.fold_stats(X, lab, K, fo, P)
    st = {k: s[k].sum(0) for k in s}
    a = {k: st[k].sum(0) for k in st}
    _, Sf = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
    Hs, hs = gls_quads(st, R, Sf)
    Hbar = Hs.sum(0) / a["n"]
    out = {}
    for V in VS:
        free = fit_sym_tp(st, R, Sf, V, np.arange(float(P)), n_iter=40, quads=(Hs, hs))
        poly = polytope_vertices(free, Hbar)
        m = fit_sym_tp_vertex(st, R, Sf, V, np.arange(float(P)), mu=mu[V], start=(poly["lam"], poly["phi"]),
                              n_iter=N_ITER, quads=(Hs, hs))
        out[V] = dict(convex=poly, vertex=m, geo_convex=geometry(poly["lam"], poly["phi"], Hbar, R, N),
                      geo_vertex=geometry(m.W, m.phi, Hbar, R, N))
    return out


def summarize(res):
    for variant in ("felix", "induced"):
        rows = [r for r in res if r["variant"] == variant]
        cnt = sum(r["cnt"].sum() for r in rows)
        label = "evoked left in" if variant == "felix" else "evoked removed"
        log(f"\n=== {label}: held-out gain over GDAR, nats/sample (fold SE); vertex distance from the nearest bin "
            f"(fraction of spread, median and max over folds and vertices); vertex spectral radius (max); sweeps (median)")
        names = [k for k in rows[0]["gain"] if k != "GDAR"]
        for name in names:
            per = np.array([r["gain"][name].sum() / r["cnt"].sum() for r in rows])
            g = sum(r["gain"][name].sum() for r in rows) / cnt
            line = f"  {name:22s} {g:+.4f} ({per.std(ddof=1) / np.sqrt(len(per)):.4f})"
            if name in rows[0]["extra"]:
                dist = np.concatenate([r["extra"][name][0] for r in rows])
                rho = np.concatenate([r["extra"][name][1] for r in rows])
                sweeps = [r["extra"][name][2] for r in rows]
                line += f"   dist {np.median(dist):.2f} / {dist.max():.2f}   rho {rho.max():.3f}   sweeps {np.median(sweeps):.0f}"
            log(line)
        for V in VS:
            chosen = []
            for r in rows:
                vals = {mu: r["val"][f"vertex r{V} mu={mu:g}"] for mu in MUS}
                chosen.append(max(vals, key=vals.get))
            g = sum(r["gain"][f"vertex r{V} mu={mu:g}"].sum() for r, mu in zip(rows, chosen)) / cnt
            per = np.array([r["gain"][f"vertex r{V} mu={mu:g}"].sum() / r["cnt"].sum() for r, mu in zip(rows, chosen)])
            log(f"  vertex r{V}, mu chosen inside the training folds: {g:+.4f} ({per.std(ddof=1) / np.sqrt(len(per)):.4f});"
                f" mu by fold {[f'{m:g}' for m in chosen]}")
            res_mu = float(np.median(chosen))
            rows[0].setdefault("median_mu", {})[V] = res_mu
    return {v: [r for r in res if r["variant"] == v][0].get("median_mu", {}) for v in ("felix", "induced")}


def log_full(variant, full):
    label = "evoked left in" if variant == "felix" else "evoked removed"
    for V, d in full.items():
        m = d["vertex"]
        log(f"\n  [{label}] full-data fit, V = {V}, mu = {m.mu:g}: {len(m.objective) - 1} sweeps")
        for tag, lam, geo in (("convex form of the free fit", d["convex"]["lam"], d["geo_convex"]),
                              ("vertex form", m.W, d["geo_vertex"])):
            log(f"     {tag}: vertex distance from nearest bin {np.round(geo[0], 2).tolist()}, "
                f"spectral radius {np.round(geo[1], 3).tolist()}")
            log("       lam: " + " | ".join(s13.bin_name(g) + " " + " ".join(f"{x:.2f}" for x in lam[g]) for g in range(P)))


def synthetic():
    s15 = importlib.import_module("15_nine_node_control")
    from gdar_tp.graph import grid_graph
    g = grid_graph(3, 3)
    A = np.asarray(g.adjacency, bool)
    R = symmetric_basis(A, s15.K, node_inputs=True)
    log("synthetic: script 15's plants, V = 2; errors relative to the truth, medians over 20 plants")
    for case, targets in s15.SCENARIOS.items():
        errs = {k: [] for k in ["convex of free r2"] + [f"vertex mu={m:g}" for m in SYN_MUS]}
        lam_err = {k: [] for k in errs}
        for seed in range(s15.N_PLANTS):
            rng = np.random.default_rng(1000 + seed)            # as script 15's one_plant
            ph0, ph1 = s15.make_plant(seed, R, A, targets)
            st, _, Sig, models, poly = s15.learn(ph0, ph1, R, rng)
            truth = np.stack([ph0, ph1])
            cands = {"convex of free r2": (poly["lam"], poly["phi"])}
            for mu in SYN_MUS:
                m = fit_sym_tp_vertex(st, R, Sig, 2, s15.P_GRID, mu=mu, start=(poly["lam"], poly["phi"]), n_iter=N_ITER)
                cands[f"vertex mu={mu:g}"] = (m.W, m.phi)
            for name, (lam, phi) in cands.items():
                order = min(((0, 1), (1, 0)), key=lambda o: np.linalg.norm(phi[list(o)] - truth))
                errs[name].append(np.linalg.norm(phi[list(order)] - truth) / np.linalg.norm(truth))
                lam_err[name].append(np.abs(lam[:, order[1]] - s15.P_GRID).max())
        log(f"  {case}:")
        for name in errs:
            log(f"    {name:20s} vertex error {np.median(errs[name]):.4f} (max {np.max(errs[name]):.4f})   "
                f"weight error, max over bins {np.median(lam_err[name]):.4f} (max {np.max(lam_err[name]):.4f})")
    write_log(SYN_LOG)


def main():
    t0 = time.time()
    log(f"H 2021-11-12, 16 warped bins, K = {K}, V = {VS}, mu grid {MUS}")
    jobs = [(v, f) for v in ("felix", "induced") for f in range(s12.NFOLD)]
    res = s12.fold_map(fold_job, jobs)
    med = summarize(res)
    full = {}
    for variant in ("felix", "induced"):
        full[variant] = full_fit(variant, med[variant])
        log_full(variant, full[variant])
    with open(CACHE, "wb") as fh:
        pickle.dump(dict(res=res, full=full, mus=MUS), fh)
    log(f"done in {time.time() - t0:.0f}s")
    write_log(LOG)


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        synthetic()
    else:
        main()
