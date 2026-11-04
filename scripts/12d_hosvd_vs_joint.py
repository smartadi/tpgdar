r"""HOSVD against the joint fit on IOS Reach, with the same data, bins, folds and score.

Every low-rank step starts from the same per-bin GLS fits and differs only in how it reduces them:

  HOSVD        Baranyi's Method 4, step 2: SVD of [vec Theta_1; ...; vec Theta_P] over the bins, Frobenius metric.
               With one parameter mode HOSVD is this one SVD, so it is the best rank-r fit in that metric.
  metric SVD   the same SVD after changing coordinates to the pooled data metric, ||phi||^2_Hbar.  Closed form.
  joint        fit_sym_tp: alternating least squares on the data loss, each bin in its own metric H_g.
  separate     the per-bin fits themselves, 16 operators.

Hosvd is scored before any sum-to-one or nonnegative step, which only re-express the same family.
Scored as in scripts 12 and 13: held-out one-step likelihood over GDAR, nats/sample, each model with its own Sigma.

Run:  .venv\Scripts\python scripts/12d_hosvd_vs_joint.py     Log: cache/12d_run.log
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s12b = importlib.import_module("12b_tp_vs_gdar_robust")
s13 = importlib.import_module("13_tp_vertices_ios")
from gdar_tp.sym_tp import _quad, _solve, fit_gls_pooled, fit_sym_tp, symmetric_basis, theta_of  # noqa: E402
from gdar_tp.tpgdar import resid_cov_sum  # noqa: E402

RANKS = (1, 2, 3)
JITTER = 1e-10
LOG = os.path.join(ROOT, "cache", "12d_run.log")


def train_loss(Th, tr, Sinv):
    """The data loss the joint fit minimises, without its constant."""
    return sum(np.sum(Sinv * resid_cov_sum(Th[g], tr["Gzz"][g], tr["Gzy"][g], tr["Syy"][g])) for g in range(len(Th)))


def one(job):
    variant, f = job
    D = s12.load()
    lab = s13.warped_labels(D)
    K = s13.K
    fo = s12.folds(D["ang"])
    N = D["X"].shape[2]
    P = lab.max() + 1
    R = symmetric_basis(D["A"], K)
    grid = np.arange(float(P))
    X = D["X"] if variant == "left in" else s12.remove_evoked(D["X"], s12.evoked_design(D), fo != f)
    sf = s12b.fold_stats(X, lab, K, fo, P)
    tr, te = s12b.sel(sf, f), s12b.sel(sf, f, False)
    a = {k: tr[k].sum(0) for k in tr}
    phi, S0 = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
    Sinv = np.linalg.pinv(S0)
    Base = np.stack([theta_of(phi, R, N)] * P)
    base = s12b.ll(Base, te, [s12.sigma_of(Base, tr)] * P)

    quads = [_quad(tr["Gzz"][g], tr["Gzy"][g], R, Sinv) for g in range(P)]
    fits = np.stack([_solve(H, h, JITTER) for H, h in quads])

    vec = np.asarray(R @ fits.T).T
    U, S, Vt = np.linalg.svd(vec, full_matrices=False)
    Hbar = sum(H for H, _ in quads)
    L = np.linalg.cholesky(Hbar + JITTER * np.trace(Hbar) / len(Hbar) * np.eye(len(Hbar)))
    U2, S2, Vt2 = np.linalg.svd(fits @ L, full_matrices=False)

    out = dict(variant=variant, fold=f, count=float(te["n"].sum()), n_train=float(tr["n"].sum()), gain={}, loss={}, angle={},
               sv=S / S[0], sv_metric=S2 / S2[0])

    def score(name, Th):
        out["gain"][name] = float((s12b.ll(Th, te, [s12.sigma_of(Th, tr)] * P) - base).sum())
        out["loss"][name] = float(train_loss(Th, tr, Sinv))

    score("GDAR", Base)
    score("separate", np.stack([theta_of(fits[g], R, N) for g in range(P)]))
    for r in RANKS:
        low = (U[:, :r] * S[:r]) @ Vt[:r]
        score(f"HOSVD {r}", np.stack([row.reshape(N, -1, order="F") for row in low]))
        low2 = (U2[:, :r] * S2[:r]) @ Vt2[:r]
        phi2 = np.linalg.solve(L.T, low2.T).T
        score(f"metric SVD {r}", np.stack([theta_of(phi2[g], R, N) for g in range(P)]))
        m = fit_sym_tp(tr, R, S0, r, grid, n_iter=25)
        score(f"joint {r}", m.theta_grid())
        # do the two fits pick the same weights over the trial?  Cosines of the principal angles between spans.
        Qj = np.linalg.qr(m.W)[0]
        out["angle"][r] = np.linalg.svd(U[:, :r].T @ Qj, compute_uv=False)
    return out


def main():
    jobs = [(v, f) for v in ("left in", "removed") for f in range(s12.NFOLD)]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(one, jobs))
    lines = []
    for v in ("left in", "removed"):
        rows = [q for q in res if q["variant"] == v]
        count, n_train = sum(q["count"] for q in rows), sum(q["n_train"] for q in rows)
        lines.append(f"===== evoked {v}: held-out gain over GDAR, nats/sample; training loss above the joint fit of that rank, nats/sample")
        for name in ["separate"] + [f"{k} {r}" for r in RANKS for k in ("HOSVD", "metric SVD", "joint")]:
            gain = sum(q["gain"][name] for q in rows) / count
            r = name.split()[-1]
            ref = f"joint {r}" if r.isdigit() else None
            extra = f"   train loss above joint {sum(q['loss'][name] - q['loss'][ref] for q in rows) / 2 / n_train:+.4f}" if ref else ""
            lines.append(f"  {name:14s} {gain:+.4f}{extra}")
        lines.append("  singular values, normalised, Frobenius: " + np.array2string(np.mean([q["sv"] for q in rows], 0)[:8], precision=3, suppress_small=True))
        lines.append("  singular values, normalised, data metric: " + np.array2string(np.mean([q["sv_metric"] for q in rows], 0)[:8], precision=3, suppress_small=True))
        for r in RANKS:
            lines.append(f"  rank {r} weights, HOSVD vs joint, cosines of principal angles: " + np.array2string(np.mean([q["angle"][r] for q in rows], 0), precision=3))
    text = "\n".join(lines)
    print(text, flush=True)
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
