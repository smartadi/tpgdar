"""Stacked TP-GDAR: two-stage (HOSVD) vs joint (G-weighted ALS) fits.  docs/MODEL.md.

Three tests, all scored by held-out one-step Gaussian log-likelihood per sample
(each model's Sigma from its own training residuals), reported as Delta-ell vs a
pooled (p-blind) GDAR:

  A. synthetic linear, truly rank-2 Theta(p) = [A(p) | B(p)], conditions with
     uneven data length / input power (the case two-stage HOSVD is not built for);
     also the error of the recovered operators against the truth;
  B. Wilson-Cowan (g0 = 3.0), the script-06 set-up, A only, fresh test segments at
     off-grid p;
  C. real OS-Pulse 091114 M1, stimulation window 1-2 s (script 10's cached per-trial
     Grams), stacked [A | B] with TP ranks (r_A, r_B), p = stimulation frequency:
     held-out trials (train 0-39, test 40-59) and leave-one-frequency-out.

Run:  .venv\\Scripts\\python scripts/11_tp_joint.py [A] [B] [C]
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gdar_tp.graph import grid_graph, support_mask  # noqa: E402
from gdar_tp.sim.linear import simulate_gdar  # noqa: E402
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams  # noqa: E402
from gdar_tp.tpgdar import (fit_joint, fit_per_condition, fit_pooled,  # noqa: E402
                            fit_two_stage, resid_cov_sum, row_columns, stack_stats,
                            trial_stats)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = os.path.join(ROOT, "cache")
SIG_SHRINK = 0.01
RIDGE = 1e-6


# ----------------------------------------------------------------------------- scoring
def sigma_of(Theta_list, stats_list):
    """Innovation covariance from training residuals summed over (Theta_g, stats_g)."""
    S = sum(resid_cov_sum(T, s["Gzz"], s["Gzy"], s["Syy"]) for T, s in zip(Theta_list, stats_list))
    n = sum(s["n"] for s in stats_list)
    S = S / n
    return (1 - SIG_SHRINK) * S + SIG_SHRINK * np.diag(np.diag(S))


def ll_per_sample(Theta, s, Sigma):
    S = resid_cov_sum(Theta, s["Gzz"], s["Gzy"], s["Syy"])
    _, ld = np.linalg.slogdet(Sigma)
    N = Sigma.shape[0]
    return float(-0.5 * (np.trace(np.linalg.solve(Sigma, S)) / s["n"] + N * np.log(2 * np.pi) + ld))


def cond(stats, g):
    return {k: stats[k][g] for k in stats}


def score(models, train, test, p_test_idx=None):
    """models: name -> (P_train, N, D) operators on the training grid, or a callable
    g_test -> Theta for the test conditions.  Returns name -> per-test-condition ll."""
    P = len(train["n"])
    out = {}
    for name, M in models.items():
        if callable(M):
            Th_tr, f = M("train"), M
        else:
            Th_tr = M
            f = lambda g, M=M: M[g]
        Sig = sigma_of([Th_tr[g] for g in range(P)], [cond(train, g) for g in range(P)])
        nt = len(test["n"])
        out[name] = np.array([ll_per_sample(f(g), cond(test, g), Sig) for g in range(nt)])
    return out


def report(title, res, base="pooled"):
    print(f"\n{title}   (Delta-ell per sample vs {base}, nats)")
    b = res[base]
    for k, v in res.items():
        d = v - b
        print(f"  {k:<22} mean {d.mean():+.4f}   per-cond " + " ".join(f"{x:+.3f}" for x in d))


# ----------------------------------------------------------------------------- A
def part_a(n_seeds=5):
    print("=" * 78 + "\nA. synthetic linear, rank-2 stacked [A|B](p), uneven conditions")
    g = grid_graph(4, 4); N = g.n; mask = support_mask(g.adjacency)
    K, Q, P = 2, 1, 7
    D = K * N + Q
    cols = row_columns(mask, K, n_inputs=1)
    blocks = [np.arange(K * N), np.arange(K * N, D)]
    p_grid = np.linspace(0, 1, P)
    errs = {"per-condition": [], "two-stage r2": [], "joint r2": []}
    lofo = {"pooled": [], "nearest": [], "two-stage r2": [], "joint r2": [], "oracle": []}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        def op():
            A = [np.where(mask, rng.standard_normal((N, N)) * 0.10, 0.0) for _ in range(K)]
            A[0] += 0.5 * np.eye(N)
            return A, rng.standard_normal((N, 1))
        (A0, B0), (A1, B1) = op(), op()

        def theta(p):
            return np.concatenate([(1 - p) * a + p * b for a, b in zip(A0, A1)]
                                  + [(1 - p) * B0 + p * B1], axis=1)

        def simulate(p, T, upow, n_tr):
            th = theta(p)
            A = [th[:, k * N:(k + 1) * N] for k in range(K)]; B = th[:, K * N:]
            out = []
            for _ in range(n_tr):
                U = upow * rng.standard_normal((T + 50, 1))
                X, U = simulate_gdar(A, T, B=B, U=U, noise_std=0.3, rng=rng, burn_in=50)
                out.append(trial_stats(X, U, K=K, Q=Q))
            return out
        lengths = [int(300 * (0.3 + 2.0 * (i % 2))) for i in range(P)]
        upows = [0.2 if i % 3 == 0 else 2.0 for i in range(P)]
        train = stack_stats([simulate(p, T, u, 4) for p, T, u in zip(p_grid, lengths, upows)])
        truth = np.stack([theta(p) for p in p_grid])
        per = fit_per_condition(train, cols, RIDGE)
        two = fit_two_stage(train, cols, blocks, [2, 2], p_grid, RIDGE)
        joint = fit_joint(train, cols, blocks, [2, 2], p_grid, RIDGE, n_iter=60)
        rel = lambda T: np.linalg.norm(T - truth) / np.linalg.norm(truth)
        errs["per-condition"].append(rel(per))
        errs["two-stage r2"].append(rel(two.theta_grid()))
        errs["joint r2"].append(rel(joint.theta_grid()))
        # leave-one-condition-out on the interior grid points, fresh test data
        for h in range(1, P - 1):
            tr = [j for j in range(P) if j != h]
            sub = {k: train[k][tr] for k in train}
            test = stack_stats([simulate(p_grid[h], 600, 1.0, 4)])
            pooled = fit_pooled(sub, cols, RIDGE)
            per_s = fit_per_condition(sub, cols, RIDGE)
            two_s = fit_two_stage(sub, cols, blocks, [2, 2], p_grid[tr], RIDGE)
            joi_s = fit_joint(sub, cols, blocks, [2, 2], p_grid[tr], RIDGE, n_iter=60)
            near = int(np.argmin(np.abs(p_grid[tr] - p_grid[h])))
            orc = fit_per_condition(test, cols, RIDGE)
            P8 = len(tr)
            models = {
                "pooled": lambda w, M=pooled: np.stack([M] * P8) if w == "train" else M,
                "nearest": lambda w, M=per_s, j=near: M if w == "train" else M[j],
                "two-stage r2": lambda w, m=two_s: m.theta_grid() if w == "train" else m.theta_at(p_grid[h]),
                "joint r2": lambda w, m=joi_s: m.theta_grid() if w == "train" else m.theta_at(p_grid[h]),
                "oracle": lambda w, M=per_s, O=orc: M if w == "train" else O[0],
            }
            r = score(models, sub, test)
            for k in lofo:
                lofo[k].append(r[k][0])
    print("  relative operator error vs truth (mean over seeds):")
    for k, v in errs.items():
        print(f"    {k:<16} {np.mean(v):.4f}  (+/- {np.std(v):.4f})")
    res = {k: np.array(v) for k, v in lofo.items()}
    report("  leave-one-condition-out, interior p, all seeds", res)
    return errs, res


# ----------------------------------------------------------------------------- B
def part_b():
    print("=" * 78 + "\nB. Wilson-Cowan 5x5, g0 = 3.0, A only (script-06 set-up)")
    g = grid_graph(5, 5); N = g.n; mask = support_mask(g.adjacency)
    net = WCNetwork(g.adjacency, WCParams(g0=3.0))
    K = 2
    cols = row_columns(mask, K)
    blocks = [np.arange(K * N)]
    p_train = np.linspace(0.0, 1.0, 7)
    p_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def seg_stats(p, seed, n=6000):
        E = net.simulate(n_steps=n, p=p, noise_std=0.03, rng=seed, burn_in=800)
        return trial_stats(E - E.mean(0), K=K)
    train = stack_stats([[seg_stats(p, 10 + i)] for i, p in enumerate(p_train)])
    test = stack_stats([[seg_stats(p, 500 + int(round(p * 100)))] for p in p_test])
    pooled = fit_pooled(train, cols, RIDGE)
    per = fit_per_condition(train, cols, RIDGE)
    oracle = fit_per_condition(test, cols, RIDGE)
    out = {"pooled": lambda w: np.stack([pooled] * 7) if w == "train" else pooled,
           "nearest": lambda w: per if w == "train" else per[int(np.argmin(np.abs(p_train - p_test[w])))]}
    for r in (1, 2, 3):
        two = fit_two_stage(train, cols, blocks, [r], p_train, RIDGE)
        joi = fit_joint(train, cols, blocks, [r], p_train, RIDGE, n_iter=60)
        out[f"two-stage r{r}"] = lambda w, m=two: m.theta_grid() if w == "train" else m.theta_at(p_test[w])
        out[f"joint r{r}"] = lambda w, m=joi: m.theta_grid() if w == "train" else m.theta_at(p_test[w])
    out["oracle"] = lambda w: per if w == "train" else oracle[w]
    res = score(out, train, test)
    report("  held-out p (fresh segments)", res)
    return res


# ----------------------------------------------------------------------------- C
def part_c():
    print("=" * 78 + "\nC. real OS-Pulse 091114 M1, stim window 1-2 s, stacked [A | B](f)")
    path = os.path.join(CACHE, "10_grams_091114_M1_K5_Ku5.npz")
    if not os.path.exists(path):
        print("  no cached Grams; run scripts/10_ospulse_tp_input.py first")
        return None
    t0 = time.time()
    z = np.load(path)
    G, cnd, trl, freqs = z["G"], z["cond"], z["trial"], z["freqs"]
    K, KU = 5, 5
    N = (G.shape[1] - KU - 1) // (K + 1)
    D = N * K + KU
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    s10 = importlib.import_module("10_ospulse_tp_input")
    d10 = s10.build_grams()
    _, mask = s10.knn_support(d10["pos"])
    cols = row_columns(mask, K, n_inputs=KU)
    blocks = [np.arange(N * K), np.arange(N * K, D)]
    conds = sorted(set(cnd.tolist()))
    f = freqs[conds]
    print(f"  N={N}, D={D}, {len(conds)} conditions, {G.shape[0]} trials, loaded {time.time()-t0:.0f}s")

    def stats_of(sel_fn):
        out = {k: [] for k in ("Gzz", "Gzy", "Syy", "n")}
        for c in conds:
            Gc = G[(cnd == c) & sel_fn(trl)].sum(0)
            out["Gzz"].append(Gc[:D, :D]); out["Gzy"].append(Gc[:D, D:D + N])
            out["Syy"].append(Gc[D:D + N, D:D + N]); out["n"].append(Gc[-1, -1])
        return {k: np.asarray(v) for k, v in out.items()}
    train = stats_of(lambda t: t < 40)
    test = stats_of(lambda t: t >= 40)

    # ridge by validation INSIDE the training trials (fit 0-29, score 30-39); the
    # test trials 40-59 are never looked at.  Script 10's 1e-6 is ~4 decades too
    # strong on these lag Grams (cond ~5e10).
    fit_tr, val = stats_of(lambda t: t < 30), stats_of(lambda t: (t >= 30) & (t < 40))
    best = None
    for rho in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12):
        pm = fit_pooled(fit_tr, cols, rho)
        v = score({"m": np.stack([pm] * len(conds))}, fit_tr, val)["m"].mean()
        print(f"  ridge {rho:.0e}: validation ell {v:.3f}", flush=True)
        if best is None or v > best[1]:
            best = (rho, v)
    RIDGE_C = best[0]
    print(f"  -> ridge {RIDGE_C:.0e} (pooled model, inner validation)")

    # B = 0 baseline: zero the input columns
    cols_noB = row_columns(mask, K, n_inputs=0)
    Pn = len(conds)
    a_only = fit_pooled(train, cols_noB, RIDGE_C)
    configs = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 3)]

    # (i) held-out trials, every condition seen in training
    pooled = fit_pooled(train, cols, RIDGE_C)
    per = fit_per_condition(train, cols, RIDGE_C)
    models = {"B=0 (pure GDAR)": np.stack([a_only] * Pn), "pooled": np.stack([pooled] * Pn),
              "per-condition": per}
    fitted = {}
    for ra, rb in configs:
        t1 = time.time()
        two = fit_two_stage(train, cols, blocks, [ra, rb], f, RIDGE_C)
        joi = fit_joint(train, cols, blocks, [ra, rb], f, RIDGE_C, n_iter=40)
        fitted[(ra, rb)] = (two, joi)
        models[f"two-stage A{ra}B{rb}"] = two.theta_grid()
        models[f"joint A{ra}B{rb}"] = joi.theta_grid()
        print(f"  fit A{ra}B{rb}: {time.time()-t1:.0f}s, joint J {joi.objective[0]:.6g} -> "
              f"{joi.objective[-1]:.6g} ({len(joi.objective)-1} sweeps)", flush=True)
    res_i = score(models, train, test)
    report("  (i) held-out trials (0-39 train, 40-59 test)", res_i)

    # (ii) leave-one-frequency-out
    lofo = {k: [] for k in ["pooled", "nearest"] + [f"{m} A{a}B{b}" for a, b in configs
                                                   for m in ("two-stage", "joint")] + ["oracle"]}
    for h in range(Pn):
        tr = [j for j in range(Pn) if j != h]
        sub = {k: train[k][tr] for k in train}
        tst = {k: test[k][[h]] for k in test}
        pooled = fit_pooled(sub, cols, RIDGE_C)
        per_s = fit_per_condition(sub, cols, RIDGE_C)
        near = int(np.argmin(np.abs(np.log(f[tr]) - np.log(f[h]))))
        m = {"pooled": lambda w, M=pooled: np.stack([M] * len(tr)) if w == "train" else M,
             "nearest": lambda w, M=per_s, j=near: M if w == "train" else M[j],
             "oracle": lambda w, M=per_s, O=per[h]: M if w == "train" else O}
        for ra, rb in configs:
            two = fit_two_stage(sub, cols, blocks, [ra, rb], f[tr], RIDGE_C)
            joi = fit_joint(sub, cols, blocks, [ra, rb], f[tr], RIDGE_C, n_iter=40)
            m[f"two-stage A{ra}B{rb}"] = lambda w, mm=two: mm.theta_grid() if w == "train" else mm.theta_at(f[h], log=True)
            m[f"joint A{ra}B{rb}"] = lambda w, mm=joi: mm.theta_grid() if w == "train" else mm.theta_at(f[h], log=True)
        r = score(m, sub, tst)
        for k in lofo:
            lofo[k].append(r[k][0])
        print(f"  LOFO held {int(f[h])} Hz done", flush=True)
    res_ii = {k: np.array(v) for k, v in lofo.items()}
    report("  (ii) leave-one-frequency-out (all 9; 10 and 150 Hz are extrapolation)", res_ii)
    inner = {k: v[1:-1] for k, v in res_ii.items()}
    report("  (ii) LOFO interior 20-100 Hz", inner)
    np.savez(os.path.join(CACHE, "11_results_091114_M1.npz"),
             **{f"i::{k}": v for k, v in res_i.items()},
             **{f"ii::{k}": v for k, v in res_ii.items()}, freqs=f, ridge=RIDGE_C)
    return res_i, res_ii


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"A", "B", "C"}
    t00 = time.time()
    if "A" in which:
        part_a()
    if "B" in which:
        part_b()
    if "C" in which:
        part_c()
    print(f"\ndone in {time.time()-t00:.0f}s")
