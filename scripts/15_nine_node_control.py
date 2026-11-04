r"""Worked example: learn a 9-node TP-GDAR with an input at each node, then control it.

Synthetic, known truth.  Sparse 3 x 3 graph (12 edges), K = 2, one input per node:

    x(t+1) = sum_k A_k(p) x(t-k) + B u(t) + e(t),   Theta(p) = (1-p) Theta_0 + p Theta_1

Two designed cases, not findings about cortex:
  one unstable   Theta_0 stable (rho 0.90), Theta_1 mildly unstable (rho 1.03)
  two unstable   both rho 1.03, added as a stress test because in the first case a pooled
                 model already sees the instability and pooled LQR does fine

Learn: random-input trials at 7 values of p, symmetric TP-GDAR by the joint fit, rank 2,
vertices from `polytope_vertices`.  Both starts of `fit_sym_tp` are used: with two unstable
ends the data at those ends hold about 10^4 times the energy of the rest, and the pooled
start alone stalls at an operator error of 0.21.
Control, one fixed gain u = F xi, on the true plant only (never the fitted model):
  polytopic   the LMI of `control.synthesize_quadratic_feedback` on the fitted vertices
  pooled LQR  LQR on a single p-blind GDAR fitted to the same data
  open loop   u = 0

Run:  .venv\Scripts\python scripts/15_nine_node_control.py [--plot | --lqr-cert]
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
from gdar_tp.control import closed_loop_max_rho, companion_form, synthesize_quadratic_feedback  # noqa: E402
from gdar_tp.graph import grid_graph  # noqa: E402
from gdar_tp.sym_tp import _quad, fit_gls_pooled, fit_sym_tp, polytope_vertices, symmetric_basis, theta_of  # noqa: E402
from gdar_tp.tpgdar import resid_cov_sum, stack_stats, trial_stats  # noqa: E402

K, N = 2, 9
P_GRID = np.linspace(0, 1, 7)
P_DENSE = np.linspace(0, 1, 41)
N_TRIALS, T_TRIAL, NOISE, U_STD = 30, 200, 0.3, 1.0
N_PLANTS = 20
SCENARIOS = {"one unstable": (0.90, 1.03), "two unstable": (1.03, 1.03)}
DECAY = 0.9
SWEEP_DECAY = (0.995, 0.98, 0.95, 0.9, 0.8, 0.6)
LQR_R = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
CACHE = os.path.join(ROOT, "cache", "15_nine_node.pkl")
log = s12.log


def rho(th):
    return float(np.abs(s12.rho(th[:, :K * N], K)).max())


def scaled(phi, R, target):
    """Scale the lag coefficients so the operator has spectral radius ``target``."""
    n_lag = K * (R.shape[1] - N) // K
    lo, hi = 0.0, 5.0
    for _ in range(60):
        s = (lo + hi) / 2
        ph = phi.copy()
        ph[:n_lag] *= s
        if rho(theta_of(ph, R, N)) > target:
            hi = s
        else:
            lo = s
    ph = phi.copy()
    ph[:n_lag] *= (lo + hi) / 2
    return ph


def make_plant(seed, R, A, targets):
    rng = np.random.default_rng(seed)
    E = int(A.sum() // 2)
    n_lag = K * (N + E)

    def draw():
        ph = np.zeros(R.shape[1])
        ph[:N] = rng.uniform(0.3, 0.8, N)                               # lag 1 self terms
        ph[N:N + E] = rng.standard_normal(E) * 0.25                     # lag 1 edges
        ph[N + E:n_lag] = rng.standard_normal(N + E) * 0.06             # lag 2
        return ph
    b = rng.uniform(0.5, 1.5, N)
    ph0, ph1 = draw(), draw()
    ph0[n_lag:] = ph1[n_lag:] = b
    return scaled(ph0, R, targets[0]), scaled(ph1, R, targets[1])


def simulate(th, T, rng, U=None, x0=None, noise=NOISE):
    x = np.zeros((T, N))
    if x0 is not None:
        x[:K] = x0
    if U is None:
        U = rng.standard_normal((T, N)) * U_STD
    for t in range(K - 1, T - 1):
        z = np.concatenate([x[t - k] for k in range(K)] + [U[t]])
        x[t + 1] = th @ z + noise * rng.standard_normal(N)
    return x, U


def truth_at(ph0, ph1, R, p):
    return theta_of((1 - p) * ph0 + p * ph1, R, N)


def learn(ph0, ph1, R, rng):
    per = []
    for p in P_GRID:
        th = truth_at(ph0, ph1, R, p)
        per.append([trial_stats(*simulate(th, T_TRIAL, rng), K=K, Q=1) for _ in range(N_TRIALS)])
    st = stack_stats(per)
    tot = {k: st[k].sum(0) for k in st}
    phi_pool, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    models = {r: fit_sym_tp(st, R, Sig, r, P_GRID, starts=("pooled", "svd")) for r in (1, 2, 3)}
    Hbar = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sig))[0] / tot["n"]
    poly = polytope_vertices(models[2], Hbar)
    return st, phi_pool, Sig, models, poly


def score_new_p(ph0, ph1, R, st, phi_pool, models, rng, p_test=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Held-out ell at values of p the fit never saw, nats/sample, per model."""
    out = {k: [] for k in ["pooled", "rank 1", "rank 2", "rank 3"]}
    n_g = len(P_GRID)

    def sigma(Thetas):
        S = sum(resid_cov_sum(Thetas[g], st["Gzz"][g], st["Gzy"][g], st["Syy"][g]) for g in range(n_g))
        return S / st["n"].sum()
    Th_pool = theta_of(phi_pool, R, N)
    Sigs = {"pooled": sigma([Th_pool] * n_g), **{f"rank {r}": sigma(m.theta_grid()) for r, m in models.items()}}
    for p in p_test:
        th = truth_at(ph0, ph1, R, p)
        s = stack_stats([[trial_stats(*simulate(th, T_TRIAL, rng), K=K, Q=1) for _ in range(N_TRIALS)]])
        s = {k: s[k][0] for k in s}

        def ll(Th, Sg):
            S = resid_cov_sum(Th, s["Gzz"], s["Gzy"], s["Syy"])
            return -0.5 * (np.sum(np.linalg.inv(Sg) * S) / s["n"] + N * np.log(2 * np.pi) + np.linalg.slogdet(Sg)[1])
        out["pooled"].append(ll(Th_pool, Sigs["pooled"]))
        for r, m in models.items():
            out[f"rank {r}"].append(ll(m.theta_at(p), Sigs[f"rank {r}"]))
    return {k: np.array(v) for k, v in out.items()}


def comp(th):
    return companion_form([th[:, k * N:(k + 1) * N] for k in range(K)], th[:, K * N:])


def lqr(phi_pool, R, r=1.0):
    Ac, Bc = comp(theta_of(phi_pool, R, N))
    X = solve_discrete_are(Ac, Bc, np.eye(K * N), r * np.eye(N))
    return -np.linalg.solve(r * np.eye(N) + Bc.T @ X @ Bc, Bc.T @ X @ Ac)


def design(poly, phi_pool, R, decay=DECAY, min_gain=False):
    ths = [theta_of(ph, R, N) for ph in poly["phi"]]
    B_hat = np.mean([th[:, K * N:] for th in ths], 0)
    res = synthesize_quadratic_feedback([[th[:, k * N:(k + 1) * N] for k in range(K)] for th in ths], B_hat,
                                        decay=decay, min_gain=min_gain)
    return res, B_hat


def effort_sweep(ph0, ph1, R, phi_pool, poly, seed):
    """Control effort against worst true stability, each controller at several settings."""
    p_t = schedule()
    pts = {"pooled LQR": [], "polytopic": []}

    def evaluate(F):
        wr = true_rho_curve(ph0, ph1, R, F).max()
        x, u = run_loop(ph0, ph1, R, F, p_t, np.random.default_rng(5000 + seed))
        ok = np.isfinite(x).all()
        return dict(worst_rho=wr, diverged=not ok, rms_x=np.sqrt(np.nanmean(x ** 2)), rms_u=np.sqrt(np.nanmean(u ** 2)))
    for r in LQR_R:
        pts["pooled LQR"].append(dict(setting=r, **evaluate(lqr(phi_pool, R, r))))
    for a in SWEEP_DECAY:
        res, _ = design(poly, phi_pool, R, decay=a, min_gain=True)
        if res.F is None:
            pts["polytopic"].append(dict(setting=a, worst_rho=np.inf, diverged=True, rms_x=np.nan, rms_u=np.nan))
        else:
            pts["polytopic"].append(dict(setting=a, **evaluate(res.F)))
    return pts


def true_rho_curve(ph0, ph1, R, F):
    out = []
    for p in P_DENSE:
        Ac, Bc = comp(truth_at(ph0, ph1, R, p))
        out.append(np.abs(np.linalg.eigvals(Ac + (Bc @ F if F is not None else 0))).max())
    return np.array(out)


def certificate(ph0, ph1, R, F, Q):
    """Largest eigenvalue of A_cl' P A_cl - P over p on the true plant, P = Q^-1 (< 0 certifies)."""
    P = np.linalg.inv(Q)
    worst = -np.inf
    for p in P_DENSE:
        Ac, Bc = comp(truth_at(ph0, ph1, R, p))
        Acl = Ac + Bc @ F
        worst = max(worst, np.linalg.eigvalsh(Acl.T @ P @ Acl - P).max())
    return float(worst)


def run_loop(ph0, ph1, R, F, p_t, rng, u_limit=None, stop=1e4):
    """Closed loop on the true plant.  The noise stream is set by ``rng``."""
    T = len(p_t)
    x, u = np.zeros((T, N)), np.zeros((T, N))
    e = NOISE * rng.standard_normal((T, N))
    for t in range(K - 1, T - 1):
        xi = np.concatenate([x[t - k] for k in range(K)])
        if F is not None:
            u[t] = F @ xi
        z = np.concatenate([xi, u[t]])
        x[t + 1] = truth_at(ph0, ph1, R, p_t[t]) @ z + e[t]
        if np.abs(x[t + 1]).max() > stop:
            x[t + 1:] = np.nan
            u[t + 1:] = np.nan
            break
    return x, u


def schedule(T=1500):
    """p ramps up and down, dwelling at the unstable end."""
    return np.clip(0.5 - 0.75 * np.cos(2 * np.pi * np.arange(T) / 500.0), 0, 1)


def one_plant(seed, A, R, targets):
    rng = np.random.default_rng(1000 + seed)
    ph0, ph1 = make_plant(seed, R, A, targets)
    st, phi_pool, Sig, models, poly = learn(ph0, ph1, R, rng)
    res, B_hat = design(poly, phi_pool, R)
    res_boundary, _ = design(poly, phi_pool, R, decay=1.0)
    F_lqr = lqr(phi_pool, R)
    truth = np.stack([truth_at(ph0, ph1, R, p) for p in P_GRID])
    err = {f"rank {r}": np.linalg.norm(m.theta_grid() - truth) / np.linalg.norm(truth) for r, m in models.items()}
    err["pooled"] = np.linalg.norm(np.stack([theta_of(phi_pool, R, N)] * len(P_GRID)) - truth) / np.linalg.norm(truth)
    new_p = score_new_p(ph0, ph1, R, st, phi_pool, models, rng)
    out = dict(seed=seed, ph0=ph0, ph1=ph1, phi_pool=phi_pool, poly=poly, models=models, err=err, new_p=new_p,
               feasible=res.feasible, F_poly=res.F, F_lqr=F_lqr, Q=res.Q)
    Fs = {"open loop": None, "pooled LQR": F_lqr, "polytopic": res.F}
    out["rho"] = {k: true_rho_curve(ph0, ph1, R, F) for k, F in Fs.items()}
    out["cert"] = certificate(ph0, ph1, R, res.F, res.Q) if res.F is not None else np.nan
    out["cert_boundary"] = certificate(ph0, ph1, R, res_boundary.F, res_boundary.Q) if res_boundary.F is not None else np.nan
    th_v = [theta_of(ph, R, N) for ph in poly["phi"]]
    out["b_true"] = np.diag(truth[0][:, K * N:])
    out["b_hat"] = np.diag(B_hat)
    out["b_spread"] = float(np.abs(np.diag(th_v[0][:, K * N:]) - np.diag(th_v[1][:, K * N:])).max())
    out["pooled_model_rho"] = rho(theta_of(phi_pool, R, N))
    out["sweep"] = effort_sweep(ph0, ph1, R, phi_pool, poly, seed)
    p_t = schedule()
    sims = {}
    for k, F in Fs.items():
        sims[k] = run_loop(ph0, ph1, R, F, p_t, np.random.default_rng(5000 + seed))
    out["sims"] = sims
    out["p_t"] = p_t
    return out


def main():
    g = grid_graph(3, 3)
    A = np.asarray(g.adjacency, bool)
    R = symmetric_basis(A, K, node_inputs=True)
    log(f"N = {N}, E = {int(A.sum()/2)}, K = {K}, free numbers per operator {R.shape[1]}")
    results = {}
    for name, targets in SCENARIOS.items():
        log(f"\n===== {name} =====")
        plants = []
        for seed in range(N_PLANTS):
            plants.append(one_plant(seed, A, R, targets))
            r = plants[-1]
            fin = {k: np.isfinite(v[0]).all() for k, v in r["sims"].items()}
            rms = {k: float(np.sqrt(np.nanmean(v[0] ** 2))) if fin[k] else np.nan for k, v in r["sims"].items()}
            log(f"plant {seed}: feasible {r['feasible']}, cert {r['cert']:.4f}; worst true rho polytopic {r['rho']['polytopic'].max():.3f}, "
                f"LQR {r['rho']['pooled LQR'].max():.3f}, open {r['rho']['open loop'].max():.3f}; err r2 {r['err']['rank 2']:.3f} pooled {r['err']['pooled']:.3f}; "
                f"rms {rms}")
        results[name] = plants
        with open(CACHE, "wb") as f:
            pickle.dump(dict(scenarios=results, A=A), f)
        sm = lambda f: np.array([f(r) for r in plants])
        log("\nacross plants")
        for k in ("rank 1", "rank 2", "rank 3", "pooled"):
            log(f"  operator error {k}: median {np.median(sm(lambda r: r['err'][k])):.3f}")
        for k in ("rank 1", "rank 2", "rank 3"):
            d = sm(lambda r: (r['new_p'][k] - r['new_p']['pooled']).mean())
            log(f"  gain at unseen p, {k} vs pooled: median {np.median(d):+.3f}, positive {int((d > 0).sum())}/{N_PLANTS}")
        log(f"  LMI feasible on fitted vertices: {int(sm(lambda r: r['feasible']).sum())}/{N_PLANTS}")
        log(f"  fitted certificate holds on true plant: decay {DECAY} {int(sm(lambda r: r['cert'] < 0).sum())}/{N_PLANTS}, "
            f"decay 1 {int(sm(lambda r: r['cert_boundary'] < 0).sum())}/{N_PLANTS}")
        log(f"  input gains: max |b_hat - b| median {np.median(sm(lambda r: np.abs(r['b_hat'] - r['b_true']).max())):.3f}, "
            f"spread between vertices median {np.median(sm(lambda r: r['b_spread'])):.3f}")
        log(f"  pooled model's own spectral radius: median {np.median(sm(lambda r: r['pooled_model_rho'])):.3f}, "
            f"below 1 in {int(sm(lambda r: r['pooled_model_rho'] < 1).sum())}/{N_PLANTS}")
        for c, settings in (("pooled LQR", LQR_R), ("polytopic", SWEEP_DECAY)):
            for i, s in enumerate(settings):
                wr = sm(lambda r: r['sweep'][c][i]['worst_rho'])
                log(f"  sweep {c} {s}: stable for all p {int((wr < 1).sum())}/{N_PLANTS}, worst rho median {np.median(wr):.3f}, "
                    f"diverged {int(sm(lambda r: r['sweep'][c][i]['diverged']).sum())}, rms u median {np.nanmedian(sm(lambda r: r['sweep'][c][i]['rms_u'])):.3f}, "
                    f"rms x median {np.nanmedian(sm(lambda r: r['sweep'][c][i]['rms_x'])):.3f}")
        for k in ("open loop", "pooled LQR", "polytopic"):
            wr = sm(lambda r: r['rho'][k].max())
            log(f"  worst true rho, {k}: median {np.median(wr):.3f}, stable for all p {int((wr < 1).sum())}/{N_PLANTS}")
            div = sm(lambda r: not np.isfinite(r['sims'][k][0]).all())
            rms = sm(lambda r: np.sqrt(np.nanmean(r['sims'][k][0] ** 2)))
            rmu = sm(lambda r: np.sqrt(np.nanmean(r['sims'][k][1] ** 2)))
            log(f"    diverged {int(div.sum())}/{N_PLANTS}; rms x median {np.median(rms):.3f}, rms u median {np.median(rmu):.3f}")


C_OPEN, C_LQR, C_POLY = "#8a8f98", s12.C_FELIX, s12.C_IND
CTRL = (("open loop", C_OPEN), ("pooled LQR", C_LQR), ("polytopic", C_POLY))
EXAMPLE = 0


def _graph(ax, th, A, xy, vmax, title):
    """Edges summed over lags: colour = sign, width = size.  The number in a node is its input gain."""
    S = sum(th[:, k * N:(k + 1) * N] for k in range(K))
    for i, j in zip(*np.where(np.triu(A, 1))):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=s12.C_FELIX if S[i, j] > 0 else s12.C_IND,
                lw=0.5 + 6 * abs(S[i, j]) / vmax, solid_capstyle="round", zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=560, c="white", edgecolors="#333333", linewidths=0.9, zorder=2)
    for i in range(N):
        ax.text(xy[i, 0], xy[i, 1], f"{th[i, K * N + i]:.2f}", ha="center", va="center", fontsize=8, zorder=3)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-2.45, 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def _strip(ax, rows, log_y=False):
    """One dot per plant, a bar at the median."""
    rng = np.random.default_rng(0)
    for i, (name, v) in enumerate(rows):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=s12.C_FELIX, alpha=0.55, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="#222222", lw=2, solid_capstyle="round")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    if log_y:
        ax.set_yscale("log")


def fig_learning(plants, A):
    import matplotlib.pyplot as plt
    g = grid_graph(3, 3)
    xy = np.column_stack([g.coords()[:, 1], -g.coords()[:, 0]]).astype(float)
    R = symmetric_basis(A, K, node_inputs=True)
    r = plants[EXAMPLE]
    poly = r["poly"]
    order = np.argsort(-poly["lam"][0])
    th = [theta_of(r["ph0"], R, N), theta_of(r["ph1"], R, N), theta_of(poly["phi"][order[0]], R, N), theta_of(poly["phi"][order[1]], R, N)]
    edge = ~np.eye(N, dtype=bool)
    vmax = max(np.abs(sum(t[:, k * N:(k + 1) * N] for k in range(K))[edge]).max() for t in th)
    fig, axs = plt.subplots(2, 4, figsize=(13.5, 6.8), constrained_layout=True)
    fig.suptitle("Learning a 9-node TP-GDAR with an input at each node: 3 x 3 graph, K = 2, known truth, 20 plants",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    for ax, t, title in zip(axs[:2, :2].ravel(), th, (r"A  True $\Theta_0$ (p = 0)", r"B  True $\Theta_1$ (p = 1)",
                                                      "C  Learned vertex 1", "D  Learned vertex 2")):
        _graph(ax, t, A, xy, vmax, title)
    fig.text(0.02, 0.012, "One plant. Edges summed over lags: blue +, orange -, width = size. Number in a node = its input gain.",
             fontsize=8.5, color=s12.MUTED)
    ax = axs[0, 2]
    for q in plants:
        o = np.argsort(-q["poly"]["lam"][0])
        ax.plot(P_GRID, q["poly"]["lam"][:, o[1]], color=s12.C_FELIX, lw=1, alpha=0.5)
    ax.plot([0, 1], [0, 1], color="#222222", lw=1.2, ls="--")
    ax.text(0.62, 0.42, "dashed: true (= p)", fontsize=8.5, color=s12.MUTED)
    ax.set_xlabel("p")
    ax.set_ylabel("learned weight of vertex 2")
    ax.set_title("E  Weights, 20 plants")
    ax = axs[1, 2]
    bt = np.concatenate([q["b_true"] for q in plants])
    bh = np.concatenate([q["b_hat"] for q in plants])
    ax.plot([0.4, 1.6], [0.4, 1.6], color="#222222", lw=1.2, ls="--")
    ax.scatter(bt, bh, s=12, color=s12.C_FELIX, alpha=0.6, lw=0)
    ax.set_xlabel("true input gain")
    ax.set_ylabel("learned input gain")
    ax.set_title(f"F  Input gains, worst error {np.abs(bt - bh).max():.3f}")
    ax = axs[0, 3]
    _strip(ax, [(k, np.array([q["err"][k] for q in plants])) for k in ("pooled", "rank 1", "rank 2", "rank 3")], log_y=True)
    ax.set_ylabel("relative operator error")
    ax.set_title("G  Distance from the true operators")
    ax = axs[1, 3]
    _strip(ax, [(k, np.array([(q["new_p"][k] - q["new_p"]["pooled"]).mean() for q in plants])) for k in ("rank 1", "rank 2", "rank 3")])
    ax.set_ylim(0, None)
    ax.set_ylabel("gain over pooled (nats/sample)")
    ax.set_title("H  Fresh data at p never fitted")
    return fig


def fig_control(scenarios):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig = plt.figure(figsize=(13.5, 7.6), constrained_layout=True)
    fig.suptitle("One fixed gain on the true plant: 9 nodes, an input at each node, 20 plants",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    outer = fig.add_gridspec(len(scenarios), 3, width_ratios=[1, 1.25, 1])
    for row, (name, plants) in enumerate(scenarios.items()):
        letters = "ABCDEF"[3 * row:3 * row + 3]
        ax = fig.add_subplot(outer[row, 0])
        for k, c in CTRL:
            curves = np.stack([q["rho"][k] for q in plants])
            ax.fill_between(P_DENSE, curves.min(0), curves.max(0), color=c, alpha=0.2, lw=0)
            ax.plot(P_DENSE, np.median(curves, 0), color=c, lw=2, label=k)
        ax.axhline(1, color="#222222", lw=1, ls="--")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("p")
        ax.set_ylabel("true spectral radius")
        ax.set_title(f"{letters[0]}  {name}: median and range")
        handles = ax.get_legend_handles_labels()
        inner = outer[row, 1].subgridspec(2, 1, height_ratios=[1, 3.2])
        r = plants[EXAMPLE]
        a0 = fig.add_subplot(inner[0])
        a0.plot(r["p_t"], color="#222222", lw=1.2)
        a0.set_ylabel("p")
        a0.set_ylim(-0.05, 1.05)
        a0.set_xlim(0, len(r["p_t"]))
        a0.tick_params(labelbottom=False)
        a0.set_title(f"{letters[1]}  Closed loop as p moves, one plant")
        a1 = fig.add_subplot(inner[1], sharex=a0)
        for k, c in CTRL:
            x = r["sims"][k][0]
            trace = np.sqrt(np.convolve(np.mean(x ** 2, 1), np.ones(25) / 25, mode="valid"))
            a1.plot(np.arange(len(trace)) + 12, trace, color=c, lw=1.4, label=k)
        a1.set_yscale("log")
        a1.set_xlabel("time step")
        a1.set_ylabel("rms x over nodes")
        ax = fig.add_subplot(outer[row, 2])
        for k, c in (("pooled LQR", C_LQR), ("polytopic", C_POLY)):
            n_set = len(plants[0]["sweep"][k])
            wr = np.stack([[q["sweep"][k][i]["worst_rho"] for i in range(n_set)] for q in plants])
            ru = np.stack([[q["sweep"][k][i]["rms_u"] for i in range(n_set)] for q in plants])
            med_u = np.nanmedian(ru, 0)
            ax.plot(med_u, np.median(wr, 0), "-o", ms=5, color=c, lw=1.6, label=k)
            ax.vlines(med_u, wr.min(0), np.minimum(wr.max(0), 1.25), color=c, lw=1, alpha=0.5)
        ax.axhline(1, color="#222222", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([0.02, 0.05, 0.1, 0.2], ["0.02", "0.05", "0.1", "0.2"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("rms input (median over plants)")
        ax.set_ylabel("worst true spectral radius")
        ax.set_title(f"{letters[2]}  Effort for each controller setting")
    fig.legend(*handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    return fig


def plot():
    with open(CACHE, "rb") as f:
        out = pickle.load(f)
    s12._style()
    for fname, fig in (("15_nine_node_learning.png", fig_learning(out["scenarios"]["one unstable"], out["A"])),
                       ("15_nine_node_control.png", fig_control(out["scenarios"]))):
        fig.savefig(os.path.join(s12.FIG, fname), dpi=170)
        log("wrote", fname)


def common_lyapunov(closed_loops, alpha):
    """P >= I with A_cl' P A_cl < alpha^2 P at every closed-loop vertex, or None."""
    import cvxpy as cp
    n = closed_loops[0].shape[0]
    P = cp.Variable((n, n), symmetric=True)
    cons = [P >> np.eye(n)]
    for Acl in closed_loops:
        cons.append(cp.bmat([[alpha * P, (P @ Acl).T], [P @ Acl, alpha * P]]) >> 1e-6 * np.eye(2 * n))
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    for solver in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=solver)
            if prob.status in ("optimal", "optimal_inaccurate", "infeasible"):
                break
        except cp.error.SolverError:
            pass
    return P.value if prob.status in ("optimal", "optimal_inaccurate") else None


def lqr_cert(n_plants=N_PLANTS, alphas=(0.9, 0.8, 0.6)):
    """Can the pooled LQR gain be certified?  Same test as the polytopic gain: rate alpha on the fitted
    vertices, then rate 1 on the true plant.  About 30 minutes for all plants."""
    with open(CACHE, "rb") as f:
        out = pickle.load(f)
    R = symmetric_basis(out["A"], K, node_inputs=True)

    def closed(phis, F):
        return [comp(theta_of(ph, R, N))[0] + comp(theta_of(ph, R, N))[1] @ F for ph in phis]
    for name, plants in out["scenarios"].items():
        for alpha in alphas:
            feasible = holds = 0
            for r in plants[:n_plants]:
                P = common_lyapunov(closed(r["poly"]["phi"], r["F_lqr"]), alpha)
                if P is not None:
                    feasible += 1
                    worst = max(np.linalg.eigvalsh(A.T @ P @ A - P).max() for A in closed((r["ph0"], r["ph1"]), r["F_lqr"]))
                    holds += worst < 0
            log(f"{name}, rate {alpha}: feasible on fitted vertices {feasible}/{n_plants}, holds on true plant at rate 1 {holds}/{n_plants}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    elif "--lqr-cert" in sys.argv:
        lqr_cert()
    else:
        main()
