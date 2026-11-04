r"""Few inputs: the 9-node example of script 15 with only some nodes actuated.

Same 20 plants per case, same data amounts, loop closed on the true plant only.  The lag operators
of a plant don't depend on the input nodes.  Input nodes come from a rule fixed before any result:
    M = 1 [4]   M = 2 [0, 8]   M = 3 [0, 4, 8]   M = 5 [0, 2, 4, 6, 8]   M = 9 all

Controllers, each one fixed gain u = F xi that does not know p:
  polytopic        the LMI on the fitted vertices.  An LMI counts as solved only if it holds when
                   checked on the fitted system (`lmi_residual`), since a near-infeasible problem can
                   come back "optimal_inaccurate" with a useless F.  Headline design: alpha = 0.9 as
                   in script 15, or else the next looser alpha where the LMI holds (uses the fit only)
  pooled           LQR on a single p-blind GLS fit, as in script 15
  pooled, trials normalised
                   the same, each trial scaled to unit state energy, so that unstable trials don't
                   set the fit.  Still blind to p
  mean vertex      LQR on the mean of the two fitted vertices.  Needs the TP-GDAR fit but not the LMI
  vertex p=0, p=1  LQR on one fitted vertex
  LQR family       every LQR above plus lower R and a prescribed decay rate, up to 160 gains per plant, the one
                   with the lowest spectral radius across the fitted model chosen.  Needs the TP-GDAR fit
                   but not the LMI, and is the strongest rival here.  Its failures are visible on the fit
Reference only: LQR on the true operator at each frozen p, to show each p is stabilisable.

Run:  .venv\Scripts\python scripts/16_nine_node_few_inputs.py [--family | --summary | --plot]
      --family redoes only the LQR family from the cached fits
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import solve_discrete_are  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s15 = importlib.import_module("15_nine_node_control")
from gdar_tp.control import synthesize_quadratic_feedback  # noqa: E402
from gdar_tp.graph import grid_graph  # noqa: E402
from gdar_tp.sym_tp import _quad, fit_gls_pooled, fit_sym_tp, polytope_vertices, symmetric_basis, theta_of  # noqa: E402
from gdar_tp.tpgdar import stack_stats, trial_stats  # noqa: E402

K, N = s15.K, s15.N
PLACEMENT = {1: [4], 2: [0, 8], 3: [0, 4, 8], 5: [0, 2, 4, 6, 8], 9: list(range(9))}
ALPHAS = (0.6, 0.8, 0.9, 0.95, 0.98, 0.995, 1.0)
HEADLINE_ALPHAS = (0.9, 0.95, 0.98, 0.995, 1.0)
Q_MAX = 1e4                 # for the min-gain LMI; script 15's 100 can make few inputs falsely infeasible
LMI_TOL = 1e-6
LQR_KINDS = ("pooled", "pooled, trials normalised", "mean vertex", "vertex p=0", "vertex p=1")
FAMILY_R = (1e-3, 1e-2) + tuple(s15.LQR_R)
FAMILY_SHIFT = (1.0, 0.95, 0.9, 0.8)
TARGETS = (0.98, 0.95, 0.9, 0.8)
EXAMPLE = dict(case="two unstable", M=2, seed=s15.EXAMPLE)
CACHE = os.path.join(ROOT, "cache", "16_few_inputs.pkl")
log = s12.log
warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def learn(ph0, ph1, R, rng):
    """As `s15.learn`, same random stream, and also the p-blind fit with trials normalised."""
    per = []
    for p in s15.P_GRID:
        th = s15.truth_at(ph0, ph1, R, p)
        per.append([trial_stats(*s15.simulate(th, s15.T_TRIAL, rng), K=K, Q=1) for _ in range(s15.N_TRIALS)])
    st = stack_stats(per)
    tot = {k: st[k].sum(0) for k in st}
    phi_pool, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    model = fit_sym_tp(st, R, Sig, 2, s15.P_GRID, starts=("pooled", "svd"))
    Hbar = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sig))[0] / tot["n"]
    poly = polytope_vertices(model, Hbar)
    acc = {k: 0.0 for k in ("Gzz", "Gzy", "Syy", "n")}
    for trials in per:
        for t in trials:
            w = t["n"] / np.trace(t["Gzz"][:K * N, :K * N])
            for k in acc:
                acc[k] = acc[k] + w * t[k]
    phi_norm, _ = fit_gls_pooled(acc["Gzz"], acc["Gzy"], acc["Syy"], acc["n"], R)
    return model, poly, phi_pool, phi_norm


def lmi(poly, R, alpha, min_gain=False):
    ths = [theta_of(ph, R, N) for ph in poly["phi"]]
    B_hat = np.mean([th[:, K * N:] for th in ths], 0)
    res = synthesize_quadratic_feedback([[th[:, k * N:(k + 1) * N] for k in range(K)] for th in ths], B_hat,
                                        decay=alpha, min_gain=min_gain, q_max=Q_MAX)
    return res, res.F is not None and res.lmi_residual <= LMI_TOL


def lqr_or_none(phi, R, r):
    try:
        F = s15.lqr(phi, R, r)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return F if np.isfinite(F).all() else None


def evaluate(F, ph0, ph1, R, seed, p_t, keep=False):
    """Worst true spectral radius over p, and the closed loop on the true plant."""
    curve = s15.true_rho_curve(ph0, ph1, R, F)
    x, u = s15.run_loop(ph0, ph1, R, F, p_t, np.random.default_rng(5000 + seed))
    diverged = not np.isfinite(x).all()
    out = dict(worst_rho=float(curve.max()), diverged=diverged, norm_F=float(np.linalg.norm(F, 2)) if F is not None else 0.0,
               rms_x=np.nan if diverged else float(np.sqrt(np.mean(x ** 2))),
               rms_u=np.nan if diverged else float(np.sqrt(np.mean(u ** 2))))
    if keep:
        out["x"] = x
    return out


def failed():
    return dict(worst_rho=np.inf, diverged=True, norm_F=np.nan, rms_x=np.nan, rms_u=np.nan)


def reach(phi, R):
    """Least input reach of any unstable mode: min ||w' B_c|| / ||w||, w a left eigenvector.  NaN if none is unstable."""
    Ac, Bc = s15.comp(theta_of(phi, R, N))
    lam, W = np.linalg.eig(Ac.T)
    idx = np.where(np.abs(lam) >= 1)[0]
    return float(min(np.linalg.norm(W[:, i].conj() @ Bc) / np.linalg.norm(W[:, i]) for i in idx)) if len(idx) else np.nan


def frozen_rho(ph0, ph1, R):
    worst = 0.0
    phi_at = [(1 - p) * ph0 + p * ph1 for p in s15.P_DENSE[::4]]
    for phi in phi_at:
        F = lqr_or_none(phi, R, 1.0)
        Ac, Bc = s15.comp(theta_of(phi, R, N))
        worst = max(worst, np.inf if F is None else np.abs(np.linalg.eigvals(Ac + Bc @ F)).max())
    return float(worst)


def one(job):
    warnings.filterwarnings("ignore")
    case, M, seed = job
    A = np.asarray(grid_graph(3, 3).adjacency, bool)
    nodes = np.array(PLACEMENT[M])
    R = symmetric_basis(A, K, node_inputs=nodes)
    ph0, ph1 = s15.make_plant(seed, R, A, s15.SCENARIOS[case], nodes=nodes)
    model, poly, phi_pool, phi_norm = learn(ph0, ph1, R, np.random.default_rng(1000 + seed))
    truth = np.stack([s15.truth_at(ph0, ph1, R, p) for p in s15.P_GRID])
    relerr = lambda th, n=None: float(np.linalg.norm(th[..., :n] - truth[..., :n]) / np.linalg.norm(truth[..., :n]))
    order = np.argsort(-poly["lam"][0])
    phis = {"pooled": phi_pool, "pooled, trials normalised": phi_norm, "mean vertex": poly["phi"].mean(0),
            "vertex p=0": poly["phi"][order[0]], "vertex p=1": poly["phi"][order[1]]}
    keep = case == EXAMPLE["case"] and M == EXAMPLE["M"] and seed == EXAMPLE["seed"]
    p_t = s15.schedule()
    out = dict(case=case, M=M, seed=seed, nodes=nodes, ph0=ph0, ph1=ph1, phis=phis)
    out["err_r2"] = relerr(model.theta_grid())
    out["lag_err"] = {"rank 2": relerr(model.theta_grid(), K * N)}
    for k in ("pooled", "pooled, trials normalised"):
        out["lag_err"][k] = relerr(np.stack([theta_of(phis[k], R, N)] * len(s15.P_GRID)), K * N)
    out["b_err"] = float(np.abs(np.mean(poly["phi"], 0)[s15.N_LAG:] - ph0[s15.N_LAG:]).max())
    out["model_rho"] = {k: s15.rho(theta_of(phis[k], R, N)) for k in ("pooled", "pooled, trials normalised", "mean vertex")}
    out["reach"] = {"pooled": reach(phi_pool, R), "pooled, trials normalised": reach(phi_norm, R), "mean vertex": reach(phis["mean vertex"], R),
                    "true p=0": reach(ph0, R), "true p=1": reach(ph1, R)}
    out["open"] = evaluate(None, ph0, ph1, R, seed, p_t, keep)
    out["frozen"] = frozen_rho(ph0, ph1, R)
    out["lqr"] = {}
    for kind in LQR_KINDS:
        out["lqr"][kind] = {}
        for r in s15.LQR_R:
            F = lqr_or_none(phis[kind], R, r)
            out["lqr"][kind][r] = failed() if F is None else evaluate(F, ph0, ph1, R, seed, p_t, keep and r == 1.0)
    out["lmi"], out["lmi_gain"] = {}, {}
    for a in ALPHAS:
        res, ok = lmi(poly, R, a)
        row = dict(ok=ok, status=res.status, residual=res.lmi_residual, cond_Q=float(np.linalg.cond(res.Q)) if res.Q is not None else np.nan)
        if ok:
            row.update(evaluate(res.F, ph0, ph1, R, seed, p_t), cert=s15.certificate(ph0, ph1, R, res.F, res.Q) < 0)
        elif res.F is not None:
            row["raw_rho"] = float(s15.true_rho_curve(ph0, ph1, R, res.F).max())
        out["lmi"][a] = row
        res, ok = lmi(poly, R, a, min_gain=True)
        row = dict(ok=ok, status=res.status, residual=res.lmi_residual)
        if ok:
            row.update(evaluate(res.F, ph0, ph1, R, seed, p_t))
        out["lmi_gain"][a] = row
    head = next((a for a in HEADLINE_ALPHAS if out["lmi"][a]["ok"]), None)
    out["alpha"] = head
    if head is None:
        out["poly"] = failed() | dict(cert=False)
    else:
        res, _ = lmi(poly, R, head)
        out["poly"] = evaluate(res.F, ph0, ph1, R, seed, p_t, keep) | dict(cert=out["lmi"][head]["cert"])
    if keep:
        out["p_t"] = p_t
    return out


def shifted_lqr(phi, R, r, a):
    """LQR with a prescribed decay rate: solving on (A/a, B/a) puts the closed loop inside radius a on that model."""
    Ac, Bc = s15.comp(theta_of(phi, R, N))
    Ac, Bc = Ac / a, Bc / a
    I = r * np.eye(Bc.shape[1])
    X = solve_discrete_are(Ac, Bc, np.eye(K * N), I)
    return -np.linalg.solve(I + Bc.T @ X @ Bc, Bc.T @ X @ Ac)


def fixed_gain_cert(F, v0, v1, ph0, ph1, R, on_fit):
    """The polytopic gain's test applied to a gain that didn't come from the LMI: a common Lyapunov
    function at the first rate that holds on the fitted vertices (checked), then rate 1 on the true plant."""
    closed = []
    for ph in (v0, v1):
        Ac, Bc = s15.comp(theta_of(ph, R, N))
        closed.append(Ac + Bc @ F)
    for a in HEADLINE_ALPHAS:
        if a <= on_fit:
            continue
        P = s15.common_lyapunov(closed, a)
        if P is None:
            continue
        P = np.asarray(P)
        if max(np.linalg.eigvalsh(c.T @ P @ c - a ** 2 * P).max() for c in closed) / np.linalg.norm(P, 2) <= LMI_TOL:
            return a, bool(s15.certificate(ph0, ph1, R, F, np.linalg.inv(P)) < 0)
    return None, False


def family(r):
    """All the LQR gains on one plant, the one to use chosen on the fitted model.  The truth only scores it."""
    warnings.filterwarnings("ignore")
    A = np.asarray(grid_graph(3, 3).adjacency, bool)
    R = symmetric_basis(A, K, node_inputs=r["nodes"])
    ph0, ph1, seed = r["ph0"], r["ph1"], r["seed"]
    v0, v1 = r["phis"]["vertex p=0"], r["phis"]["vertex p=1"]
    fit = [s15.comp(theta_of((1 - p) * v0 + p * v1, R, N)) for p in np.linspace(0, 1, 11)]
    p_t = s15.schedule()
    rows, gains = [], []
    for kind in LQR_KINDS:
        for q in FAMILY_R:
            for a in FAMILY_SHIFT:
                try:
                    F = shifted_lqr(r["phis"][kind], R, q, a)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                if not np.isfinite(F).all():
                    continue
                on_fit = max(np.abs(np.linalg.eigvals(Ac + Bc @ F)).max() for Ac, Bc in fit)
                rows.append(evaluate(F, ph0, ph1, R, seed, p_t) | dict(kind=kind, r=q, shift=a, on_fit=float(on_fit)))
                gains.append(F)
    i = int(np.argmin([x["on_fit"] for x in rows]))
    pick = dict(rows[i])
    pick["alpha"], pick["cert"] = fixed_gain_cert(gains[i], v0, v1, ph0, ph1, R, pick["on_fit"]) if pick["on_fit"] < 1 else (None, False)
    if "p_t" in r:
        pick["x"] = evaluate(gains[i], ph0, ph1, R, seed, p_t, keep=True)["x"]
    return (r["case"], r["M"], seed), rows, pick


def add_family(res):
    by_job = {(r["case"], r["M"], r["seed"]): r for r in res}
    with ProcessPoolExecutor(14) as ex:
        for n, (job, rows, pick) in enumerate(ex.map(family, res), 1):
            by_job[job]["family"], by_job[job]["pick"] = rows, pick
            log(f"family {n}/{len(res)} {job}: {pick['kind']} R {pick['r']:g} shift {pick['shift']}: on fit {pick['on_fit']:.3f}, true {pick['worst_rho']:.3f}, certificate {pick['cert']}")


def stable(row):
    return row["worst_rho"] < 1


def cheapest(rows, target):
    """Least rms input over a controller's settings whose worst true rho meets the target, or NaN."""
    vals = [r["rms_u"] for r in rows if r.get("worst_rho", np.inf) <= target and not r.get("diverged", True)]
    return min(vals) if vals else np.nan


def summary(res):
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(np.asarray(v, float)).any() else np.nan
    for case in s15.SCENARIOS:
        for M in PLACEMENT:
            rs = sorted((r for r in res if r["case"] == case and r["M"] == M), key=lambda r: r["seed"])
            n = len(rs)
            log(f"\n===== {case}, M = {M}, inputs at {PLACEMENT[M]}, {n} plants =====")
            log(f"operator error rank 2 {med([r['err_r2'] for r in rs]):.4f}; lag error: rank 2 {med([r['lag_err']['rank 2'] for r in rs]):.4f}, "
                f"pooled {med([r['lag_err']['pooled'] for r in rs]):.3f}, normalised {med([r['lag_err']['pooled, trials normalised'] for r in rs]):.3f}; "
                f"input gain worst error {med([r['b_err'] for r in rs]):.4f}")
            log("model's own rho, median: " + ", ".join(f"{k} {med([r['model_rho'][k] for r in rs]):.3f}" for k in rs[0]["model_rho"]))
            log("input reach of the least reachable unstable mode, median: " + ", ".join(f"{k} {med([r['reach'][k] for r in rs]):.4f}" for k in rs[0]["reach"]))
            log(f"frozen-p LQR on the truth (reference): stable {sum(r['frozen'] < 1 for r in rs)}/{n}, worst rho median {med([r['frozen'] for r in rs]):.3f}")
            log(f"open loop: stable at all p {sum(stable(r['open']) for r in rs)}/{n}, worst rho {med([r['open']['worst_rho'] for r in rs]):.3f}, diverged {sum(r['open']['diverged'] for r in rs)}, rms x {med([r['open']['rms_x'] for r in rs]):.2f}")
            log("LMI, trace objective, per alpha:")
            for a in ALPHAS:
                ok = [r for r in rs if r["lmi"][a]["ok"]]
                log(f"  alpha {a}: LMI holds on fit {len(ok)}/{n}" + (f"; of those stable on true plant {sum(stable(r['lmi'][a]) for r in ok)}, certificate at rate 1 {sum(r['lmi'][a]['cert'] for r in ok)}, "
                    f"worst rho {med([r['lmi'][a]['worst_rho'] for r in ok]):.3f}, rms u {med([r['lmi'][a]['rms_u'] for r in ok]):.3f}" if ok else "")
                    + f"; solver said optimal or inaccurate on {sum(r['lmi'][a]['status'] in ('optimal', 'optimal_inaccurate') for r in rs)}")
            al = [r["alpha"] for r in rs]
            log(f"polytopic headline: alpha used {sorted((a, al.count(a)) for a in set(al) if a is not None)}, no design {al.count(None)}; "
                f"stable {sum(stable(r['poly']) for r in rs)}/{n}, certificate {sum(r['poly']['cert'] for r in rs)}/{n}, worst rho {med([r['poly']['worst_rho'] for r in rs]):.3f}, "
                f"diverged {sum(r['poly']['diverged'] for r in rs)}, rms x {med([r['poly']['rms_x'] for r in rs]):.3f}, rms u {med([r['poly']['rms_u'] for r in rs]):.3f}, ||F|| {med([r['poly']['norm_F'] for r in rs]):.2f}")
            for kind in LQR_KINDS:
                rows = {q: [r["lqr"][kind][q] for r in rs] for q in s15.LQR_R}
                one_ = rows[1.0]
                some = sum(any(stable(r["lqr"][kind][q]) for q in s15.LQR_R) for r in rs)
                every = sum(all(stable(r["lqr"][kind][q]) for q in s15.LQR_R) for r in rs)
                log(f"LQR {kind}: R = 1 stable {sum(stable(x) for x in one_)}/{n}, worst rho {med([x['worst_rho'] for x in one_]):.3f}, diverged {sum(x['diverged'] for x in one_)}, "
                    f"rms x {med([x['rms_x'] for x in one_]):.3f}, rms u {med([x['rms_u'] for x in one_]):.3f}, ||F|| {med([x['norm_F'] for x in one_]):.2f}; "
                    f"stable at every R {every}, at some R {some}; per R " + " ".join(f"{q:g}:{sum(stable(x) for x in rows[q])}" for q in s15.LQR_R))
            pk = [r["pick"] for r in rs]
            bad = [x for x in pk if not stable(x)]
            log(f"LQR family, up to {max(len(r['family']) for r in rs)} gains, one chosen on the fit: stable {sum(stable(x) for x in pk)}/{n}, certificate {sum(x['cert'] for x in pk)}/{n}, worst rho {med([x['worst_rho'] for x in pk]):.3f}, "
                f"diverged {sum(x['diverged'] for x in pk)}, rms x {med([x['rms_x'] for x in pk]):.3f}, rms u {med([x['rms_u'] for x in pk]):.3f}, ||F|| {med([x['norm_F'] for x in pk]):.2f}; "
                f"of its failures the fit already showed rho >= 1 on {sum(x['on_fit'] >= 1 for x in bad)}/{len(bad)}; "
                f"any gain in the family stable on the true plant (needs the truth to pick) {sum(any(stable(x) for x in r['family']) for r in rs)}/{n}, "
                f"any from the p-blind fits only {sum(any(stable(x) for x in r['family'] if x['kind'].startswith('pooled')) for r in rs)}/{n}")
            log("input needed for a worst true rho at or below the target, comparing each LQR with the min-gain LMI, least over settings:")
            for t in TARGETS:
                pol = np.array([cheapest(list(r["lmi_gain"].values()), t) for r in rs])
                parts = [f"polytopic reaches it {int(np.isfinite(pol).sum())}/{n}"]
                for kind in ("pooled", "pooled, trials normalised", "mean vertex"):
                    l = np.array([cheapest(list(r["lqr"][kind].values()), t) for r in rs])
                    both = np.isfinite(l) & np.isfinite(pol)
                    parts.append(f"{kind} {int(np.isfinite(l).sum())}/{n}" + (f" (on {int(both.sum())} shared: {med(l[both]):.3f} vs {med(pol[both]):.3f})" if both.any() else ""))
                l = np.array([cheapest(r["family"], t) for r in rs])
                both = np.isfinite(l) & np.isfinite(pol)
                parts.append(f"LQR family {int(np.isfinite(l).sum())}/{n}" + (f" (on {int(both.sum())} shared: {med(l[both]):.3f} vs {med(pol[both]):.3f})" if both.any() else ""))
                log(f"  rho <= {t}: " + "; ".join(parts))
    with open(s15.CACHE, "rb") as f:
        c15 = pickle.load(f)["scenarios"]
    for case in s15.SCENARIOS:
        rs = [r for r in res if r["case"] == case and r["M"] == 9]
        log(f"check against script 15 at M = 9, {case}: pooled LQR worst rho {np.median([r['lqr']['pooled'][1.0]['worst_rho'] for r in rs]):.3f} "
            f"vs {np.median([r['rho']['pooled LQR'].max() for r in c15[case]]):.3f}; polytopic alpha 0.9 {np.median([r['lmi'][0.9]['worst_rho'] for r in rs]):.3f} "
            f"vs {np.median([r['rho']['polytopic'].max() for r in c15[case]]):.3f}")


def main():
    jobs = [(c, M, s) for c in s15.SCENARIOS for M in PLACEMENT for s in range(s15.N_PLANTS)]
    res = []
    with ProcessPoolExecutor(14) as ex:
        futures = {ex.submit(one, j): j for j in jobs}
        for f in as_completed(futures):
            r = f.result()
            res.append(r)
            log(f"{len(res)}/{len(jobs)} {r['case']} M={r['M']} seed {r['seed']}: polytopic {r['poly']['worst_rho']:.3f} at alpha {r['alpha']}, pooled LQR {r['lqr']['pooled'][1.0]['worst_rho']:.3f}")
    add_family(res)
    with open(CACHE, "wb") as f:
        pickle.dump(res, f)
    summary(res)


def plot():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    s12._style()
    with open(CACHE, "rb") as f:
        res = pickle.load(f)
    rows = (("polytopic LMI: stable", lambda r: stable(r["poly"])),
            ("polytopic LMI: certificate holds", lambda r: r["poly"]["cert"]),
            ("LQR family, picked on the fit: stable", lambda r: stable(r["pick"])),
            ("LQR family: certificate holds", lambda r: r["pick"]["cert"]),
            ("LQR on mean vertex, R = 1: stable", lambda r: stable(r["lqr"]["mean vertex"][1.0])),
            ("LQR on normalised pooled fit, R = 1: stable", lambda r: stable(r["lqr"]["pooled, trials normalised"][1.0])),
            ("LQR on pooled fit, R = 1: stable", lambda r: stable(r["lqr"]["pooled"][1.0])),
            ("open loop: stable", lambda r: stable(r["open"])))
    Ms = list(PLACEMENT)
    cmap = LinearSegmentedColormap.from_list("count", ["#eef3fa", "#1f5fb0"])
    fig = plt.figure(figsize=(13.5, 4.9), constrained_layout=True)
    fig.suptitle("Few inputs: one fixed gain on the true plant, 20 plants per case. Each count is plants out of 20",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    outer = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45])
    for col, (case, title) in enumerate((("one unstable", "A  One unstable end"), ("two unstable", "B  Two unstable ends"))):
        ax = fig.add_subplot(outer[0, col])
        counts = np.array([[sum(bool(f(r)) for r in res if r["case"] == case and r["M"] == M) for M in Ms] for _, f in rows])
        ax.imshow(counts, cmap=cmap, vmin=0, vmax=20, aspect="auto")
        for i, j in np.ndindex(counts.shape):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if counts[i, j] >= 12 else s12.INK)
        ax.set_xticks(range(len(Ms)), Ms)
        ax.set_yticks(range(len(rows)), [name for name, _ in rows] if col == 0 else [])
        ax.set_xlabel("input nodes")
        ax.tick_params(length=0)
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_title(title)
    r = next(q for q in res if "p_t" in q)
    inner = outer[0, 2].subgridspec(2, 1, height_ratios=[1, 3.6])
    a0 = fig.add_subplot(inner[0])
    a0.plot(r["p_t"], color=s12.INK, lw=1.2)
    a0.set_ylabel("p")
    a0.set_ylim(-0.05, 1.05)
    a0.set_xlim(0, len(r["p_t"]))
    a0.tick_params(labelbottom=False)
    a0.set_title(f"C  One plant, two unstable ends, inputs at nodes {', '.join(map(str, PLACEMENT[r['M']]))}")
    a1 = fig.add_subplot(inner[1], sharex=a0)
    series = (("open loop", r["open"], s15.C_OPEN), ("LQR family, picked on the fit", r["pick"], "#6250d6"), ("polytopic LMI", r["poly"], s12.C_IND))
    for name, row, c in series:
        trace = np.sqrt(np.convolve(np.mean(row["x"] ** 2, 1), np.ones(25) / 25, mode="valid"))
        a1.plot(np.arange(len(trace)) + 12, trace, color=c, lw=1.4, label=name)
    a1.set_yscale("log")
    a1.set_ylim(top=500)
    a1.set_xlabel("time step")
    a1.set_ylabel("rms x over nodes")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    gone = np.argmax(~np.isfinite(r["lqr"]["pooled"][1.0]["x"]).all(1))
    a1.text(0.98, 0.97, f"LQR on the pooled fit, R = 1, blows up by step {gone}: not drawn", transform=a1.transAxes, ha="right", va="top", color=s12.C_FELIX)
    out = os.path.join(ROOT, "figures", "16_few_inputs.png")
    fig.savefig(out, dpi=170)
    log(f"saved {out}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot()
    elif "--family" in sys.argv:
        with open(CACHE, "rb") as f:
            cached = pickle.load(f)
        add_family(cached)
        with open(CACHE, "wb") as f:
            pickle.dump(cached, f)
        summary(cached)
    elif "--summary" in sys.argv:
        with open(CACHE, "rb") as f:
            summary(pickle.load(f))
    else:
        main()
