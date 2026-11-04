r"""Consensus demo: every node of the 9-node network is driven to the same moving target.

Same plants, fits and vertices as script 15 (cache/15_nine_node.pkl, an input at every node).
Exact consensus needs an input at every node: M inputs can hold only M nodes at a chosen value.
To reach the target at any p, the gain also acts on the running error at each node:

    z(t+1) = z(t) + r(t) - x(t),    u(t) = F_x xi(t) + F_z z(t)

xi and z together still form a polytope, so the same LMI applies, one gain for all p:
  polytopic   LMI on the fitted vertices, rate alpha
  pooled LQR  LQR on a single p-blind GDAR fitted to the same data
  no control  u = 0

Loop closed on the true plant only.  p moves through its range while the target steps.

Run:  .venv\Scripts\python scripts/17_consensus_demo.py [--movie | --mp4]
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s15 = importlib.import_module("15_nine_node_control")
s12 = s15.s12
from gdar_tp.control import _solve  # noqa: E402
from gdar_tp.graph import grid_graph  # noqa: E402
from gdar_tp.sym_tp import symmetric_basis, theta_of  # noqa: E402

K, N = s15.K, s15.N
CACHE = os.path.join(ROOT, "cache", "17_consensus.pkl")
MOVIE = os.path.join(s12.FIG, "17_consensus.gif")
ALPHA = 0.95
T = 1000
STEPS = ((0, 0.0), (100, 3.0), (350, -3.0), (600, 3.0), (850, 0.0))     # (time, target)
log = s15.log


def target():
    r = np.zeros(T)
    for t0, value in STEPS:
        r[t0:] = value
    return r


def p_schedule():
    return s15.schedule(T)


def augment(th):
    """Companion form with one running-error state per node."""
    Ac, Bc = s15.comp(th)
    nx, M = Ac.shape[0], Bc.shape[1]
    C = np.hstack([np.eye(N), np.zeros((N, nx - N))])
    A = np.block([[Ac, np.zeros((nx, N))], [-C, np.eye(N)]])
    return A, np.vstack([Bc, np.zeros((N, M))])


def lmi_gain(thetas, alpha=ALPHA):
    import cvxpy as cp
    comps = [augment(th) for th in thetas]
    B = np.mean([c[1] for c in comps], 0)
    n, m = B.shape
    Q, Y = cp.Variable((n, n), symmetric=True), cp.Variable((m, n))
    cons = [Q >> np.eye(n)]
    for A, _ in comps:
        M = A @ Q + B @ Y
        cons.append(cp.bmat([[alpha * Q, M.T], [M, alpha * Q]]) >> 1e-6 * np.eye(2 * n))
    prob = cp.Problem(cp.Minimize(cp.trace(Q)), cons)
    _solve(prob, None)
    if prob.status not in ("optimal", "optimal_inaccurate") or Q.value is None:
        return None, None
    return Y.value @ np.linalg.inv(Q.value), Q.value


def lqr_gain(th, q_error=0.1, r=1.0):
    A, B = augment(th)
    Qw = np.diag(np.r_[np.ones(K * N), q_error * np.ones(N)])
    X = solve_discrete_are(A, B, Qw, r * np.eye(B.shape[1]))
    return -np.linalg.solve(r * np.eye(B.shape[1]) + B.T @ X @ B, B.T @ X @ A)


def run(ph0, ph1, R, F, seed, stop=1e3):
    """True plant.  Returns x (T, N), u (T, M).  x is clipped to NaN once it passes ``stop``."""
    rng = np.random.default_rng(7000 + seed)
    p_t, r_t = p_schedule(), target()
    e = s15.NOISE * rng.standard_normal((T, N))
    M = R.shape[1] - s15.N_LAG
    x, u, z = np.zeros((T, N)), np.zeros((T, M)), np.zeros(N)
    for t in range(K - 1, T - 1):
        xi = np.concatenate([x[t - k] for k in range(K)])
        if F is not None:
            u[t] = F @ np.r_[xi, z]
        z = z + r_t[t] - x[t]
        x[t + 1] = s15.truth_at(ph0, ph1, R, p_t[t]) @ np.r_[xi, u[t]] + e[t]
        if np.abs(x[t + 1]).max() > stop:
            x[t + 1:], u[t + 1:] = np.nan, np.nan
            break
    return x, u


def worst_rho(ph0, ph1, R, F):
    out = 0.0
    for p in s15.P_DENSE:
        A, B = augment(s15.truth_at(ph0, ph1, R, p))
        out = max(out, np.abs(np.linalg.eigvals(A + (B @ F if F is not None else 0))).max())
    return float(out)


def certificate(ph0, ph1, R, F, Q):
    """Largest eigenvalue of A_cl' P A_cl - P on the true plant across p (< 0 certifies)."""
    P = np.linalg.inv(Q)
    worst = -np.inf
    for p in s15.P_DENSE:
        A, B = augment(s15.truth_at(ph0, ph1, R, p))
        Acl = A + B @ F
        worst = max(worst, np.linalg.eigvalsh(Acl.T @ P @ Acl - P).max())
    return float(worst)


def scores(x, u):
    """Held windows: the last 40 steps before each target change."""
    if not np.isfinite(x).all():
        return dict(diverged=True, err=np.nan, spread=np.nan, rms_u=np.nan)
    r_t = target()
    idx = np.concatenate([np.arange(t1 - 40, t1) for (t1, _) in STEPS[1:]])
    d = x[idx] - r_t[idx, None]
    return dict(diverged=False, err=float(np.sqrt(np.mean(d ** 2))), spread=float(np.sqrt(np.mean(np.var(x[idx], axis=1)))),
                rms_u=float(np.sqrt(np.mean(u ** 2))))


def one(job):
    name, seed = job
    with open(s15.CACHE, "rb") as f:
        out = pickle.load(f)
    A = out["A"]
    r = out["scenarios"][name][seed]
    R = symmetric_basis(A, K, node_inputs=True)
    verts = [theta_of(ph, R, N) for ph in r["poly"]["phi"]]
    F_poly, Q = lmi_gain(verts)
    F_lqr = lqr_gain(theta_of(r["phi_pool"], R, N))
    res = dict(name=name, seed=seed, feasible=F_poly is not None, sims={}, rho={}, score={})
    res["cert"] = certificate(r["ph0"], r["ph1"], R, F_poly, Q) if F_poly is not None else np.nan
    for k, F in (("no control", None), ("pooled LQR", F_lqr), ("polytopic", F_poly)):
        if k == "polytopic" and F is None:
            continue
        x, u = run(r["ph0"], r["ph1"], R, F, seed)
        res["rho"][k] = worst_rho(r["ph0"], r["ph1"], R, F)
        res["score"][k] = scores(x, u)
        if seed == s15.EXAMPLE:
            res["sims"][k] = (x, u)
    return res


def main():
    jobs = [(c, seed) for c in s15.SCENARIOS for seed in range(s15.N_PLANTS)]
    with ProcessPoolExecutor(14) as ex:
        res = list(ex.map(one, jobs))
    with open(CACHE, "wb") as f:
        pickle.dump(res, f)
    for c in s15.SCENARIOS:
        rows = [q for q in res if q["name"] == c]
        log(f"===== {c}: LMI feasible {sum(q['feasible'] for q in rows)}/{len(rows)}, "
            f"certificate holds on true plant at rate 1: {sum(q['cert'] < 0 for q in rows)}/{len(rows)}")
        for k in ("no control", "pooled LQR", "polytopic"):
            have = [q for q in rows if k in q["score"]]
            wr = np.array([q["rho"][k] for q in have])
            div = sum(q["score"][k]["diverged"] for q in have)
            med = {m: np.nanmedian([q["score"][k][m] for q in have]) for m in ("err", "spread", "rms_u")}
            log(f"  {k}: stable at all p {int((wr < 1).sum())}/{len(have)}, worst rho {np.median(wr):.3f}, diverged {div}; "
                f"held windows, median: rms error {med['err']:.3f}, spread across nodes {med['spread']:.3f}, rms input {med['rms_u']:.3f}")


def movie(case="two unstable", stride=4, fps=20):
    """Animated GIF: graph colours chase the target ring, with and without control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from PIL import Image
    with open(CACHE, "rb") as f:
        res = pickle.load(f)
    with open(s15.CACHE, "rb") as f:
        out = pickle.load(f)
    q = next(r for r in res if r["name"] == case and r["seed"] == s15.EXAMPLE)
    plant = out["scenarios"][case][s15.EXAMPLE]
    A = out["A"]
    R = symmetric_basis(A, K, node_inputs=True)
    g = grid_graph(3, 3)
    xy = np.column_stack([g.coords()[:, 1], -g.coords()[:, 0]]).astype(float)
    iu, ju = np.where(np.triu(A, 1))
    p_t, r_t = p_schedule(), target()
    x_off = np.nan_to_num(q["sims"]["no control"][0], nan=1e3)
    x_on = q["sims"]["polytopic"][0]
    th_t = [sum(s15.truth_at(plant["ph0"], plant["ph1"], R, p)[:, k * N:(k + 1) * N] for k in range(K)) for p in p_t]
    w_max = max(np.abs(S[iu, ju]).max() for S in th_t)
    cmap = LinearSegmentedColormap.from_list("bwr_", [s12.C_FELIX, "#f0efec", "#e34948"], N=19)    # stepped, so the GIF keeps it
    norm = Normalize(-4.5, 4.5)
    s12._style()
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.2, 1.5, 0.55])
    panels = []
    for col, (title, x) in enumerate((("no control", x_off), ("with the polytopic gain", x_on))):
        ax = fig.add_subplot(gs[0, col])
        lc = LineCollection([(xy[i], xy[j]) for i, j in zip(iu, ju)], colors="#b5b3ad", linewidths=2, zorder=1)
        ax.add_collection(lc)
        ring = ax.scatter(*xy.T, s=1150, c=np.zeros(N), cmap=cmap, norm=norm, edgecolors="#555555", linewidths=0.8, zorder=2)
        dots = ax.scatter(*xy.T, s=480, c=np.zeros(N), cmap=cmap, norm=norm, edgecolors="#ffffff", linewidths=1.6, zorder=3)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-2.5, 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontweight="bold")
        tr = fig.add_subplot(gs[1, col])
        tr.plot(x, color="#7a7f88", lw=0.7, alpha=0.8)
        tr.plot(r_t, color="#222222", lw=1.6, ls="--")
        tr.set_xlim(0, T)
        tr.set_ylim(-6.5, 6.5)
        tr.set_xticks([])
        tr.set_yticks([-3, 0, 3])
        tr.grid(False)
        if col == 0:
            tr.set_ylabel("activity")
        cursor = tr.axvline(0, color="#e34948", lw=1.6)
        pax = fig.add_subplot(gs[2, col])
        pax.plot(p_t, color=s12.C_FELIX, lw=1.6)
        pax.set_xlim(0, T)
        pax.set_ylim(-0.1, 1.1)
        pax.set_xticks([])
        pax.set_yticks([0, 1])
        pax.grid(False)
        if col == 0:
            pax.set_ylabel("p")
        cursor2 = pax.axvline(0, color="#e34948", lw=1.6)
        panels.append((lc, dots, ring, cursor, cursor2, x))
    frames = []
    for t in range(0, T, stride):
        for lc, dots, ring, cursor, cursor2, x in panels:
            dots.set_array(np.clip(x[t], -4.5, 4.5))
            ring.set_array(np.full(N, r_t[t]))
            lc.set_linewidths(0.6 + 5.5 * np.abs(th_t[t][iu, ju]) / w_max)
            cursor.set_xdata([t, t])
            cursor2.set_xdata([t, t])
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    sample = frames[::6]        # one palette from the whole movie, so no colour is lost in any frame
    montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, f in enumerate(sample):
        montage.paste(f, (0, i * f.height))
    palette = montage.quantize(colors=255, dither=Image.Dither.NONE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(MOVIE, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)
    log(f"wrote {MOVIE}, {len(frames)} frames, {os.path.getsize(MOVIE) / 1e6:.1f} MB")


def mp4(ms_per_frame=50):
    """MP4 and poster from the GIF, through desktop PowerPoint (scripts/movie_to_mp4.ps1).

    PowerPoint for the web plays embedded video but not animated GIFs.
    """
    import subprocess
    import tempfile

    from lxml import etree
    from PIL import Image
    from pptx import Presentation
    from pptx.util import Inches
    g = Image.open(MOVIE)
    tmp = tempfile.mkdtemp()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    ns = "http://schemas.openxmlformats.org/presentationml/2006/main"
    for f in range(g.n_frames):
        g.seek(f)
        path = os.path.join(tmp, f"{f:04d}.png")
        g.convert("RGB").save(path)
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(path, 0, 0, width=prs.slide_width, height=prs.slide_height)
        tr = etree.Element(f"{{{ns}}}transition", advClick="0", advTm=str(ms_per_frame))
        s._element.find(f"{{{ns}}}clrMapOvr").addnext(tr)
    deck = os.path.join(tmp, "frames.pptx")
    prs.save(deck)
    out = MOVIE.replace(".gif", ".mp4")
    g.seek(g.n_frames * 3 // 10)        # a frame that shows the point, for the still view
    g.convert("RGB").save(MOVIE.replace(".gif", "_poster.png"))
    ps1 = os.path.join(ROOT, "scripts", "movie_to_mp4.ps1")
    res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1, deck, out], capture_output=True, text=True)
    log(res.stdout.strip(), res.stderr.strip()[:300])
    log(f"wrote {out}, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    if "--movie" in sys.argv:
        movie()
    elif "--mp4" in sys.argv:
        mp4()
    else:
        main()
