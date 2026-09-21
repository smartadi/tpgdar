"""Stage 3 -- TP-GDAR on the real OS-Pulse ECoG dataset.

Monkey Jalapeno, 96-ch uECoG (M1 / S1), microstimulation parameter study.  We use
**stimulation frequency** (conditions 1-9: 10..150 Hz) as the scheduling variable
p, and ask the Stage-2b question on real data: as stim frequency changes, how does
the effective communication graph reshape, is that dependence low-rank, and do the
recovered vertex graphs respect the array geometry?

Anatomical support: **8 nearest neighbours** by electrode distance (per the array
geometry).  Phase-locked stimulation artifact is removed by subtracting the
per-condition trial average (we model the *induced* ongoing activity).

Data path is auto-detected; override with $OSPULSE_DIR.
Run:  python scripts/09_ospulse_real.py
"""

from __future__ import annotations

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdar_tp import metrics
from gdar_tp.gdar import GDARModel, fit_gdar
from gdar_tp.tp import convex_hull, hosvd

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)

DATADIR = os.environ.get("OSPULSE_DIR",
    r"C:\Users\aditya\Documents\projects\collab_export_OSPulse")
SESSION = "091114_M1"
FS = 3051.76 / 3
K = 3
DECIM = 2
KNN = 8


def knn_support(pos, k=KNN):
    """Symmetric k-nearest-neighbour adjacency + support mask from positions."""
    N = pos.shape[0]
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((N, N), bool)
    for i in range(N):
        nn = np.argsort(D[i])[:k]
        adj[i, nn] = True
    adj = adj | adj.T                        # symmetrize
    mask = adj | np.eye(N, dtype=bool)
    return adj, mask


def gap_rank(s, kmax=12):
    r = s[:kmax] / s[0]
    ratios = r[1:] / r[:-1]
    k = int(np.argmin(ratios[1:])) + 1
    return k + 1


def load_condition(X, c, good, t0=1.5, t1=5.5):
    """Induced (trial-mean-subtracted), z-scored, decimated series for condition c."""
    lo, hi = int(t0 * FS), int(t1 * FS)
    seg = np.asarray(X[c][:, good, lo:hi], dtype=float)   # (trials, N, samples)
    seg = seg - seg.mean(axis=0, keepdims=True)           # remove phase-locked artifact
    seg = seg[:, :, ::DECIM]
    ntr, N, ns = seg.shape
    Xc = seg.transpose(0, 2, 1).reshape(ntr * ns, N)      # concat trials over time
    Xc = Xc - Xc.mean(0, keepdims=True)
    Xc = Xc / (Xc.std(0, keepdims=True) + 1e-9)
    return Xc


def main():
    import pandas as pd
    meta = os.path.join(DATADIR, "metadata")
    pos = pickle.load(open(os.path.join(meta, "electrode_positions.pkl"), "rb"))
    pos = np.array([pos[i] for i in range(len(pos))])
    cond = pd.read_csv(os.path.join(meta, "conditions.csv"))
    bad = pickle.load(open(os.path.join(meta, "bad_channels.pkl"), "rb"))
    date, site = SESSION.split("_")
    bad_ch = np.array(bad.get(date, {}).get(site, []), dtype=int) - 1   # 1- -> 0-indexed
    good = np.array([i for i in range(96) if i not in set(bad_ch.tolist())])
    pos_g = pos[good]
    adj, mask = knn_support(pos_g, KNN)
    N = len(good)

    print("=== Stage 3: TP-GDAR on real OS-Pulse ECoG ===")
    print(f"session {SESSION}: {N} good channels ({len(bad_ch)} bad dropped), "
          f"8-NN support ({int(adj.sum()//2)} edges), fs={FS/DECIM:.0f}Hz (decim {DECIM}), K={K}")

    freqs = cond.stim_freq_Hz.values[1:10].astype(float)   # conditions 1..9
    X = np.load(os.path.join(DATADIR, "data", f"{SESSION}.npy"), mmap_mode="r")

    series = {}
    A_list = []
    for ci in range(1, 10):
        Xc = load_condition(X, ci, good)
        series[ci] = Xc
        m = fit_gdar(Xc, K=K, support=mask, param="edgewise", ridge=1e-2)
        A_list.append(np.stack(m.A, axis=-1))
    coeffs = np.stack(A_list, axis=0)                      # (9, N, N, K)

    # --- how many modes does the frequency dependence have? ---
    tp = hosvd(coeffs, n_param_modes=1, energy=1.0)
    s = tp.singular_values[0]
    r_gap = gap_rank(s)
    print(f"\nstim-frequency sweep: {freqs.astype(int)} Hz")
    print(f"HOSVD singular values (norm): {np.round(s[:8]/s[0],4)}")
    print(f"gap-detected rank (effective # of graphs): {r_gap}")

    # --- recover the vertex graphs ---
    tp_r = hosvd(coeffs, n_param_modes=1, max_rank=r_gap, energy=1.0)
    poly = convex_hull(tp_r.reconstruct(), n_param_modes=1, method="spa",
                       n_vertices=r_gap)
    vert_f = freqs[poly.vertex_grid_index]
    print(f"recovered {poly.n_vertices} vertex graphs, anchored near {vert_f.astype(int)} Hz")

    # --- leave-one-frequency-out: does TP beat a pooled (freq-blind) GDAR? ---
    print("\nleave-one-frequency-out held-out 1-step NRMSE:")
    print(f"{'held f':>8}{'pooled':>10}{'TP-interp':>11}")
    pooled_errs, tp_errs = [], []
    for hi in range(9):
        train = [j for j in range(9) if j != hi]
        # pooled: one GDAR on all training conditions concatenated
        Xpool = np.vstack([series[j + 1] for j in train])
        mp = fit_gdar(Xpool, K=K, support=mask, param="edgewise", ridge=1e-2)
        ep = metrics.one_step_nrmse(mp, series[hi + 1])
        # TP-interp: rank-r reconstruction, interpolate coeffs to held-out freq
        c_tr = coeffs[train]
        tp_tr = hosvd(c_tr, n_param_modes=1, max_rank=min(r_gap, len(train)), energy=1.0)
        rec = tp_tr.reconstruct()
        A_int = _interp(freqs[train], rec, freqs[hi])
        mi = GDARModel(A=[A_int[..., k] for k in range(K)], B=np.zeros((N, 0)),
                       K=K, param="edgewise", support=mask)
        ei = metrics.one_step_nrmse(mi, series[hi + 1])
        pooled_errs.append(ep); tp_errs.append(ei)
        print(f"{int(freqs[hi]):>8}{ep:>10.4f}{ei:>11.4f}")
    print(f"{'mean':>8}{np.mean(pooled_errs):>10.4f}{np.mean(tp_errs):>11.4f}  "
          f"(TP {'wins' if np.mean(tp_errs)<np.mean(pooled_errs) else 'ties/loses'} "
          f"by {np.mean(pooled_errs)-np.mean(tp_errs):+.4f})")

    # --- does estimated coupling respect array geometry? ---
    C = np.sqrt((coeffs.mean(0) ** 2).sum(-1)); np.fill_diagonal(C, 0)
    D = np.linalg.norm(pos_g[:, None] - pos_g[None], axis=2)
    off = ~np.eye(N, dtype=bool)
    rho = np.corrcoef(C[off], -D[off])[0, 1]
    print(f"\ncoupling strength vs -distance corr: {rho:.3f} "
          f"(positive => nearby channels couple more, as anatomy predicts)")

    _figures(pos_g, adj, tp, poly, coeffs, freqs, C, D, off)
    print(f"\nfigures -> {FIGDIR}")


def _interp(fs, coeffs, f_star):
    P = coeffs.shape[0]; flat = coeffs.reshape(P, -1)
    order = np.argsort(fs)
    out = np.array([np.interp(f_star, fs[order], flat[order, j]) for j in range(flat.shape[1])])
    return out.reshape(coeffs.shape[1:])


def _figures(pos, adj, tp, poly, coeffs, freqs, C, D, off):
    # array + 8-NN graph
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if adj[i, j]:
                ax[0].plot(pos[[i, j], 0], pos[[i, j], 1], "-", color="0.8", lw=0.6, zorder=1)
    ax[0].scatter(pos[:, 0], pos[:, 1], s=30, c="#065A82", zorder=2)
    ax[0].set_title("96-ch array, 8-NN support"); ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)")

    s = tp.singular_values[0]
    ax[1].semilogy(np.arange(1, len(s) + 1), s / s[0], "o-")
    ax[1].axvline(poly.n_vertices + 0.5, color="C3", ls=":",
                  label=f"detected rank = {poly.n_vertices}")
    ax[1].set_title("HOSVD spectrum vs stim frequency")
    ax[1].set_xlabel("component"); ax[1].set_ylabel("norm. singular value")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].scatter(D[off], C[off], s=4, alpha=0.3, c="#1C7293")
    ax[2].set_title("coupling vs inter-electrode distance")
    ax[2].set_xlabel("distance (mm)"); ax[2].set_ylabel("|coupling|"); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "09_ospulse_overview.png"), dpi=130)

    # vertex coupling maps on the array
    V = poly.vertex_systems
    nv = V.shape[0]
    fig2, axes = plt.subplots(1, nv, figsize=(3.4 * nv, 3.4))
    axes = np.atleast_1d(axes)
    vf = freqs[poly.vertex_grid_index]
    for v in range(nv):
        A0 = V[v, :, :, 0].copy(); np.fill_diagonal(A0, 0)
        strength = np.abs(A0).sum(1)
        sc = axes[v].scatter(pos[:, 0], pos[:, 1], c=strength, s=45, cmap="magma")
        axes[v].set_title(f"vertex {v} (~{int(vf[v])} Hz)"); axes[v].set_aspect("equal")
        axes[v].set_xticks([]); axes[v].set_yticks([])
        fig2.colorbar(sc, ax=axes[v], fraction=0.046)
    fig2.suptitle("recovered vertex graphs: per-channel coupling strength on the array")
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGDIR, "09_ospulse_vertices.png"), dpi=130)


if __name__ == "__main__":
    main()
