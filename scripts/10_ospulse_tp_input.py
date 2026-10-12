"""Stage 3b -- TP-GDAR WITH INPUT on the real OS-Pulse ECoG (091114 M1).

Model (one-step, strictly causal, tpgdar timing convention):

    x_t = sum_{k=1..K} A_k x_{t-k} + sum_{q=1..Ku} b_q(p) u_{t-q} + e_t ,   e_t ~ N(0, Sigma)

A_k is SHARED across conditions and graph-supported (8-NN electrode graph, edge-wise
row-constrained, as in script 09).  The input operator B(p) = [b_1..b_Ku] (N x Ku) is
the thing that varies with the scheduling variable p, written as a tensor product

    B(p) = sum_r w_r(p) B_r          (HOSVD of the B tensor over p)

Why: sysID found A is stim-invariant (corr >= 0.997) and the stim effect lives in B,
whose gain falls ~20x from 10 to 150 Hz.  Script 09 put TP on A with B = 0 and lost.

Variant (i)  p = stimulation frequency (conds 1..9, 10..150 Hz, 1000 us).
    Baselines: B = 0 (pure GDAR), pooled single B (one LTI input kernel for all
    frequencies), per-condition B (no interpolation; upper bound).  TP rank r = 1..3.
    Evaluated (a) on held-out trials (train 0-39, test 40-59 of every condition) and
    (b) leave-one-frequency-out (LOFO): fit on 8 frequencies, predict the 9th; TP
    interpolates the weights w_r(p) in log-frequency.
Variant (ii) p = pre-stimulation band power (beta 13-30 Hz, theta 4-8 Hz; log power,
    mean over channels, 0.50-0.95 s, i.e. ending 50 ms before stim onset; filtered on
    the pre-stim segment only so the stim artifact cannot leak backward).  Per-trial
    state; tested on held-out trials with B (and optionally A) depending on it.

Metric: one-step held-out Gaussian log-likelihood per sample, each model with its own
innovation covariance from its own TRAIN residuals (as sysID/run_gdar_input_ll.py),
reported as Delta-ell vs B = 0 (or vs per-frequency B for variant ii), plus the
tpgdar one-step NRMSE (per-channel RMSE / std, averaged over channels).

Everything is computed from per-trial Gram matrices of W = [x-lags | u-lags | x_t | 1],
cached under cache/ (git-ignored).  Data: the local collab export (read-only use);
nothing is written to Z:.

Pulse train: sysID `make_pulse_train(..., align='exact')` -- the 2026-08-04 fixed
version (pulse k at round(k fs / f); the legacy fixed-integer-period train drifted
anti-phase at >= 40 Hz).

Run:  .venv\\Scripts\\python scripts/10_ospulse_tp_input.py
"""

from __future__ import annotations

import os
# small dense solves are ~100x slower under BLAS thread oversubscription on a busy box
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gdar_tp.tp import hosvd  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FIGDIR = os.path.join(ROOT, "figures")
CACHE = os.path.join(ROOT, "cache")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

DATADIR = os.environ.get("OSPULSE_DIR",
                         r"C:\Users\aditya\Documents\projects\collab_export_OSPulse")
SYSID = os.environ.get("SYSID_DIR",
                       r"C:\Users\aditya\Documents\projects\yazdanlab\sysID")
SESSION = "091114_M1"
FS = 3051.7578125 / 3          # export is already decimated x3 -> ~1017.25 Hz
STIM0_S, STIM1_S = 1.0, 2.0    # sysID stim window (do not use 1-6 s)
BAND = (0.1, 150.0)
SCALE = 1e6                    # V -> uV
K = 5                          # AR lags (sysID GDAR_K)
KU = 5                         # input lags (sysID K_U)
KNN = 8
CONDS = list(range(1, 10))     # 10..150 Hz at 1000 us
N_TRAIN = 40                   # trials 0..39 train, 40..59 test
PRE_WIN = (0.50, 0.95)         # s, pre-stim power window
BANDS = {"beta": (13.0, 30.0), "theta": (4.0, 8.0)}
RIDGE_REL = 1e-6               # Levenberg-style ridge, relative to Gram diagonal
SIG_SHRINK = 0.01              # Sigma shrinkage toward its diagonal


# ----------------------------------------------------------------------------- input
def _pulse_train_fallback(freq_hz, pw_us, n_samples, fs):
    """Verbatim logic of sysid_utils.make_pulse_train(align='exact')."""
    seq = np.zeros(n_samples)
    if freq_hz <= 0 or pw_us <= 0:
        return seq
    period = fs / float(freq_hz)
    pw_samp = max(1, int(round(float(pw_us) * 1e-6 * fs)))
    k = np.arange(int(np.floor((n_samples - 1) / period)) + 1)
    for start in np.round(k * period).astype(int):
        seq[start:min(start + pw_samp, n_samples)] = 1.0
    return seq


def make_pulse_train(freq_hz, pw_us, n_samples, fs):
    try:
        if SYSID not in sys.path:
            sys.path.insert(0, SYSID)
        import sysid_utils as su
        return su.make_pulse_train(freq_hz, pw_us, n_samples, fs, align="exact")
    except Exception:
        return _pulse_train_fallback(freq_hz, pw_us, n_samples, fs)


def knn_support(pos, k=KNN):
    N = pos.shape[0]
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    adj = np.zeros((N, N), bool)
    for i in range(N):
        adj[i, np.argsort(D[i])[:k]] = True
    adj = adj | adj.T
    return adj, adj | np.eye(N, dtype=bool)


# ----------------------------------------------------------------------------- grams
def build_grams():
    """Per-trial Grams of W=[Z(NK) | U(KU) | Y(N) | 1] over the stim window."""
    import pandas as pd
    meta = os.path.join(DATADIR, "metadata")
    pos = pickle.load(open(os.path.join(meta, "electrode_positions.pkl"), "rb"))
    pos = np.array([pos[i] for i in range(len(pos))])
    cond = pd.read_csv(os.path.join(meta, "conditions.csv"))
    bad = pickle.load(open(os.path.join(meta, "bad_channels.pkl"), "rb"))
    date, site = SESSION.split("_")
    bad_ch = set((np.array(bad.get(date, {}).get(site, []), int) - 1).tolist())
    good = np.array([i for i in range(96) if i not in bad_ch])
    N = len(good)
    path = os.path.join(CACHE, f"10_grams_{SESSION}_K{K}_Ku{KU}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return dict(G=z["G"], cond=z["cond"], trial=z["trial"], pw=dict(
            beta=z["p_beta"], theta=z["p_theta"]), freqs=z["freqs"], good=good,
            pos=pos[good], N=N, n=int(z["n"]))

    X = np.load(os.path.join(DATADIR, "data", f"{SESSION}.npy"), mmap_mode="r")
    stim0, stim1 = int(round(STIM0_S * FS)), int(round(STIM1_S * FS))
    nrow = stim1 - stim0
    sos = signal.butter(4, list(BAND), btype="bandpass", fs=FS, output="sos")
    sos_b = {b: signal.butter(4, list(f), btype="bandpass", fs=FS, output="sos")
             for b, f in BANDS.items()}
    p0, p1 = int(round(PRE_WIN[0] * FS)), int(round(PRE_WIN[1] * FS))
    NK = N * K
    D = NK + KU + N + 1
    Gs, cs, ts, pb, pt = [], [], [], [], []
    freqs = cond.stim_freq_Hz.values.astype(float)
    pws = cond.pulse_width_us.values.astype(float)
    t_lag = np.arange(stim0, stim1)
    for c in CONDS:
        t0 = time.time()
        raw = np.asarray(X[c][:, good, :], dtype=np.float64) * SCALE   # (tr, N, T)
        xf = signal.sosfiltfilt(sos, raw, axis=-1)
        u = np.zeros(raw.shape[-1])
        u[stim0:] = make_pulse_train(freqs[c], pws[c], raw.shape[-1] - stim0, FS)
        U = np.stack([u[t_lag - q] for q in range(1, KU + 1)], axis=1)  # (nrow, KU)
        pre = raw[:, :, :stim0]
        lp = {}
        for b in BANDS:
            pf = signal.sosfiltfilt(sos_b[b], pre, axis=-1)[:, :, p0:p1]
            lp[b] = np.log((pf ** 2).mean(-1)).mean(-1)                # (tr,)
        for tr in range(raw.shape[0]):
            x = xf[tr]                                                 # (N, T)
            Z = np.concatenate([x[:, t_lag - k].T for k in range(1, K + 1)], axis=1)
            W = np.concatenate([Z, U, x[:, t_lag].T, np.ones((nrow, 1))], axis=1)
            Gs.append(W.T @ W); cs.append(c); ts.append(tr)
        pb.append(lp["beta"]); pt.append(lp["theta"])
        print(f"  grams cond {c} ({freqs[c]:.0f} Hz): {raw.shape[0]} trials, "
              f"{time.time()-t0:.1f}s, pulses in window={int(u[stim0:stim1].sum())}",
              flush=True)
    G = np.stack(Gs)
    np.savez(path, G=G, cond=np.array(cs), trial=np.array(ts),
             p_beta=np.concatenate(pb), p_theta=np.concatenate(pt), freqs=freqs,
             n=nrow)
    return dict(G=G, cond=np.array(cs), trial=np.array(ts),
                pw=dict(beta=np.concatenate(pb), theta=np.concatenate(pt)),
                freqs=freqs, good=good, pos=pos[good], N=N, n=nrow)


# ----------------------------------------------------------------------------- fitting
class Fitter:
    def __init__(self, G, N, mask):
        self.G, self.N, self.mask = G, N, mask
        self.NK = N * K
        self.iz = slice(0, self.NK)
        self.iu = slice(self.NK, self.NK + KU)
        self.iy = slice(self.NK + KU, self.NK + KU + N)
        self.Gzz = G[:, self.iz, self.iz]
        self.Gzu = G[:, self.iz, self.iu]
        self.Guu = G[:, self.iu, self.iu]
        self.Gzy = G[:, self.iz, self.iy]
        self.Guy = G[:, self.iu, self.iy]
        self.zcols = [np.concatenate([k * N + np.where(mask[i])[0] for k in range(K)])
                      for i in range(N)]

    def fit(self, sel, F=None, H=None):
        """Least squares for shared-by-group A and feature-weighted B.

        sel : trial indices.  F : (len(sel), Gb) B-features (None -> B = 0).
        H : (len(sel), Ha) A-features (None -> a single shared A).
        Regressor = [kron(h, z), kron(f, u)].  Returns A (Ha,N,NK), B (Gb,N,KU).
        """
        N, NK = self.N, self.NK
        H = np.ones((len(sel), 1)) if H is None else H
        Ha = H.shape[1]
        Gb = 0 if F is None else F.shape[1]
        dim = Ha * NK + Gb * KU
        R = np.zeros((dim, dim)); r = np.zeros((dim, N))
        Gzz, Gzu, Guu = self.Gzz[sel], self.Gzu[sel], self.Guu[sel]
        Gzy, Guy = self.Gzy[sel], self.Guy[sel]
        td = lambda w, M: np.tensordot(w, M, axes=1)
        for h in range(Ha):
            a = slice(h * NK, (h + 1) * NK)
            r[a] = td(H[:, h], Gzy)
            for h2 in range(h, Ha):
                w = H[:, h] * H[:, h2]
                if np.any(w):
                    blk = td(w, Gzz); a2 = slice(h2 * NK, (h2 + 1) * NK)
                    R[a, a2] = blk; R[a2, a] = blk.T
            for g in range(Gb):
                w = H[:, h] * F[:, g]
                if np.any(w):
                    b = slice(Ha * NK + g * KU, Ha * NK + (g + 1) * KU)
                    blk = td(w, Gzu); R[a, b] = blk; R[b, a] = blk.T
        for g in range(Gb):
            b = slice(Ha * NK + g * KU, Ha * NK + (g + 1) * KU)
            r[b] = td(F[:, g], Guy)
            for g2 in range(g, Gb):
                w = F[:, g] * F[:, g2]
                if np.any(w):
                    b2 = slice(Ha * NK + g2 * KU, Ha * NK + (g2 + 1) * KU)
                    blk = td(w, Guu); R[b, b2] = blk; R[b2, b] = blk.T
        A = np.zeros((Ha, N, NK)); B = np.zeros((Gb, N, KU))
        bcols = np.arange(Ha * NK, dim)
        for i in range(N):
            cols = np.concatenate([h * NK + self.zcols[i] for h in range(Ha)] + [bcols])
            Ri = R[np.ix_(cols, cols)]
            Ri = Ri + RIDGE_REL * np.diag(np.diag(Ri) + 1e-12)
            beta = np.linalg.solve(Ri, r[cols, i])
            nz = len(self.zcols[i])
            for h in range(Ha):
                A[h, i, self.zcols[i]] = beta[h * nz:(h + 1) * nz]
            if Gb:
                B[:, i, :] = beta[Ha * nz:].reshape(Gb, KU)
        return A, B

    def fit_B_given_A(self, sel, A):
        """Per-set B with A fixed (A: N x NK): solve Guu B^T = Guy - Guz A^T."""
        Guu = self.Guu[sel].sum(0); Guy = self.Guy[sel].sum(0)
        Guz = self.Gzu[sel].sum(0).T
        rhs = Guy - Guz @ A.T
        Guu = Guu + RIDGE_REL * np.diag(np.diag(Guu) + 1e-12)
        return np.linalg.solve(Guu, rhs).T                      # (N, KU)

    def resid(self, idx, A, B):
        """Residual outer-product sums per trial for model (A: N x NK, B: N x KU)."""
        N = self.N
        M = np.zeros((N, self.G.shape[1]))
        M[:, self.iz] = -A; M[:, self.iu] = -B
        M[:, self.iy] = np.eye(N)
        return np.einsum("ia,tab,jb->tij", M, self.G[idx], M, optimize=True)


def sigma_from(S_sum, n):
    S = S_sum / n
    return (1 - SIG_SHRINK) * S + SIG_SHRINK * np.diag(np.diag(S))


def trial_ll(S_t, Sig, n):
    """Per-trial Gaussian log-likelihood from residual outer-product sums."""
    Si = np.linalg.inv(Sig)
    _, ld = np.linalg.slogdet(Sig)
    N = Sig.shape[0]
    return -0.5 * (np.einsum("ij,tji->t", Si, S_t) + n * (N * np.log(2 * np.pi) + ld))


def nrmse_from(fit, idx, S_sum):
    """tpgdar one_step_nrmse (per-channel RMSE/std, mean over channels) on idx."""
    G = fit.G[idx].sum(0)
    n = G[-1, -1]
    ysum = G[fit.iy, -1]
    yy = np.diag(G[fit.iy, fit.iy])
    var = yy / n - (ysum / n) ** 2
    return float(np.mean(np.sqrt(np.diag(S_sum) / n) / np.sqrt(var)))


def tp_truncate(Bt, r):
    """Rank-r HOSVD truncation of B tensor (P, N, KU) along the parameter mode."""
    P = Bt.shape[0]
    U, s, Vt = np.linalg.svd(Bt.reshape(P, -1), full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r], U, s, Vt


def tp_interp(Bt, logf_tr, logf_star, r):
    """TP: weights w_r(p) = U_r s_r, interpolated piecewise-linearly in log f
    (clamped at the ends), times the vertex/basis operators V_r."""
    P = Bt.shape[0]
    U, s, Vt = np.linalg.svd(Bt.reshape(P, -1), full_matrices=False)
    o = np.argsort(logf_tr)
    w = np.array([np.interp(logf_star, logf_tr[o], (U[:, j] * s[j])[o])
                  for j in range(r)])
    return (w @ Vt[:r]).reshape(Bt.shape[1:])


# ----------------------------------------------------------------------------- main
def main():
    t00 = time.time()
    D = build_grams()
    G, cond, trial, N, n = D["G"], D["cond"], D["trial"], D["N"], D["n"]
    freqs = D["freqs"]
    adj, mask = knn_support(D["pos"])
    fit = Fitter(G, N, mask)
    print(f"=== TP-GDAR with input: {SESSION}, N={N} good ch, 8-NN ({int(adj.sum()//2)} "
          f"edges), fs={FS:.1f} Hz, K={K}, Ku={KU}, window {STIM0_S}-{STIM1_S}s "
          f"({n} rows/trial) ===")
    ci = {c: j for j, c in enumerate(CONDS)}
    is_tr = trial < N_TRAIN
    tr_idx = np.where(is_tr)[0]
    te_idx = np.where(~is_tr)[0]
    onehot = lambda idx: np.eye(len(CONDS))[[ci[c] for c in cond[idx]]]
    fC = np.array([freqs[c] for c in CONDS])
    lf = np.log(fC)
    out = {}

    def eval_model(A, Bfun, tr_sel, te_sel):
        """Own-Sigma from train residuals; return per-trial LL on te, NRMSE per cond.
        Bfun(c) -> (N, KU) input operator for condition c (A: N x NK)."""
        S_tr = np.zeros((N, N)); ntr = 0
        for c in np.unique(cond[tr_sel]):
            s = tr_sel[cond[tr_sel] == c]
            S_tr += fit.resid(s, A, Bfun(c)).sum(0); ntr += n * len(s)
        Sig = sigma_from(S_tr, ntr)
        ll = np.zeros(len(te_sel)); nr = {}
        for c in np.unique(cond[te_sel]):
            m = cond[te_sel] == c
            St = fit.resid(te_sel[m], A, Bfun(c))
            ll[m] = trial_ll(St, Sig, n)
            nr[int(c)] = nrmse_from(fit, te_sel[m], St.sum(0))
        return ll, nr

    zeroB = np.zeros((N, KU))

    # ================= Variant (i)a: held-out trials =================
    print("\n[i-a] held-out trials (train 0-39 / test 40-59), p = stim frequency")
    A0, _ = fit.fit(tr_idx)
    Ap, Bp = fit.fit(tr_idx, F=np.ones((len(tr_idx), 1)))
    Ac, Bc = fit.fit(tr_idx, F=onehot(tr_idx))
    A0, Ap, Ac = A0[0], Ap[0], Ac[0]
    tpA = hosvd(Bc, n_param_modes=1, energy=1.0)
    sv = tpA.singular_values[0]
    print(f"  HOSVD singular values of B over freq (raw): {np.round(sv, 3)}")
    print(f"  normalised: {np.round(sv / sv[0], 4)}   energy rank-1 = "
          f"{sv[0]**2/np.sum(sv**2):.3f}")
    gain = np.linalg.norm(Bc.reshape(9, -1), axis=1)
    print(f"  ||B_c||_F vs freq: {np.round(gain, 3)}  (10->150 Hz ratio "
          f"{gain[0]/gain[-1]:.1f}x)")
    _, U_all, s_all, Vt_all = tp_truncate(Bc, 1)

    models = {"B=0": (A0, lambda c: zeroB), "pooled B": (Ap, lambda c: Bp[0]),
              "per-cond B": (Ac, lambda c: Bc[ci[c]])}
    for r in (1, 2, 3):
        Br = tp_truncate(Bc, r)[0].reshape(Bc.shape)
        models[f"TP r={r}"] = (Ac, (lambda Br: lambda c: Br[ci[c]])(Br))
    res_a = {k: eval_model(A, Bf, tr_idx, te_idx) for k, (A, Bf) in models.items()}
    ll0 = res_a["B=0"][0]
    print(f"  {'model':<11}{'dLL/sample vs B=0':>20}{'(SE over trials)':>18}"
          f"{'NRMSE':>9}  per-freq dLL/sample")
    for k, (ll, nr) in res_a.items():
        d = (ll - ll0) / n
        perf = [d[cond[te_idx] == c].mean() for c in CONDS]
        print(f"  {k:<11}{d.mean():>20.4f}{d.std(ddof=1)/np.sqrt(len(d)):>18.4f}"
              f"{np.mean(list(nr.values())):>9.4f}  " + " ".join(f"{v:+.3f}" for v in perf))
    out["ia"] = {k: dict(dll=((ll - ll0) / n), nrmse=nr) for k, (ll, nr) in res_a.items()}
    out["sv_freq"] = sv; out["gain"] = gain; out["U"] = U_all; out["s"] = s_all

    # ================= Variant (i)b: LOFO =================
    print("\n[i-b] leave-one-frequency-out (fit 8 freqs on trials 0-39; score held-out "
          "freq on trials 40-59)")
    names = ["B=0", "pooled B", "TP r=1", "TP r=2", "TP r=3", "full interp",
             "nearest B", "oracle B"]
    lofo = {k: [] for k in names}; lofo_nr = {k: [] for k in names}; lofo_sv = []
    for h in CONDS:
        trs = tr_idx[cond[tr_idx] != h]
        tes = te_idx[cond[te_idx] == h]
        tr_c = [c for c in CONDS if c != h]
        oh = np.eye(8)[[tr_c.index(c) for c in cond[trs]]]
        a0 = fit.fit(trs)[0][0]
        ap, bp = fit.fit(trs, F=np.ones((len(trs), 1)))
        ac, bc = fit.fit(trs, F=oh)
        ap, ac = ap[0], ac[0]
        lofo_sv.append(np.linalg.svd(bc.reshape(8, -1), compute_uv=False))
        lf_tr = np.log([freqs[c] for c in tr_c]); lfs = np.log(freqs[h])
        Bh = {"B=0": zeroB, "pooled B": bp[0]}
        for r in (1, 2, 3):
            Bh[f"TP r={r}"] = tp_interp(bc, lf_tr, lfs, r)
        Bh["full interp"] = tp_interp(bc, lf_tr, lfs, 8)
        Bh["nearest B"] = bc[int(np.argmin(np.abs(lf_tr - lfs)))]
        Bh["oracle B"] = fit.fit_B_given_A(tr_idx[cond[tr_idx] == h], ac)
        Amap = {"B=0": a0, "pooled B": ap}
        # Sigma: own train residuals; per-cond-family models share the per-cond fit's
        S_tr = {}
        for k in names:
            A = Amap.get(k, ac)
            if k in ("B=0", "pooled B"):
                Bf = lambda c, k=k: Bh[k]
            else:
                Bf = lambda c: bc[tr_c.index(c)]
            key = k if k in ("B=0", "pooled B") else "percond"
            if key not in S_tr:
                s = np.zeros((N, N))
                for c in tr_c:
                    s += fit.resid(trs[cond[trs] == c], A, Bf(c)).sum(0)
                S_tr[key] = sigma_from(s, n * len(trs))
            St = fit.resid(tes, A, Bh[k])
            lofo[k].append(trial_ll(St, S_tr[key], n))
            lofo_nr[k].append(nrmse_from(fit, tes, St.sum(0)))
    base = np.array(lofo["B=0"])
    print(f"  {'held f':>7}" + "".join(f"{k:>12}" for k in names[1:]) +
          "   (dLL/sample vs B=0, LOFO)")
    for j, h in enumerate(CONDS):
        print(f"  {freqs[h]:>7.0f}" + "".join(
            f"{(np.mean(lofo[k][j]) - base[j].mean())/n:>12.4f}" for k in names[1:]))
    D_lofo = {k: (np.array(lofo[k]) - base) / n for k in names}   # (9, 20)
    interior = slice(1, 8)
    for lab, sl in (("mean all", slice(0, 9)), ("mean 20-100", interior)):
        print(f"  {lab:>7}" + "".join(f"{D_lofo[k][sl].mean():>12.4f}" for k in names[1:]))
    print(f"  {'NRMSE':>7}" + "".join(f"{np.mean(lofo_nr[k]):>12.4f}" for k in names[1:])
          + f"   (B=0 NRMSE {np.mean(lofo_nr['B=0']):.4f})")
    d_tp = D_lofo["TP r=1"] - D_lofo["pooled B"]
    for r in (1, 2, 3):
        dd = (D_lofo[f"TP r={r}"] - D_lofo["pooled B"]).ravel()
        wins = sum(D_lofo[f"TP r={r}"][j].mean() > D_lofo["pooled B"][j].mean()
                   for j in range(9))
        print(f"  TP r={r} - pooled: {dd.mean():+.4f} nats/sample "
              f"(SE {dd.std(ddof=1)/np.sqrt(dd.size):.4f}), wins {wins}/9 held-out freqs")
    out["lofo"] = D_lofo; out["lofo_nrmse"] = lofo_nr
    out["lofo_sv"] = np.array(lofo_sv)

    # ================= Variant (ii): pre-stim band power =================
    print("\n[ii] p = pre-stim band power (per trial), held-out trials")
    out["ii"] = {}
    for band in BANDS:
        p = D["pw"][band]
        pz = np.zeros_like(p); pbin = np.zeros(len(p), int)
        for c in CONDS:
            m = cond == c; mt = m & is_tr
            mu, sd = p[mt].mean(), p[mt].std()
            pz[m] = (p[m] - mu) / sd
            q1, q2 = np.quantile(p[mt], [1 / 3, 2 / 3])
            pbin[m] = np.digitize(p[m], [q1, q2])
        ohc = lambda idx: onehot(idx)
        ohb = lambda idx: np.eye(3)[pbin[idx]]
        F_cb = lambda idx: np.eye(27)[[ci[c] * 3 + b for c, b in zip(cond[idx], pbin[idx])]]
        mods = {}
        # reference: per-frequency B (state-blind)
        mods["per-freq B"] = (fit.fit(tr_idx, F=ohc(tr_idx)), ohc, None)
        mods["B(f,bin)"] = (fit.fit(tr_idx, F=F_cb(tr_idx)), F_cb, None)
        Fls = lambda idx: np.hstack([ohc(idx), pz[idx][:, None]])
        mods["B_f + p*D"] = (fit.fit(tr_idx, F=Fls(tr_idx)), Fls, None)
        Flc = lambda idx: np.hstack([ohc(idx), ohc(idx) * pz[idx][:, None]])
        mods["B_f + p*D_f"] = (fit.fit(tr_idx, F=Flc(tr_idx)), Flc, None)
        mods["A(bin)+B_f"] = (fit.fit(tr_idx, F=ohc(tr_idx), H=ohb(tr_idx)), ohc, ohb)
        Fl1 = lambda idx: np.hstack([ohc(idx), ohc(idx) * pz[idx][:, None]])
        Hl = lambda idx: np.stack([np.ones(len(idx)), pz[idx]], 1)
        mods["A+p*A1, B_f+p*D_f"] = (fit.fit(tr_idx, F=Fl1(tr_idx), H=Hl(tr_idx)), Fl1, Hl)
        # TP over the power-bin mode of the B(f,bin) tensor
        Bcb = mods["B(f,bin)"][0][1].reshape(9, 3, N, KU)
        tpb = hosvd(Bcb, n_param_modes=2, energy=1.0)
        sv_b = tpb.singular_values[1]

        def per_trial_ll(A, B, Ff, Hf, idx_sets):
            """LL per trial with trial-specific A_t, B_t (features)."""
            out_ll = []
            for idx in idx_sets:
                F = Ff(idx); Hh = np.ones((len(idx), 1)) if Hf is None else Hf(idx)
                S = np.zeros((len(idx), N, N))
                for t in range(len(idx)):
                    At = np.tensordot(Hh[t], A, axes=1)
                    Bt = np.tensordot(F[t], B, axes=1)
                    S[t] = fit.resid(idx[t:t + 1], At, Bt)[0]
                out_ll.append(S)
            return out_ll
        res = {}
        for k, ((A, B), Ff, Hf) in mods.items():
            S_tr, S_te = per_trial_ll(A, B, Ff, Hf, [tr_idx, te_idx])
            Sig = sigma_from(S_tr.sum(0), n * len(tr_idx))
            res[k] = trial_ll(S_te, Sig, n)
        # TP rank-1 in power mode (B(f,bin) projected on the leading power-mode vector)
        Ub = tpb.factors[1] if tpb.factors[1].shape[1] >= 1 else None
        U1 = np.linalg.svd(np.moveaxis(Bcb, 1, 0).reshape(3, -1), full_matrices=False)[0][:, :1]
        Bcb1 = np.einsum("pb,cbnq->cpnq", U1 @ U1.T, Bcb).reshape(27, N, KU)
        A_cb = mods["B(f,bin)"][0][0]
        S_tr, S_te = per_trial_ll(A_cb, Bcb1, F_cb, None, [tr_idx, te_idx])
        res["TP power r=1"] = trial_ll(S_te, sigma_from(S_tr.sum(0), n * len(tr_idx)), n)
        ref = res["per-freq B"]
        r_pb = np.corrcoef(p[cond > 0], np.array([freqs[c] for c in cond]))[0, 1]
        print(f"  --- {band} {BANDS[band]} Hz: log-power range "
              f"{p.min():.2f}..{p.max():.2f}; power-mode HOSVD sv (norm) "
              f"{np.round(sv_b / sv_b[0], 3)}")
        print(f"  {'model':<20}{'dLL/sample vs per-freq B':>26}{'SE':>9}")
        dres = {}
        for k, ll in res.items():
            d = (ll - ref) / n
            dres[k] = d
            print(f"  {k:<20}{d.mean():>26.4f}{d.std(ddof=1)/np.sqrt(len(d)):>9.4f}")
        out["ii"][band] = dict(dll=dres, sv=sv_b, p=p, pbin=pbin)
    with open(os.path.join(CACHE, f"10_results_{SESSION}.pkl"), "wb") as f:
        pickle.dump(out, f)
    figures(out, freqs, n)
    print(f"\ndone in {time.time()-t00:.0f}s; figures -> {FIGDIR}")


# ----------------------------------------------------------------------------- figures
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
     "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"


def _style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "lines.linewidth": 1.5,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})


def figures(out, freqs, n):
    _style()
    fC = np.array([freqs[c] for c in CONDS])
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
    # (a) HOSVD spectrum of B over freq
    sv = out["sv_freq"]; a = ax[0, 0]
    a.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], "o-", color=C[0], ms=4,
               label="all 9 freqs")
    for j, s in enumerate(out["lofo_sv"]):
        a.semilogy(np.arange(1, len(s) + 1), s / s[0], "-", color="0.75", lw=0.6,
                   zorder=0, label="LOFO (8 freqs)" if j == 0 else None)
    a.set_xlabel("component r"); a.set_ylabel("normalised singular value")
    a.set_title("a  HOSVD of B tensor over stim frequency", loc="left")
    a.legend(frameon=False); a.grid(alpha=0.25, lw=0.5)
    # (b) weights w_r(f)
    a = ax[0, 1]; U, s = out["U"], out["s"]
    for r in range(3):
        w = U[:, r] * s[r]
        if w[0] < 0 and r == 0:
            w = -w
        a.plot(fC, w, "o-", color=C[r], ms=4, label=f"w$_{r+1}$(f)")
    a.set_xscale("log"); a.set_xticks(fC); a.set_xticklabels([f"{int(f)}" for f in fC])
    a.minorticks_off(); a.axhline(0, color=MUTED, lw=0.5)
    a.set_xlabel("stim frequency (Hz)"); a.set_ylabel("TP weight")
    a.set_title("b  TP weight functions", loc="left"); a.legend(frameon=False)
    # (c) held-out trials dLL per freq
    a = ax[1, 0]; keys = ["pooled B", "per-cond B", "TP r=1", "TP r=2"]
    te_c = None
    x = np.arange(9); w = 0.2
    for j, k in enumerate(keys):
        d = out["ia"][k]["dll"].reshape(9, -1)   # te trials ordered by cond
        a.bar(x + (j - 1.5) * w, d.mean(1), w * 0.9, color=C[j], label=k)
    a.set_xticks(x); a.set_xticklabels([f"{int(f)}" for f in fC])
    a.axhline(0, color=MUTED, lw=0.5)
    a.set_xlabel("stim frequency (Hz)"); a.set_ylabel("ΔLL / sample vs B=0 (nats)")
    a.set_title("c  held-out trials", loc="left"); a.legend(frameon=False, ncol=2)
    # (d) LOFO
    a = ax[1, 1]; keys = ["pooled B", "TP r=1", "TP r=2", "oracle B"]
    cols = [C[0], C[2], C[3], "0.6"]
    for j, k in enumerate(keys):
        a.bar(x + (j - 1.5) * w, out["lofo"][k].mean(1), w * 0.9, color=cols[j], label=k)
    a.set_xticks(x); a.set_xticklabels([f"{int(f)}" for f in fC])
    a.axhline(0, color=MUTED, lw=0.5)
    a.set_xlabel("held-out stim frequency (Hz)"); a.set_ylabel("ΔLL / sample vs B=0 (nats)")
    a.set_title("d  leave-one-frequency-out", loc="left"); a.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"10_ospulse_tp_input.{ext}"), dpi=200)
    plt.close(fig)

    # variant (ii)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for j, band in enumerate(out["ii"]):
        a = ax[j]; dres = out["ii"][band]["dll"]
        ks = [k for k in dres if k != "per-freq B"]
        m = [dres[k].mean() for k in ks]
        se = [dres[k].std(ddof=1) / np.sqrt(len(dres[k])) for k in ks]
        a.barh(np.arange(len(ks)), m, xerr=se, color=C[j], height=0.6,
               error_kw=dict(lw=0.8, ecolor=MUTED))
        a.set_yticks(np.arange(len(ks))); a.set_yticklabels(ks)
        a.invert_yaxis(); a.axvline(0, color=MUTED, lw=0.5)
        sv = out["ii"][band]["sv"]
        a.set_xlabel("ΔLL / sample vs per-freq B (nats)")
        a.set_title(f"{'ab'[j]}  p = pre-stim {band} power  (power-mode sv "
                    f"{', '.join(f'{v:.2f}' for v in sv / sv[0])})", loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"10_ospulse_prestim_state.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
