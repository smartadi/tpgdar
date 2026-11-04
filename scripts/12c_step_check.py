"""Is there a step artefact in lfp noart at target onset?  Does masking it change the rank-2 gains on 11-12?"""
import importlib
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
s12b = importlib.import_module("12b_tp_vs_gdar_robust")
s13 = importlib.import_module("13_tp_vertices_ios")
from gdar_tp.sym_tp import fit_gls_pooled, fit_sym_tp, polytope_vertices, symmetric_basis, theta_of, _quad

for date in ("2021-11-12", "2021-08-23"):
    z = np.load(s12.SEGS.replace("2021-11-12", date), allow_pickle=True)
    print(date, "fields", list(z.keys())[:14])
    keep = ~np.isin(z["channels"], s12.BAD)
    for field in ("noart", "lfp"):
        if field not in z:
            print("  no field", field)
            continue
        X = z[field][keep].astype(float)
        d = np.abs(np.diff(X, axis=1))
        med = np.median(d, axis=(0, 2))
        top = np.argsort(med)[::-1][:6]
        t = np.arange(X.shape[1]) - 500
        print(f"  {field}: typical |step| {np.median(med):.2f}; largest median steps at " +
              ", ".join(f"{t[i]}->{t[i + 1]} ms: {med[i]:.1f}" for i in sorted(top)))

D = s12.load()
lab = s13.warped_labels(D)
t = D["t"]
K = s13.K
fo = s12.folds(D["ang"])
Z = s12.evoked_design(D)
N = D["X"].shape[2]
P = lab.max() + 1
R = symmetric_basis(D["A"], K)
grid = np.arange(float(P))
# drop any target whose sample or lags touch the steps (decimated grid: 3 ms)
art = np.zeros(len(t), bool)
for a, b in ((-3.0, 15.0), (897.0, 915.0)):
    art |= (t >= a) & (t < b)
bad = np.convolve(art, np.ones(K + 1), mode="full")[:len(t)] > 0
lab_m = lab.copy()
lab_m[:, bad] = -1
print("masked samples of the time axis:", int(bad.sum()), "of", len(t), "; bins per sample count after mask:", [int((lab_m == g).sum()) for g in range(P)])
for variant in ("felix", "induced"):
    for name, L in (("as reported", lab), ("steps masked", lab_m)):
        gain = {2: 0.0, 3: 0.0}
        per_bin = np.zeros(P)
        count = 0.0
        cnt_bin = np.zeros(P)
        for f in range(s12.NFOLD):
            Xf = D["X"] if variant == "felix" else s12.remove_evoked(D["X"], Z, fo != f)
            sf = s12b.fold_stats(Xf, L, K, fo, P)
            tr, te = s12b.sel(sf, f), s12b.sel(sf, f, False)
            a = {k: tr[k].sum(0) for k in tr}
            phi, Sf = fit_gls_pooled(a["Gzz"], a["Gzy"], a["Syy"], a["n"], R)
            Base = np.stack([theta_of(phi, R, N)] * P)
            base = s12b.ll(Base, te, [s12.sigma_of(Base, tr)] * P)
            count += te["n"].sum()
            cnt_bin += te["n"]
            for r in (2, 3):
                m = fit_sym_tp(tr, R, Sf, r, grid, n_iter=25)
                Th = m.theta_grid()
                g = s12b.ll(Th, te, [s12.sigma_of(Th, tr)] * P) - base
                gain[r] += g.sum()
                if r == 2:
                    per_bin += g
        # full-day weights, rank 2: does the jump at plan 1/4 survive?
        s = s12b.fold_stats(D["X"] if variant == "felix" else s12.remove_evoked(D["X"], Z, np.ones(len(D["X"]), bool)), L, K, fo, P)
        st = {k: s[k].sum(0) for k in s}
        tot = {k: st[k].sum(0) for k in st}
        _, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
        Hbar = _quad(tot["Gzz"], tot["Gzy"], R, np.linalg.pinv(Sig))[0] / tot["n"]
        m3 = fit_sym_tp(st, R, Sig, 3, grid, n_iter=40)
        lam3 = polytope_vertices(m3, Hbar)["lam"]
        print(f"{variant} {name}: held-out gain r2 {gain[2] / count:+.4f}, r3 {gain[3] / count:+.4f}; "
              f"r2 gain in plan 1/4 {per_bin[4] / max(cnt_bin[4], 1):+.3f}; rank-3 lam at plan 1/4: {np.round(lam3[4], 2)}, hold 4/4: {np.round(lam3[3], 2)}", flush=True)
print("done")
