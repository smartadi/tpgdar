r"""Null test for the evoked response, H 2021-11-12: a network whose coupling never changes, plus each trial's real evoked wave.

Approved by Aditya 2026-11-23 (gate block "add scripts/20_null_evoked.py, which builds the null data you approved as (b)").
For each recorded trial n, with its own go and touch times and reach direction:

    x~_n(t) = y_n(t) + m_n(t)
    y_n(t+1) = sum_k A^_k y_n(t-k+1) + e_n(t),   e ~ N(0, Sigma^) independent,   2,000 burn-in samples dropped, seed 0
    A^_k, Sigma^: the GDAR fit (MATH §4.3) to all trials with the evoked response removed (MATH §3, kernels fitted on all trials)
    m_n(t): trial n's fitted evoked wave Z beta^, the part MATH §3 removes

Each simulated trial is then mean-centred per channel, as the recordings are (MATH §1.1).  X_nowave is y alone, centred the same
way.  Script 13 fits them with --null and --null-nowave and is otherwise unchanged.  The data have no state change, so any gain
of TP-GDAR over GDAR on X_wave comes from the evoked wave, and on X_nowave it should be about 0.

Checks printed: the spectral radius of A^ (must be below 1); the trial mean of x~ against the trial mean of the wave; the
variance of each data set against the recordings.

Run:  .venv\Scripts\python scripts/20_null_evoked.py
"""

from __future__ import annotations

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
from gdar_tp.sym_tp import fit_gls_pooled, symmetric_basis, theta_of  # noqa: E402

SEED = 0
BURN = 2000
log = s12.log


def simulate(Theta, Sigma, n, T, burn, rng):
    """n independent runs of x(t+1) = Theta z(t) + e(t), e ~ N(0, Sigma), from zeros; the first ``burn`` samples dropped."""
    N, KN = Theta.shape
    lags = KN // N
    L = np.linalg.cholesky(Sigma)
    y = np.zeros((n, burn + T, N))
    for t in range(lags - 1, burn + T - 1):
        z = np.concatenate([y[:, t - k] for k in range(lags)], 1)
        y[:, t + 1] = z @ Theta.T + rng.standard_normal((n, N)) @ L.T
    return y[:, burn:]


def main():
    D = s12.load()
    X = D["X"]
    n, T, N = X.shape
    lags = s13.K
    lab = s13.warped_labels(D)
    fo = s12.folds(D["ang"])
    Z = s12.evoked_design(D)
    Xr = s12.remove_evoked(X, Z, np.ones(n, bool))     # as script 13's full-data "evoked removed" fit
    m = X - Xr                                          # the fitted evoked wave of every trial
    s = s12b.fold_stats(Xr, lab, lags, fo, lab.max() + 1)
    tot = {k: s[k].sum((0, 1)) for k in s}
    R = symmetric_basis(D["A"], lags)
    phi, Sig = fit_gls_pooled(tot["Gzz"], tot["Gzy"], tot["Syy"], tot["n"], R)
    Th = theta_of(phi, R, N)
    rho = float(np.abs(s12.rho(Th, lags)).max())
    log(f"fixed network: GDAR on all {n} trials, evoked removed, {lags} lags, {N} channels; spectral radius {rho:.4f}")
    assert rho < 1, "the fixed network is unstable; can't simulate it"

    y = simulate(Th, Sig, n, T, BURN, np.random.default_rng(SEED))
    x_wave = y + m
    x_wave = x_wave - x_wave.mean(1, keepdims=True)
    x_nowave = y - y.mean(1, keepdims=True)

    mc = m - m.mean(1, keepdims=True)
    a, b = x_wave.mean(0), mc.mean(0)
    log(f"check, trial mean of the null data against the trial mean of the wave: relative difference "
        f"{np.linalg.norm(a - b) / np.linalg.norm(b):.3f}, correlation {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.3f}")
    for name, A in (("recordings", X), ("recordings, evoked removed", Xr), ("null, fixed GDAR plus wave", x_wave),
                    ("null, fixed GDAR only", x_nowave), ("the wave alone", mc)):
        log(f"check, variance per sample and channel, {name}: {A.var(axis=(0, 1)).mean():.2f} uV^2")
    np.savez(s13.NULL_DATA, X_wave=x_wave, X_nowave=x_nowave, seed=SEED, burn=BURN, rho=rho, Sigma=Sig, phi=phi)
    log("wrote", s13.NULL_DATA)


if __name__ == "__main__":
    main()
