"""Overlapping bins on the warped trial axis.  MATH §2.4, approved by Aditya 2026-11-23.

A target sample at warped time tau (bins, 0 <= tau < P) spreads its weight over a window of width delta bins centred on it;
its weight in bin g is the share of that window, clipped to [0, P), that falls in [g, g + 1).  The weights sum to 1, so each
sample counts once over all bins, and delta = 0 is the hard bin of MATH §2.3.
"""

from __future__ import annotations

import numpy as np


def overlap_weights(tau, delta, P, label=None):
    """Weights a_g(tau), shape tau.shape + (P,).  NaN tau (an unused sample) gets all zeros.

    With delta = 0 the weight is 1 in the sample's own bin: ``label`` if given (the scripts' §2.3 labels, -1 = unused),
    else floor(tau) capped at P - 1.
    """
    tau = np.asarray(tau, dtype=float)
    out = np.zeros(tau.shape + (P,))
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if delta == 0:
        g = (np.asarray(label) if label is not None else
             np.where(np.isnan(tau), -1, np.minimum(np.floor(np.nan_to_num(tau, nan=-1.0)), P - 1))).astype(int)
        ok = g >= 0
        out[ok, g[ok]] = 1.0
        return out
    ok = ~np.isnan(tau)
    lo = np.clip(tau[ok] - delta / 2, 0, P)
    hi = np.clip(tau[ok] + delta / 2, 0, P)
    edges = np.arange(P)
    share = np.clip(np.minimum(hi[:, None], edges + 1) - np.maximum(lo[:, None], edges), 0, None)
    out[ok] = share / (hi - lo)[:, None]
    return out


def weighted_bin_stats(X, weights, K, K0, fold, nfold):
    """Per (fold, bin) sufficient statistics with each target sample weighted by ``weights`` (MATH §2.4).

    X (trials, T, N); weights (trials, T, P), the weight of target x(t+1) at index t+1.  Targets start at index K0, as in
    ``12b: fold_stats``, which this equals when the weights are one-hot.
    """
    n, T, N = X.shape
    P = weights.shape[2]
    D = K * N
    s = dict(Gzz=np.zeros((nfold, P, D, D)), Gzy=np.zeros((nfold, P, D, N)), Syy=np.zeros((nfold, P, N, N)),
             n=np.zeros((nfold, P)))
    ts = np.arange(K0 - 1, T - 1)
    for i in range(n):
        Z = np.concatenate([X[i, ts - k] for k in range(K)], 1)
        Y = X[i, ts + 1]
        w = weights[i, ts + 1]
        f = fold[i]
        for g in range(P):
            m = w[:, g] > 0
            if not m.any():
                continue
            Zm, Ym, wm = Z[m], Y[m], w[m, g][:, None]
            s["Gzz"][f, g] += (Zm * wm).T @ Zm
            s["Gzy"][f, g] += (Zm * wm).T @ Ym
            s["Syy"][f, g] += (Ym * wm).T @ Ym
            s["n"][f, g] += wm.sum()
    return s
