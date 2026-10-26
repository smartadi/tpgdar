"""Evaluation metrics: prediction error, flow-signal correlation, rank/energy."""

from __future__ import annotations

import numpy as np


def nrmse(pred, true, eps=1e-12):
    """Normalized RMSE (by the std of the truth), averaged over channels.

    Returns a single float in [0, inf); 0 is perfect, ~1 is "no better than the
    mean".  Computed per channel then averaged so loud channels don't dominate.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    scale = np.std(true, axis=0)
    return float(np.mean(rmse / (scale + eps)))


def one_step_nrmse(model, X, U=None):
    pred, true = model.predict_one_step(X, U)
    return nrmse(pred, true)


def multi_step_nrmse(model, X, U=None, horizon=10):
    """Free-run NRMSE at a given horizon.

    Rolls the model out ``horizon`` steps from many starting points and compares
    the horizon-step-ahead prediction to truth.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    K = model.K
    preds, trues = [], []
    for t0 in range(K - 1, T - 1 - horizon):
        hist = X[t0 - K + 1:t0 + 1]
        u = None
        if U is not None and model.m > 0:
            # rollout step s produces x(t0+1+s), driven by u(t0+s)
            u = U[t0:t0 + horizon]
        roll = model.rollout(hist, horizon, U=u)
        preds.append(roll[-1])
        trues.append(X[t0 + horizon])
    return nrmse(np.array(preds), np.array(trues))


def flow_signal_correlation(estimated_coupling, true_coupling, mask=None):
    """Pearson correlation between estimated and true off-diagonal coupling.

    ``mask`` (bool NxN) restricts to particular entries (e.g. graph edges).
    Diagonal is always excluded.
    """
    est = np.asarray(estimated_coupling, dtype=float)
    tru = np.asarray(true_coupling, dtype=float)
    n = est.shape[0]
    off = ~np.eye(n, dtype=bool)
    if mask is not None:
        off &= mask
    a = est[off].ravel()
    b = tru[off].ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def energy_spectrum(singular_values):
    """Cumulative energy (fraction of squared-singular-value mass)."""
    s = np.asarray(singular_values, dtype=float)
    e = s ** 2
    return np.cumsum(e) / e.sum()


def rank_at_energy(singular_values, thresh=0.999):
    """Smallest number of components capturing ``thresh`` of the energy."""
    cum = energy_spectrum(singular_values)
    return int(np.searchsorted(cum, thresh) + 1)
