"""Plotting helpers (matplotlib, Agg-friendly)."""

from __future__ import annotations

import numpy as np


def _ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
    return ax


def plot_array_map(values, rows, cols, ax=None, title=None, cmap="viridis"):
    """Heat-map a per-channel scalar onto the array grid layout."""
    ax = _ax(ax)
    img = np.asarray(values, dtype=float).reshape(rows, cols)
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(title or "")
    ax.set_xticks([]); ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_singular_spectrum(singular_values, ax=None, title="HOSVD spectrum"):
    ax = _ax(ax)
    s = np.asarray(singular_values, dtype=float)
    ax.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-")
    ax.set_xlabel("component"); ax.set_ylabel("normalized singular value")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return ax


def plot_weight_functions(p_grid, weights, ax=None, title="weight functions"):
    """weights: (P, r) or (P, V) weighting/convex functions over the schedule."""
    ax = _ax(ax)
    weights = np.asarray(weights)
    for j in range(weights.shape[1]):
        ax.plot(p_grid, weights[:, j], "-o", ms=3, label=f"w{j}")
    ax.set_xlabel("scheduling p"); ax.set_ylabel("weight")
    ax.set_title(title); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    return ax


def plot_coupling(C, ax=None, title="coupling", cmap="magma"):
    ax = _ax(ax)
    im = ax.imshow(np.asarray(C), cmap=cmap)
    ax.set_title(title); ax.set_xlabel("from j"); ax.set_ylabel("to i")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_traces(traces_dict, ax=None, title="", xlabel="step", ylabel="value"):
    ax = _ax(ax)
    for label, y in traces_dict.items():
        ax.plot(y, label=label, lw=1.2)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    return ax
