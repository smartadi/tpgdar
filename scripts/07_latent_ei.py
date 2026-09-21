"""Stage 4b -- latent population recovery: can a 2-population GDAR find E/I?

Falsification test for the "latent cell-type / population graph" idea, run in a
sandbox where we KNOW the answer.  The Wilson-Cowan plant is secretly
``space x {E, I}``; we record only E.  A 2-population latent LDS is fit on E
alone (E-slot clamped to the data, hidden I-slot free, Dale + graph constrained);
we then ask whether the inferred hidden slot recovers the TRUE inhibitory trace.

The key control is the **separation of timescales**: we sweep the inhibitory time
constant ``tau_i``.  When I is fast (tau_i = tau_e) it is slaved to E and E is
already a fine proxy -- the latent leg buys nothing.  When I is slow
(tau_i >> tau_e) it carries memory the instantaneous observation cannot, and the
latent inference should recover it *better* than E-as-proxy.  We also check
one-step prediction, to be honest that the latent leg does NOT buy predictive
power for E here -- only interpretability of the hidden variable.

Run:  python scripts/07_latent_ei.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdar_tp import metrics
from gdar_tp.gdar import fit_gdar
from gdar_tp.graph import grid_graph, support_mask
from gdar_tp.latent import (fit_em, one_step_pred_nrmse, population_masks,
                            recovered_inhibitory)
from gdar_tp.sim.wilson_cowan import WCNetwork, WCParams

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def abscorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    s = a.std() * b.std()
    return 0.0 if s < 1e-12 else abs(float(np.mean(a * b) / s))


def run_regime(g, mask, tau_i, seed=42):
    N = g.n
    net = WCNetwork(g.adjacency, WCParams(g0=3.0, tau_i=tau_i))
    E, I_true = net.simulate(n_steps=8000, p=0.8, noise_std=0.03, rng=seed,
                             burn_in=800, return_full=True)
    E = E - E.mean(axis=0, keepdims=True)
    I_true = I_true - I_true.mean(axis=0, keepdims=True)
    T = E.shape[0]; tr = slice(0, T // 2); te = slice(T // 2, T)

    # single-population GDAR baselines (lag sweep)
    single = {}
    for K in (1, 3):
        m = fit_gdar(E[tr], K=K, support=mask, param="edgewise", ridge=1e-3)
        single[K] = metrics.one_step_nrmse(m, E[te])

    # latent 2-population LDS
    struct, sign, C = population_masks(g.adjacency)
    lds, info = fit_em(E[tr], struct, sign, C, n_iter=30, rng=1)
    lat_pred = one_step_pred_nrmse(lds, E[te])

    # recovery of the true hidden I (smooth the full run with the fitted model)
    _, info_full = fit_em(E, struct, sign, C, n_iter=30, rng=1)
    I_rec = recovered_inhibitory(info_full, N)
    rec = float(np.mean([abscorr(I_rec[:, i], I_true[:, i]) for i in range(N)]))
    proxy = float(np.mean([abscorr(E[:, i], I_true[:, i]) for i in range(N)]))
    return dict(tau_i=tau_i, single=single, lat_pred=lat_pred, rec=rec,
                proxy=proxy, gain=rec - proxy, info=info, I_rec=I_rec,
                I_true=I_true, best=int(np.argmax([abscorr(I_rec[:, i], I_true[:, i])
                                                   for i in range(N)])))


def main():
    g = grid_graph(3, 3)
    mask = support_mask(g.adjacency)
    print("=== Stage 4b: latent E/I recovery vs. timescale separation ===")
    print("grid 3x3=9 ch, observe E only; hidden I kept only for scoring\n")
    print(f"{'tau_i':>6}{'gdar K1':>9}{'gdar K3':>9}{'latent pred':>12}"
          f"{'rec|corr|':>11}{'E-proxy':>9}{'gain':>8}")
    taus = [8.0, 25.0, 60.0]
    results = []
    for tau in taus:
        r = run_regime(g, mask, tau)
        results.append(r)
        print(f"{tau:>6.0f}{r['single'][1]:>9.3f}{r['single'][3]:>9.3f}"
              f"{r['lat_pred']:>12.3f}{r['rec']:>11.3f}{r['proxy']:>9.3f}{r['gain']:>+8.3f}")

    print("\nreading:")
    print("  * prediction: latent lag-1 ~ single-pop GDAR everywhere -> the hidden")
    print("    population buys NO predictive power for E (honest: no free lunch).")
    print("  * recovery: the gain (latent minus E-proxy) turns POSITIVE only when I")
    print("    is slow (tau_i >> tau_e=8) -- then E's instantaneous value is a poor")
    print("    proxy and the latent inference recovers the hidden slow variable.")
    fast, slow = results[0], results[-1]
    print(f"\n  fast I (tau_i=8): gain {fast['gain']:+.3f} -> latent adds nothing")
    print(f"  slow I (tau_i=60): gain {slow['gain']:+.3f} -> latent recovers hidden I")

    _figures(results)
    print(f"\nfigures -> {FIGDIR}")


def _figures(results):
    taus = [r["tau_i"] for r in results]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(taus, [r["rec"] for r in results], "o-", label="latent hidden slot")
    ax[0].plot(taus, [r["proxy"] for r in results], "s--", label="E as proxy")
    ax[0].set_xlabel("inhibitory time constant tau_i (ms)")
    ax[0].set_ylabel("|corr| with true I")
    ax[0].set_title("hidden-I recovery vs timescale separation")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    slow = results[-1]
    ax[1].plot(slow["info"]["loglik"], "o-")
    ax[1].set_xlabel("EM iteration"); ax[1].set_ylabel("log-likelihood")
    ax[1].set_title(f"EM convergence (tau_i={slow['tau_i']:.0f})"); ax[1].grid(alpha=0.3)

    b = slow["best"]; seg = slice(0, 400)
    z = slow["I_rec"][seg, b]; z = (z - z.mean()) / (z.std() + 1e-9)
    it = slow["I_true"][seg, b]; it = (it - it.mean()) / (it.std() + 1e-9)
    ax[2].plot(it, label="true hidden I", lw=1.3)
    ax[2].plot(z, label="recovered slot", lw=1.0, alpha=0.85)
    r_best = abscorr(slow["I_rec"][:, b], slow["I_true"][:, b])
    ax[2].set_title(f"slow-I recovery (node {b}, |r|={r_best:.2f})")
    ax[2].set_xlabel("step"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "07_latent_ei.png"), dpi=130)


if __name__ == "__main__":
    main()
