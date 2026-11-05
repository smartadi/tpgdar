r"""Three checks of the IOS Reach papers against our no-stim days.  Prints only; no model is fitted.

Papers: Griggs, Stanis et al., Nat Commun 2026; Stanis et al., EMBC 2025.

  theta   4-10 Hz, plan vs hold, per channel, with the estimator of make_deck_figure_spectra.py.  The papers report a rise
          near the IPS and in 7a, 7b during planning, against the 500 ms before the target.
  steps   median absolute sample-to-sample jump at 0, 10, 900 and 910 ms in `noart` and raw `lfp`.  The Nat Commun methods
          split trials at light on and off and rejoin them by matching 50 ms means.
  notch   power at 60 and 120 Hz against their neighbours.  EMBC lists notches at 60, 120 and 180 Hz.

Run:  .venv\Scripts\python scripts/ios_paper_checks.py
"""

import importlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sp = importlib.import_module("make_deck_figure_spectra")
s12 = sp.s12
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def channel_name(c):
    return f"A{c + 1}" if c < 16 else f"B{c - 15}"


for date in ("2021-11-12", "2021-08-23"):
    z = np.load(s12.SEGS.replace("2021-11-12", date), allow_pickle=True)
    keep = ~np.isin(z["channels"], s12.BAD)
    names = [channel_name(int(c)) for c in z["channels"][keep]]
    power, f, _ = sp.trial_power(sp.load(date))
    theta = power[:, :, (f >= 4) & (f < 10), :].mean(2)      # phases, trials, channels
    change = 10 * np.log10(np.nanmean(theta[1], 0) / np.nanmean(theta[0], 0))
    rng = np.random.default_rng(0)
    trials = theta.shape[1]
    boot = []
    for _ in range(300):
        pick = rng.integers(0, trials, trials)
        boot.append(10 * np.log10(np.nanmean(theta[1][pick], 0) / np.nanmean(theta[0][pick], 0)))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"H {date}, theta 4-10 Hz, plan vs hold: all-channel mean {change.mean():+.2f} dB")
    print("  95% above 0:", [f"{names[c]} {change[c]:+.1f}" for c in np.argsort(-change) if lo[c] > 0])
    print("  95% below 0:", [f"{names[c]} {change[c]:+.1f}" for c in np.argsort(change) if hi[c] < 0])
    t = np.arange(2000) - 500
    for key in ("noart", "lfp"):
        X = z[key][keep].astype(float)
        jump = np.median(np.abs(np.diff(X, axis=1)), axis=(0, 2))
        at = ", ".join(f"{e} ms {jump[np.where(t == e)[0][0]]:.1f}" for e in (0, 10, 900, 910))
        seg = X[:, 600:1600, :] - X[:, 600:1600, :].mean(1, keepdims=True)
        db = 10 * np.log10((np.abs(np.fft.rfft(seg, axis=1)) ** 2).mean(2)).mean(0)
        hz = np.fft.rfftfreq(1000, 1 / 1000)
        notch = [db[hz == c][0] - 0.5 * (db[hz == c - 10][0] + db[hz == c + 10][0]) for c in (60, 120)]
        print(f"  {key}: jump typical {np.median(jump):.2f} uV, {at}; 60 Hz {notch[0]:+.1f} dB, 120 Hz {notch[1]:+.1f} dB re neighbours")
