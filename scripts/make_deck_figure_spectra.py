r"""Frequency spectrum of the LFP in each trial phase, IOS Reach, no-stim days, for the slides.

Data as in script 12 (`noart`, decimated x3 to 333 Hz, B14 dropped, move time under 3 s), evoked response left in.
Phases as in script 12: hold -400..0 ms from target on, plan to the go cue, move to touch, post touch..touch+500 ms.

Estimator, with the lab's spectrogram window (iosreach/scripts/session_report.py), approved by Aditya 2026-11-05:
  200 ms Hann windows, here every 50 ms, each wholly inside the phase and demeaned
  per channel, the trial-mean power, then dB, then the mean over channels
A phase shorter than 200 ms on a trial contributes no window from that trial.

Also: the fitted rank-2 family from cache/13 (evoked left in), as the spectrum of the AR model frozen at each bin, with
the same 200 ms window applied in frequency so that data and model have the same resolution.  Both are shown relative
to the mean of the four phases, in dB.  No model is fitted here.

Writes figures/deck/spectra_phase.json, spectra_phase_0823.json, spectra_model.json.  Log: cache/spectra_run.log.

Run:  .venv\Scripts\python scripts/make_deck_figure_spectra.py
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import sys

import numpy as np
from scipy.signal import decimate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
s12 = importlib.import_module("12_tp_vs_gdar_ios")
from gdar_tp.sym_tp import theta_of  # noqa: E402

DECK = os.path.join(ROOT, "figures", "deck")
LOG = os.path.join(ROOT, "cache", "spectra_run.log")
CACHE13 = os.path.join(ROOT, "cache", "13_vertices_H_2021-11-12.pkl")
WIN_MS, HOP_MS = 200.0, 50.0
NFFT = 512
FMIN, FMAX = 3.0, 100.0         # the 120 Hz line and the edge of the x3 decimation filter lie above this
COLORS = {"hold": "#8a8f98", "plan": "#2a78d6", "move": "#eb6834", "post": "#1baf7a"}
BANDS = {"theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-30": (13, 30), "gamma 30-60": (30, 60), "high gamma 65-115": (65, 115)}
NBOOT = 200

lines = []


def log(*a):
    text = " ".join(str(x) for x in a)
    print(text, flush=True)
    lines.append(text)


def load(date, key="noart"):
    z = np.load(s12.SEGS.replace("2021-11-12", date), allow_pickle=True)
    assert int(z["stimmed"].sum()) == 0
    keep = ~np.isin(z["channels"], s12.BAD)
    ok = z["move_ms"] < 3000
    X = decimate(z[key][keep][:, :, ok].astype(float), s12.DEC, axis=1, zero_phase=True).transpose(2, 1, 0)
    t = (np.arange(2000) - 500)[::s12.DEC].astype(float)[:X.shape[1]]
    go = z["plan_ms"][ok].astype(float)
    return dict(X=X, t=t, go=go, touch=go + z["move_ms"][ok])


def phase_limits(D, k):
    go, touch = D["go"][k], D["touch"][k]
    return ((-400.0, 0.0), (0.0, go), (go, touch), (touch, min(touch + 500.0, 1400.0)))


def trial_power(D):
    """(phases, trials, freqs, channels) window-mean power, NaN where a phase has no window.  Also window counts."""
    X, t = D["X"], D["t"]
    dt = t[1] - t[0]
    nwin, nhop = int(round(WIN_MS / dt)), int(round(HOP_MS / dt))
    taper = np.hanning(nwin)
    norm = (taper ** 2).sum() * s12.FS
    freqs = np.fft.rfftfreq(NFFT, 1 / s12.FS)
    out = np.full((4, X.shape[0], len(freqs), X.shape[2]), np.nan)
    count = np.zeros((4, X.shape[0]), int)
    for k in range(X.shape[0]):
        for ph, (a, b) in enumerate(phase_limits(D, k)):
            idx = np.where((t >= a) & (t < b))[0]
            starts = idx[0] + np.arange(0, max(len(idx) - nwin + 1, 0), nhop) if len(idx) else []
            if len(starts) == 0:
                continue
            seg = np.stack([X[k, s:s + nwin] for s in starts])          # windows, time, channels
            seg = seg - seg.mean(1, keepdims=True)
            power = np.abs(np.fft.rfft(seg * taper[None, :, None], NFFT, axis=1)) ** 2 * 2 / norm
            out[ph, k] = power.mean(0)
            count[ph, k] = len(starts)
    return out, freqs, count


def phase_db(power, weights=None):
    """(phases, freqs) in dB: the trial-mean power per channel, dB, then the mean over channels."""
    if weights is None:
        mean = np.nanmean(power, axis=1)
    else:
        w = weights[None, :, None, None] * ~np.isnan(power)
        mean = np.nansum(power * weights[None, :, None, None], axis=1) / w.sum(1)
    return (10 * np.log10(mean)).mean(-1)


def spectra(date, key="noart"):
    D = load(date, key)
    power, freqs, count = trial_power(D)
    db = phase_db(power)
    rng = np.random.default_rng(0)
    n = power.shape[1]
    boot = np.stack([phase_db(power, np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(NBOOT)])
    log(f"H {date} {key}: {n} trials, {power.shape[3]} channels; windows per phase {count.sum(1).tolist()}; "
        f"trials with no window {(count == 0).sum(1).tolist()}")
    return dict(db=db, boot=boot, freqs=freqs, power=power, n=n, ch=power.shape[3])


def band_table(res):
    db, boot, f = res["db"], res["boot"], res["freqs"]
    log("  band, change from hold in dB (bootstrap 95% over trials): plan, move, post")
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        cells = []
        for ph in (1, 2, 3):
            d = (db[ph] - db[0])[m].mean()
            b = (boot[:, ph] - boot[:, 0])[:, m].mean(1)
            cells.append(f"{d:+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")
        log(f"    {name:18s} " + "   ".join(cells))


def freq_panel(cell, title, f, series, ylabel, ylim=None, hline=None, legend=True, **kw):
    m = (f >= FMIN) & (f <= FMAX)
    panel = {"kind": "xy", "cell": cell, "title": title, "xlabel": "frequency (Hz)", "ylabel": ylabel, "xlog": True,
             "xlim": [FMIN, FMAX],
             "series": [{"name": nm if legend else None, "x": [round(float(v), 2) for v in f[m]],
                         "y": [round(float(v), 2) for v in y[m]], "color": COLORS.get(nm, c), "width": 2.0, "line": ls}
                        for nm, y, c, ls in series], "legend": legend}
    if ylim:
        panel["ylim"] = ylim
    if hline is not None:
        panel["hlines"] = [{"y": hline, "color": "#555555"}]
    panel.update(kw)
    return panel


def write(name, spec):
    with open(os.path.join(DECK, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh)


def phase_figure(res, name):
    db, f = res["db"], res["freqs"]
    names = s12.PHASES
    write(name, {"grid": [1, 2], "panels": [
        freq_panel([0, 0, 1, 1], "Power in each phase", f, [(names[p], db[p], None, "solid") for p in range(4)], "power (dB)"),
        freq_panel([0, 1, 1, 1], "Change from hold", f, [(names[p], db[p] - db[0], None, "solid") for p in (1, 2, 3)],
                   "dB re hold", hline=0.0),
    ]})


def model_spectra(f):
    """(phases, freqs) dB, the frozen-bin AR spectrum of the rank-2 family, with the data's window applied in frequency."""
    with open(CACHE13, "rb") as fh:
        res = pickle.load(fh)["felix"]
    model = res["full"][2]["model"]
    Sig = res["Sigma"]
    N = Sig.shape[0]
    nwin = int(round(WIN_MS / (1000.0 / s12.FS)))
    kernel = np.abs(np.fft.fft(np.hanning(nwin), NFFT)) ** 2
    kernel /= kernel.sum()
    thetas = model.theta_grid()
    out = np.zeros((len(thetas), len(f)))
    for g, theta in enumerate(thetas):
        s = s12.ar_spectrum(theta, Sig, theta.shape[1] // N, f)
        full = np.concatenate([s, s[-2:0:-1]])
        out[g] = 10 * np.log10(np.real(np.fft.ifft(np.fft.fft(full) * np.fft.fft(kernel)))[:len(f)])
    return out.reshape(4, 4, -1).mean(1)


def model_figure(res):
    db, f = res["db"], res["freqs"]
    data = db - db.mean(0)
    mod = model_spectra(f)
    mod = mod - mod.mean(0)
    m = (f >= FMIN) & (f <= FMAX)
    log("  data vs model, phase minus mean of phases, dB:")
    for p, nm in enumerate(s12.PHASES):
        r = np.corrcoef(data[p][m], mod[p][m])[0, 1]
        at = {lo: (data[p][(f >= lo) & (f < hi)].mean(), mod[p][(f >= lo) & (f < hi)].mean()) for lo, hi in ((13, 30), (65, 115))}
        log(f"    {nm}: r over {FMIN:.0f}-{FMAX:.0f} Hz {r:+.2f}; 13-30 Hz data {at[13][0]:+.2f}, model {at[13][1]:+.2f}; "
            f"65-115 Hz data {at[65][0]:+.2f}, model {at[65][1]:+.2f}")
    ylim = [float(np.floor(min(data[:, m].min(), mod[:, m].min()))), float(np.ceil(max(data[:, m].max(), mod[:, m].max())))]
    panels = []
    for p, nm in enumerate(s12.PHASES):
        panels.append(freq_panel([0, p, 1, 1], nm, f, [("data", data[p], "#222222", "solid"), ("model", mod[p], COLORS[nm], "solid")],
                                 "dB re mean of phases" if p == 0 else "", ylim=ylim, hline=0.0))
    write("spectra_model", {"grid": [1, 4], "col_widths": [1.14, 1, 1, 1], "panels": panels})


def main():
    res = spectra("2021-11-12")
    band_table(res)
    phase_figure(res, "spectra_phase")
    raw = spectra("2021-11-12", "lfp")
    log(f"  noart vs raw lfp, largest gap in change from hold over {FMIN:.0f}-{FMAX:.0f} Hz, dB: " + ", ".join(
        f"{s12.PHASES[p]} {np.abs(((res['db'][p] - res['db'][0]) - (raw['db'][p] - raw['db'][0]))[(res['freqs'] >= FMIN) & (res['freqs'] <= FMAX)]).max():.2f}"
        for p in (1, 2, 3)))
    model_figure(res)
    other = spectra("2021-08-23")
    band_table(other)
    phase_figure(other, "spectra_phase_0823")
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
