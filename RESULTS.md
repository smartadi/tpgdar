# GDAR-TP — results summary (for the explainer deck)

All numbers below are reproduced by the scripts in `scripts/`. Figures land in
`figures/`; rendered LaTeX equation PNGs are in `figures/eq/`. Run any script with
`.venv\Scripts\python scripts/<name>.py`.

## One-paragraph story

Cortex has **one anatomy** but its dynamics change with state. GDAR fits a
graph-constrained autoregressive model on that fixed anatomy; **GDAR-TP** lets the
model's coefficients vary with a measured state `p` and uses a tensor-product
(HOSVD) decomposition to express that variation as a few **vertex graphs** blended
by convex weights `w_v(p)`. Crucially this is **one graph, state-modulated gains** —
not many anatomies. We show (a) TP beats a state-blind GDAR and pins the number of
state modes, (b) on a realistic 8-neighbour array we **recover 5 planted graphs at
0.998**, (c) a latent-population variant recovers a hidden slow inhibitory
population only when its timescale is separable, and (d) the whole thing runs on
the **real OS-Pulse ECoG** with stim frequency as the state variable.

## Equations (figures/eq/)

| file | meaning |
|---|---|
| `gdar.png` | GDAR model: `x(t+1)=Σ_k A_k x(t−k)+B u(t)+e(t)` |
| `support.png` | graph support + strict diffusion `A_k=a_k I−b_k L` |
| `tp_param.png` | parametric TP: `A_k(p)=Σ_j w_j(p) A_k^(j)`, convex weights |
| `hosvd.png` | HOSVD of the coefficient tensor |
| `polytope.png` | convex/polytopic form `A(p)=Σ_v λ_v(p) A_v` |
| `eig.png` | diffusion form ⇒ fixed graph modes, moving eigenvalues λ_i(p) |
| `product.png` | structural TP: product graph `G_space ⊗ G_pop` |
| `latent.png` | latent LDS: `x(t+1)=A x(t)+w`, `y=C x+v` |
| `dale.png` | Dale sign constraint |
| `lmi.png` | LMI state-feedback synthesis |
| `recover.png` | recovery question: `Â(p)=Σ λ_v Â_v ↔ true graphs?` |

## Stage 1 — GDAR beats plain VAR on flow recovery  (`01_synthetic_gdar.py`)
Linear graph-diffusion process, 6×6 grid, K=2.
| model | flow-corr | #params |
|---|---|---|
| GDAR edge-wise | **0.99** | 312 |
| GDAR diffusion | **1.00** | 4 |
| unconstrained VAR | 0.90 | 2592 |

Residuals white (Ljung-Box median Q=16 < χ²₀.₉₅(20)=31.4). One-step NRMSE ≈0.86
(the process is noise-driven, so it is intrinsically hard to predict — the point
here is *structure* recovery, not prediction). Figs: `01_coupling_recovery.png`.

## Stage 2 — TP finds a clean rank-2 parameter dependence  (`02_synthetic_tp.py`)
Wilson–Cowan network, coupling varies with `p` (g0=3.0), 5×5 grid.
- HOSVD singular values (norm): `[1, 0.117, 0.009, ...]` → **rank 2**.
- Vertex-reconstruction error vs #modes: rank1 0.117 → **rank2 0.017** (7× drop) → plateau (noise floor). Elbow at 2 = true mode count.
- Convex hull: **2 vertices**, affine dim 1.
- Held-out `p`: TP-interp NRMSE ≈ oracle (mean gap **+0.0002**).
Figs: `02_tp_spectrum_weights.png`, `02_vertex_mode_maps.png`.

## Diagnostic — does TP buy anything over plain GDAR?  (`06_tp_diagnostic.py`)
Held-out-`p` one-step NRMSE (fresh WC segments):
| model | mean NRMSE |
|---|---|
| pooled GDAR (state-blind null) | 0.1913 |
| per-condition (nearest) | 0.1817 |
| **TP low-rank interp** | **0.1815** |
| oracle | 0.1812 |

TP beats the pooled null by **+0.0098** and sits only **+0.0002** above the oracle
— recovers essentially all achievable gain. Effect concentrated at schedule
extremes (p=0.9: pooled 0.418 vs TP 0.381). **Rank stable at 2** across all 5
independent noise realizations (sv2/sv1 = 0.116–0.118). *Honest caveat:* margins
are modest because the WC network largely synchronizes. Fig: `06_tp_diagnostic.png`.

## Headline — recover 5 planted graphs on the OS-Pulse-style array  (`08_five_graph_recovery.py`)
8×8 Utah-style array, **8-nearest-neighbour** support (420 directed edges). Five
distinct ground-truth operators on that one support — isotropic diffusion,
rightward & upward advection, a focal hotspot, a global synchrony mode — blended by
a partition-of-unity schedule so each is the sole-active operator at one knot
(making them the true convex-hull vertices; **identifiability condition = the
recording visits each regime**).
- Gap-detected rank: **5** (true 5); sv5/sv1=0.042 → sv6/sv1=0.014 (2.9× drop).
- Convex hull via **SPA** (successive-projection simplex-corner finder): **5
  vertices**, affine dim 4, reconstruct A(p) to rel. residual **6e-3**.
- Recovered-vs-true operator match (best assignment): isotropic 0.998, rightward
  1.000, upward 1.000, focal 0.995, synchrony 0.997 → **mean |corr| = 0.998**.
Figs: `08_recovery_spectrum.png`, `08_recovery_maps.png` (true vs recovered maps).

## Latent populations / cell types — regime of validity  (`07_latent_ei.py`)
The WC plant is secretly `space ⊗ {E,I}`; we observe only E and fit a 2-population
latent LDS (E-slot clamped to data, hidden I-slot free, Dale + graph constrained,
EM). Sweeping the inhibitory timescale `tau_i` (tau_e = 8 ms):
| tau_i | latent 1-step pred | recover-I \|corr\| | E-as-proxy \|corr\| | gain |
|---|---|---|---|---|
| 8 | 0.298 | 0.369 | 0.499 | **−0.131** |
| 25 | 0.090 | 0.560 | 0.709 | −0.149 |
| 60 | 0.077 | 0.532 | 0.492 | **+0.040** |

*Re-run on 2026-10-26 with the exact constrained M-step (`latent.constrained_m_step`); the earlier
project-after-least-squares M-step gave 0.371 / 0.613 / 0.560 and gains −0.128 / −0.096 / +0.068.
The conclusions stand, with a smaller slow-I benefit.*

Two honest findings: (1) **no free lunch for prediction** — the latent lag-1 model
ties single-population GDAR everywhere (hidden pop adds no predictive power for E);
(2) **recovery works only with timescale separation** — the hidden slot beats
E-as-proxy for the true inhibitory trace only when I is slow (`tau_i ≫ tau_e`).
EM log-likelihood is monotone. Fig: `07_latent_ei.png`.

## Control (in silico)  (`05_control_insilico.py`)
Single LMI state-feedback gain quadratically stabilizes the 2-vertex polytope
(closed-loop ρ ≈ 0.75); on the nonlinear WC plant with noise, loop delay and
actuator limits it cuts activity fluctuation **~23%**, degrading gracefully with
delay. Fig: `05_closed_loop.png`.

## Real data — OS-Pulse ECoG  (`09_ospulse_real.py`)  — SUPERSEDED (invalid window)

> **2026-10-26: this section is not a valid test.** Script 09 fits on 1.5–5.5 s. The
> response locked to the stimulation frequency shows that stimulation lasts only
> 1.0–2.0 s (40 Hz: 12–17× baseline at 1.0–2.0 s, about 1× from 2.0 s on), so about
> 87 % of the window is post-stimulation. The script also z-scores each condition
> separately, which rescales A per condition, and concatenates trials. See
> `docs/MODEL.md` §5. The real-data result is **§ Stacked TP-GDAR** below. The text is
> kept for the record.
Monkey Jalapeno, 96-ch uECoG (session 091114_M1), stim frequency (10–150 Hz, 9
conditions) as the scheduling variable; **78 good channels** (18 SNR-bad dropped);
**8-NN support** from electrode positions (362 edges); fs 339 Hz (decim 3), K=2;
**induced activity** (per-condition trial-mean removed) to drop the phase-locked
stim artifact.

What TP-GDAR finds on real data:
- **Coupling respects anatomy:** |coupling| vs −distance corr = **+0.33** — nearby
  channels couple more, as the array geometry predicts. The GDAR fit is capturing
  real spatial structure.
- **But the frequency dependence is NOT strongly low-rank.** HOSVD spectrum:
  `[1, 0.110, 0.094, 0.075, 0.072, 0.064, ...]` — a gradual decay with **no clean
  gap** (contrast the synthetic 2.9× gap at rank 5). The gap heuristic returns 3,
  but that is weak.
- **TP does not beat the state-blind model.** Leave-one-frequency-out held-out
  1-step NRMSE: pooled **0.1615** vs TP-interp **0.1629** (TP loses by −0.0013).

**Read this honestly:** the pipeline runs end-to-end on real 96-ch data and the
spatial structure is sensible, but on *this* session, with induced activity and
stim frequency as the knob, the effective connectivity does **not** reorganize in
a strongly low-rank, predictive way — so TP earns nothing over a frequency-blind
GDAR. This is exactly the diagnostic behaving as designed (it tells you when TP is
unnecessary). Contrast with the synthetic 5-graph case where TP wins decisively.
Next things to try before concluding: a different scheduling variable (stim
amplitude, per-band power, a low-dim latent), the **phase-locked evoked** component
instead of induced, or conditioning on stim-site — and the lab's reproducibility-
based artifact handling. Figs: `09_ospulse_overview.png`, `09_ospulse_vertices.png`.

## Real data — TP on the INPUT operator B(p) (10_ospulse_tp_input.py) — 2026-10-12

> **2026-10-26: numbers superseded.** The ridge used here (1e-6 × each Gram diagonal
> entry) is about four decades too strong for these lag Grams (condition number ~5e10).
> In a ridge scan, held-out ℓ rises by +84 nats/sample as the ridge drops toward 1e-10.
> All Δℓ below were computed on an over-shrunk A, and the TP models were two-stage. The
> window, input and Gram construction here are correct, and script 11 reuses them. See
> **§ Stacked TP-GDAR** for the valid comparison.

Session 091114 M1, 78 good ch, 8-NN graph (362 edges), fs 1017.25 Hz (the export is
already ÷3; script 09 wrongly decimated again). Data are bandpassed from 0.1 to 150 Hz
and scaled ×1e6. Only the stim window (1–2 s) is used.

Model: x_t = Σ_{k≤5} A_k x_{t−k} + Σ_{q≤5} b_q(p) u_{t−q}.
- A is shared and graph-supported.
- u is the fixed pulse train (`make_pulse_train(align='exact')`), 1000 µs pulses, conditions 1–9 (10–150 Hz).
- Training uses trials 0–39 and testing uses trials 40–59.
- The metric is the held-out one-step Gaussian log-likelihood per sample. Each model uses its own Σ, taken from its own training residuals. Results are reported as Δℓ vs B = 0.

**HOSVD of the B tensor (9 freq × 78 ch × 5 lags) over frequency:**
- s/s₀ = [1, 0.281, 0.130, 0.100, 0.068, 0.056, 0.036, 0.031, 0.030].
- Rank 1 holds 89.6 % of the energy. There is a moderate drop after r = 1 and no clean gap after that.
- The 8-frequency LOFO spectra look the same.
- w₁(f) falls monotonically from 10 to 150 Hz (‖B_f‖_F drops 2.4×). w₂ takes over at 100–150 Hz.

**(i-a) Held-out trials, Δℓ (nats/sample) vs B = 0:**

| model | mean ± SE |
|---|---|
| pooled B | 0.325 ± 0.024 |
| per-condition B | 0.466 ± 0.035 |
| TP r = 1 | 0.120 ± 0.031 |
| TP r = 2 | 0.332 ± 0.027 |
| TP r = 3 | 0.436 ± 0.034 |

- TP r = 1 is wrecked by 150 Hz (−0.95).
- TP needs r = 3 to approach the per-condition model.
- The one-step NRMSE is 0.0034 for every model, so it cannot tell them apart; that metric is dominated by power below 2 Hz.

**(i-b) Leave-one-frequency-out. TP does NOT beat pooled.**

Mean Δℓ vs B = 0:

| model | all 9 freqs | interior 20–100 Hz |
|---|---|---|
| pooled B | **0.366** | **0.499** |
| TP r = 1 | −0.150 | 0.200 |
| TP r = 2 | 0.169 | 0.381 |
| TP r = 3 | 0.171 | 0.420 |
| full-rank interpolation | 0.130 | 0.390 |
| nearest-frequency B | 0.150 | 0.415 |
| oracle (own B) | 0.827 | 0.726 |

- Paired TP − pooled: r = 1 −0.516 ± 0.064, r = 2 −0.197 ± 0.026, r = 3 −0.196 ± 0.033.
- TP wins only 2 of 9 held-out frequencies, at 10 and 20 Hz (for example 0.21 vs 0.09 at 10 Hz).
- 150 Hz is extrapolation, and every interpolating model fails there: TP r = 2 scores −1.34, while the oracle scores +2.22.

**(ii) p = pre-stim power, per trial.** The power is the log band power in the 0.50–0.95 s window, averaged over channels.

Δℓ vs per-frequency B, beta / theta:

| model | beta | theta |
|---|---|---|
| B(f, tertile) | −0.055 ± 0.011 | −0.097 ± 0.015 |
| B_f + p̃·D, shared | +0.005 ± 0.005 | −0.015 ± 0.004 |
| B_f + p̃·D_f | −0.007 | −0.059 |
| TP rank-1 over the power mode | −0.035 | −0.034 |
| A per tertile | −5.9 ± 1.3 | −0.03 ± 0.62 |
| A + p̃A₁ | −0.02 ± 0.08 | −0.09 ± 0.12 |

- The power-mode singular values are [1, 0.14, 0.09] for beta and [1, 0.13, 0.11] for theta.
- Pre-stim state adds nothing to predicting the stim response. This is a null result.

**Read:**
- Moving TP from A to B is the right place: B carries real, frequency-ordered structure with a leading monotone gain mode.
- But under LOFO a single pooled B beats interpolating B(p). The extra frequency-specific components (w₂, w₃) are not smooth in f, and they are dominated by 100–150 Hz.

**Caveat:** the large Δℓ at 70–150 Hz (per-condition 0.48 / 1.53 / 0.69) contradicts sysID's ≈0 above 40 Hz. With 5 input lags and dense pulse trains, B can absorb residual stim artifact, so treat the high-frequency gains as suspect until the artifact is handled.

Figures: `figures/10_ospulse_tp_input.png/.pdf`, `figures/10_ospulse_prestim_state.png/.pdf`. The cache is in `cache/` (git-ignored, 0.97 GB of Gram matrices).

## Stacked TP-GDAR: two-stage vs joint fit  (`11_tp_joint.py`) — 2026-10-26

Model and learning: `docs/MODEL.md`. One stacked operator
Θ(p) = [A_1..A_K | B_1..B_Q] per condition, with TP ranks (r_A, r_B) along p. Two
learners are compared:
- two-stage: per-condition fits, then an unweighted SVD over p;
- joint: alternating least squares on the data loss, which is a Gram-weighted
  low-rank fit with the ridge on the grid-point operators.

Every score is held-out one-step Gaussian log-likelihood per sample, each model with
its own Σ from its training residuals, reported as Δℓ against a pooled (p-blind) model.

**A. Synthetic linear, true rank 2, uneven conditions.** 4×4 grid, K = 2, one input,
7 conditions with data length ×0.3/×2.3 and input power ×0.2/×2, 5 seeds.
- Relative operator error against the truth: per-condition 0.109, two-stage 0.068,
  **joint 0.036**.
- Leave-one-condition-out, interior p: two-stage +3.78, **joint +3.98**, oracle +4.06.
  Joint closes about 70 % of the gap between two-stage and the oracle.

**B. Wilson–Cowan 5×5, g0 = 3.0, A only, held-out p on fresh segments.**

| | r = 1 | r = 2 | r = 3 |
|---|---|---|---|
| two-stage | **−7.85** | +0.745 | +0.744 |
| joint | +0.16 | +0.751 | +0.747 |

Nearest grid point scores +0.72 and the oracle +0.81. Rank 2 is recovered. With equal
data per condition the two learners tie, as the theory predicts. At rank 1 the
unweighted SVD fails badly: the synchronised network makes one regressor direction
dominate G, and a small coefficient error along it is expensive.

**C. Real OS-Pulse 091114 M1.** 78 channels, 8-NN graph, K = 5, Q = 5, 0.1–150 Hz,
**stimulation window 1–2 s**, p = stimulation frequency (9 conditions, 10–150 Hz).
- **Ridge** chosen by validation inside the training trials (fit 0–29, score 30–39):
  1e-5 −2.7, 1e-6 44.6, 1e-8 108.1, **1e-10 114.4** (saturated).
- Test trials 40–59 were used only for the final scores.

| model | held-out trials | LOFO all 9 | LOFO interior 20–100 Hz |
|---|---|---|---|
| B = 0 (no light) | −0.53 | | |
| pooled [A\|B] | 0 | 0 | 0 |
| per-condition | −1.08 | (oracle −0.85) | (oracle −1.60) |
| nearest frequency | | −6.74 | −2.47 |
| two-stage A1B1 … A3B3 | −45 … −7.5 | −38 … −13 | −34 … −6.9 |
| **joint A1B1** | +0.45 | +0.34 | +0.37 (6/7 wins) |
| **joint A1B2** | +0.46 | +0.35 | **+0.37 (7/7 wins)** |
| joint A1B3 | **+0.52** | +0.26 | +0.33 |
| joint A2B1 | +0.38 | −4.73 | −0.68 |
| joint A2B2 | +0.34 | −4.72 | −0.75 |
| joint A3B3 | −0.11 | −5.91 | −1.26 |

**Read:**
- **The light is a real input:** +0.53 nats/sample over B = 0.
- **TP does beat a frequency-blind model, in the right place.** The winning form is a
  shared A with B(f) of rank 1–2 (a frequency-dependent gain on one input kernel), and
  it generalises to *unseen* frequencies: all 7 interior frequencies improve at A1B2.
  This reverses the script 10 conclusion, which came from the over-strong ridge plus
  the two-stage learner.
- **No support for A varying with frequency.** A2/A3 help some conditions in-sample
  but lose under leave-one-frequency-out, which agrees with sysID's finding that A is
  stimulation-invariant.
- **Two-stage HOSVD fails on real data at every rank** (−7 to −45 nats). This is the
  practical case for the joint learner: decomposing per-condition estimates without
  their Gram weighting spends rank on poorly determined coefficients.
- **Per-condition fits overfit** (395 parameters per row, about 40 trials per condition).
  Joint TP sits between pooled and per-condition, which is where a regularised model
  should be.
- **Caveats:**
  - 150 Hz is extrapolation and every interpolating model loses there;
  - the A2 models' leave-one-frequency-out scores are dominated by 70 Hz and 150 Hz;
  - this is one session and one site.

## Caveats to state plainly
- Vertex systems are a **representation, not unique** in general; the robust
  readouts are the graph, the **rank**, the weight trajectory, and (diffusion form)
  the fixed spatial modes. Recovery of specific vertices needs the identifiability
  condition (schedule visits each regime).
- On real data the stim artifact is only partially handled here (induced-activity
  subtraction); a publication analysis needs the lab's reproducibility-based
  artifact treatment. Units/scaling of the ECoG to be confirmed with the owner.

## TP-GDAR vs Felix's GDAR, no input, IOS Reach  (`12_tp_vs_gdar_ios.py`, `12b_tp_vs_gdar_robust.py`) — 2026-11-02

Hydrogen 2021-11-12, a day with no stim: 320 reach trials, 22 channels, 333 Hz, 50 graph
edges. Felix's positions (`32ch_mmad`, identical to `priming`'s Hydrogen grid) and Felix's
1.5-unit threshold. The baseline is Felix's model and estimator (symmetric, graph-supported,
GLS), reimplemented on trial-wise Grams in `gdar_tp/sym_tp.py`; on one trial it matches
`CVARModel.fit_gdar` to a relative difference of 1e-11. TP-GDAR here stays in the same
symmetric class. Held-out trials, 5-fold, K = 16 (48 ms) chosen by CV of the baseline only.
Gain = held-out one-step log-likelihood over GDAR, nats/sample (over 22 channels).

| p = task phase (hold/plan/move/post) | evoked left in (Felix) | evoked removed |
|---|---|---|
| rank 1 | +0.087 ± 0.013 | +0.113 ± 0.016 |
| rank 2 | +0.138 ± 0.009 | +0.218 ± 0.013 |
| rank 3 | +0.167 ± 0.009 | +0.259 ± 0.013 |
| separate fit per phase (rank 4) | +0.180 ± 0.010 | +0.297 ± 0.012 |

| p = 12 bins of 100 ms around go | evoked left in | evoked removed |
|---|---|---|
| rank 3 (3,483 params) | +0.137 | +0.168 |
| rank 4 (4,640) | +0.127 | +0.178 |
| 12 separate fits (13,824) | +0.098 | +0.173 |
| leave-one-bin-out: rank 3, weights interpolated | +0.151 | +0.144 |
| leave-one-bin-out: separate fit on the neighbouring bin | -0.019 | -0.080 |

Trial-level Wilcoxon p < 1e-36 for every TP row of the fitted grids (not across sessions).

**What TP adds:**
- A single GDAR misses real task dependence. The gain is not the evoked response entering
  through A: it grows when evoked kernels are removed.
- Rank 2 takes about 75 % of the per-phase gain with half the parameters.
- It predicts task times it never saw (interior bins). Separate fits cannot: the
  neighbouring-bin fit is worse than GDAR. On 12 bins rank 3-4 equals or beats 12 separate fits at a quarter to a third
  of the parameters. This is the clearest advantage over both GDAR and separate fits.
- The operators give a readable effect: the rank-2 model spectrum near 15 Hz is about +3.7 dB
  in hold and about -4 dB in move and post, relative to GDAR, the pattern expected of beta
  desynchronisation (read from the model, not yet checked against the data spectrum).

**How big, and how robust:**
- Small in absolute terms: 0.138 nats/sample is 0.006 per channel, SSE down 1.9 %. Raising K
  from 5 to 32 gives GDAR +8.3 nats/sample. Forecasts: VAF 18 ms ahead 0.18 to 0.21; 3 ms
  ahead no real change (0.974 to 0.975).
- Holds at K up to 32 (+0.13 evoked left in, +0.25 removed) and with one noise covariance
  per phase, so it is not only a power difference.
- Hold carries a large share: dropping it gives +0.077 / +0.071. Trimming it to -300..-20 ms
  gives +0.122 / +0.173. Plan barely gains at rank 2.
- Spectral radius barely moves with phase with evoked left in (0.964 to 0.965). With evoked
  removed, hold is slower (0.95 vs 0.93).
- Split-half reliability of Θ(p) minus GDAR at rank 2: 0.83 with evoked left in. The 0.99 with
  evoked removed is inflated, since those kernels were fitted on all trials before splitting.

**Edge-wise TP-GDAR (`tpgdar.fit_joint`, not symmetric, row-wise OLS).** Against its own pooled
fit it gains +0.099 / +0.087 / +0.045 (ranks 1-3, evoked left in). Against Felix's GLS fit the
whole family loses on full-covariance likelihood (pooled -0.51) while winning on SSE (+1.9 %)
and diagonal-covariance likelihood (+0.21). OLS ignores noise correlations, which is what
this score weights, so the comparison mostly separates OLS from GLS. Results above use the symmetric fit.

**Limits:** one session, one monkey. K is at the edge of the grid (GDAR likelihood still rises
at 32). The channel-to-position map assumes Noah's channel numbers follow the `32ch_mmad` order
(not confirmed with Noah). Felix fits per trial or segment; here the baseline is pooled over the
day, since a fixed day-wide operator is what TP is compared with.

Figures: `figures/12_tp_vs_gdar_prediction.png`, `figures/12_tp_vs_gdar_operators.png`.

## Vertex graphs and weights across the trial  (`13_tp_vertices_ios.py`) — 2026-11-04

Same day and model as above (H 2021-11-12, K = 16, symmetric TP-GDAR). Each trial is warped
onto p = trial phase: hold, plan, move, post, 4 bins of each phase's own duration. Convex form
Θ(p) = Σ_v λ_v(p) Θ_v, distances in the pooled data metric. Rank 2 vertices are the two
extreme bins. Rank 3 vertices are the corners of the smallest triangle on hull edges that
contains the trajectory, so they need not be visited by the data.

| held-out gain over GDAR, nats/sample | evoked left in | evoked removed |
|---|---|---|
| rank 2 | +0.190 ± 0.009 | +0.248 ± 0.012 |
| rank 3 | +0.222 ± 0.009 | +0.323 ± 0.013 |
| 16 separate fits | +0.211 | +0.392 |

(± is SE over 5 folds. Not comparable with the 4-phase table above: this grid and sample set differ.)

- **Rank 2 reads as a switch.** One vertex holds through hold and early plan, and the other takes
  over during late plan, before the go cue, and stays through move and early post. After touch
  the weights go back: fully with evoked left in, about half way with evoked removed.
- **Vertices differ near 15 Hz.** Rank-2 model spectrum about +3 dB / -5 dB re GDAR at the two
  vertices (evoked left in). Model only, not yet checked against the data spectrum.
- **Weights are tight:** median jackknife SE 0.016 to 0.027, though the 5 training splits overlap.
- **Stability:** vertex spectral radii 0.964, 0.965 with evoked left in; 0.947 (hold), 0.929
  (move) with evoked removed. Single vertices only; no common Lyapunov test was run.

**Limits:** vertices are one choice of coordinates for the fitted family; only the family is
unique. Edge maps sum over lags and hide lag structure. Evoked-left-in rank-3 vertex 2 is 1.01
of the trajectory spread from any bin, so its strong coupling is an extrapolation. With evoked
removed, plan 1/4 is an abrupt outlier at target onset (it becomes a rank-3 vertex), not explained
yet. One session; channel map not confirmed with Noah.

Figures: `figures/13_vertices_{felix,induced}_r{2,3}.png`.

## Does the switch repeat? Other IOS Reach days  (`14_tp_vertices_days.py`) — 2026-11-04

Script 13's fit (K = 16, warped trial phase, 16 bins, 5-fold over trials) on two more Hydrogen days.
Lit and unlit trials of the stim day are fitted separately. On that day a sample is dropped if it or
any of its K lags falls in the light artefact windows (-20..60 and 860..960 ms re target), on both
trial types. Each cell: evoked left in / evoked removed. 11-11 was chosen over 11-16 because it shares
20 channels with 11-12 (11-16 only 14).

| day | trials, channels | rank 2 | rank 3 | 16 separate fits | move-minus-rest edges, r with 11-12 |
|---|---|---|---|---|---|
| 11-12 no stim (reference) | 320, 22 | +0.190 / +0.248 | +0.222 / +0.323 | +0.211 / +0.392 | |
| 08-23 no stim | 404, 16 | +0.221 / +0.266 | +0.254 / +0.317 | +0.278 / +0.384 | 0.15 / 0.47 (16 edges) |
| 11-11 no light | 303, 22 | +0.193 / +0.277 | +0.234 / +0.335 | +0.210 / +0.389 | 0.12 / 0.15 (40 edges) |
| 11-11 light | 307, 22 | +0.242 / +0.400 | +0.287 / +0.496 | +0.292 / +0.610 | 0.21 / 0.13 (40 edges) |

- **The gain and the timing repeat.** Rank 2 gains are similar on every day, and the weight time
  courses overlap closely, especially with evoked left in.
- **The edge maps mostly do not repeat.** Move-minus-rest edge correlation with 11-12 is low (0.12 to
  0.21 with evoked left in); only 08-23 with evoked removed reaches 0.47, on 16 shared edges.
- **Lit trials differ with evoked removed:** the weights jump at plan 1/4, at light onset. There is no
  input term here, so lit vertices can absorb the light; they are not intrinsic dynamics.

**Limits:** one monkey. Weights are 0 to 1 by the choice of extreme bins, so the overlap in range is
partly by construction; the timing is the part that is informative. Some 11-11 rank-3 extrapolated
corners have spectral radius above 1 (1.002, 1.026). Channel map not confirmed with Noah.

Figures: `figures/14_days_comparison.png`, `figures/14_vertices_{08-23_no_stim,11-11_no_light,11-11_light}_{felix,induced}_r2.png`.

## Worked control example: 9 nodes, an input at each node  (`15_nine_node_control.py`) — 2026-11-04

Synthetic, known truth. 3 x 3 grid (12 edges), K = 2, symmetric operators plus one input gain per node
(51 numbers per operator; `symmetric_basis(..., node_inputs=True)`), Theta(p) = (1-p) Theta_0 + p Theta_1.
Learned from 30 trials of 200 samples at each of 7 values of p, random input at every node. 20 plants per
case. One unstable: spectral radius 0.90 at p = 0, 1.03 at p = 1. Two unstable: 1.03 at both, added after
seeing the first case as a stress test. Instability is by design, so this says nothing about cortex.

| median over 20 plants | one unstable | two unstable |
|---|---|---|
| operator error: pooled / rank 1 / rank 2 / rank 3 | 0.177 / 0.169 / 0.004 / 0.005 | 0.305 / 0.281 / 0.004 / 0.005 |
| gain at unseen p over pooled, rank 2 (positive in 20/20) | +1.883 | +4.258 |
| input gains: worst error / spread between vertices | 0.003 / 0.010 | 0.003 / 0.010 |
| LMI feasible on the fitted vertices | 20/20 | 20/20 |
| fitted certificate holds on the true plant: alpha = 0.9 / alpha = 1 | 20/20, 0/20 | 20/20, 0/20 |

| closed loop on the true plant, median | stable at all p | worst true rho | rms x | rms input |
|---|---|---|---|---|
| one unstable: open loop | 0/20 | 1.030 | 9.76 | 0 |
| pooled LQR, R = 1 | 20/20 | 0.534 | 0.311 | 0.126 |
| polytopic, alpha = 0.9, min trace | 20/20 | 0.501 | 0.307 | 0.171 |
| two unstable: open loop | 0/20 | 1.030 | 14.2 | 0 |
| pooled LQR, R = 1 | 20/20 | 0.567 | 0.314 | 0.169 |
| polytopic, alpha = 0.9, min trace | 20/20 | 0.506 | 0.308 | 0.184 |

- **Learning works.** Rank 2 reaches the truth; a single model cannot, and rank 3 adds nothing.
- **Pooled LQR also stabilises every plant, at every setting tried, in both cases.** With an input at every
  node the gain can nearly cancel the mean operator, so this example does not need a polytopic controller
  for stability. The pooled model's own spectral radius is above 1 in all 40 plants, so it sees the
  instability.
- **Effort.** With the LMI set to minimise the gain, the same worst-case rho takes less input: 15 to 35 %
  less with two unstable ends; with one, even down to rho = 0.8 and 10 to 17 % less below that. Comparing
  rms x per unit of input, the two are about even. The polytopic worst rho lands on the design alpha.
- **Margin matters.** At alpha = 1 the solution sits on the LMI boundary and the fitted certificate never
  survives on the true plant. At alpha = 0.9 it always does.
- **The certificate comes from the fitted vertices, not from the LMI.** Testing the pooled LQR gain (R = 1) on
  the fitted vertices the same way, then on the true plant at rate 1: certified at rate 0.9 in 20/20 and 19/20,
  and at 0.8 in 20/20 and 19/20 (one unstable, two unstable). A rate-0.6 certificate exists on the fitted
  vertices in only 10/20 and 0/20, whereas the LMI reaches alpha = 0.6 in all 40 (`--lqr-cert`, slow).
- **So on this example** TP-GDAR's vertices are what matter. They let any fixed gain be certified across p. The
  LMI on top reaches a chosen rate directly and saves some input for it; it does not stabilise anything LQR can't.

**A learner bug this exposed.** With two unstable ends, the data at those ends hold about 10^4 times the
energy of the rest. `fit_sym_tp` from its original (pooled) start stalled at rank-2 error 0.21 even though the
truth is rank 2, and the certificate then held in only 2 of 20 plants. A strict stopping rule did not fully cure
it: 2 of 6 plants tested stayed worse than the truth's own objective. Added `starts=("pooled", "svd")`: the
unweighted SVD of the per-condition fits as a second start, lowest objective wins. Error 0.004 on 20/20.
**The real-data numbers don't change:** on H 2021-11-12 (16 bins, K = 16, rank 2 and 3, evoked in and removed)
a strict stop and the svd start move the held-out gain by at most 0.0001. Scripts 12 to 14 keep the default
start. The other days were not re-checked.

**Limits:** designed plants, linear, full state, no delay or actuator limits, one input per node. Real
stimulation has few sites and a nonlinear plant. The closed-loop traces use a slow schedule of p; arbitrarily
fast p is covered by the certificate only, not simulated.

Figures: `figures/15_nine_node_learning.png`, `figures/15_nine_node_control.png`. Log: `cache/15_run.log`.
