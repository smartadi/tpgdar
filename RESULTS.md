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
