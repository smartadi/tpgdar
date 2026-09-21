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
| 8 | 0.298 | 0.371 | 0.499 | **−0.128** |
| 25 | 0.091 | 0.613 | 0.709 | −0.096 |
| 60 | 0.077 | 0.560 | 0.492 | **+0.068** |

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

## Real data — OS-Pulse ECoG  (`09_ospulse_real.py`)
Monkey Jalapeno, 96-ch uECoG (session 091114_M1), stim frequency (10–150 Hz) as
the scheduling variable; 8-NN support from electrode positions; induced activity
(trial-mean removed) to drop the phase-locked stim artifact.

*(Filled in when the run completes — see the script's printed output: HOSVD rank
over frequency, recovered vertex graphs, leave-one-frequency-out TP-vs-pooled
prediction, and coupling-vs-distance correlation.)*
Figs: `09_ospulse_overview.png`, `09_ospulse_vertices.png`.

## Caveats to state plainly
- Vertex systems are a **representation, not unique** in general; the robust
  readouts are the graph, the **rank**, the weight trajectory, and (diffusion form)
  the fixed spatial modes. Recovery of specific vertices needs the identifiability
  condition (schedule visits each regime).
- On real data the stim artifact is only partially handled here (induced-activity
  subtraction); a publication analysis needs the lab's reproducibility-based
  artifact treatment. Units/scaling of the ECoG to be confirmed with the owner.
