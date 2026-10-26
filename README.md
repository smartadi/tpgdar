# GDAR-TP

**Graph Diffusion AutoRegression with Tensor-Product vertex structure** — a
parameter-varying GDAR model on array (ECoG / Utah) graphs, decomposed via TP
model transformation (HOSVD) into vertex LTI systems, for (a) interpretable
dynamic communication modes and (b) LMI-based closed-loop stimulation control.

The full synthetic pipeline (graph → parameter-varying fit → TP decomposition →
LMI control, all on a Wilson–Cowan example plant) is built and tested, and it runs
on the real OS-Pulse ECoG (`scripts/10_*`, `scripts/11_*`).

## Model

**Full math and the learning procedure: [`docs/MODEL.md`](docs/MODEL.md).** In short,
the lag matrices and input matrices form **one stacked operator**, and the TP
structure lives on that operator along the p mode only:

```
z(t)   = [x(t); …; x(t-K+1); u(t); …; u(t-Q+1)]
x(t+1) = Θ(p) z(t) + e(t),        Θ(p) = [A_0(p) … A_{K-1}(p) | B_0(p) … B_{Q-1}(p)]
Θ(p)   = Σ_j w_j(p) Θ_j           # TP form; A and B blocks may carry their own ranks
supp(A-blocks of Θ_j) ⊆ E ∪ diag  # graph-supported; strict diffusion A_k = a_k I − b_k L also available
```

**Learning.** Minimise the data loss of the whole TP model,
`J = Σ_g ‖x(t+1) − Θ(p_g) z(t)‖² + ridge`. That is a G_g-weighted low-rank fit
(`gdar_tp/tpgdar.py: fit_joint`, alternating least squares from per-trial
sufficient statistics). The older two-stage route (per-condition fits, then an
unweighted HOSVD) minimises the wrong distance and is kept only as a baseline;
see `RESULTS.md` § Stacked TP-GDAR.

**Data rules that change the answer if broken.** These are in `docs/MODEL.md` §5:
- never fit across trial boundaries;
- never z-score per condition;
- u(t) drives x(t+1);
- the OS-Pulse stimulation window is **1.0–2.0 s**. This was measured from the
  response locked to the stimulation frequency; the export README's 1–6 s is wrong.

## Install

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # numpy scipy networkx matplotlib tensorly cvxpy pytest
```

(The repo ships a `.venv/` locally; it is git-ignored.)

## Quickstart

```bash
.venv\Scripts\python scripts/01_synthetic_gdar.py    # Stage 1: GDAR beats VAR on flow recovery
.venv\Scripts\python scripts/02_synthetic_tp.py      # Stage 2: HOSVD -> rank-2, held-out p generalization
.venv\Scripts\python scripts/04_interpret_modes.py   # Stage 4: vertex maps + weight tracking
.venv\Scripts\python scripts/05_control_insilico.py  # Stage 5: LMI feedback closes the loop on WC
.venv\Scripts\python scripts/11_tp_joint.py      # stacked TP-GDAR: two-stage vs joint (synthetic, WC, OS-Pulse)
.venv\Scripts\python -m pytest                       # 44 tests
```

Figures land in `figures/` (git-ignored).

## What the example shows

- **Stage 1** — on a linear graph-diffusion process, structural GDAR recovers the
  coupling (flow-signal corr 0.99 edge-wise / 1.00 diffusion) and beats
  unconstrained VAR (0.90) with 4 vs 2592 parameters; residuals are white.
- **Stage 2** — a WC network whose coupling varies with a scheduling variable `p`:
  HOSVD finds a clean **rank-2** parameter dependence (reconstruction error drops
  7× from 1→2 vertices, then plateaus); the low-dim TP model predicts held-out
  `p` as well as an oracle fit at that `p`.
- **Stage 4** — each vertex is a spatial diffusion mode; the convex weight
  `w_v(t)` tracks a time-varying schedule (corr 0.997).
- **Stage 5** — a single LMI state-feedback gain quadratically stabilizes the
  2-vertex polytope (closed-loop ρ ≈ 0.75) and, run on the *nonlinear* WC plant
  with noise, loop delay and actuator limits, cuts the activity fluctuation ~23%.
- **Real OS-Pulse ECoG** (`scripts/11_tp_joint.py`, stimulation window 1–2 s,
  validated ridge): the light is a real input (+0.53 nats/sample). The jointly fitted
  TP model with a shared A and a frequency-dependent B (r_B = 1–2) beats a
  frequency-blind GDAR on held-out trials (+0.46) **and on held-out frequencies**
  (+0.37, all 7 interior frequencies). There is no support for A varying with
  frequency. Two-stage HOSVD fails here at every rank. Details: `RESULTS.md`.

## Layout

```
gdar_tp/
  graph.py       grid graph, Laplacian, Kronecker-sum, support mask
  gdar.py        GDAR fit (edge-wise masked ridge / strict diffusion), flow signal
  qlpv.py        parameter-varying fit (gridded / windowed) -> coefficient tensor
  tp.py          HOSVD over parameter modes; convex-hull polytope for control
  tpgdar.py      stacked TP-GDAR: sufficient stats, two-stage and joint (ALS) fits
  control.py     LMI polytopic state feedback (cvxpy), companion form, checks
  metrics.py     prediction error, flow correlation, rank/energy
  io.py          placeholder (real-data loading lives in scripts/09-11)
  viz.py         plotting helpers
  sim/
    linear.py        linear GDAR generator (exact-recovery ground truth)
    wilson_cowan.py  WC oscillator network on a graph, p-varying coupling + stim
    closed_loop.py   controller-on-plant loop with delay, noise, actuator limits
scripts/  01..11  (03 is a stub; 09 is superseded, see RESULTS.md; 10-11 real OS-Pulse)
docs/     MODEL.md (the math), PLAN.md (original staging)
tests/            graph / gdar / tp / tpgdar / latent / control / closed_loop / review fixes
```

## Open decision (Stage 1)

Strict diffusion (`A_k = a_k I − b_k L`, 2 params/lag) vs. edge-wise + ridge.
Both are implemented and pass tests; on the synthetic linear data the diffusion
form recovers coupling exactly with 4 parameters. Revisit against real-data
residuals before committing.
