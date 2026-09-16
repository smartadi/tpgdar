# GDAR-TP

**Graph Diffusion AutoRegression with Tensor-Product vertex structure** — a
parameter-varying GDAR model on array (ECoG / Utah) graphs, decomposed via TP
model transformation (HOSVD) into vertex LTI systems, for (a) interpretable
dynamic communication modes and (b) LMI-based closed-loop stimulation control.

The full synthetic pipeline (graph → parameter-varying fit → TP decomposition →
LMI control, all on a Wilson–Cowan example plant) is built and tested. Real
ECoG/Utah data is Stage 3 — deferred (see `gdar_tp/io.py`).

## Model

```
x(t+1) = sum_k A_k(p(t)) x(t-k) + B(p(t)) u(t) + e(t)
A_k(p) = sum_j w_j(p) A_k^(j)          # TP form, w_j convex
supp(A_k^(j)) ⊆ E ∪ diag                # graph-supported
A_k^(j) = a_k^(j) I - b_k^(j) L         # strict diffusion, or edge-wise + ridge
```

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
.venv\Scripts\python -m pytest                       # 27 tests
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

## Layout

```
gdar_tp/
  graph.py       grid graph, Laplacian, Kronecker-sum, support mask
  gdar.py        GDAR fit (edge-wise masked ridge / strict diffusion), flow signal
  qlpv.py        parameter-varying fit (gridded / windowed) -> coefficient tensor
  tp.py          HOSVD over parameter modes; convex-hull polytope for control
  control.py     LMI polytopic state feedback (cvxpy), companion form, checks
  metrics.py     prediction error, flow correlation, rank/energy
  io.py          real-data loaders (Stage 3, deferred)
  viz.py         plotting helpers
  sim/
    linear.py        linear GDAR generator (exact-recovery ground truth)
    wilson_cowan.py  WC oscillator network on a graph, p-varying coupling + stim
    closed_loop.py   controller-on-plant loop with delay, noise, actuator limits
scripts/  01..05  (03 real-data + 04 interpret; 03 is a deferred stub)
tests/            graph / gdar / tp / control / closed_loop
```

## Open decision (Stage 1)

Strict diffusion (`A_k = a_k I − b_k L`, 2 params/lag) vs. edge-wise + ridge.
Both are implemented and pass tests; on the synthetic linear data the diffusion
form recovers coupling exactly with 4 parameters. Revisit against real-data
residuals before committing.
