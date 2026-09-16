# CLAUDE.md — GDAR-TP (tpgdar)

Project-specific notes. The global `~/.claude/CLAUDE.md` (Research Hub loop) still
applies; this file adds detail and overrides it on conflict.

## What this is

A parameter-varying **G**raph **D**iffusion **A**uto**R**egression model on array
graphs, decomposed via **T**ensor-**P**roduct model transformation (HOSVD) into
vertex LTI systems, for interpretable dynamics + LMI stimulation control. This
project is the concrete home for the "GDAR model fits" placeholder in the brain-
paper grant (Aim 3). See `README.md` for the model and stage results, and
`gdar_tp_plan.md`-style staging (Stages 1–2 + 4–5 built on synthetic data;
Stage 3 real data is deferred).

## Two-PC workflow (per global setup)

- **Pull before you start, push before you leave.** Code + docs sync via git; the
  60 GB of real recordings never enters git and lives on the lab PC only.
- Data, `figures/*.png`, and `.venv/` are git-ignored. Keep the repo code-only.

## Environment

- Python **3.14** in `.venv/` (bare system Python otherwise). Deps:
  numpy, scipy, networkx, matplotlib, tensorly, cvxpy (SCS/Clarabel), pytest.
- Run things through `.venv\Scripts\python`. Tests: `.venv\Scripts\python -m pytest`.

## Conventions that matter (don't re-derive)

- **Input timing:** `x(t+1) = Σ_k A_k x(t-k) + B u(t)` — the input at time `t`
  drives the state at `t+1`. `sim/linear.py` and `GDARModel.predict_one_step`
  both follow this; a mismatch here silently zeroes the recovered `B`.
- **Fitting WC:** always mean-center each segment before fitting GDAR (the linear
  model describes deviations from the operating point).
- **WC coupling regime:** the demo uses `WCParams(g0=3.0)`, not the dataclass
  default `0.9`. At `g0≈0.9` the network coupling is only ~6% of the local E-I
  circuit, so the scheduling variable barely moves the fitted coefficients
  (HOSVD looks rank-1). `g0≈3.0` puts coupling on par with the local circuit and
  gives the clean rank-2 parameter dependence the TP story needs.
- **Control loop is closed on the WC plant, never on the fitted model** (plan
  risk row 4). `sim/closed_loop.py` drives the true nonlinear network.
- At high `p` the WC synchronizes, so the *fitted* inter-channel AR coupling
  shrinks — the low-`p` vertex has the stronger coupling map. Report this
  honestly; it's not a bug.

## Open decision

Strict diffusion (`a_k I − b_k L`) vs. edge-wise + ridge parameterization — both
implemented; decide against real-data residuals (Stage 1 item in the plan).
