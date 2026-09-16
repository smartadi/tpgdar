# GDAR-TP: Graph Diffusion Autoregression with Tensor Product Vertex Structure

Implementation plan for a parameter-varying GDAR model on ECoG/Utah array graphs,
decomposed via TP model transformation (HOSVD) into vertex LTI systems, for
(a) interpretable dynamic communication modes and (b) LMI-based closed-loop stimulation control.

---

## 0. Model

```
x(t+1) = sum_k A_k(p(t)) x(t-k) + B(p(t)) u(t) + e(t)

A_k(p) = sum_j w_j(p) A_k^(j)          # TP form, w_j convex weights
supp(A_k^(j)) ⊆ E ∪ diag                # graph-supported (array edges)
A_k^(j) = a_k^(j) I - b_k^(j) L         # diffusion constraint (L = graph Laplacian), or edge-wise free with L-regularizer
```

- `x` : channels (N)          `u` : stimulation inputs (M)
- `p` : scheduling variable(s), measurable online (stim amplitude, band power, behavioral phase, latent state)
- `L` : Laplacian of array grid; for a rectangular grid `L = L_row ⊕ L_col` (Kronecker sum)

Open decision (resolve in Stage 1): strict diffusion parameterization vs. edge-wise coefficients with a
Laplacian penalty. Strict form gives tiny parameter count and clean LMIs; edge-wise form matches GDAR's
flow-signal derivation more directly.

---

## 1. Repo layout

```
gdar_tp/
  graph.py         # grid/array graph construction, Laplacian, Kronecker-sum helpers
  gdar.py          # GDAR fit (LS with structural constraint), flow signal, residual diagnostics
  qlpv.py          # parameter-varying fit: gridded / windowed / basis-expanded coefficients
  tp.py            # HOSVD over the parameter grid, convex hull (SNNN / CNO / IRNO), vertex systems
  control.py       # LMI synthesis (cvxpy), polytopic state feedback, stability/robustness checks
  sim/
    wilson_cowan.py   # WC oscillator network on a graph, with parameter-varying coupling + stim input
    closed_loop.py    # plant-controller loop with delay, noise, actuator limits
  metrics.py       # prediction error, flow-signal correlation, rank/energy, tracking error
  io.py            # loaders for ECoG / Utah data, alignment to stim + behavior
  viz.py
scripts/
  01_synthetic_gdar.py
  02_synthetic_tp.py
  03_fit_real_data.py
  04_interpret_modes.py
  05_control_insilico.py
tests/
notebooks/
```

Dependencies: numpy, scipy, tensorly (HOSVD), cvxpy (+ SCS/MOSEK/Clarabel for LMIs), networkx, matplotlib, pytest.

---

## 2. Stages and deliverables

### Stage 1 – Baseline GDAR on synthetic data
- [ ] `graph.py`: build 8x8 / 16x16 grid, return `A_adj`, `L`, edge list; test Kronecker-sum identity.
- [ ] `sim/wilson_cowan.py`: WC network on the grid, fixed coupling, optional external drive.
- [ ] `gdar.py`: fit lag-K model with edge support + Laplacian constraint; derive edge-wise flow signal.
- [ ] Reproduce sanity results: GDAR flow signal vs. true coupling beats unconstrained VAR.
- [ ] Residual diagnostics: whiteness (Ljung-Box), stationarity across windows.
- **Metric**: 1-step / 10-step NRMSE; flow-signal correlation with ground-truth coupling.

### Stage 2 – Parameter-varying fit + TP decomposition (synthetic)
- [ ] Extend WC sim: coupling varies with a scheduling variable `p` (1D first, then 2D, e.g. stim amp × arousal).
- [ ] `qlpv.py`: fit `A_k(p)` on a grid of `p` values (windowed fits) → coefficient tensor `[p1, p2, ..., N, N, K]`.
- [ ] `tp.py`: HOSVD along parameter modes; retain singular values by energy; build SNNN and CNO hulls; return vertex systems + weight functions.
- [ ] Check: recovered number of vertices matches true mode count; interpolated model on held-out `p` values predicts well.
- **Metric**: singular-value spectrum, held-out `p` NRMSE, vertex-system reconstruction error vs. `# vertices`.

### Stage 3 – Real data (existing NHP ECoG / Utah recordings)
- [ ] `io.py`: load, bandpass, downsample, align to stim events and behavioral labels.
- [ ] Choose `p` candidates: stim amplitude, per-band power, reach phase, low-dim latent (PCA/LDS).
- [ ] Windowed GDAR fits → coefficient tensor → TP.
- [ ] Cross-validation: held-out trials and held-out conditions.
- [ ] Baselines: static GDAR, plain VAR, per-condition GDAR without TP.
- **Metric**: CV prediction error; rank/energy curve; generalization gap to unseen condition.

### Stage 4 – Interpretability of vertex systems
- [ ] `04_interpret_modes.py`: map each vertex system's flow pattern onto the array layout; compare to stim-evoked flows and anatomy.
- [ ] Baseline: spectral clustering of raw GDAR flow signals.
- [ ] Track weight functions `w_j(p)` over trials: do they switch with behavior/stim in a sensible way?
- **Output**: figure set (mode maps, weight trajectories, singular-value spectrum).

### Stage 5 – Control in silico
- [ ] `control.py`: polytopic state feedback via LMIs (quadratic stability first, then parameter-dependent Lyapunov).
- [ ] Actuator model: stim input `u` mapped to channels via `B(p)`; enforce amplitude/charge limits.
- [ ] `sim/closed_loop.py`: run controller on WC plant (not the fitted model) with loop delay and observation noise.
- [ ] Target: drive edge-wise flow toward a desired pattern / suppress a pathological mode.
- [ ] Robustness sweeps: unmodeled nonlinearity, delay 5–50 ms, hull type, number of retained vertices.
- **Metric**: tracking error, settling time, LMI feasibility vs. hull, stability margin.

### Stage 6 – Toward in vivo
- [ ] Latency budget: acquisition → `p` estimate → `w_j(p)` → control law → stim, must fit in one control tick.
- [ ] Safety envelope: hard clamps on `u`, watchdog on residual energy.
- [ ] Offline replay of the controller on recorded data before any live test.

---

## 3. Key risks / open questions

| Risk | Mitigation |
|---|---|
| `p` not measurable online | Prefer band power or stim amplitude; latent-state option needs a fast estimator |
| Residuals non-white under parameter variation | Increase K, add basis expansion in `p`, or move `p` to a finer grid |
| LMI infeasible / too conservative | Compare SNNN vs. CNO hulls; reduce vertices; parameter-dependent Lyapunov |
| Fitted model ≠ plant | Always close the loop on the WC simulator, never on the fitted model |
| Overfitting on windowed fits | Ridge on coefficients, shared support across windows, nested CV |

---

## 4. Testing

- `tests/test_graph.py`: Laplacian properties, Kronecker-sum equivalence.
- `tests/test_gdar.py`: recovers known coefficients on linear synthetic data; support constraint respected.
- `tests/test_tp.py`: exact recovery when coefficient tensor is truly low-rank; weights sum to one and are non-negative.
- `tests/test_control.py`: closed loop stable on a toy 2-vertex polytope; controller respects limits.
- `tests/test_closed_loop.py`: end-to-end on small WC grid, tracking error below threshold.

---

## 5. First week

1. Stage 1 complete with tests passing.
2. WC simulator with 1D scheduling variable.
3. Coefficient tensor + HOSVD spectrum plotted (Stage 2 partial).
4. Decide diffusion parameterization (strict vs. edge-wise) based on Stage 1 residuals and flow accuracy.
