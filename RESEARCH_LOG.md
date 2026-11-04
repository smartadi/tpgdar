# Research log: GDAR-TP

Newest on top. Format and rules: research-hub `agents/logging.md`. One entry per logical unit of work, committed together with that work. Issues and the hub worklog cite entries by ID.

### L-2026-11-04-e · Deck with the MPC replay, TP-GDAR, other days and the 9-node control example
commit: ef89a4a+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Added `scripts/make_mpc_tp_deck.py` and `GDAR-TP_MPC_and_TP-GDAR.pptx` (30 slides). Part 1: MPC replay from the NC draft (`Closedloop_nc/results.tex`, 5a955c7), every number on those slides matched against that text; panel m of Figure 6 left out because `brain_paper/TASKS.md` flags it. Part 2: TP-GDAR model, training, HOSVD vs joint fit, vertices, result figures, other days. Part 3: the 9-node example. Refactored `scripts/make_tp_math_deck.py` into slide functions so both decks share them, and rebuilt `GDAR-TP_model_training.pptx` for the wording and limitations changes.
**State:** done. Exported through PowerPoint and new slides checked by eye. The pptx skill's schema validator can't load its schemas on this PC, so no schema check ran. The MPC slides use the locked numbers of 2026-10-16; the 2026-10-10 audit (0.78x, 0.65x at lag 0) predates them and wasn't re-checked. "About 40% / 80%" are ratios of medians (0.61, 0.19); the per-trial median ratios are 0.68, 0.22, and the slide says which.
**Resume:** edit the script and re-run `.venv\Scripts\python scripts/make_mpc_tp_deck.py`; it needs `figures/12_*`, `13_*`, `14_days_comparison.png`, `15_*` and `pdftoppm` for Figure 6.

### L-2026-11-04-d · 9-node control example: learning works, and the certificate comes from the vertices
commit: e9e6971+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:**
- Added an input at each node to symmetric TP-GDAR (`symmetric_basis(..., node_inputs=True)`), a decay rate and a min-gain objective to the LMI (`synthesize_quadratic_feedback(decay=, min_gain=)`), and `scripts/15_nine_node_control.py`: sparse 3 x 3 graph, K = 2, 20 plants, two designed cases (one or two unstable ends; the second added after seeing the first), loop closed on the true plant only.
- Rank 2 recovers the truth: operator error 0.004 against 0.177 and 0.305 pooled, input gains within 0.003. LMI feasible 20/20. Certificate holds on the true plant at alpha 0.9 in 20/20, at alpha 1 in 0/20 (the solution sits on the LMI boundary).
- Pooled LQR also stabilises every plant at every setting, since an input at every node makes this easy. For the same worst-case rho the polytopic gain saves some input (15 to 35 % with two unstable ends; even down to rho 0.8 and 10 to 17 % below that with one). rms x per unit of input is about even.
- The fitted vertices certify the LQR gain too: rate 0.9 and 0.8 hold on the true plant in 19 or 20 of 20. A rate-0.6 certificate exists for it in only 10/20 and 0/20, whereas the LMI reaches 0.6 in all 40. So the certificate comes from TP-GDAR's vertices, and the LMI adds a little on top.
- Learner bug found: with two unstable ends (energy 10^4 times the rest) the pooled start of `fit_sym_tp` stalled at rank-2 error 0.21, and the certificate then held in 2/20. Added `starts=("pooled", "svd")`, default unchanged: error 0.004 on 20/20. On H 2021-11-12 a strict stop and the svd start move the held-out gains by at most 0.0001, so scripts 12 to 14 stand; the other days were not re-checked. 50 tests pass.

**State:** done. Numbers in `RESULTS.md`, figures `figures/15_*`, cache `cache/15_nine_node.pkl`, log `cache/15_run.log` (git-ignored). Synthetic, linear, full state, no delay or actuator limits, one input per node, slow p schedule: says nothing about cortex.
**Resume:** the case that could separate the controllers is few inputs, as with real stimulation. `.venv\Scripts\python scripts/15_nine_node_control.py [--plot | --lqr-cert]` (full run about 10 min, `--lqr-cert` about 30).

### L-2026-11-04-c · TP-GDAR on other IOS Reach days: the timing repeats, the edge maps mostly don't
commit: 5fef213+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:**
- Added `scripts/14_tp_vertices_days.py`: script 13's fit on H 2021-08-23 (no stim, 404 trials, 16 channels) and H 2021-11-11 (stim day; 303 unlit and 307 lit trials, 22 channels, fitted separately, artefact windows dropped).
- Rank 2 held-out gain over GDAR, evoked left in / removed: 08-23 +0.221 / +0.266, 11-11 no light +0.193 / +0.277, 11-11 light +0.242 / +0.400 (11-12: +0.190 / +0.248). Weight time courses overlap closely across days.
- Move-minus-rest edge correlation with 11-12 is low: 0.15, 0.12, 0.21 with evoked left in; 0.47, 0.15, 0.13 with evoked removed. So timing repeats, edge maps mostly don't.
- On lit trials the evoked-removed weights jump at light onset. There is no input term, so lit vertices can absorb the light.
- Small wording fixes: Felix's name instead of a guessed pronoun in script 12 and `RESULTS.md`, and the broken run line in script 12b.

**State:** done for these days. Numbers in `RESULTS.md`, figures `figures/14_*` and cache `cache/14_days.pkl` (git-ignored). Weights run 0 to 1 by the choice of extreme bins, so overlap in range is partly by construction. One monkey; channel map not confirmed with Noah.
**Resume:** a fit with an input term on the lit trials, to separate the light from the intrinsic change. `.venv\Scripts\python scripts/14_tp_vertices_days.py --plot` redraws from the cache.

### L-2026-11-04-b · Plain deck on the TP-GDAR model and training, with rendered equations
commit: 0225898+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Added `scripts/make_tp_math_deck.py` and `GDAR-TP_model_training.pptx` (11 slides: Felix's GDAR, TP structure, warped trial phase, training in 3 steps, vertices, evaluation, result, limitations). Equations are images (matplotlib mathtext, LaTeX in the alt text), so they are not editable in PowerPoint. No new analysis.
**State:** done. Opened and exported through PowerPoint and checked by eye. The pptx skill's schema validator could not load its schemas on this PC (same error on the older overview deck), so no schema check ran.
**Resume:** edit the script and re-run `.venv\Scripts\python scripts/make_tp_math_deck.py`; it needs `figures/13_vertices_felix_r2.png` from script 13.

### L-2026-11-04-a · Vertex graphs of TP-GDAR and their weights across the trial (no-stim IOS Reach)
commit: e75b0aa+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:**
- Added `polytope_vertices` to `gdar_tp/sym_tp.py` (convex form in the pooled data metric; rank 2 extreme grid points, rank 3 smallest enclosing triangle on hull edges) and `scripts/13_tp_vertices_ios.py`. A first try with 3 grid points as rank-3 vertices left the trajectory up to 52 % of its spread outside (evoked left in), so it was replaced. 48 tests pass.
- H 2021-11-12, K = 16, p = trial phase warped per trial (hold, plan, move, post, 4 bins each). Held-out gain over GDAR, evoked left in / removed: rank 2 +0.190 / +0.248, rank 3 +0.222 / +0.323, 16 separate fits +0.211 / +0.392.
- Rank 2 is a two-state switch: one vertex through hold and early plan, the other from late plan (before the go cue) through move and early post, then back or part way back after touch. Vertex 1 vs 2 model spectrum differs by about 8 dB near 15 Hz. Weights are tight (median jackknife SE 0.016 to 0.027). Sum-to-one error 0.033 / 0.037.
- Spectral radius barely separates the vertices with evoked left in (0.964, 0.965); with evoked removed the hold vertex is slower (0.947 vs 0.929).

**State:** done for one session. Numbers in `RESULTS.md`, figures `figures/13_vertices_{felix,induced}_r{2,3}.png`, cache `cache/13_*` (git-ignored). Vertices are one coordinate choice for the family; rank-3 corners are not visited by the data (evoked left in, vertex 2 is 1.01 of the spread from any bin). With evoked removed, plan 1/4 is an abrupt outlier: not explained yet. Channel map still unconfirmed.
**Resume:** replicate on H 2021-08-23 to see whether the switch and vertex maps repeat; look into plan 1/4; check the 15 Hz change against the data spectrum. `.venv\Scripts\python scripts/13_tp_vertices_ios.py [--plot]`.

### L-2026-11-02-a · TP-GDAR vs Felix's GDAR, no input, on a no-stim IOS Reach day
commit: 5dcdc2e+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:**
- Added `gdar_tp/sym_tp.py`: TP-GDAR in Felix's model class (symmetric, graph-supported, GLS), 3 tests, 47 pass. The pooled fit matches `CVARModel.fit_gdar` to 1e-11 on one trial. Added scripts 12 (comparison, plots) and 12b (robustness).
- H 2021-11-12, 320 trials, 22 ch, K = 16, 5-fold over trials. p = task phase: rank 2 +0.138 ± 0.009 nats/sample over GDAR with evoked left in, +0.218 ± 0.013 with evoked removed; per-phase +0.180 / +0.297.
- p = 12 time bins around go, leave-one-bin-out: rank 3 with interpolated weights +0.151 / +0.144; a separate fit on the neighbouring bin -0.019 / -0.080. On the fitted grid rank 3-4 equals or beats 12 separate fits at a quarter to a third of the parameters.
- Holds to K = 32 and with per-phase noise covariance. Hold is a large share (hold dropped: +0.077 / +0.071). Absolute size is small: 0.006 nats per channel, SSE -1.9 %; K 5 to 32 gives GDAR +8.3 nats. Rank-2 model spectrum near 15 Hz: about +3.7 dB in hold, about -4 dB in move and post.
- Edge-wise `fit_joint` loses to Felix's GLS on full-covariance likelihood (pooled -0.51) but wins on SSE and diagonal likelihood: mostly OLS vs GLS, not TP.

**State:** done for one session. Full numbers in `RESULTS.md`. Figures `figures/12_tp_vs_gdar_{prediction,operators}.png`, caches `cache/12_*`, `cache/12b_*` (both git-ignored). Channel-to-position map assumes Noah's numbering follows `32ch_mmad`: not confirmed.
**Resume:** replicate on H 2021-08-23 (406 trials, no stim), check the 15 Hz change against the data spectrum, confirm the channel map with Noah. `.venv\Scripts\python scripts/12_tp_vs_gdar_ios.py` takes about 1.8 h; add `--plot` to redraw from cache.

### L-2026-10-26-c · Spec: latent directed graph (d nodes × n-dim LTI) as a low-d representation of the array graph
commit: 09ca61b+this · branch: master · issues: — · pc: Pinnacle
**Did/Found:**
- Wrote `docs/LATENT_GRAPH.md`. The model has d latent patches, each an n-dim LTI node, coupled by a directed sparse W through the node outputs: A = blkdiag(F_i) + E W C_ξ. The observation is x = P C_ξ ξ, with P ≥ 0 a smooth footprint on the electrode graph. W is restricted to the coarsened support supp(Pᵀ A_N P); there is a directed-Laplacian variant.
- The doc also covers gauges (observer-canonical nodes, ‖P_i‖ = 1), the init (graph coarsening → patch means → masked VARX), EM with structured M-steps, model selection, the TP/control tie-in and a literature map.
- The node dynamics carry the pulse-response shape, so the input needs no time embedding.

**State:** specification only, no code.
**Resume:** implement Stage 0 (coarsen L_N → y₀ = P₀⁺x → masked VARX via `tpgdar.fit_static`) on 091114 M1 and compare its held-out log-likelihood with script 11; then EM (extend `latent.py`).

### L-2026-10-26-b · Stacked TP-GDAR: math written down, joint (Gram-weighted) learner, real OS-Pulse re-test
commit: (this commit) · branch: master · issues: #66, #65 · pc: Pinnacle
**Did:**
- Wrote `docs/MODEL.md`. It covers the stacked operator Θ(p) = [A lags | B lags] with TP along p only, the learning objective, the preprocessing rules and the control caveat (the companion lift is affine, so control needs convex weights).
- Added `gdar_tp/tpgdar.py`:
  - per-trial sufficient statistics;
  - a two-stage learner (per-condition fits, then an unweighted SVD);
  - a joint ALS learner on the data loss, with the ridge on the grid-point operators (augmented Gram), so it is gauge-invariant.
- Added 6 tests, including the per-condition ≤ joint ≤ pooled bracket. 44 tests pass.
- Added `scripts/11_tp_joint.py`, which runs the comparison on synthetic data, on Wilson–Cowan and on real OS-Pulse.
- Re-ran script 07 under the exact M-step.
- Updated `README.md` and `RESULTS.md`.

**Found:**
- **Two-stage HOSVD minimises the wrong distance.** The data loss is ‖(Θ − Θ̂_g) G_g^{1/2}‖², and the unweighted SVD ignores the G_g weighting. Operator error on synthetic data with uneven conditions: joint 0.036, two-stage 0.068. On Wilson–Cowan at rank 1, two-stage scores −7.85 nats against +0.16 for joint. On real data, two-stage fails at every rank (−7 to −45).
- **Stimulation window.** Stimulation lasts 1.0–2.0 s, measured from the response locked to the stimulation frequency. Script 09 used 1.5–5.5 s and is invalid (marked superseded).
- **Ridge.** Script 10's ridge of 1e-6 is about four decades too strong (held-out +84 nats/sample at 1e-10), so script 10's numbers are superseded. The ridge is now chosen inside the training trials.
- **Real result.**
  - The light adds +0.53 nats.
  - Joint A1B2 (shared A, B rank 2 over frequency) beats pooled on held-out trials (+0.46) and on unseen frequencies (+0.37, interior 7/7). This reverses script 10's null.
  - A varying with frequency does not generalise (leave-one-frequency-out −0.7 to −1.3).
- **First bug, caught and fixed.** A ridge on the TP factors leaks through the W/Θ gauge: the joint fit beat the per-condition fit in-sample, which is impossible.
- **Latent re-run.** The slow-I gain falls from +0.068 to +0.040. The conclusions stand.

**State:** committed and pushed. `cache/11_results_091114_M1.npz` and the logs are git-ignored.
**Resume:**
- Apply stacked TP-GDAR to IOS Reach with p = task phase (Aditya's priority; Hydrogen 2021-11-16 first).
- On OS-Pulse: second session/site; pulse width as a second p (2-D TP); deal with 150 Hz extrapolation.

### L-2026-10-26-a · Code review of `gdar_tp/` + fixes (input timing, hull containment, latent M-step)
commit: (this commit) · branch: master · issues: #65 · pc: Pinnacle
**Did/Found:** Reviewed the whole package (no pending diff). Fixed: (1) `metrics.multi_step_nrmse` fed inputs one step late (`U[t0+1:]` → `U[t0:]`); (2) `WCNetwork.simulate` applied `drive[t]` to x(t), against the x(t+1)=f(x(t),u(t)) convention, which zeroes any B fitted on WC with input (no script passed U, so no published number moves); (3) `convex_hull(method="box")` is *not* guaranteed to contain A(p): on the test data it misses by 1.12 against a spread of 3. The docstring is fixed and a containment warning added; (4) `qlpv` crashed with neither support nor Laplacian; (5) latent EM used project-after-LS, replaced by an exact bounded-LS M-step; also a warning when B is dropped, a narrowed except, no double fit in `fit_grid_2d`, and a bounded obs buffer. 38/38 tests pass (6 new in `tests/test_review_fixes.py`).
**State:** Committed. `scripts/07_latent_ei.py` was not re-run under the new M-step, so its `RESULTS.md` numbers may shift slightly. `edge_flow` is still slow (pure Python).
**Resume:** Re-run 07 and refresh `RESULTS.md` § latent. If the box hull is ever used for control, build true box corners as synthetic vertices. Then #66 (validity and results).

### L-2026-10-15-a · Baseline: log started; state as of the input-operator run
commit: e60b53c · branch: master · issues: #11, #12, #13 · pc: Pinnacle
**Did/Found:** Log started. Up to here the history lives in `RESULTS.md` (per-stage results) and the hub worklog. Latest work: TP-GDAR with input B(p) on OS-Pulse 091114 M1 (`scripts/10_ospulse_tp_input.py`). B has a monotone gain mode (rank 1 = 89.6 %), but pooled B beats TP under leave-one-frequency-out, and pre-stim band power adds nothing.
**State:** e60b53c is committed but not yet pushed (master). `GDAR-TP_overview.pptx` is untracked; `CLAUDE.md` has local edits. The Gram-matrix cache (0.97 GB) is in `cache/`, git-ignored.
**Resume:** `RESULTS.md` § TP on the INPUT operator. Next knobs: stim amplitude, a latent, the evoked window (#12). Handle stim artefact above 40 Hz first.
