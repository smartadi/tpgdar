# Research log: GDAR-TP

Newest on top. Format and rules: research-hub `agents/logging.md`. One entry per logical unit of work, committed together with that work. Issues and the hub worklog cite entries by ID.

### L-2026-11-04-l · HOSVD vs the joint fit on IOS Reach: the metric does most of the work
commit: 57d82a5+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Read Baranyi 2004 (`docs/`): it transforms a known model, with no data or noise, so it has no system identification of its own. `scripts/12d_hosvd_vs_joint.py` compares the low-rank step on H 11-12 from the same per-bin GLS fits. Rank 2 gain, evoked left in / removed: HOSVD +0.016 / +0.003, SVD in the pooled data metric +0.180 / +0.211, joint +0.190 / +0.248. HOSVD also gives different weights across the trial. Sixteen separate fits reach +0.211 / +0.392. Table in RESULTS.md.
**State:** done for 11-12 only, with the target-onset steps unmasked. The joint numbers reproduce those of L-2026-11-04-j to 4 places. Deck left alone, parked at Aditya's request; its HOSVD slide still shows OS-Pulse only.
**Resume:** `.venv\Scripts\python scripts/12d_hosvd_vs_joint.py`; log in `cache/12d_run.log`.

### L-2026-11-04-k · Few inputs on the 9-node example: the LMI separates only with two unstable ends and at most 3 inputs
commit: e191a77+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** `scripts/16_nine_node_few_inputs.py`: script 15's 20 plants per case with inputs at M = 1, 2, 3, 5, 9 nodes by a fixed rule, loop closed on the true plant. M = 9 reproduces script 15. Plants of 20 stable with two unstable ends, M = 1 / 2 / 3 / 5 / 9: LMI 20 / 20 / 20 / 20 / 20 (certificate 17 / 18 / 19 / 20 / 20); pooled LQR at R = 1 0 / 1 / 6 / 17 / 20; normalised pooled LQR 10 / 7 / 17 / 20 / 20; a family of up to 160 LQR gains with one picked on the fitted model (no LMI) 16 / 14 / 19 / 20 / 20, and no other gain in the family does better even with the truth to pick. So on 11 of 60 plants only the LMI stabilises, 6 of them certified on the true plant, and one plant's rms x ends above open loop. With one unstable end, or M >= 5, stability doesn't separate them. Rho in the hundreds is mostly an artefact: the energy-weighted pooled fit has a spurious unstable mode the inputs hardly reach, so LQR uses a huge gain; the input gains were fine. Tighter alpha looking worse on M = 1, seed 0 was the solver: `optimal_inaccurate` answers that don't satisfy the LMI. `SynthesisResult.lmi_residual` now checks this, and `synthesize_quadratic_feedback` takes `q_max`; older scripts are unchanged. Effort has no consistent winner, and script 15's saving at M = 9 against pooled LQR mostly goes against the family. Table and limits in RESULTS.md.
**State:** results and `figures/16_few_inputs.png` done; cache and figure are local and git-ignored. Not on a slide yet: the Part 3 slide waits for the deck rework. One placement rule, grid of LQR gains with Q = I, no direct search over F, slow p schedule, synthetic plants.
**Resume:** add the figure to the deck's Part 3 after the rework. If this matters for the argument, try a direct search over F on the 11 plants, and other placements.

### L-2026-11-04-j · Step in lfp noart at target onset: gains hold, the plan 1/4 vertex was mostly the step
commit: b54c7d9+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** `lfp noart` has a step at 0->1 ms after target on (23.5 uV vs 2.5 typical on 11-12, 29.0 vs 3.4 on 08-23), and smaller ones at 10, 900 and 910 ms, on these no-stim days; raw `lfp` has none. Masking on 11-12 moves held-out gains little (rank 2 +0.190 to +0.196 evoked left in, +0.248 to +0.259 removed; rank 3 +0.222 to +0.225, +0.323 to +0.320). With evoked removed, plan 1/4 stops being a rank-3 vertex (weights 0, 1, 0 become 0.66, 0, 0.34), which accounts for most of the outlier left unexplained in L-2026-11-04-a. Table in RESULTS.md.
**State:** check only, 11-12 only. Scripts 12, 13 and 14 (08-23) still run unmasked; vertex maps not refit. `scripts/12c_step_check.py` is the scratch script with its root path made relative, not rerun after that one edit. Where the step comes from isn't known. The IOS slide's traces come from `noart` through `s12.load()`.
**Resume:** ask Noah about the step; decide whether to mask in scripts 12 to 14 and rerun.

### L-2026-11-04-i · Deck: earlier LMI slides back, as editable text
commit: bd0b86d+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** At Aditya's request, put back the polytopic control slide with its LMI equations (slide 28) and the table of medians (slide 32), now 32 slides. Wording as in 9d3a0c5, except that "(panels C, F)" is dropped because those panels are gone. Table numbers match RESULTS.md.
**State:** done; the changed slides exported through desktop PowerPoint and viewed. Not opened in PowerPoint for the web.
**Resume:** as in L-2026-11-04-g.

### L-2026-11-04-h · Deck: slide 14 shows the loss as the weighted prediction error
commit: d13c60d+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Slide 14 now writes `L_g` as the Sigma-inverse-weighted one-step error instead of its expansion in `H_g` and `h_g`, at Aditya's request. Slide 15 still solves with `H_g` and `h_g`, so it gains one line saying what they are.
**State:** done; slides 14 and 15 exported through desktop PowerPoint and viewed.
**Resume:** as in L-2026-11-04-g.

### L-2026-11-04-g · Talk deck rebuilt: less text, editable shapes, IOS Reach, graphs and GDAR, consensus movie
commit: 9d3a0c5+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Rebuilt `GDAR-TP_MPC_and_TP-GDAR.pptx` (30 slides) for PowerPoint for the web: `scripts/deck_native.py` draws plots as grouped shapes and equations as text, because that app can't insert charts or equations and doesn't play GIFs in Slide Show. Only the two Figure 6 slides stay pictures; the movie is an embedded MP4, made through desktop PowerPoint (no ffmpeg here). Added slides on IOS Reach, graphs and GDAR; new low-text figures from `make_deck_figures_real.py`, `_nine.py` and `make_deck_figure_ios.py`. New `scripts/17_consensus_demo.py`: both gains reach the target about equally (RESULTS.md). The corner slide holds p at 1, since under the moving schedule without noise no-control energy also decays.
**State:** deck done, exported through desktop PowerPoint and viewed; not opened in PowerPoint for the web, and the schema validator can't run on this PC. Plotted numbers aren't linked to the shapes. Open: a possible step in `lfp noart` at target onset, reported by a read of the data and not yet checked against the gains; the channel map and brain area need Noah. Trial-level prediction was set aside at Aditya's request, unfinished. The few-input files (`gdar_tp/control.py`, script 16) belong to another session and aren't in this commit.
**Resume:** run the three figure scripts, then `.venv\Scripts\python scripts/make_mpc_tp_deck.py`.

### L-2026-11-04-f · Inputs at chosen nodes, for the few-input control example
commit: 75f5d2b+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** `symmetric_basis(..., node_inputs=[...])` now takes a list of input nodes as well as True; one test added. The functions in `scripts/15_nine_node_control.py` work for any number of inputs; results at 9 inputs are unchanged by construction, and the lag operators of a plant don't depend on the input nodes, so plants are nested across input counts.
**State:** code only; `tests/test_sym_tp.py` passes. No results recorded yet.
**Resume:** the few-input example, `scripts/16_nine_node_few_inputs.py`.

### L-2026-11-04-e · Deck with the MPC replay, TP-GDAR, other days and the 9-node control example
commit: ef89a4a+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Added `scripts/make_mpc_tp_deck.py` and `GDAR-TP_MPC_and_TP-GDAR.pptx` (30 slides). Part 1: MPC replay from the NC draft (`Closedloop_nc/results.tex`, 5a955c7), every number on those slides matched against that text; panel m of Figure 6 left out because `brain_paper/TASKS.md` flags it. Part 2: TP-GDAR model, training, HOSVD vs joint fit, vertices, result figures, other days. Part 3: the 9-node example. Refactored `scripts/make_tp_math_deck.py` into slide functions so both decks share them, and rebuilt `GDAR-TP_model_training.pptx` for the wording and limitations changes.
**State:** done. Exported through PowerPoint and new slides checked by eye. The pptx skill's schema validator can't load its schemas on this PC, so no schema check ran. The MPC slides use the locked numbers of 2026-10-16; the 2026-10-10 audit (0.78x, 0.65x at lag 0) predates them and wasn't re-checked. "About 40% / 80%" are ratios of medians (0.61, 0.19); the per-trial median ratios are 0.68, 0.22, and the slide says which. The command-limit bullet follows the JN correction of 2026-11-03, not the NC sentence ("only slightly"), which doesn't hold for the worst quarter of trials (89% of their squared error at a limit, perfect prediction, Mouse 1e).
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
