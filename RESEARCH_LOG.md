# Research log: GDAR-TP

Newest on top. Format and rules: research-hub `agents/logging.md`. One entry per logical unit of work, committed together with that work. Issues and the hub worklog cite entries by ID.

### L-2026-11-06-a · Code and project review: a unit test for every library function, 8 bugs pinned, structure assessed
commit: f6bf8ba+this · branch: master · issues: #65 · pc: Pinnacle
**Did/Found:** Aditya asked for a full code and project review with unit tests for each method, and whether the structure is proper OOP. 115 new tests in 10 files; every function and method in `gdar_tp/` now has a direct test, and so do the script-12 helpers that 11 other scripts import (`tests/test_script12_helpers.py`). Strong checks include the Kalman filter and RTS smoother against the brute-force joint Gaussian, `_quad` against the explicit Kronecker form, the Theta- and W-steps as exact block minimisers, and `ll_matrix` against `gaussian_ll`. 8 bugs confirmed by probes and pinned as strict xfails (`tests/test_review_2026_11_06.py`); the two silent ones: a `(V, N, N)` vertex array gives an N² companion in `control`, and `SymTP.theta_at` doesn't sort `p_grid`. No current script hits either. 5 method questions for Aditya, mainly: script 12 picks the edge-wise ridge with fold 1, later a test fold (edge-wise rows only), and the rank-3 enclosing triangle puts all three sides on hull edges, so it is not the smallest (near-square hull: corner at (0, -5) for data in 0 to 1.2). Structure: the dataclass-plus-fit-function style is fine; the problems are scripts used as a library through `importlib`, duplicated helpers, three operator layouts with no shared interface, and private names used as public API. Full write-up: `docs/REVIEW_2026-11-06.md`. Also fixed `tests/test_method_gate.py`, which failed whenever the newest transcript was a live session with no chat text yet.
**State:** 219 tests: 211 pass, 8 strict xfails (about 25 s). Nothing in `gdar_tp/` or `scripts/` changed: the bug fixes and the proposed restructure (phase 1: move script 12's helpers into `gdar_tp/evaluate.py`, `operators.py`, `data/ios_reach.py`, numbers unchanged; phase 2: one `Protocol` for operator-grid models, fix R1 to R8) wait for Aditya's approval through the gate.
**Resume:** `.venv\Scripts\python -m pytest -rxX`. After approval, start phase 1 from section 4 of `docs/REVIEW_2026-11-06.md`; check that script 12's cache reproduces exactly.

### L-2026-11-05-f · Single-trial states: they predict better than the clock but follow the target, not the go cue
commit: 6d591c9+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** At Aditya's approval, `scripts/18_single_trial_states.py` keeps script 13's joint rank 3 fit (K = 16, 16 warped bins, enclosing-triangle vertices, 5 folds) and fits each held-out trial's three weights to its own data over 30, 60 or 120 ms windows, on the simplex, with no go or touch times. The exact 3-weight solver matched brute force and SLSQP on 4,000 and 200 random cases, and recovered known weights within 0.03. Weights from the previous 60 ms beat the clock's: one-step gain over GDAR +0.193 vs +0.156 (evoked left in), +0.239 vs +0.207 (removed), all with GDAR's Sigma; most of the extra gain is in move. But go time isn't recoverable from them (r +0.01 to +0.04, error no better than the median guess); touch partly is (r +0.29 to +0.43, p 0.0001). Movement time isn't (r about 0). The move vertex comes on 150 to 400 ms after target at the same time for early, middle and late go. A clock built from median go and touch matches the weights nearly as well as the trial's own (0.72 to 0.85 vs 0.73 to 0.86). One weight sits at 0 on 47 to 58% of samples.
**State:** done for H 2021-11-12, rank 3. Not separated: how much of the gain is just trial-by-trial adaptation (a control needs approval). Evoked removed, the timing test wasn't run, because its kernels use each trial's own events. Figure `figures/18_single_trial_states.png`, log `cache/18_run.log`, both git-ignored.
**Resume:** `.venv\Scripts\python scripts/18_single_trial_states.py` (about 2 min), `--plot` redraws. Next: the adaptation control, then graph constraints so no vertex dominates, keeping the convex hull.

### L-2026-11-05-e · Deck's unseen-time panel redrawn with the corrected numbers
commit: 54f4ddf+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** At Aditya's yes, redrew panel C of slide 24 of `TP-GDAR_IOS_Reach.pptx` ("TP beats one GDAR on held-out trials and unseen task times") from the corrected script 12 cache. With evoked left in, rank 3 lies above the fit on that bin, which lies above the neighbouring-bin fit, in all 10 inner bins. The three floating labels are now a key in the empty lower right, because their old positions assumed the neighbour's dip below zero. The speaker notes now give the corrected numbers, the shared-Sigma scoring and the own-Sigma sensitivity row. Panels A and B are unchanged. The online copy had changed since L-2026-11-05-c (saved at 10:42): Aditya removed four control slides (learn it from data; one gain for every p; both gains keep the network stable; vertices matter more than the LMI) and nudged slide 8's state-space equation up 0.03 in. Both changes are now in the script, so the deck has 30 slides.
**State:** the online copy was overwritten only after its hash matched the version diffed; OneDrive's version history has Aditya's 10:42 version. Against that version, only slide 24 (panel C and its notes) differs, plus a text box holding only "s" over the consensus movie, which wasn't carried over because it looks accidental. Slides 8, 24 and 30 were rendered through desktop PowerPoint and viewed; the deck wasn't opened online. `GDAR-TP_MPC_and_TP-GDAR.pptx` (`make_deck_figures_real.py`, `make_mpc_tp_deck.py`) still shows the old curve.
**Resume:** before the next rebuild, compare the OneDrive copy's MD5 with 77a40e6fca474c3800d5e2d30a302a1b. `.venv\Scripts\python scripts/make_deck_figures_r3.py tp_vs_gdar_r3`, then `.venv\Scripts\python scripts/make_ios_tp_deck.py`.

### L-2026-11-05-d · Leave-one-bin-out Sigma corrected; fits about 6x faster on CPU, same numbers
commit: f453ed2+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:**
- Code review of the IOS Reach fits, 10 findings. Sigma fix approved by Aditya: all leave-one-bin-out rows in script 12 now share GDAR's Sigma. Neighbouring bin -0.019 / -0.080 becomes +0.044 / +0.004, rank 3 +0.151 / +0.144 becomes +0.122 / +0.110. "The neighbouring-bin fit is worse than GDAR" was an artefact of an inflated Sigma; rank 3 still leads it by +0.079 +/- 0.007 / +0.105 +/- 0.008. Own-bin Sigma kept as a sensitivity row (+0.089 / +0.082).
- GPU tried and not adopted, at Aditya's approval: on the real matrices the RTX 3060 in double precision is no faster than 6 CPU threads (n = 3456 solve 0.094 s vs 0.121 s; Z'Z 0.375 s vs 0.231 s), and single precision fails (condition number 5e8: Cholesky fails, and a general solve makes training loss 0.25 nats/sample worse).
- `fit_sym_tp` instead assembles the phi-step block by block, solves by Cholesky (LU if that fails) and takes `quads` from the new `gls_quads`. Script 12 runs folds in worker processes with per-trial Grams only for test trials; 12b and 13 compute Grams once and run folds in parallel. Script 12 6,408 s to 960 s, 12b fold loops 1,659 s to 275 s, 13 now 90 s, 14 now 258 s.
- Same numbers: fitted grids and VAF of script 12 within 3e-8 nats/sample, 13 and 14 gains to 5 decimals with weights within 4e-6 and the same vertices, 12d log byte-identical. Largest change is 12b at K = 32, evoked left in, rank 2: +0.12971 to +0.12944; K = 24 and below within 5e-6.
- Measured, not changed (RESULTS.md, last section): the sum-to-one vertex model scores rank 2 +0.142 / +0.224 against +0.190 / +0.248; within-day edge r is 0.72 to 0.93, and static GDAR edges repeat poorly with the next day (0.18 to 0.37) but better with 08-23 (0.66, 0.72), so channel identity is a live question.

**State:** done for the Sigma fix and the speedup; 104 tests pass. The 2.8e-4 change at K = 32 isn't explained; the ALS stop on that badly conditioned system is my guess, unchecked. Scripts 15 and 16 weren't rerun; the match test covers their worst case. Deck figures weren't rebuilt, so `TP-GDAR_IOS_Reach.pptx` and `GDAR-TP_MPC_and_TP-GDAR.pptx` still show the old neighbouring-bin curve under "unseen task times", and their tag position in `make_deck_figures_r3.py` and `_real.py` assumes the old curve. The other 8 findings stand open; the review checks' scripts were throwaway and aren't in the repo.
**Resume:** `.venv\Scripts\python scripts/12_tp_vs_gdar_ios.py` takes about 16 min and needs about 5 GB free; `--plot` redraws. Next decisions: what to do about the vertex model's gain, and the channel map with Noah.

### L-2026-11-05-c · Deck rework: rank 3 only, GDAR from Felix's paper, LaTeX equations, control example
commit: 83d73e2+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Reworked `TP-GDAR_IOS_Reach.pptx` (34 slides) to Aditya's list, on top of the edits Aditya made online (task, setup and TP paper slides; Limits slide removed). Slide 8 has the state-space S(p) (the paper's eqs. 1 and 2). Bins now come before the first mention of them. New GDAR slides cite Schwock et al. 2024 with their Eqs. 1 to 5 and 7. The data step shows only the objective, and the noise and metric moved to the joint-fit slides, which need them. The HOSVD gain table is gone and convex weights sit next to HOSVD. Everything rank-dependent on real data is rank 3, from cached fits (`scripts/make_deck_figures_r3.py`, nothing refit). All 57 equations are LaTeX pictures with their source as alt text (`scripts/deck_latex.py`). The control example is back, with a slide on the system, its inputs and the goal, and the consensus movie. Slide 10's worked example is now S(p) = S0 + p S1 + p^2 S2 (approved through the gate): after Transformation 2 the weights stay in 0.14 to 0.61, and the triangle's area is 59 against 0.19 for the tightest. Rank 3 model spectrum vs data: r 0.88 to 0.98 (rank 2: 0.81 to 0.97), beta in move -1.9 vs -3.6 dB, and post still has the wrong sign, -0.1 vs +0.7 dB. Separate fits match or beat rank 3 except on 12 bins with evoked left in (+0.137 vs +0.098).
**State:** the data-step objective is written as a plain norm, as Aditya asked; the real fit weights by Sigma^-1, and the notes say so. The control example keeps 2 vertices because its true plant is a mix of 2 operators; rank 3 there would be a new experiment. The online copy was overwritten only after its hash matched Aditya's 09:45 version. Checked in desktop PowerPoint, not online, and the movie wasn't played. Equations can't be edited online. Uncommitted edits to `gdar_tp/sym_tp.py`, scripts 12, 12b, 13 and `tests/test_sym_tp.py` aren't from this work and are left out of this commit. That other work rewrote `cache/13` at 09:38; its rank 3 weights and vertices match the earlier log.
**Resume:** `.venv\Scripts\python scripts/make_ios_tp_deck.py` writes only the repo copy. Before any overwrite, compare the OneDrive copy's MD5 with 86284f6f593e4cf69fa8b40c1f56e562; if it differs, Aditya has edited it, so carry those edits into the script first.

### L-2026-11-05-b · IOS Reach facts from the two papers, added to the same deck
commit: fd91ca8+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Read Griggs, Stanis et al. (Nat Commun 2026) and Stanis et al. (EMBC 2025) and added them to `TP-GDAR_IOS_Reach.pptx`, now 24 slides: a setup slide, a findings slide, the real task timing (1 s hold, 50 to 700 ms delay, 900 ms light) and updated limits. The recording is over left PPC (7a, 7b near the IPS) with Jaws; the iosreach README says sensorimotor and opsin unconfirmed. Likely source of the `noart` steps: the methods split each trial at light on and off, zero each epoch and rejoin by matching 50 ms means, and no-stim trials go through the same processing. On 11-12 the median jump is 23.5 uV at 0 ms and 13.9 at 900 ms, against 2.5 typical, and raw `lfp` has none; the smaller steps at 10 and 910 ms aren't explained. The 60 Hz dip in the spectra is the notch, about -9 dB in both `lfp` and `noart`. Theta 4-10 Hz, plan vs hold: the all-channel mean is flat (+0.11 dB on 11-12, -0.25 on 08-23), but channels split (8 up, 7 down; 5 up, 8 down), which fits a local rise but can't be placed without the channel map. `scripts/ios_paper_checks.py`.
**State:** the same OneDrive file was overwritten, after checking its hash still matched what was published, so no online edits were lost. Checked in desktop PowerPoint; not opened online. The papers disagree on light intensity (about 12 mW/mm2 vs 1.2 mW/cm2) and EMBC's 800 um pitch doesn't fit 32 electrodes over 314 mm2, so neither is on a slide. Their trial counts differ from Noah's curated set. The iosreach README is unchanged (read-only for this work).
**Resume:** ask Noah to confirm where the steps come from, which channels sit over 7a, 7b and Jaws cortex, and the intensity. Masking window if wanted: the paper drops the 50 ms after light on and off. `.venv\Scripts\python scripts/ios_paper_checks.py`.

### L-2026-11-05-a · Fresh simple deck, and the data's spectrum in each trial phase
commit: 0622ca8+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** `scripts/make_ios_tp_deck.py` writes `TP-GDAR_IOS_Reach.pptx` (22 slides, text and shapes only): the IOS Reach task, spectra by phase, the TP paper's steps, what data adds to them, the GDAR and joint-fit math, HOSVD vs joint, vertices, results and limits. `scripts/make_deck_figure_spectra.py` computes the first per-phase spectra of the data, with the estimator Aditya approved through the gate (200 ms Hann windows every 50 ms inside each phase; trial-mean power per channel, dB, mean over channels; evoked left in). Move vs hold on 11-12: alpha -7.9 dB, beta -5.5, high gamma +1.2. 08-23 repeats it: -8.2, -7.1, +1.8. Raw `lfp` differs by under 0.7 dB from 3 to 100 Hz. This answers the open check of the 15 Hz change against the data: the rank-2 fit follows the data's shape across frequency (r 0.81 to 0.97) but gives half the beta drop in move (-1.8 vs -3.6 dB re the mean of phases) and the wrong sign for beta in post (-0.3 vs +0.7). Table in RESULTS.md.
**State:** every slide rendered through desktop PowerPoint and viewed. Copy at `OneDrive\Talks\TP-GDAR_IOS_Reach.pptx`, which OneDrive shows as synced; the script writes only the repo copy, so a rebuild can't overwrite edits there. Not opened in PowerPoint for the web, because the app's browser isn't signed in to OneDrive. Plotted numbers aren't linked to the shapes. The data vs model comparison is one day, evoked left in, with AR spectra frozen at each bin and nothing refit. The gate refused the spectrum script over a variable named `starts`, a false match; the block went to Aditya instead of a rename.
**Resume:** edit the OneDrive copy. To rebuild: `.venv\Scripts\python scripts/make_deck_figure_spectra.py`, then `.venv\Scripts\python scripts/make_ios_tp_deck.py`. The other figure data (`figures/deck/*.json`, git-ignored) comes from `make_deck_figure_ios.py` and `make_deck_figures_real.py`.

### L-2026-11-04-n · Method gate: first real approval, and the fix it held back
commit: 898372b+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Aditya approved the block for the one-line fix in `why_gated`, and the same edit the gate had refused then went through, so both the refusal and the opening have now run in the real harness. Only write tools are now checked by file path, and the expected-failure mark is gone.
**State:** done. `tests/test_method_gate.py`: 50 pass. Closing on Aditya's next message is covered by the tests, not yet seen in use.
**Resume:** `.venv\Scripts\python -m pytest tests/test_method_gate.py`.

### L-2026-11-04-m · Method gate: fitting and scoring edits wait for Aditya's approval
commit: 33219c2+this · branch: master · issues: #75 · pc: Pinnacle
**Did/Found:** Aditya asked for a hook after I picked the joint fit on my own. `.claude/hooks/method_gate.py` is a PreToolUse hook that refuses edits to `gdar_tp/` (except `io.py`, `viz.py`, `__init__.py`), edits to `scripts/*.py` that change a fitting or scoring term, shell commands that visibly write to those, and edits to the gate. It reads the session transcript and opens only when a complete explanation block is followed by a plain approval from Aditya; their next message closes it. `tests/test_method_gate.py`: 49 pass, 1 expected failure. The hook fired through the real harness on my next edit, which was to the gate itself, under auto mode.
**State:** live on this PC and synced by git. The expected failure is a one-line fix to the gate, refused by the gate and put to Aditya. A Read call would be gated if it ever reached the script, but the settings matcher doesn't send Read. The gate can't see intent: a comment fix in `gdar_tp/` is stopped, and a method choice in a scratchpad script isn't.
**Resume:** after approval, add the tool check to `why_gated` and remove the xfail in `tests/test_method_gate.py`. `.venv\Scripts\python -m pytest tests/test_method_gate.py`.

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
