# Research log: GDAR-TP

Newest on top. Format and rules: research-hub `agents/logging.md`. One entry per logical unit of work, committed together with that work. Issues and the hub worklog cite entries by ID.

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
