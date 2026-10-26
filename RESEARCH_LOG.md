# Research log: GDAR-TP

Newest on top. Format and rules: research-hub `agents/logging.md`. One entry per logical unit of work, committed together with that work. Issues and the hub worklog cite entries by ID.

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
