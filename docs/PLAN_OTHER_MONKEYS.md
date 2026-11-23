# Plan: TP-GDAR runs on monkey L and more H sessions

Asked by Aditya on 2026-11-22: "start plan for runs on other monley data". Started 2026-11-22. Nothing here is run yet.
Choices marked **D1** to **D6** change what is analysed, so they go through the method gate before any code uses them.

## 1. What data there is

The IOS Reach dataset has two monkeys, H (2021) and L (2022) (`yazdanlab/iosreach/README.md`). Every session is already cut
into trial segments on this PC, in `yazdanlab/iosreach/cache/segs/{M}_{date}.npz`: `lfp` and `noart` as channels × 2000 ms ×
trials at 1 kHz, from 500 ms before the target to 1499 ms after, plus `channels`, `stimmed`, `plan_ms`, `move_ms`. The raw
recordings are on the lab share (`Z:`), which is not mapped here; the segments are enough.

Read from each segment file on 2026-11-22 (channels before any bad-channel drop):

| session | channels | trials | lit | unlit | median plan (ms) | median move (ms) |
|---|---|---|---|---|---|---|
| H 2021-07-28 | 32 | 90 | 0 | 90 | 251 | 588 |
| H 2021-08-04 | 25 | 49 | 0 | 49 | 252 | 411 |
| H 2021-08-18 | 22 | 211 | 0 | 211 | 357 | 338 |
| H 2021-08-23 (fitted, script 14) | 17 | 406 | 0 | 406 | 349 | 386 |
| H 2021-09-22 | 8 | 494 | 241 | 253 | 376 | 296 |
| H 2021-09-24 | 8 | 522 | 265 | 257 | 373 | 332 |
| H 2021-09-29 | 4 | 493 | 241 | 252 | 452 | 337 |
| H 2021-11-11 (fitted, script 14) | 23 | 616 | 311 | 305 | 378 | 439 |
| H 2021-11-12 (fitted, scripts 12 to 19) | 23 | 321 | 0 | 321 | 399 | 409 |
| H 2021-11-16 | 15 | 663 | 337 | 326 | 450 | 266 |
| L 2022-04-25 | 7 | 232 | 112 | 120 | 332 | 252 |
| L 2022-04-27 | 5 | 701 | 366 | 335 | 386 | 166 |
| L 2022-04-29 | 6 | 958 | 488 | 470 | 394 | 130 |
| L 2022-05-02 | 5 | 1229 | 0 | 1229 | 392 | 87 |
| L 2022-05-06 | 3 | 1155 | 586 | 569 | 394 | 82 |
| L 2022-08-10 | 21 | 841 | 440 | 401 | 409 | 106 |

L 2022-08-10 channels (0-based, A = 0 to 15, B = 16 to 31): 2, 3, 4, 5, 7, 8, 10, 11, 12, 18 to 29. The iosreach README
says L has about 3 to 7 good channels per day until the A5 array (2022-08-08 onward).

The light on lit trials is already available as an input: `iosreach/stim_timing.py`, `stim_input(s)` gives a 0/1 light
signal per trial and ms (trials × 2000). A `B u(t)` term for the light would be a model change and is not in this plan; it
is one answer to the 11-11 lit fits that have no stable rotation.

## 2. Which sessions

1. **L 2022-08-10.** The only L session with enough channels for a graph. 401 unlit trials for the main fits, 440 lit.
2. **H 2021-11-16.** A stim day like 11-11: 326 unlit, 337 lit, 15 channels.
3. **H 2021-08-18.** No stim, 22 channels, 211 trials.
4. Maybe **H 2021-07-28** (32 channels, 90 trials) and **H 2021-08-04** (25 channels, 49 trials): few trials for 5 folds
   (**D5**).

Left out: L sessions before 08-10 (3 to 7 channels) and H 09-22, 09-24, 09-29 (4 to 8 channels).

## 3. Steps

**Step 0: plumbing.** The scripts hard-code one H session. No math changes here, but edits to `scripts/*.py` trip the method
gate, so they go to Aditya with a block that says what they are.
- One `--session` argument (e.g. `L_2022-08-10`) for scripts 12, 13, 14 and 18. Script 12's `SEGS` points at
  `H_2021-11-12.npz`, and script 14 swaps only the date, so the `H_` prefix stays. Cache, log and figure names carry the
  session, so H's results are not overwritten.
- Bad channels per monkey. Today `BAD = [29]` (H's B14) is applied to every session; on L 08-10 channel 29 is a good channel
  (**D1**).
- Lit trials. Script 12's `load()` asserts that no trial is lit, and scripts 13 and 18 use it. L 08-10 has lit trials, so
  `load()` needs the unlit filter that script 14 already applies (**D2**).
- Electrode map. The graph comes from `32ch_mmad_electrode_positions.pkl` and a 1.5 distance threshold. Whether that map
  holds for L's new array is a question for Noah (**D6**).

**Step 1: checks before any fit** (read-only, nothing fitted):
- Samples per warped bin on L 08-10 against H 11-12. L's median move is 106 ms against 409 ms on H, so each move bin is
  short (**D3**).
- The graph on L's 21 channels: number of edges, connected or not.
- The `noart` steps at 0 and 900 ms seen on H (5 November deck). Measure them on L the same way (`12c_step_check.py`).
- The evoked kernels for the "evoked removed" variant (MATH §3) on L.

**Step 2: fits** (approved math, re-run as is):
- Script 12: GDAR against TP-GDAR on held-out trials, and the K scan (**D4**).
- Script 13: rank 2 and 3 joint fits on 16 warped bins, the stable regular-simplex hull (MATH §9.6), the rotation sweep.
- Script 18: single-trial weights from 30, 60, 120 ms windows against the clock, and the go and touch timing tests.
- Script 14 style: lit against unlit trials on L 08-10 and H 11-16, each fitted on its own.

**Step 3: compare monkeys.** Edge maps can't be compared across arrays, so compare what doesn't depend on the array:
- held-out gain of rank 2 and 3 over GDAR;
- whether hold and move each get their own vertex, and how the weights move through the trial;
- whether a stable rotation exists, unlit and lit;
- the vertices' spectra relative to GDAR;
- single-trial gain over the clock, and whether the move vertex is locked to the target or to the go cue.

Each run gets a RESULTS.md section and a research-log entry. Cross-monkey numbers come from the scripts' logs.

## 4. Decisions for Aditya (method gate)

- **D1. L's bad channels.** Use Noah's good-channel list as is (21 channels), or also drop channels by our own check?
- **D2. Lit trials.** Main fits on unlit trials only, with lit trials fitted separately as on 11-11?
- **D3. Bins on L.** The same 16 warped bins (4 per phase), or fewer move bins because L's moves are short?
- **D4. Lag order K.** Keep K = 16 for every session, so models are comparable, or choose K per session from script 12's scan?
- **D5. Short H sessions.** Fit 07-28 (90 trials) and 08-04 (49 trials), or skip them?
- **D6. Array map for L.** Ask Noah whether the 32-channel map applies to the array L got in August 2022.

## 5. Order

Step 0 (one commit, after approval), then step 1 on L 08-10, then step 2 on L 08-10 (scripts 12, 13, 18, then lit against
unlit), then H 11-16 and H 08-18. Each session writes its own caches, so runs don't overwrite each other's results.
