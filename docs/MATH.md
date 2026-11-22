# GDAR-TP: the math as the code runs it

This is the one math document for the repo. It replaces `docs/MODEL.md` (2026-10-26), whose content is kept here in §1.2, §4.4,
§5.2, §6.8 and §7.4. Each section describes what the code does at commit 5651b9e, with every symbol defined, and names the code
(`file: function`). Where an older document, log entry or slide says something different, the section says so under **Differs**.
Each section ends with its approval status (research-hub `rules/hub-math.md`), and §13 collects them. A change to anything here
goes through the method gate first. Code comments cite sections as `MATH §n`.

Written 2026-11-21 from the code, not from earlier descriptions. Nothing in this document changes a method.

## 0. Notation and conventions

| symbol | meaning |
|---|---|
| N, E | channels, graph edges (undirected, i < j) |
| K | state lags; lag k is 3k ms at 333.33 Hz |
| Q, M | input lags, inputs (OS-Pulse only; the IOS Reach work has no input) |
| g = 1..P | grid points (bins) of the scheduling variable p |
| r | TP rank |
| x(t) ∈ R^N | channel vector at sample t |
| z(t) = [x(t); x(t−1); …; x(t−K+1)] ∈ R^{KN} | regressor; with input, [ … ; u(t); …; u(t−Q+1)] is appended |
| Θ ∈ R^{N×KN} | one-step operator, Θ = [A_1 … A_K]; A_k multiplies x(t−k+1) |
| Σ ∈ R^{N×N} | noise covariance |

Model at one operating point: `x(t+1) = Θ z(t) + e(t)`, `e(t) ~ N(0, Σ)`. **Input timing:** u(t) drives x(t+1), in every simulator
and fitter.

**Sufficient statistics,** summed within a trial (never across a trial boundary) and per bin. A sample belongs to the bin of its
target x(t+1):

```
G = Σ_t z(t) z(t)ᵀ        C = Σ_t z(t) x(t+1)ᵀ        S_yy = Σ_t x(t+1) x(t+1)ᵀ        n = number of targets
```

Code names: `Gzz`, `Gzy` (= C, shape KN × N), `Syy`, `n`. The residual outer-product sum of an operator Θ is
`S(Θ) = S_yy − Θ C − Cᵀ Θᵀ + Θ G Θᵀ` (`tpgdar.py: resid_cov_sum`).

**First target.** Targets start at sample index K0 so that every K ≤ K0 sees the same samples: K0 = 16 in script 12
(`max(KS)`), K0 = 33 in scripts 12b, 12d, 13, 14 and 18 (`12b: fold_stats`). The two sample sets differ, so script 12's tables
and script 13's are not comparable.

## 1. Data and preprocessing

### 1.1 IOS Reach (scripts 12 to 18)

- **Source:** `yazdanlab/iosreach/cache/segs/H_<date>.npz`, key `noart` (µV), one segment per trial from −500 to +1500 ms re
  target onset at 1 kHz (2000 samples). Read only.
- **Days:** H 2021-11-12 and H 2021-08-23 (no stimulation); H 2021-11-11 (stimulation day, lit and unlit trials fitted
  separately). Trial and channel counts are in `RESULTS.md`.
- **Exclusions:** channel 29 (B14) dropped (`BAD`); trials with movement time ≥ 3000 ms dropped.
- **Decimation:** by 3 with `scipy.signal.decimate(..., zero_phase=True)` (default IIR filter), giving 333.33 Hz.
- **Centring:** each trial's channel means over the whole 2 s segment are subtracted. No other scaling (data stay in µV).
- **Event times:** go = `plan_ms`, touch = go + `move_ms`, both re target onset. Reach direction = `angle` mod 360.
- **Graph:** electrode positions from Felix's `32ch_mmad` file; channels i, j share an edge if 0 < ‖pos_i − pos_j‖ ≤ 1.5
  position units (`GRAPH_TH`).
- **Stimulation day (script 14):** a target is dropped, on lit and unlit trials alike, if it or any of its K lags falls in
  −20..60 ms or 860..960 ms re target (`ART`, `mask_artefact`).
- **Not done:** the steps in `noart` at 0→1, 10→11, 900→901 and 910→911 ms re target are not masked in scripts 12 to 14.
  Script 12c tests a mask (targets whose sample or lags touch −3..15 or 897..915 ms) on 11-12 only.

Code: `scripts/12_tp_vs_gdar_ios.py: load`, `scripts/14_tp_vertices_days.py: load, mask_artefact`.

**Differs:** none from the code. Not reported elsewhere: on 11-11 the artefact drop leaves plan 1/4 (§2.3) with 246 unlit and
228 lit targets against about 10,000 in the other bins (`cache/14_run.log`).

**Status:** not yet approved.

### 1.2 OS-Pulse rules (scripts 10 and 11)

Kept from `MODEL.md` §5. Each changes the fitted operator if broken.

| rule | why |
|---|---|
| Accumulate G, C per trial; never concatenate trials into one series | concatenation adds K false transitions per boundary |
| Don't z-score per condition; scale every condition by one per-channel std from training data | per-condition scaling turns A into D_g⁻¹ A D_g, a false p-dependence |
| Mean-centre per segment (or include an intercept column) | the linear model describes deviations from the operating point |
| Stimulation window 1.0 to 2.0 s | the response locked to the stimulation frequency is present only in 1.0 to 2.0 s; the export README's 1 to 6 s is wrong |
| u(t) → x(t+1) | a one-sample shift makes the fitted B absorb nothing |

Script 09 broke rules 1, 2 and 4 and is superseded by script 11.

**Status:** not yet approved (written 2026-10-26, before the gate).

## 2. The scheduling variable p

Labels are per target sample; −1 means the sample is not used. Times t are in ms re target onset.

### 2.1 Four task phases (script 12, grid `phase`)

```
hold   −400 ≤ t < 0
plan      0 ≤ t < go
move     go ≤ t < touch
post  touch ≤ t < touch + 500,  and t ≤ 1400
```

p_grid = (0, 1, 2, 3). Categorical: no interpolation between phases.

### 2.2 Twelve go-locked bins (script 12, grid `time`)

Bin b = ⌊(t − go)/100⌋ + 4 for −400 ≤ t − go < 800 and −400 ≤ t ≤ 1400, so b = 0..11. p_grid = bin centres, −350 to +750 ms
re go. Ordered: used for leave-one-bin-out (§8.6), with linear interpolation of the weights between bins (`SymTP.theta_at`).

### 2.3 Sixteen warped bins (script 13, `warped_labels`; also 12c, 12d, 14, 18)

Each phase interval [a, b) of §2.1 is cut into 4 equal fractions of its own duration on that trial:
`q = min(⌊4 (t − a)/(b − a)⌋, 3)`, label = 4·phase + q, for t ≤ 1400. For post, b = touch + 500 even when samples past 1400 ms
are cut, so late-touch trials have a short or empty post 4/4. p_grid = (0, …, 15), named "hold 1/4" to "post 4/4".

**Status:** not yet approved.

## 3. Evoked kernels (the "evoked removed" variant)

The task-evoked response is an input the model doesn't have, and a state-dependent operator can absorb it. The "evoked removed"
variant subtracts a linear fit of event-locked kernels first.

**Design** (`evoked_design`). For event e ∈ {target onset (t = 0), go, touch}, window [lo_e, hi_e) = [−500, 1500), [−300, 800),
[−400, 600) ms, and 9 ms bins: a sample at time t of a trial with direction d and event time t_e gets a 1 in column
(e, d, ⌊(t − t_e − lo_e)/9⌋) when that bin index lies in 0..⌊(hi_e − lo_e)/9⌋ − 1. Z is sparse, (trials × samples) × columns.

**Fit** (`remove_evoked`), on training trials only, every channel at once:

```
β = (Z_trᵀ Z_tr + λ d I)⁻¹ Z_trᵀ Y_tr,      d = median of the positive diagonal entries of Z_trᵀ Z_tr,   λ = 10⁻²
X_removed = X − Z β        (applied to training and test trials)
```

Held-out scores refit β on the training trials of each fold (scripts 12, 12b, 12c, 12d, 13 folds, 18). Full-data fits use all
trials: script 12's interpretation operators and split-half check, and the vertex fits of scripts 13 and 14. The kernels sit at
each trial's own go and touch, so the removed data carry those event times (why script 18 skips its timing test for this variant).

**Status:** not yet approved. The design follows `iosreach lti_stim`.

## 4. GDAR baseline: Felix's model and estimator

Reference: Schwock, Bloch, Khateeb, Zhou, Atlas, Yazdan-Shahmorad, "Inferring neural communication dynamics from field
potentials using graph diffusion autoregression", bioRxiv 2024, doi:10.1101/2024.02.26.582177 (PDF in `docs/`).

### 4.1 Model class

Each A_k is symmetric and is zero off the graph and its diagonal. Parameterisation `vec Θ = R φ` (column-major vec), where φ holds
per lag the N self terms, then one number per edge i < j used at both (i, j) and (j, i): n_φ = K (N + E) numbers
(`sym_tp.py: symmetric_basis`). Script 15 adds one input gain per listed node (`node_inputs`).

### 4.2 GLS loss for a fixed Σ

```
L(φ) = Σ_t ‖x(t+1) − Θ(φ) z(t)‖²_{Σ⁻¹} = tr(Σ⁻¹ S(Θ(φ))) = φᵀ H φ − 2 φᵀ h + tr(Σ⁻¹ S_yy)

H = Rᵀ (G ⊗ Σ⁻¹) R,      h = Rᵀ vec(Σ⁻¹ Cᵀ)
```

(`sym_tp.py: _quad`.) Linear systems are solved as `(H + 10⁻¹⁰ (tr H / dim H) I) x = h` by Cholesky, falling back to LU
(`_solve`). That jitter is the only regularisation: the symmetric fits have no ridge.

### 4.3 Estimator (`fit_gls_pooled`)

1. φ_OLS minimises L with Σ = I.
2. Σ̂ = S(Θ(φ_OLS)) / n.
3. φ_GDAR minimises L with Σ = Σ̂.

One operator for the whole day, from statistics summed over all training trials and bins. On one trial it matches
`CVARModel.fit_gdar` to a relative 1e-11 (L-2026-11-02-a). Σ̂ is returned and reused as the fixed metric of every TP fit (§6.1).

### 4.4 Edge-wise class (`tpgdar.py`, OS-Pulse work, "edge-wise" rows of script 12)

Row i of every A block may be nonzero on Ω_i = N(i) ∪ {i}; input columns are free. Fitted row by row by ridge least squares
(Σ = I) with one weight per coefficient, λ_ic = ρ G_cc (`_ridge`), so the ridge has no units and keeps its meaning across data
lengths. ρ must be chosen by validation inside the training data. In script 12 ρ ∈ {1e-6, 1e-8, 1e-10, 1e-12} is chosen by
`pick_ridge`. Completing the square gives `L(Θ) = ‖(Θ − Θ̂) G^{1/2}‖²_F + const`: the data measure operator distance in the G
metric. A strict-diffusion form A_k = a_k I − b_k L (L the graph Laplacian) is also implemented (`gdar.py`).

**Differs:** `pick_ridge` validates on fold 1, which is later a test fold (review M1, `docs/REVIEW_2026-11-06.md` §3). It affects
the edge-wise rows only.

**Status:** not yet approved (4.1 to 4.3 reimplemented 2026-11-02 and 4.4 written 2026-10-26, both before the gate).

## 5. The TP-GDAR model

### 5.1 Symmetric class (all IOS Reach results)

```
Θ(p_g) = Σ_{j=1..r} W_gj Θ_j,      Θ_j = Θ(φ_j),   so  φ(p_g) = Σ_j W_gj φ_j = (W Φ)_g
W ∈ R^{P×r},   Φ = [φ_1; …; φ_r] ∈ R^{r×n_φ}
```

Every Θ(p) is symmetric and graph-supported. The rank is along p only; the channel and lag structure of each Θ_j is never
compressed, so each Θ_j is a complete lag-K system. Off the grid, w(p) is linear between grid points (`SymTP.theta_at`).
Parameter count: r n_φ + P r − r² (`SymTP.n_params`; r² for the gauge, §6.4).

### 5.2 Stacked [A | B] form (edge-wise, OS-Pulse)

Kept from `MODEL.md` §2. Θ(p) = [A_1(p) … A_K(p) | B_1(p) … B_Q(p)] with its own rank per block:
`Θ(p) = [Σ_j h_j(p) A^(j) | Σ_l f_l(p) B^(l)]`, ranks (r_A, r_B). r_A = 1 is a shared A with a p-dependent input (script 10's
model); r_A = r_B with h = f is one joint mode set.

**Status:** not yet approved. Context: on 2026-11-05 Aditya wrote "so for now lets stick with joint fit setup", which keeps the
joint fit as the working setup; it was not a gate approval of these equations. L-2026-11-04-m records that the joint fit was
first chosen without asking him.

## 6. Learning the TP-GDAR (`sym_tp.py: fit_sym_tp`)

### 6.1 Objective

H_g, h_g are §4.2's quadratic form on bin g's training statistics, with Σ fixed at Σ̂ from the pooled fit (§4.3, step 2; not
shrunk):

```
J(W, Φ) = Σ_g [ (W_g Φ) H_g (W_g Φ)ᵀ − 2 (W_g Φ) h_g ]  =  Σ_g L_g(φ(p_g)) − const
L_g(φ) = ‖φ − φ̂_g‖²_{H_g} + const,      φ̂_g = H_g⁻¹ h_g   (the separate GLS fit of bin g)
```

So the joint fit is the H_g-weighted rank-r approximation of the per-bin fits, each bin in its own data metric.

### 6.2 W-step (Φ fixed)

For each g separately: `(Φ H_g Φᵀ) w_g = Φ h_g`, an r × r system (least squares, rcond 1e-12).

### 6.3 Φ-step (W fixed)

```
( Σ_g w_g w_gᵀ ⊗ H_g ) vec Φ = Σ_g w_g ⊗ h_g          (block (j, k) = Σ_g W_gj W_gk H_g)
```

Assembled block by block and solved by Cholesky with the §4.2 jitter (L-2026-11-05-d). Both steps are exact block minimisers
(`tests/test_tpgdar_units.py`, `tests/test_sym_tp_units.py`), so J never increases.

### 6.4 Gauge

W ↦ W M, Φ ↦ M⁻¹ Φ leaves Θ(p) unchanged. After every sweep each weight column is rescaled to unit RMS:
s_j = ‖W_{:,j}‖ / √P, W_{:,j} ← W_{:,j} / s_j, φ_j ← s_j φ_j. No rotation.

### 6.5 Starts

- `pooled` (default in every IOS script): φ_1 = H̄⁻¹ h̄ with H̄ = Σ_g H_g, h̄ = Σ_g h_g; W_{:,1} = 1, other columns 0. φ_2..φ_r are
  the leading right singular vectors of the deviations [φ̂_g − φ_1] in the pooled metric (SVD of the deviations times
  chol(H̄)), mapped back. The first W-step sets the other weights.
- `svd`: Φ = the top r singular vectors of [φ̂_1; …; φ̂_P] scaled by their singular values (unweighted); W by the W-step.

With several starts the lowest final J wins (`starts=("pooled", "svd")` in scripts 15 to 17). On H 2021-11-12 the svd start and a
strict stop moved the held-out gains by at most 0.0001 (L-2026-11-04-d); other days weren't checked.

### 6.6 Stopping

Stop when |J_{k−1} − J_k| ≤ 10⁻⁷ |J_k| or after n_iter sweeps: 60 by default (script 12, leave-one-bin-out), 25 in the fold fits
of scripts 12d, 13 and 18, 40 in script 13's full-data fit. Script 13 and 14's full-data fits stopped after 5 to 15 sweeps
(`cache/13_run.log`, `cache/14_run.log`), so the tolerance, not the cap, ended them.

### 6.7 Bracket

On the same objective, separate per-bin fits ≤ joint ≤ pooled. A test checks it for the edge-wise learner
(`test_joint_objective_bracketed_by_per_condition_and_pooled`); a gauge leak in a ridge breaks the lower bound.

### 6.8 Edge-wise joint learner (`tpgdar.py: fit_joint`)

Kept from `MODEL.md` §3b. Σ = I (row-wise least squares). The ridge sits on the grid-point operators as an augmented Gram
G_g + ρ diag G_g, never on the factors, where rescaling W against Θ_j would escape it.

```
Θ-step, row i:   ( Σ_g (w_g w_gᵀ) ⊗ G̃_g[Ω_i, Ω_i] ) φ_i = Σ_g w_g ⊗ C_g[Ω_i, i],      G̃_g = G_g + ρ diag G_g
W-step, per g:   H_g w_g = h_g,   H_g[j, l] = tr(Θ_j G_g Θ_lᵀ),   h_g[j] = tr(Θ_j C_g)
```

Start: the pooled operator with constant weight, plus the leading unweighted deviations. Gauge: each block's W is rotated to
orthonormal columns by QR (`_normalize_gauge`). n_iter 200, tol 1e-8 by default; script 12 uses n_iter 40.

**Differs:** `MODEL.md` §3b gave the orthonormal gauge as the TP-GDAR convention. That holds for the edge-wise learner only; the
symmetric learner rescales (§6.4). The choice matters for the convex form (§9.1). `MODEL.md` §1's validated ridge also applies to
the edge-wise class only (§4.2).

**Status:** 6.3's numerics (block assembly, Cholesky, the CPU route) approved 2026-11-05 (§13). The rest is not yet approved.

## 7. Low-rank alternatives kept as baselines

### 7.1 Separate per-bin fits

φ̂_g = H_g⁻¹ h_g with the GDAR Σ̂ ("per-condition" in script 12, "separate" in 12d and 13). P n_φ numbers.

### 7.2 HOSVD (script 12d)

SVD of the stacked entries [vec Θ̂_1; …; vec Θ̂_P] in the Frobenius metric, truncated to rank r. With one parameter mode HOSVD is
this one SVD, the TP model transformation step of Baranyi, "TP model transformation as a way to LMI-based controller design",
IEEE Trans. Ind. Electron. 51(2), 2004, doi:10.1109/TIE.2003.822037. That paper transforms a known model and has no
identification from data (L-2026-11-04-l).

### 7.3 Metric SVD (script 12d)

The same SVD in the pooled data metric: with L L ᵀ = H̄ (Cholesky, H̄ = Σ_g H_g), truncate the SVD of [φ̂_g L] to rank r and map
back by L⁻¹. Closed form; one metric for all bins, where the joint fit (§6) uses each bin's own.

### 7.4 Two-stage (edge-wise, script 11)

Per-condition ridge fits, then the unweighted SVD over p. It minimises Σ_g ‖Θ − Θ̂_g‖²_F, which equals the data loss only when every
G_g is the same multiple of I. Real recordings break this (different trial counts and power per condition, correlated channels,
A and B columns in different units), so it spends rank on poorly determined directions. Kept from `MODEL.md` §3a.

**Status:** not yet approved (baselines).

## 8. Scoring

### 8.1 Held-out log-likelihood

For an operator Θ_g, a noise covariance Σ and a test trial's statistics in bin g:

```
ℓ = −½ [ tr(Σ⁻¹ S(Θ_g)) + n (N log 2π + log det Σ) ]
```

(`12: ll_matrix`, `12b: ll`.) **Gain** = (Σ ℓ_model − Σ ℓ_GDAR) / (test targets), in nats per sample, summed over channels.

### 8.2 Which Σ

- **Default:** each model gets one Σ from its own training residuals over all bins, shrunk 1% toward its diagonal:
  Σ = 0.99 Ŝ + 0.01 diag(Ŝ), Ŝ = Σ_g S(Θ_g) / n_train (`12: sigma_of`, `SIG_SHRINK = 0.01`). Scripts 12 (phase and time grids),
  12b, 12d, 13 and 14.
- **Per-bin Σ** (12b's `per-phase-Sigma` rows): the same, one per bin (`12b: sigma_per`).
- **GDAR's Σ for every row:** leave-one-bin-out (§8.6) and all of script 18 (§10).

The fit's metric (§6.1, unshrunk GLS Σ̂) is not the scoring Σ.

### 8.3 Folds

5 folds over trials, stratified by reach direction: within each direction a random permutation (seed 0), assigned in turn
(`12: folds`).

### 8.4 Order K

Chosen by held-out ℓ of GDAR alone, evoked left in, on one sample set (K0 = 16), over K ∈ {1, 2, 3, 5, 8, 12, 16}; the best is 16,
the top of the grid (`12: select_K`, `cache/12_run.log`). Then fixed for every model. 12b re-checks K = 5, 16, 24, 32.

### 8.5 Uncertainty

- Script 12: mean and SE over test trials of each trial's per-sample gain; p from a Wilcoxon signed-rank test over trials.
- Scripts 13, 14, 18: fold SE, the SD over the 5 folds of each fold's gain divided by √5.
- Weights in script 13: jackknife SE √((F−1)/F Σ_f (λ_f − λ̄)²) over the F = 5 training-split fits, each expressed on the full-data
  vertices (`in_vertices`). The splits share 3 of 4 folds.

### 8.6 Leave-one-bin-out (12 go-locked bins, interior bins h = 1..10)

With bin h removed from training: GDAR and its Σ̂ on the other bins; TP rank r on the other bins, its operator at p_h by linear
interpolation of w; "nearest bin" = the GLS fit on bin h − 1 alone (always the earlier neighbour); "oracle" = the GLS fit on bin
h's training data (seen, for reference). Each is scored on bin h of the test trials with GDAR's Σ from the other bins
(`12: leave_one_bin_out`). The sensitivity row "nearest bin, own Sigma" uses that fit's own Σ on bin h − 1.

**Differs:** `MODEL.md` §3c said each model always uses its own Σ. §8.6 and §10 don't, and `RESULTS.md` mixes the two rules: the
same sum-to-one rank 3 model scores +0.214 / +0.313 with its own Σ (review checks, 2026-11-05) and +0.156 / +0.207 with GDAR's
(script 18's clock row, `cache/18_run.log`).

**Status:** §8.6's shared Σ approved 2026-11-05 (§13). The rest is not yet approved.

## 9. Convex form and vertices (`sym_tp.py: polytope_vertices`)

Control needs convex weights (§12.1). The fitted W is turned into vertices Θ_v and weights λ_gv ≥ 0, Σ_v λ_gv = 1.

### 9.1 Sum-to-one projection

```
a = argmin ‖W a − 1‖,   e = 1 − W a,   affine_err = max_g |e_g|
W_a = W + e aᵀ / (aᵀ a)                (then W_a a = 1 exactly)
```

When affine_err ≠ 0 this changes the family: the vertices describe W_a Φ, not the scored W Φ. The projection depends on the gauge
(§6.4): W ↦ W M, Φ ↦ M⁻¹ Φ leaves W_a Φ unchanged when M is a multiple of an orthogonal matrix, and changes it otherwise, including
under the column rescaling of §6.4. Checked on random W, Φ (P = 16, r = 3): change 1e-14 for orthogonal and 2 × orthogonal M,
1.6 for a diagonal rescaling (a scratch check, not in the repo).

### 9.2 Coordinates

Y = W_a L with L Lᵀ = Φ H̄ Φᵀ (+10⁻¹² I) and H̄ the quadratic form of the pooled statistics with Σ̂, divided by the number of targets.
Euclidean distance between rows of Y is the distance between operators in the pooled data metric, per sample.

### 9.3 Vertices

- **Rank 2:** the row farthest from the centroid, then the row farthest from it. The projected bins lie on a line, so these two
  bins contain all others.
- **Rank 3:** the centred rows are projected onto their top two principal directions. Every triple of convex-hull edge lines is
  intersected; among the bounded triangles (outward normals positively spanning the plane) the smallest area is kept
  (`_enclosing_triangle`). Its corners are generally not bins. φ_v = (Y_v L⁻¹) Φ.
- **Rank ≥ 4:** successive projection onto bins, which need not contain the trajectory (`outside` reports the miss).

### 9.4 Weights

λ_g = nonnegative least squares of [Y_vᵀ; 10³·1ᵀ] λ ≈ [y_g; 10³], then divided by its sum (`tp.py: _barycentric`). Exact for
points inside the polytope; for points outside, the reported miss depends on the scale of Y (review M4).

**Differs:**
- Log L-2026-11-05-f, the script 18 docstring and `RESULTS.md` (single-trial section) call the rank 3 triangle the "smallest
  enclosing triangle". It is the smallest triangle with all three sides on hull edges, which can be much larger: on a near-square
  hull it puts a corner at (0, −5) (review M2).
- `MODEL.md` §4 described `tp.convex_hull` (tight, spa, box). The IOS work uses `polytope_vertices` instead.
- The vertex model scores below the fit it comes from (`RESULTS.md`, review checks 2026-11-05).

**Status:** not yet approved (written 2026-11-04, before the gate).

### 9.5 Vertex form, fitted directly (`sym_tp.py: fit_sym_tp_vertex, simplex_qp`)

Added 2026-11-21. The convex form is built into the fit, so the vertices describe the model that is scored. `fit_sym_tp`
(§6) and `polytope_vertices` (§9.1 to §9.4) are unchanged.

```
Θ(p_g) = Σ_{v=1..V} λ_gv Θ_v,      λ_g ∈ Δ_V = { λ : λ_v ≥ 0, Σ_v λ_v = 1 }

J(Λ, Φ) = Σ_g L_g( Σ_v λ_gv φ_v )  +  μ n Σ_v ‖φ_v − φ̄‖²_H̄
```

| symbol | meaning |
|---|---|
| v = 1..V | vertices (V = 2, 3 in script 19) |
| Θ_v = Θ(φ_v) | vertex operator, symmetric and graph-supported (§4.1) |
| λ_gv, Λ | weight of vertex v in bin g; Λ = [λ_gv] (stored in `W`) |
| Φ = [φ_1; …; φ_V] | the vertices |
| L_g | bin g's GLS loss with Σ fixed from the GDAR fit, as in §6.1 |
| φ̄ = (1/V) Σ_v φ_v | mean vertex |
| H̄ = Σ_g H_g / n | pooled data metric per sample (§9.2); ‖φ‖²_H̄ = φᵀH̄φ |
| n | number of training targets, so μ is a per-sample weight with no units |
| μ ≥ 0 | penalty weight |

Σ_v ‖φ_v − φ̄‖²_H̄ = (1/V) Σ_{v<w} ‖φ_v − φ_w‖²_H̄: the penalty is the summed squared distance between vertices, in how
differently they predict.

**Steps,** each an exact block minimiser, so J never increases:

```
λ-step, each bin:  λ_g = argmin_{λ ∈ Δ_V}  λᵀ (Φ H_g Φᵀ) λ − 2 λᵀ Φ h_g
Φ-step:            ( Σ_g λ_g λ_gᵀ ⊗ H_g + μ n (I_V − 11ᵀ/V) ⊗ H̄ ) vec Φ = Σ_g λ_g ⊗ h_g
```

The λ-step is solved exactly by enumerating the faces of the simplex (`simplex_qp`): for each support S, the KKT system of the
problem restricted to S with Σλ = 1; infeasible solutions are dropped and the best feasible one kept (2^V − 1 small solves). A
new λ_g is kept only if it lowers the bin's objective. The Φ-step uses the §4.2 jitter.

**Start:** the convex form of the free rank-V fit (§9.1 to §9.3). **Stop:** |ΔJ| ≤ 10⁻⁷ |J| or n_iter sweeps (100 in script 19).

**Properties.** V = 1 is GDAR. At μ = 0 the vertex form spans the same models as a sum-to-one family: any simplex that contains the
fitted bins gives the same loss, so the vertices stay where the start put them. μ > 0 breaks that tie by pulling the vertices
together as far as the data allow, which keeps every vertex near states the data visit (Aditya's "one vertex does not dominate",
read as: no vertex far outside the data). At V = 2 and small μ the vertices are the two end bins, as in §9.3. With finite μ the
penalty also shrinks the bins at the ends slightly toward the centre. At μ = 0 the training data loss satisfies
free rank-V fit ≤ vertex form ≤ convex form of the free fit (the free family contains the vertex family; the vertex form starts at
the convex form).

**μ in script 19:** chosen from {0, 10⁻³, 10⁻², 10⁻¹, 1} by held-out likelihood inside each training set (fit on 3 of the 4
training folds, validate on fold f + 1 mod 5; for evoked removed the kernels are refitted on those 3 folds). Scored as §8.

**Checks** (`tests/test_sym_tp_vertex.py`): `simplex_qp` against SLSQP (V = 2, 3, 4, singular Q included) and a fine grid (V = 3);
V = 1 equals the GLS fit; J never increases and the weights stay on the simplex; zero gradient in Φ at the end; the data-loss
ordering above; the vertex spread falls as μ rises; recovery of a known two-vertex plant.

**First results** (`RESULTS.md`, script 19 section; L-2026-11-21-c). On script 15's synthetic plants (V = 2), μ = 10⁻³ and 10⁻²
moved the vertices away from the truth: median vertex error 0.1095 to 0.2357 against 0.0048 at μ = 0 (`cache/19_synthetic.log`).
The penalty measures spread in one pooled metric H̄, so it can bias the vertices when bins differ widely in energy, as in those
plants; that this is the cause is not tested. On H 2021-11-12 the training folds chose μ = 0 to 10⁻², and the vertex form scored
between the convex form and the free fit at ranks 2 and 3 (`cache/19_run.log`).

**Status:** Approved by Aditya 2026-11-21: "Approve" (gate block: "fit TP-GDAR in vertex form, with the weights on the simplex
inside the fit and a penalty on how far apart the vertices are"). The gate missed that reply at first because of an app note in
front of it; fixed in L-2026-11-21-b.

## 10. Single-trial weights (`scripts/18_single_trial_states.py`)

The rank 3 fold fit (§6, 25 sweeps) and its vertices (§9) are fixed. Fold vertices are put in the order of script 13's full-data
fit: the permutation whose 16-bin weights correlate best. Then on each trial, at each target index t:

```
λ̂(t) = argmin_{λ ≥ 0, Σ_v λ_v = 1}  Σ_{s ∈ window(t)} ‖ x(s+1) − Σ_v λ_v Θ_v z(s) ‖²_{Σ̂⁻¹}
```

Σ̂ is the fold's GLS Σ (§4.3), the fit's own metric. Windows of w = 10, 20, 40 targets (30, 60, 120 ms): **causal**, the w targets
before t (for prediction), or **centred** on t (for timing and agreement). Solved exactly for three vertices: the best of the three
clipped edge solutions and the interior KKT point where it is feasible (`simplex_qp`).

- **Test 3, prediction:** causal λ̂ against the clock weights λ_{bin(t)} of §9. Gain over GDAR, all rows with GDAR's scoring Σ
  (§8.2); the log det terms cancel, so the gain is −½ times the difference in Σ⁻¹-weighted squared error.
- **Test 2, agreement:** per trial, the correlation of centred λ̂ (each weight less its trial mean) with the trial's own clock, and
  with a clock built from the training median go and touch.
- **Test 1, timing (evoked left in only):** template = training trials' mean λ̂ aligned at the event over −200..+300 ms; a test
  trial's event time is the grid time minimising the mean squared distance to it (go 150..700 ms, touch 250..1135 ms, the latter on
  trials with touch ≤ 1135 ms). Scored by Pearson r with the true time, a permutation p (10,000), and the median absolute error
  against guessing the training median.

**Status:** approved 2026-11-05 (§13). The docstring records it.

## 11. Readouts

### 11.1 Spectral radius

max |eig(A_c)| of the companion matrix A_c = [Θ; I_{(K−1)N} 0] (`12: rho`).

### 11.2 Model spectrum

```
S(f) = tr( H(f) Σ̂ H(f)ᴴ ) / N,      H(f) = ( I − Σ_{k=1..K} A_k e^{−i 2π f k / f_s} )⁻¹
```

Frozen at one bin's operator (`12: ar_spectrum`). For the comparison with data, S(f) is convolved in frequency with the normalised
|FFT|² of the 200 ms Hann window (circular, NFFT 512), put in dB, and averaged over the 4 bins of a phase
(`make_deck_figure_spectra.py: model_spectra`). A frozen spectrum isn't the spectrum of a changing system.

### 11.3 Data spectrum by phase

Phase limits as in §2.1, with post ending at min(touch + 500, 1400). Windows of 200 ms (Hann) every 50 ms, each wholly inside the
phase and demeaned; one-sided power |FFT|² · 2 / (Σ w² · f_s), NFFT 512; mean over windows in a trial, then the trial mean per
channel and frequency, then 10 log₁₀, then the mean over channels. Evoked left in (`make_deck_figure_spectra.py: trial_power,
phase_db`).

### 11.4 Edge maps

A vertex's edge map is its operator summed over lags, Σ_k A_k. Script 14's "move-minus-rest" map is the rank 2 movement vertex's
map less the other vertex's (`14: rank2`).

### 11.5 h-step VAF

From starts every 3 targets, the model is run h steps ahead on its own predictions, each step with the operator of that target's
bin. VAF = 1 − Σ‖x − x̂‖² / Σ‖x‖² over scored targets (`12: h_step_vaf`).

**Differs:** where a target inside the horizon has no label, the hold operator is used; those targets aren't scored but feed later
steps (review M5).

### 11.6 Split-half reliability

The correlation between Θ(p) − Θ_GDAR (all entries) from two disjoint random halves of the trials, 5 repeats (script 12). With
evoked removed the kernels are fitted on all trials first, which inflates it.

**Status:** §11.3 approved 2026-11-05 (§13). The rest is not yet approved.

## 12. Control (9-node synthetic example, scripts 15 to 17)

### 12.1 Companion lift

ξ(t) = [x(t); …; x(t−K+1)], ξ(t+1) = A_c ξ(t) + B_c u(t), A_c = [Θ_A; S], S = [I_{(K−1)N} 0]. Because S is constant,
A_c(p) = Σ_v λ_v A_c^(v) only when Σ_v λ_v = 1. That is why control uses the convex weights of §9, never raw TP weights.

### 12.2 Quadratic stability with a decay rate (`control.py: synthesize_quadratic_feedback`)

Find Q ⪰ I (symmetric) and Y such that, at every vertex v,

```
[ α Q              (A_v Q + B_c Y)ᵀ ]
[ A_v Q + B_c Y    α Q              ]  ≻  margin · I
```

Then F = Y Q⁻¹, and with P = Q⁻¹: (A_v + B_c F)ᵀ P (A_v + B_c F) ≺ α² P at every vertex, so one gain gives decay rate α for any
p schedule inside the polytope. Objective: minimise tr Q, or (`min_gain`) minimise t with [[t I, Y], [Yᵀ, I]] ⪰ 0 and Q ⪯ q_max I.

### 12.3 Checks

Worst closed-loop spectral radius over the vertices. `lmi_residual` = max_v λ_max((A_v + B_c F)ᵀ P (A_v + B_c F) − α² P) with
‖P‖ = 1: a clearly positive value means the LMI doesn't hold, even if the solver reports `optimal_inaccurate`. The loop is always
closed on the true plant, never on the fitted model.

### 12.4 Integral action (script 17)

z(t+1) = z(t) + r(t) − x(t), u(t) = F_x ξ(t) + F_z z(t); the augmented system is still a polytope, so §12.2 applies.

### 12.5 Slide 10's worked example (deck only)

S(p) = S_0 + p S_1 + p² S_2, 1 ≤ p ≤ 2.

**Status:** §12.5 approved 2026-11-05 (§13). §12.1 to §12.4 not yet approved.

## 13. Approval register

Approvals came through the method gate: a decision block in chat, then Aditya's next message. Dates are local (Pacific); the
transcripts stamp UTC. No research-log entry has the `Math approved by Aditya` line yet; each row below gives the words.

| § | What | Status | Date | Aditya's words | Gate block (its first line) | Log |
|---|---|---|---|---|---|---|
| 1.1 | IOS Reach preprocessing | not yet approved | | | | L-2026-11-02-a |
| 1.2 | OS-Pulse rules | not yet approved | | | | L-2026-10-26-b |
| 2 | phases, go-locked bins, warped bins | not yet approved | | | | L-2026-11-02-a, L-2026-11-04-a |
| 3 | evoked kernels | not yet approved | | | | L-2026-11-02-a |
| 4 | GDAR model and GLS estimator; edge-wise class | not yet approved | | | | L-2026-11-02-a, L-2026-10-26-b |
| 5 | TP-GDAR model | not yet approved | 2026-11-05 | "so for now lets stick with joint fit setup" (context, not a gate approval) | | L-2026-11-05-f |
| 6.3 | Φ-step numerics: block assembly, Cholesky, CPU | approved | 2026-11-05 | "Approve" | "skip the GPU port for now, speed up the CPU fit, and apply the Σ fix you approved" | L-2026-11-05-d |
| 6 (rest) | objective, W-step, gauge, starts, stopping | not yet approved | | | | L-2026-11-02-a, L-2026-11-04-d |
| 7 | baselines | not yet approved | | | | L-2026-11-04-l |
| 8.6 | GDAR's Σ for every leave-one-bin-out row | approved | 2026-11-05 | "Approve" | "which Σ scores the leave-one-bin-out rows in `scripts/12_tp_vs_gdar_ios.py`" | L-2026-11-05-d |
| 8 (rest) | likelihood, 1% shrink, folds, K, SEs | not yet approved | | | | L-2026-11-02-a |
| 9.1 to 9.4 | convex form of the free fit and its vertices | not yet approved | | | | L-2026-11-04-a |
| 9.5 | vertex form fitted directly, spread penalty, μ chosen inside training folds | approved | 2026-11-21 | "Approve" | "fit TP-GDAR in vertex form, with the weights on the simplex inside the fit and a penalty on how far apart the vertices are" | L-2026-11-21-c |
| 10 | single-trial weights and tests | approved | 2026-11-05 | "Approve" (twice) | "how to tell whether the rank 3 states can be found on a single trial"; "how to test whether the rank 3 states can be found on a single held-out trial, without the task clock" | L-2026-11-05-f |
| 11.3 | data spectrum by phase | approved | 2026-11-05 | "Approve" | "not a model-fit change, in my view: this is how to compute the per-phase spectrum for the slides, and nothing gets fitted" | L-2026-11-05-a |
| 11 (rest) | model spectrum, radius, edge maps, VAF, reliability | not yet approved | | | | L-2026-11-02-a |
| 12.1 to 12.4 | companion lift, LMI, checks, integral action | not yet approved | | | | L-2026-11-04-d, k |
| 12.5 | slide 10's example model | approved | 2026-11-05 | "Approve" | "what model to use for the rank 3 worked example on slide 10" | L-2026-11-05-c |

Gate blocks waiting for an answer: the restructure, phases 1 and 2 (2026-11-07), and the R1 to R8 bug fixes (2026-11-10), both in
`docs/REVIEW_2026-11-06.md`.

## 14. Open method questions

Each needs Aditya's decision through the gate. The current method is listed first.

1. **Which model is "the" model.** The free-weight fit W Φ is scored; the sum-to-one family W_a Φ is plotted and would feed an LMI
   (§9.1). Options: report both; fit with sum-to-one as a constraint inside the joint fit; project in a gauge-invariant way.
   *2026-11-21: Aditya approved fitting in vertex form (§9.5). Whether it replaces the free fit in the reported tables is open.*
2. **One Σ rule for every table** (§8.2): own Σ per model, or GDAR's Σ for all.
3. **The rank 3 vertex rule** (§9.3): the hull-edge triangle; the true minimum-area enclosing triangle; or vertices chosen under
   graph constraints so that no vertex dominates, keeping the convex hull (Aditya's next ask, 2026-11-05).
   *2026-11-21: §9.5's spread penalty is the first such rule ("dominate" read as a vertex far outside the data). Bounds on each
   vertex's edge strengths remain a possible later constraint.*
4. **Gauge** (§6.4): unit-RMS columns or orthonormal columns. It matters only through §9.1.
5. **K = 16 at the top of the CV grid** (§8.4); GDAR's held-out ℓ still rises at K = 32.
6. **The 1% shrink of the scoring Σ** (§8.2).
7. **Starts** (§6.5): pooled only, or pooled and svd, on real data.
8. **Masking** (§1.1): the `noart` steps at target onset; the near-empty plan 1/4 bin on 11-11.
9. **Review M1, M3 to M5** (ridge chosen with a test fold; `fit_windowed` averages coefficients where its docstring says it
   concatenates; the soft sum-to-one in `_barycentric`; the hold operator inside h-step VAF).
10. **Strict diffusion or edge-wise** (§4.4), open since stage 1. The IOS work uses Felix's symmetric class instead of either.
