# GDAR-TP: the model and how it is learned

This is the reference for what the code fits. `README.md` gives the short form, and
`RESULTS.md` the numbers. Notation: N channels, K state lags, Q input lags, M inputs,
P grid points (conditions) of the scheduling variable p.

## 1. GDAR at one operating point: a single stacked LTI operator

Stack the time embedding into one regressor:

```
z(t) = [ x(t); x(t-1); ...; x(t-K+1); u(t); u(t-1); ...; u(t-Q+1) ]  ∈ R^D,   D = KN + QM

x(t+1) = Θ z(t) + e(t),     Θ = [ A_0  A_1 ... A_{K-1} | B_0 ... B_{Q-1} ]  ∈ R^{N×D},   e ~ N(0, Σ)
```

- The "layered" lag matrices are column blocks of **one** operator Θ. The lags are
  not separate systems: together they form one LTI system of order K.
- Input timing: u(t) drives x(t+1). Every simulator and fitter in the repo follows this.
- **Graph constraint.** Row i of every A block may be nonzero only on Ω_i = N(i) ∪ {i},
  the array neighbours plus the channel itself. Input columns are unconstrained. Two
  parameterisations exist (still the open decision):
  - edge-wise: every entry of Θ on the support is free;
  - strict diffusion: A_k = a_k I − b_k L.
- **State-space form.** The companion lift is ξ(t) = [x(t);…;x(t−K+1)], ξ(t+1) = A_c ξ(t) + B_c u(t),
  where A_c = [Θ_A ; S] and S = [I_{(K−1)N} 0] is the constant shift block.

**Learning (one operating point).** Everything a linear-Gaussian fit needs is in the
sufficient statistics, accumulated *within* each trial, never across trial boundaries:

```
G = Σ_t z(t) z(t)ᵀ        C = Σ_t x(t+1) z(t)ᵀ        S_yy = Σ_t x(t+1) x(t+1)ᵀ
```

The data loss is quadratic in Θ:

```
L(Θ) = Σ_t ||x(t+1) − Θ z(t)||²  =  tr(Θ G Θᵀ) − 2 tr(Θ Cᵀ) + tr(S_yy)
```

It is minimised row by row on the support, with a ridge term:
θ_i = (G_{Ω_i Ω_i} + λ_i I)⁻¹ C_{i,Ω_i}. Two properties follow:

- Completing the square gives `L(Θ) = || (Θ − Θ̂) G^{1/2} ||²_F + const`, where Θ̂ is the
  least-squares estimate. **The data measure distance between operators in the G-metric,
  not the Frobenius one.** Directions the data excite strongly are well determined;
  weakly excited ones are noisy.
- The ridge is Levenberg-style, one weight per coefficient: λ_{ic} = ρ·G_{cc}. Each
  coefficient is shrunk in proportion to how strongly its own regressor is excited.
  This makes the ridge free of units, which matters because A lags and input lags
  have different units, and it does not change meaning with data length. A single
  weight per row is **not** enough on real ECoG, where the lag Gram has condition
  number ~5e10.
- **ρ must be chosen by validation**, inside the training data. On OS-Pulse the
  inherited ρ = 1e-6 is about four decades too strong: held-out ℓ rises by about
  +84 nats/sample as ρ drops toward 1e-10, where it saturates.
- **The ridge sits on the grid-point operators**, Σ_g ρ‖Θ(p_g)‖²_{diag G_g}, which is
  an augmented Gram G_g + ρ diag G_g. It must not sit on the TP factors Θ_j: a
  penalty on the factors can be escaped by rescaling W against Θ_j (the gauge), and
  the fit then silently drops its regularisation.

## 2. Parameter variation: TP structure on the stacked operator

The operator depends on a measured scheduling variable, Θ = Θ(p). The TP model says
this dependence is low-rank:

```
Θ(p) = Σ_{j=1}^{r} w_j(p) Θ_j         (all Θ_j share the support Ω)
```

Stack the grid values into a tensor 𝒯 ∈ R^{P × N × D}, 𝒯[g] = Θ(p_g). The TP form says
𝒯 has rank r in the **p mode only**. The N and D modes are never compressed, because a
vertex has to stay a complete lag-K, graph-supported LTI system.

- **What "stacking" means.** A and B belong in the same tensor, as columns of Θ.
  Different ranks per block are allowed:
  `Θ(p) = [ Σ_j h_j(p) A^(j) | Σ_l f_l(p) B^(l) ]`, with ranks (r_A, r_B).
  - r_A = 1 gives a shared A with p-dependent input. This is script 10's model.
  - r_B = 1 gives p-dependent dynamics with a shared B.
  - r_A = r_B = r with h = f gives one joint mode set.
- **Companion lift is affine, not linear.** A_c(p) = [Σ_j w_j Θ_{A,j} ; S] equals
  Σ_j w_j A_c^(j) only when Σ_j w_j(p) = 1, because the shift block S is constant.
  HOSVD weights do not sum to one. This is why the control step uses the convex
  (polytopic) weights of §4, never the raw HOSVD factors.

## 3. Learning the TP-GDAR

Per grid point g, the training data give (G_g, C_g, S_yy,g). The quantity to minimise
is the **data loss of the whole model**:

```
J(W, Θ_1..Θ_r) = Σ_g L̃_g( Σ_j W_gj Θ_j ),    L̃_g = L_g with G_g → G_g + ρ diag G_g      (W ∈ R^{P×r},  W_gj = w_j(p_g))
```

By §1, J = Σ_g || (Σ_j W_gj Θ_j − Θ̂_g) G_g^{1/2} ||²_F + const. So **fitting TP-GDAR
is a G_g-weighted low-rank approximation of the per-condition estimates.**

### 3a. Two-stage (what the scripts did up to 2026-10-26)

1. Fit Θ̂_g independently at each grid point.
2. HOSVD (for one p, the SVD) of the unfolding [vec Θ̂_1; …; vec Θ̂_P] gives W = U_r, and the
   core gives Θ_j.

This minimises Σ_g ||Θ − Θ̂_g||²_F, which equals J only when every G_g ∝ c·I with the
same c: equal amounts of data and white, isotropic regressors in every condition. Real
recordings break all three:

- trial counts and signal power differ by condition;
- neighbouring channels are correlated;
- A and B columns have different units, and an unweighted HOSVD of the stacked [A|B]
  depends on that arbitrary scaling.

The consequence is that the unweighted SVD spends rank on the *poorly determined*
(noisy) coefficient directions. Two-stage is still the right initialiser.

### 3b. Joint fit by alternating least squares (`gdar_tp/tpgdar.py`)

Each step minimises J exactly over one block, so J never increases.

- **Θ-step** (W fixed). Row i stacks its unknowns φ_i = [θ_{1,i,Ω_i}; …; θ_{r,i,Ω_i}] and solves

  ```
  ( Σ_g (w_g w_gᵀ) ⊗ G̃_g[Ω_i,Ω_i] ) φ_i = Σ_g w_g ⊗ C_g[i,Ω_i],      G̃_g = G_g + ρ diag G_g
  ```

  This is ordinary least squares on the features w_g ⊗ z(t): a linear-in-parameters
  model (the qLPV "basis expansion" in `docs/PLAN.md`).
- **W-step** (Θ_j fixed). Each g separately solves an r×r system:

  ```
  H_g w_g = h_g,     H_g[j,l] = tr(Θ_j G_g Θ_lᵀ),     h_g[j] = tr(Θ_j C_gᵀ)
  ```

- **Start.** Component 1 is the pooled operator with constant weight; the rest
  span the leading per-condition deviations from it. The fit therefore starts at,
  and can only improve on, the pooled model. Starting from the two-stage HOSVD is
  valid, but on real data it sits so far from the optimum in the data metric that
  ALS needs hundreds of sweeps.
- **Bracket.** On the same objective, per-condition ≤ joint ≤ pooled. This is a test
  (`test_joint_objective_bracketed_by_per_condition_and_pooled`); a gauge leak in the
  ridge breaks the lower bound.
- **Gauge.** W ↦ W M, Θ ↦ M⁻¹Θ leaves Θ(p) unchanged. After convergence, W is rotated to
  orthonormal columns (the HOSVD convention) so the factors can be compared.
- **Off-grid p.** w(p) is interpolated linearly between grid points (in log-frequency
  for stimulation frequency). Θ(p) is linear in w, so this equals interpolating the
  reconstructed operators.
- **Rank.** Chosen by held-out data likelihood (held-out trials, and
  leave-one-condition-out), not by the singular values of the unweighted tensor.

### 3c. Evaluation

- **Metric.** Held-out one-step Gaussian log-likelihood per sample. Each model uses its
  own Σ, estimated from its own training residuals, and results are reported as Δℓ
  against a baseline. One-step NRMSE is reported but is nearly blind: on OS-Pulse it is
  dominated by power below 2 Hz.
- **Baselines.**
  - pooled (p-blind, one Θ);
  - per-condition Θ̂_g (the no-interpolation upper bound on the grid);
  - nearest-condition;
  - oracle (fitted on the test condition).

## 4. Polytope and control

From the fitted trajectory {Θ(p_g)}, `tp.convex_hull` finds vertices Θ_v with convex
weights λ_v(p) ≥ 0, Σ λ_v = 1, such that Θ(p) = Σ_v λ_v Θ_v. Only with Σ λ = 1 does the
companion lift give A_c(p) = Σ_v λ_v A_c^v (§2). The LMI synthesis (`control.py`) needs
every A_c(p) to lie inside the polytope:

- the "tight" and "spa" vertices are data points on the trajectory;
- the "box" vertices are snapped to the nearest data points and need **not** contain the
  trajectory. `convex_hull` warns when they do not.

The loop is always closed on the nonlinear plant, never on the fitted model.

## 5. Preprocessing rules (each one changes the fitted operator if broken)

| rule | why |
|---|---|
| Accumulate G, C per trial; never concatenate trials into one series | concatenation adds K false transitions per boundary |
| Do **not** z-score per condition; scale every condition by one per-channel std from training data | per-condition scaling turns A into D_g⁻¹ A D_g, a fake p-dependence (or a hidden real one) |
| Mean-centre per segment (or include an intercept column) | the linear model describes deviations from the operating point |
| OS-Pulse stimulation window = **1.0–2.0 s** | measured: the response locked to the stimulation frequency is present only 1.0–2.0 s (40 Hz: 12–17× baseline, back to ~1× by 2.0 s); the export README's 1–6 s is wrong |
| u(t) → x(t+1) | a one-sample shift makes the fitted B absorb nothing (B → 0) |

`scripts/09_ospulse_real.py` broke rules 1, 2 and 4: it used a 1.5–5.5 s window, about
87 % post-stimulation. Its "honest null" is therefore not a valid test of TP on A, and is
superseded by `scripts/11_tp_joint.py`.
