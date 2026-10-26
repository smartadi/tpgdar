# Latent directed-graph model: d nodes × n-dim LTI each, over the N-electrode array

Status: **specification only**. Nothing here is implemented yet. It is the next model
after the stacked TP-GDAR of `docs/MODEL.md`, and it uses the same conventions
(u(t) → state(t+1), per-trial statistics, mean-centred data, one scaling for all
conditions).

## 0. Idea in one paragraph

The array has N electrodes, N ≈ 90 on OS-Pulse, on a known spatial graph (k-NN,
Laplacian L_N). The model assumes that most of what the electrodes see comes from
**d ≪ N latent nodes**. Each latent node is a small patch of cortex with its own
**n-dimensional LTI dynamics**, for example n = 2 for an E-I pair or n = 3–4 for a
resonant population. The patches are coupled by a **directed, sparse graph W**. The
latent graph is meant to be a *low-dimensional representation of the electrode graph*:
- each latent node is a non-negative, spatially contiguous footprint on the array;
- latent edges are allowed only between patches that touch on the array.

This is DCM-like (neural masses joined by directed effective connectivity, seen
through a lead field). The differences are that it is linear, that the lead field is
learned and tied to the array graph, and that it is fitted by EM or least squares
rather than Bayesian model inversion.

## 1. Model

Notation:
- N electrodes; electrode adjacency A_N and Laplacian L_N.
- d latent nodes, n states per node; latent state ξ = [ξ_1; …; ξ_d] ∈ R^{dn}.
- u(t) ∈ R^M is the stimulation input (M = 1 for the pulse train).

**Node i** has local state ξ_i ∈ R^n and a scalar output y_i = c_iᵀ ξ_i. The output is
what the node sends to other nodes and what the electrodes see.

```
ξ_i(t+1) = F_i ξ_i(t)  +  e_i Σ_{j≠i} W_ij y_j(t)  +  g_i β_iᵀ u(t)  +  w_i(t)
y_i(t)   = c_iᵀ ξ_i(t)
```

- F_i (n×n) holds the local dynamics: oscillation, damping, the E-I loop.
- e_i (n) is where incoming coupling enters the node, for example only the E slot.
- W (d×d) is the **directed** coupling. W_ij is the edge j → i, with W_ii = 0, and
  W need not be symmetric.
- g_i (n) is where stimulation enters; β_i (M) is node i's stimulation gain.

**Stacked.** With F = blkdiag(F_i), E = blkdiag(e_i) (dn×d), C_ξ = blkdiag(c_iᵀ)
(d×dn) and G = blkdiag(g_i) (dn×d):

```
ξ(t+1) = (F + E W C_ξ) ξ(t) + G β u(t) + w(t),        w ~ N(0, Q)
x(t)   = P C_ξ ξ(t) + v(t),                           v ~ N(0, R),  R diagonal
```

Network coupling is a rank-≤ d perturbation that acts only through the node outputs.
Homogeneous special case (F_i = F, e_i = e, c_i = c): A = I_d ⊗ F + W ⊗ (e cᵀ).

**Directed-Laplacian (diffusive) variant.** Replace W by −L_W, where
L_W = diag(W 1) − W is the in-degree Laplacian and W ≥ 0. Node i then receives
Σ_j W_ij (y_j − y_i), which pulls each node toward its upstream neighbours. This is the
directed generalisation of GDAR's a_k I − b_k L. The free signed W is the edge-wise
counterpart, so the strict-diffusion vs edge-wise decision appears again here.

**Observation P ∈ R^{N×d} (the low-dimensional representation of the array graph).**
- P ≥ 0: each column is a latent node's footprint on the electrodes.
- Spatial smoothness: penalise tr(Pᵀ L_N P) = Σ_{edges (a,b)} ‖P_a − P_b‖², which keeps
  footprints contiguous on the array.
- Coarsened graph: S = supp(Pᵀ A_N P) marks the latent node pairs whose footprints touch.
  W is restricted to S (optionally plus a few long-range edges, chosen by group lasso).

## 2. What this buys over full-N GDAR

| | full-N GDAR (K lags, Q input lags) | latent graph |
|---|---|---|
| parameters | ≈ N·(deg+1)·K + N·Q (≈ 4–5 k at N=90, K=5) | N·d (P) + d·(n + 2n) (canonical F, e, g) + \|S\| + d·M |
| response shape to a pulse | input lags B_q (N×Q kernel) | node dynamics: kernel = c_iᵀ (zI − F_i)⁻¹ g_i, so **no input time embedding is needed** |
| oscillations | only through AR lags | each node can be a resonator (complex eigenvalues of F_i) |
| directed propagation / waves | edge-wise A can be asymmetric | W is directed by construction |
| what is interpretable | electrode × electrode coupling | patches (P), patch dynamics (F_i), patch → patch edges (W) |

## 3. Transfer-function view (node filters on a directed graph)

Write the node filter as h_i(z) = c_iᵀ (zI − F_i)⁻¹ e_i and set H(z) = diag(h_i). The
stimulation filter k_i(z) = c_iᵀ (zI − F_i)⁻¹ g_i β_iᵀ is collected the same way into
K_u(z). Then

```
y(z) = H(z) W y(z) + K_u(z) u(z) + noise     ⇒     y = (I − H W)⁻¹ (K_u u + noise),     x = P y + v
```

This is a dynamical structure function with diagonal node filters (Gonçalves–Warnick).
In the network-identification literature (Hendrickx, Gevers, Bazanella; Van den Hof),
the y_i are "nodes" and the W_ij h_i are "modules". Every latent node counts as
measured when P has full column rank (y = P⁺(x − v)), and every node is excited by its
own process noise. That is the favourable case for identifiability of the directed
edges. Check the exact conditions before relying on it.

## 4. Gauges and identifiability (fix these, do not hope)

1. **Per-node similarity transform.** For any invertible T_i,
   (F_i, e_i, g_i, c_i) → (T_i F_i T_i⁻¹, T_i e_i, T_i g_i, T_i⁻ᵀ c_i) leaves the model
   unchanged. Fix it with the **observer-canonical form**: c_i = e_1, F_i companion. Each
   node is then a scalar ARMA filter on its own output: n AR coefficients a_i, plus
   numerator vectors e_i and g_i.
2. **Per-node scale.** y_i → α_i y_i is absorbed by P's column i, row and column i of W,
   e_i, and β_i. Fix ‖P_{:,i}‖₂ = 1.
3. **Node permutation.** Harmless relabelling; sort nodes by footprint centroid.
4. **Mixing between nodes through P.** Non-negativity alone leaves an NMF-style
   ambiguity. It becomes unique when each node has some "pure" electrodes (the
   separability assumption). The smoothness penalty and contiguity push towards that.
   Report footprint overlap.
5. **Coupling vs local dynamics.** e_i W_ij is bilinear. Fix e_i's first entry to 1, or
   fix e_i = e_1 so coupling enters only the output slot (for an E-I node: E receives
   long-range input).

## 5. Learning

All statistics are per trial; trials are independent sequences. Data are mean-centred,
with one per-channel scale for every condition.

**Stage 0: initialise from what already works.**
1. **P₀.** Coarsen the electrode graph into d contiguous patches. Use spectral clustering
   on the fitted full-N GDAR coupling restricted to A_N, or Loukas' spectral coarsening
   of L_N. Hard assignment, column-normalised.
2. **y₀ = P₀⁺ x**, the patch-mean signals.
3. **Equation-error fit on y₀**, which is exactly the existing GDAR machinery with d
   channels. With e_i = g_i = e_1, node i's output follows
   y_i(t+1) = Σ_{k<n} a_ik y_i(t−k) + Σ_{j∈S_i} W_ij y_j(t) + β_iᵀ u(t) + noise.
   This is a masked VARX on d channels with diagonal AR(n) and one-lag directed
   coupling, solved row by row with ridge least squares (`tpgdar.fit_static` with a
   d-node mask).
4. **(Optional) dn-dim N4SID.** Run it on x only to choose dn and to check that dn
   latent dimensions are enough. Turning an unstructured subspace model into the
   block-structured form is a BMI (Massioni–Verhaegen; Yu–Verhaegen), so do not use it
   for the structure itself.

**Stage 1: EM on the full state-space model.**
- **E-step.** Kalman filter plus RTS smoother per trial (as in `latent.py`) on the
  dn-dimensional state. After a short burn-in, use the steady-state gain: with
  dn ≈ 30–60 and about 550 trials × 1000 samples, time-varying Riccati updates would
  dominate the cost.
- **M-step, observation.** With R diagonal, the rows of P separate:
  P_a = argmin_{p ≥ 0} ‖x_a − p C_ξ ξ‖² over the smoothed moments. That is NNLS with
  Gram C_ξ S_ξξ C_ξᵀ. The smoothness term μ tr(Pᵀ L_N P) couples the rows; use
  projected gradient or block coordinate descent over electrodes. Renormalise the
  columns, push the scale into W, e and β, and update R from the residual moments.
- **M-step, dynamics.** Minimise tr Q⁻¹(S₁₁ − 2 A S₁₀ᵀ + A S₀₀ Aᵀ) together with the
  input terms. In canonical form A is linear in (a_i, W) when e is fixed, and linear in
  e when W is fixed. Alternate the two exact least-squares steps, so the M-step never
  increases the loss (generalised EM). Use NNLS for W ≥ 0 in the Laplacian variant and
  a support mask on S. Update Q in closed form, diagonal or block-diagonal per node.
- **Stopping.** Stop on a small relative change in the log-likelihood. Keep the best of
  several P₀ initialisations, since EM finds a local optimum.

**Stage 2: model selection and comparison** (held-out one-step log-likelihood, as in
MODEL.md §3c).
- Grid over d ∈ {4, 6, 9, 12, 16} and n ∈ {1, 2, 3, 4}.
- Baselines:
  - full-N stacked TP-GDAR (script 11);
  - unconstrained LDS of the same dn (what the graph structure costs);
  - the same model with symmetric W (does direction matter?);
  - fixed P₀ vs learned P.

## 6. Parameter dependence (TP) and control

- **Stimulation-frequency effect.** Script 11 found that the frequency effect lives in
  the input, not the dynamics. Here that means TP on β(p), at most d·M numbers per
  condition, possibly rank 1 (one spatial gain map times a gain curve). F and W stay
  shared.
- **If dynamics do vary.** Use W(p) = Σ_v λ_v(p) W_v with convex weights. A convex
  combination of non-negative or Laplacian W stays non-negative or Laplacian, so every
  W(p) is still a graph. Raw HOSVD weights do not preserve this.
- **Control.** The LMI synthesis runs on the dn-dimensional polytope, which is much
  smaller than the N·K companion form. The loop is still closed on the plant, never on
  the model.

## 7. Open questions

- **Readout.** Is a scalar output per node enough, or do the electrodes see more than
  one state per node (c_i of rank > 1, e.g. E and I both contributing to the LFP)?
- **Delays.** Is one-lag coupling enough at ~1 kHz, or does W need delays
  (W(z) = Σ_ℓ W_ℓ z^{−ℓ}, as in DLAG)?
- **Fixed vs learned P.** Should P be fixed to the coarsening, or learned? Fixed is more
  interpretable and identifiable; learned may fit better.

## 8. Related work (where each piece comes from)

- **Model reduction of networks of identical agents by graph partitions.** This is the
  forward version of P: aggregate an N-node network of n-dim agents onto a d-node
  quotient graph.
  - Monshizadeh, Trentelman, Camlibel, *Projection-based model reduction of multi-agent
    systems using graph partitions*, IEEE TCNS 2014.
  - Directed and positive networks: Ishizaki et al., *Clustered model reduction of
    positive directed networks*, Automatica 2015 (https://hal.archives-ouvertes.fr/hal-01276233).
- **Graph coarsening with spectral guarantees (used for P₀).** Loukas, *Graph reduction
  with spectral and cut guarantees*, JMLR 2019 (https://arxiv.org/abs/1808.10650). Kron
  reduction is the resistance-preserving alternative.
- **Subspace identification of networks of homogeneous subsystems.** Shows why the
  structured realisation is a BMI and how local subsystems are identified.
  - Massioni & Verhaegen (decomposable systems).
  - Yu & Verhaegen, e.g. https://arxiv.org/abs/1309.5105 and https://arxiv.org/abs/1702.03539.
- **Identifiability of directed dynamical networks.**
  - Hendrickx, Gevers, Bazanella, *Identifiability of dynamical networks with partial
    node measurements*, IEEE TAC 2019 (https://arxiv.org/abs/1803.05885).
  - Local module identification: Van den Hof and coworkers (e.g.
    https://arxiv.org/abs/1804.10389).
  - Dynamical structure functions: Gonçalves, Howes, Warnick (IEEE TAC 2008); Yuan et al.,
    *Robust network reconstruction*, Automatica 2011.
- **Directed graph processes on time series (the GDAR lineage).** Mei & Moura,
  *Signal processing on graphs: causal modeling of unstructured data*, IEEE TSP 2017
  (https://arxiv.org/abs/1503.00173). Here x(t) is a polynomial in a directed shift
  matrix applied to past x.
- **Neuroscience analogues.**
  - DCM for M/EEG and ECoG: neural masses with directed connectivity and an estimable
    lead field (Kiebel, David, Friston; review Moran, Pinotsis, Friston 2013).
  - Multi-region latent models: DLAG (Gokcen et al., Nat. Comput. Sci. 2022, delayed
    feedforward and feedback latents); MR-SDS (Karniol-Tambour et al., ICLR 2024);
    MRM-GP (Li et al., ICML 2024).

  These assume known regions. Here the regions (P) are learned from the array graph.
