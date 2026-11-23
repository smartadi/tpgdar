r"""Update deck for 23 November 2026: the joint fit written on r + 1 stable vertices, single-trial weights, other days, and
the plan for monkey L.

Run:  .venv\Scripts\python scripts/make_deck_figures_hull.py   (figures, from the caches of scripts 13 and 18)
      .venv\Scripts\python scripts/make_stable_vertices_deck.py
Writes TP-GDAR_stable_vertices.pptx in the repo.

Same building blocks as TP-GDAR_IOS_Reach.pptx (scripts/make_ios_tp_deck.py): text and figures are shapes PowerPoint for the
web can edit (scripts/deck_native.py), except the rank 2 triangle, which is a picture; equations are LaTeX pictures with
their source as alt text (scripts/deck_latex.py).  Slide text is Claude's draft and each slide's notes say so.

Numbers: tables read the caches directly (cache/13_vertices_H_2021-11-12.pkl, cache/14_days.pkl).  Numbers typed into text
carry a src comment: cache/13_run.log, 14_run.log, 18_run.log, 19_run.log, the figure script's printout, and for monkey L
the segment files in ../yazdanlab/iosreach/cache/segs (checked 2026-11-22 by a read of each file's channels, stimmed and
move_ms).  Math: docs/MATH.md §4 to §10.
"""

import os
import pickle
import sys

import numpy as np
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the caches pickle gdar_tp objects
from deck_latex import add_latex  # noqa: E402
from make_ios_tp_deck import bins_slide, steps  # noqa: E402
from make_mpc_tp_deck import edge_key, fit_picture, native  # noqa: E402
from make_tp_math_deck import LEFT, MUTED, ROOT, WIDTH, Deck, title_slide  # noqa: E402

OUT = os.path.join(ROOT, "TP-GDAR_stable_vertices.pptx")
DECK = os.path.join(ROOT, "figures", "deck")
DRAFT = "Draft: slide text written by Claude, not yet checked by Aditya."
C13 = os.path.join(ROOT, "cache", "13_vertices_H_2021-11-12.pkl")
C14 = os.path.join(ROOT, "cache", "14_days.pkl")
SMALL_NODES = {"node_radius": 0.07}


def notes(s, text):
    s.notes_slide.notes_text_frame.text = DRAFT + "\n\n" + text


def caption(d, s, text, y=6.62):
    d.text(s, text, y, 0.5, size=16, bullet=False, color=MUTED)


def load():
    with open(C13, "rb") as fh:
        c13 = pickle.load(fh)
    with open(C14, "rb") as fh:
        c14 = pickle.load(fh)
    return c13, c14


# ----------------------------------------------------------------------------------------------- opening

def summary_slide(d):
    s = d.slide("Since 5 November")
    d.text(s, ["The math is in one document, docs/MATH.md, written from the code, with each approval recorded",
               "The fitted rank r model is now written exactly on r + 1 vertices: a regular simplex around the 16 bins. "
               "Nothing is lost from the fit",
               "Of all such simplices we keep the smallest whose vertices are all stable. At rank 3 on 11-12: one vertex "
               "for hold, one for move, two shared by plan and post",
               "Weights read off 60 ms of a single trial predict held-out data better than the task clock: "
               "+0.194 vs +0.162 nats/sample",  # src: cache/18_run.log, figure script printout
               "A stable simplex exists on 14 of 16 fits over 4 sessions. Not on 11-11 lit trials with the evoked "
               "response removed",
               "Next: the same pipeline on monkey L (2022-08-10) and three more H sessions"], 1.45, 5.0, size=22)
    notes(s, "All fits: monkey H. 11-12 is the no-stim day of the 5 November deck (320 trials, 22 channels, K = 16). Gains are one-step "
             "held-out log-likelihood over Felix's GDAR, 5-fold over trials. Evoked left in unless said otherwise. "
             "Sources: RESULTS.md sections of 2026-11-21 and 2026-11-22; logs cache/13_run.log, 14_run.log, 18_run.log, 19_run.log.")


def model_slide(d):
    s = d.slide("The model: GDARs mixed across the trial")
    y = steps(d, s, [("GDAR", r"x(t+1)=\sum_{k=1}^{K}A_k\,x(t-k+1)+e(t),\qquad A_k=A_k^{\top},\ \text{zero off the graph}"),
                     ("Free numbers", r"\mathrm{vec}\,\Theta=R\,\phi,\qquad \Theta=[\,A_1\ \cdots\ A_K\,]"),
                     ("TP-GDAR", r"\Theta(p_g)=\sum_{j=1}^{r}W_{gj}\,\Theta_j,\qquad g=1,\dots,16"),
                     ("Vertex form", r"\Theta(p_g)=\sum_{v=1}^{V}\lambda_{gv}\,\Theta_v,\qquad \lambda_{gv}\geq 0,\quad \sum_v\lambda_{gv}=1")],
              y=1.3, gap=1.12, size=24, label_w=2.4)
    d.text(s, ["Each Θⱼ and each vertex Θᵥ is a complete GDAR with all 16 lags. Only the trial direction is compressed",
               "Vertex form: every bin is a mix of V fixed systems. That is the form polytopic (LMI) control needs"],
           y + 0.15, 1.3, size=20)
    notes(s, "N = 22 channels, 50 edges, K = 16 lags (48 ms at 333 Hz): 1,152 free numbers per operator (5 November deck). "
             "R copies each edge number into (i, j) and (j, i). MATH §4 and §5.")


def joint_slide(d):
    s = d.slide("Joint fit: each bin scored on its own data")
    y = steps(d, s, [("Loss in bin g", r"L_g(\phi)=\sum_{t\in g}\|\,x(t+1)-\Theta(\phi)\,z(t)\,\|_{\Sigma^{-1}}^{2}=\|\,\phi-\hat\phi_g\,\|_{H_g}^{2}+\mathrm{const}"),
                     ("Objective", r"J(W,\Phi)=\sum_g L_g\Bigl(\sum_{j}W_{gj}\,\phi_j\Bigr)"),
                     ("W step", r"(\Phi\,H_g\,\Phi^{\top})\,w_g=\Phi\,h_g"),
                     ("Φ step", r"\Bigl(\sum_g w_g w_g^{\top}\otimes H_g\Bigr)\,\mathrm{vec}\,\Phi=\sum_g w_g\otimes h_g")],
              y=1.3, gap=0.85, size=22, label_w=2.4)
    d.table(s, [["Held-out gain over GDAR (nats/sample)", "evoked left in", "evoked removed"],
                ["Rank 2", "+0.190 (0.009)", "+0.248 (0.012)"],       # src: cache/13_run.log lines 17, 44
                ["Rank 3", "+0.222 (0.009)", "+0.323 (0.013)"],       # src: cache/13_run.log lines 22, 49
                ["Separate fit per bin", "+0.211", "+0.392"]],        # src: cache/13_run.log lines 28, 55
            y + 0.25, [5.2, 2.6, 2.6], row_h=0.44, size=18)
    caption(d, s, "H 2021-11-12, 16 bins, 5-fold over trials, fold SE in brackets. Unchanged by anything on the next slides.",
            y=6.85)
    notes(s, "H_g, h_g: bin g's data in the GLS metric, Sigma fixed from the pooled GDAR fit (MATH §6). phi-hat_g is the separate GLS fit "
             "of bin g, so the joint fit is the H_g-weighted rank r approximation of the per-bin fits. Both steps are exact block "
             "minimisers, so J never increases. Rank 3 numbers: 3,495 (5 November deck); a separate fit per bin: 18,432.")


# ----------------------------------------------------------------------------------------------- the hull

def why_slide(d):
    s = d.slide("Convex weights: r + 1 vertices cost nothing")
    y = add_latex(s, r"\Theta(p_g)=\sum_{j=1}^{r}W_{gj}\,\Theta_j=\sum_{v=1}^{r+1}\lambda_{gv}\,\Theta_v,\qquad \lambda_{gv}\geq 0,\quad\sum_v\lambda_{gv}=1",
                  LEFT + 0.1, 1.35, size=24)
    y = add_latex(s, r"\mathrm{convex}(V=r)\ \subseteq\ \mathrm{free}(r)\ \subseteq\ \mathrm{convex}(V=r+1)", LEFT + 0.1, y + 0.3, size=24)
    d.text(s, ["With V = r, sum to one is a constraint. Our old convex form with r vertices lowered the held-out gain: "
               "rank 2 from +0.190 to +0.142, rank 3 from +0.222 to +0.214",   # src: cache/19_run.log, free and convex rows
               "With V = r + 1 it is not: any r + 1 points whose simplex contains the bins give weights that rebuild the free "
               "fit exactly. Largest miss over bins: under 5e-15 of the spread",  # src: cache/13_run.log, rebuilt-bin miss
               "So the fit is done once, freely, and the vertices are a choice of coordinates. Many simplices contain the bins. "
               "We choose one by its shape, rotation and tightness"], y + 0.4, 3.0, size=21)
    notes(s, "Evoked left in, H 2021-11-12. The old convex form: sum-to-one projection of the weights, then the smallest enclosing "
             "triangle (MATH §9.1 to §9.4). Removed: rank 2 +0.248 to +0.224, rank 3 +0.323 to +0.313 (cache/19_run.log). "
             "Fitting the weights on the simplex inside the fit (script 19, MATH §9.5) recovered most of it, but on synthetic plants the "
             "spread penalty moved the vertices away from the truth, so the r + 1 hull replaced it as the default.")


def hull_slide(d):
    s = d.slide("The hull: a regular simplex in the data metric")
    y = steps(d, s, [("Coordinates", r"y_g=W_g\,L,\quad L\,L^{\top}=\Phi\,\bar H\,\Phi^{\top},\quad \tilde y_g=(y_g-\bar y)\,U"),
                     ("Facets", r"n_k=R\,\nu_k,\quad m_k=\max_g\,n_k^{\top}\tilde y_g,\quad h^{*}=\tfrac{1}{r+1}\sum_k m_k"),
                     ("Vertices", r"a_v=c-\tau\,r\,h^{*}\,n_v,\qquad n_k^{\top}c=m_k-h^{*}"),
                     ("Weights", r"\lambda_{gv}=\frac{\tau\,h^{*}-n_v^{\top}(\tilde y_g-c)}{(r+1)\,\tau\,h^{*}}"),
                     ("Operators", r"\phi_v^{\top}=(a_v\,U^{\top}+\bar y)\,L^{-1}\,\Phi")],
              y=1.25, gap=0.83, size=22, label_w=2.3)
    d.text(s, ["Distance in y is distance in one-step prediction, pooled over the 16 bins",
               "R: the rotation (one angle at rank 2, three at rank 3). τ ≥ 1: tightness. At τ = 1 each facet touches a bin",
               "All vertices are equally far apart, so the shape favours none of them"], y + 0.1, 1.6, size=19)
    notes(s, "nu_k: unit outward facet normals of a reference regular simplex, summing to zero. m_k: how far the bins reach in "
             "direction n_k. h*: centre-to-facet distance. Vertex v is opposite facet v. U: the bins' principal axes. "
             "Raising tau shrinks every weight toward 1/(r + 1): the TP paper's Transformation 2 with b = V/(tau - 1). "
             "MATH §9.6, approved by Aditya 2026-11-22. Tests: tests/test_regular_hull.py.")


def rotation_slide(d):
    s = d.slide("Rotation and tightness change vertex stability")
    fit_picture(s, os.path.join(DECK, "hull_r2_geometry.png"), LEFT, 5.4, top=1.3, bottom=6.4)
    native(s, "hull_rotation_r2", x=6.3, y=1.3, w=13.333 - 6.3 - 0.4, bottom=6.4)
    caption(d, s, "Rank 2, 11-12. Left: the 16 bins and the kept triangle (solid, τ = 1); dashed, τ = 1.5. "
                  "Right: dot = kept rotation. Every vertex is stable at 38% / 33% of rotations.")  # src: figure script printout
    notes(s, "Left, evoked left in: kept rotation 30.5 degrees, size h 0.580, vertex spectral radii 0.965, 0.964, 0.968, vertices 0.28, 0, "
             "1.05 of the spread from the nearest bin. A triangle's rotation repeats every 120 degrees. Sweep at tau = 1 over 120 rotations "
             "1 degree apart (cache/13_run.log). At rank 3 (100 random rotations) every vertex is stable at 7% / 7% of rotations at tau = 1, "
             "and at none for tau of 1.25 or more. Loosening the simplex pushes vertices further from the data, where the fitted family "
             "is extrapolated, and that is where they turn unstable.")


def rule_slide(d, c13):
    s = d.slide("The rule: the smallest simplex whose vertices are all stable")
    y = steps(d, s, [("Stability", r"\rho_v=\max\,|\,\mathrm{eig}\,A_c(\Theta_v)\,|<1\quad\text{for every vertex }v"),
                     ("Rule", r"\tau=1,\qquad \text{rotation with the smallest }h^{*}\text{ such that }\max_v\rho_v<1")],
              y=1.3, gap=0.85, size=22, label_w=2.0)
    d.text(s, "Candidates tried from the smallest up: 1,200 rotations at rank 2, 2,000 at rank 3. If none is stable, "
              "the least unstable is kept and flagged.", y + 0.05, 0.8, size=19, bullet=False)
    old = {"felix": ("0.968", "1.012", "0.966", "0.965"),     # src: cache/13_run.log line 10
           "induced": ("0.964", "1.037", "0.943", "0.941")}   # src: cache/13_run.log line 37
    rows = [["Rank 3, 11-12", "size h*", "distance from nearest bin", "spectral radius ρ"]]
    for v, lab in (("felix", "left in"), ("induced", "removed")):
        f = c13[v]["full"][3]
        sm, p = f["smallest"], f["poly"]
        rows.append([f"{lab}, smallest", f"{sm['h']:.3f}", ", ".join(f"{q:.2f}" for q in sm["dist"]), ", ".join(old[v])])
        rows.append([f"{lab}, smallest stable", f"{p['h']:.3f}", ", ".join(f"{q:.2f}" for q in p["dist"]),
                     ", ".join(f"{q:.3f}" for q in p["rho"])])
    d.table(s, rows, y + 1.0, [2.9, 1.4, 3.9, 3.9], row_h=0.44, size=17)
    caption(d, s, "Rank 2 on 11-12 needed no change: its smallest triangle already had every vertex stable.")
    notes(s, "rho_v: largest eigenvalue modulus of vertex v's companion matrix (MATH §11.1). Approved by Aditya 2026-11-22 after he "
             "chose 'All vertices stable' as the criterion (MATH §9.6). Rank 2 candidates are 0.1 degrees apart; rank 3 candidates "
             "are random rotations (seed 0) plus the unconstrained smallest. Held-out gains don't change, because the model doesn't. "
             "With the rule, two of the four vertices sit about one spread from the data, where without it one sat further out "
             "(1.60, 1.23). Tests: tests/test_stable_hull.py.")


# ----------------------------------------------------------------------------------------------- results

def weights_slide(d):
    s = d.slide("Rank 3: hold and move each get a vertex")
    native(s, "hull_weights_r3", y=1.3, bottom=6.5)
    caption(d, s, "Evoked left in: vertex 3 has 0.92 at hold 1/4; vertex 4 has 0.72 to 0.74 in move 2/4 to 4/4. "
                  "Bands: jackknife SE over 5 training splits.")  # src: figure script printout, cache/13_run.log
    notes(s, "Evoked left in: vertices 1 and 2 share plan 1/4 (0.17, 0.34) and late post (0.32, 0.40 at post 3/4). Evoked removed: "
             "vertex 3 hold (0.97), vertex 4 move (0.78 to 0.82), vertex 1 plan 1/4 (0.41) and post, vertex 2 plan 2/4 (0.34). "
             "The vertex numbers are arbitrary, and the two variants' vertices are separate fits, so vertex 1 on the left needn't match vertex 1 "
             "on the right. Source: cache/13_vertices_H_2021-11-12.pkl via scripts/make_deck_figures_hull.py.")


def maps_slide(d):
    s = d.slide("The four vertices: edges and spectra")
    native(s, "hull_maps_r3", y=1.25, bottom=6.5, panel_flags={f"Vertex {v}, ρ = {r}": SMALL_NODES for v, r in
                                                                  ((1, "0.969"), (2, "0.997"), (3, "0.969"), (4, "0.965"))})
    edge_key(s)
    caption(d, s, "Evoked left in. Vertex 2, the least stable, adds a 10 Hz peak of +20.8 dB over GDAR. "
                  "It sits 1.14 spreads from the nearest bin, a state the data don't visit.")  # src: figure script printout, cache/13_run.log
    notes(s, "Edge maps: each edge's weight summed over the 16 lags. Spectra: each vertex's AR spectrum frozen, relative to GDAR's (MATH §11.2). "
             "Nearest bins: vertex 1 post 3/4 (1.31 spreads away), vertex 2 post 3/4 (1.14), vertex 3 hold 1/4 (0.15), vertex 4 move 3/4 (0.49). "
             "Vertices far from the data are extrapolations of the fitted family: read their maps as directions in which the system "
             "moves across the trial, not as states it reaches. Vertex 4 (move) is lowest from 10 to 20 Hz, consistent with the beta drop in move "
             "on the 5 November spectrum slide.")


def days_slide(d, c13, c14):
    s = d.slide("Other days: a stable simplex on 14 of 16 fits")
    rows = [["Session", "Evoked", "Rank 2: largest ρ", "Rank 3: largest ρ"]]
    runs = [("11-12 no stim", c13, None)] + [(k, None, c14[k]) for k in ("08-23 no stim", "11-11 no light", "11-11 light")]
    n_fits = n_stable = 0
    for name, a, b in runs:
        for v, lab in (("felix", "left in"), ("induced", "removed")):
            full = a[v]["full"] if a is not None else b[v]["full"]
            cells = []
            for r in (2, 3):
                p = full[r]["poly"]
                n_fits += 1
                n_stable += bool(p["stable"])
                top = f"{max(p['rho']):.3f}"
                if p["stable"] and top == "1.000":     # 0.99987 on 11-11 no light, removed, rank 2
                    top = f"{max(p['rho']):.5f}"
                cells.append(top + ("" if p["stable"] else "  none stable"))
            rows.append([name if v == "felix" else "", lab] + cells)
    assert (n_fits, n_stable) == (16, 14), (n_fits, n_stable)   # one more run would need a new title
    d.table(s, rows, 1.35, [3.2, 2.0, 3.0, 3.0], row_h=0.44, size=17)
    d.text(s, ["Where no rotation is stable, the least unstable is kept and flagged: 1.025 at rank 2, 1.028 at rank 3",
               "Those trials carry the light, and the model has no input for it. That may be why, but it isn't tested"],
           1.35 + 0.44 * 9 + 0.25, 1.2, size=19)
    notes(s, "Sessions of monkey H. Script 14 fits each session on its own; 11-11 is split into lit and unlit trials. Largest rho over the "
             "vertices of the kept simplex; 'none stable' means no candidate rotation had every vertex below 1. On 11-11 lit with evoked left in, "
             "stable rotations exist but only far down the size order (candidates 377 and 579), so the simplex there is larger (h 0.686 vs "
             "0.658 at rank 2, 0.652 vs 0.513 at rank 3). On 11-11 unlit, removed, rank 2 the kept vertex is below 1 but 1.000 to 3 decimals. "
             "Source: cache/14_days.pkl, cache/14_run.log.")


def single_slides(d):
    s = d.slide("Single trials: read the weights off a short window of data")
    y = add_latex(s, r"\hat\lambda(t)=\arg\min_{\lambda\geq 0,\ \sum_v\lambda_v=1}\ \sum_{s\in\mathrm{window}(t)}\Bigl\|\,x(s+1)-\sum_v\lambda_v\,\Theta_v\,z(s)\,\Bigr\|_{\hat\Sigma^{-1}}^{2}",
                  LEFT + 0.1, 1.3, size=22)
    native(s, "single_trial_r3", y=y + 0.2, bottom=6.5)
    caption(d, s, "No go or touch times are used. Right: the move vertex rises at the same time after the target for early, "
                  "middle and late go (mean go 302, 401, 517 ms, dashed).")   # src: figure script printout
    notes(s, "Held-out trials (5 folds), rank 3 fold fit and its four vertices fixed. Window: 20 samples (60 ms) centred on t for this slide; "
             "the prediction test on the next slide uses only the samples before t. Example: trial 247, go 399 ms, touch 823 ms, the median go "
             "time. Go-time thirds: 107, 106, 107 trials, mean go 302, 401, 517 ms. MATH §10, approved 2026-11-05. Evoked left in.")

    s = d.slide("Weights from 60 ms of data predict better than the clock")
    native(s, "single_trial_gain", x=LEFT, y=1.3, w=6.2, bottom=6.5)
    d.text(s, ["Clock: bins from the trial's own go and touch times. Data: weights from the previous 30, 60 or 120 ms only",
               "Go time can't be read off the weights: r = −0.02 to +0.05",       # src: RESULTS.md, rotation-rule section (cache/18_run.log)
               "Touch partly can: r = +0.35, +0.40, +0.44 for 30, 60, 120 ms",   # src: same
               "Caveat: weights that adapt within a trial may gain from any slow change, not only task states. A control isn't run"],
           1.4, 5.0, x=7.0, w=13.333 - 7.0 - 0.4, size=19)
    notes(s, "Gain over GDAR, nats/sample, all rows scored with GDAR's Sigma on the same 167,932 samples, fold SE: evoked left in clock +0.1618 "
             "(0.0077), 30 ms +0.1669 (0.0080), 60 ms +0.1939 (0.0079), 120 ms +0.2013 (0.0078); removed +0.2150 (0.0100), +0.2212 (0.0074), "
             "+0.2409 (0.0070), +0.2413 (0.0072). Timing, evoked left in: go p 0.17 to 0.62, touch p 0.0001. With the stability rule the "
             "window gains are a little lower than without it (60 ms, left in: +0.2039 before). Source: cache/18_run.log.")


# ----------------------------------------------------------------------------------------------- close

def open_slide(d):
    s = d.slide("Open questions")
    d.text(s, ["11-11 lit trials, evoked removed: no stable rotation. Keep them flagged, or add the light as an input B u(t)?",
               "Comparing days needs a 'rest' vertex. Script 14's rule for it (the non-move vertex with most hold weight) isn't approved",
               "Fold to full-data vertex match fell to r = 0.70 in one fold: the rule picks different rotations in different folds",
               "Single-trial gain: task state or slow drift? Needs a control, for example one past-window gain on GDAR's prediction",
               "One monkey so far"], 1.45, 4.8, size=22)
    notes(s, "Fold match, evoked left in: 1.00, 1.00, 1.00, 0.70, 0.97; removed 1.00, 0.94, 1.00, 0.92, 0.94 (cache/18_run.log). "
             "The control for the single-trial gain needs a method approval before it is run. MATH §14 lists the open method questions.")


def next_slide(d):
    s = d.slide("Next: monkey L, and three more H sessions")
    d.table(s, [["Session", "Channels", "Trials", "Lit", "Median move (ms)"],
                ["L 2022-08-10", "21", "841", "440", "106"],
                ["H 2021-11-16", "15", "663", "337", "266"],
                ["H 2021-08-18", "22", "211", "0", "338"],
                ["H 2021-07-28", "32", "90", "0", "588"],
                ["H 2021-11-12, for reference", "23", "321", "0", "409"]],
            1.35, [4.2, 1.6, 1.4, 1.2, 2.4], row_h=0.44, size=17)   # src: segment files, read 2026-11-22
    d.text(s, ["L's other sessions have 3 to 7 channels, too few for a graph. 08-10 is the one L session to fit",
               "L moves fast, so its move bins hold few samples. Check the bin counts before trusting the fits",
               "The scripts hard-code H's file and its bad channel B14, and need a lit-trial filter",
               "Ask Noah whether the electrode map holds for L's new array"],
           1.35 + 0.44 * 6 + 0.3, 2.2, size=19)
    notes(s, "Only two monkeys are in the dataset: H (2021) and L (2022). Every session is already cut into segments on this PC "
             "(yazdanlab/iosreach/cache/segs). Channel counts are before dropping bad channels (H drops B14, so 11-12 is fitted on 22). "
             "L's channel set changed with a new array from 2022-08-08; whether the electrode map file covers it is a question for Noah. "
             "H 2021-08-04 (25 channels) has only 49 trials. The run plan is in docs/PLAN_OTHER_MONKEYS.md.")


def build():
    c13, c14 = load()
    d = Deck()
    title_slide(d, "TP-GDAR on IOS Reach: four stable states", "Monkey H. What changed since 5 November, and the plan for monkey L. 23 November 2026")
    notes(d.p.slides[0], "")
    summary_slide(d)
    model_slide(d)
    bins_slide(d)
    notes(d.p.slides[len(d.p.slides) - 1], "Reused from the 5 November deck.")
    joint_slide(d)
    why_slide(d)
    hull_slide(d)
    rotation_slide(d)
    rule_slide(d, c13)
    weights_slide(d)
    maps_slide(d)
    days_slide(d, c13, c14)
    single_slides(d)
    open_slide(d)
    next_slide(d)
    for slide in d.p.slides:        # connectors inherit a soft shadow from the default theme, which blurs thin lines
        for ref in slide._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef"):
            ref.set("idx", "0")
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
