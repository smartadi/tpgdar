r"""Simple deck: the IOS Reach task, its spectrum by phase, the TP paper's pipeline, and the TP-GDAR math.

Run:  .venv\Scripts\python scripts/make_ios_tp_deck.py
Writes TP-GDAR_IOS_Reach.pptx in the repo.  The copy to edit online lives in OneDrive\Talks; this script never touches it,
so a rebuild can't overwrite edits made there.

Text, equations and figures are ordinary text and shapes (scripts/deck_native.py), which PowerPoint for the web can edit.
Plotted numbers aren't linked to the shapes.

Figure data: scripts/make_deck_figure_spectra.py, make_deck_figure_ios.py, make_deck_figures_real.py.
Numbers: cache/spectra_run.log (spectra), RESULTS.md, L-2026-11-04-l (HOSVD vs joint).
Paper: docs/TP_model_transformation_as_a_way_to_LMI-based_controller_design.pdf (Baranyi 2004), Method 3 and 4.
Felix's GDAR: docs/2024.02.26.582177v1.full.pdf (Schwock et al. 2024).
Dataset: Griggs, Stanis et al., Nat Commun 2026, 17:3128 (s41467-026-69448-3), and Stanis et al., EMBC 2025 (10.1109/EMBC58623.2025.11254470).
"""

import os
import sys

from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_native import add_equation  # noqa: E402
from make_mpc_tp_deck import arrow, edge_key, figure, native  # noqa: E402
from make_tp_math_deck import LEFT, MUTED, ROOT, TINT, WIDTH, Deck, title_slide  # noqa: E402

OUT = os.path.join(ROOT, "TP-GDAR_IOS_Reach.pptx")
LABEL_W = 2.6
SMALL_NODES = {"node_radius": 0.07}


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def steps(d, s, rows, y=1.5, size=24, gap=0.9):
    """Rows of a grey label and an equation.  A row whose second item has no $ is plain text."""
    for label, body in rows:
        d.text(s, label, y + 0.08, 0.5, x=LEFT, w=LABEL_W, size=18, bullet=False, color=MUTED)
        if body.startswith("$"):
            add_equation(s, body, LEFT + LABEL_W, y, w=WIDTH - LABEL_W, size=size)
        else:
            d.text(s, body, y + 0.05, 0.5, x=LEFT + LABEL_W, w=WIDTH - LABEL_W, size=20, bullet=False)
        y += gap
    return y


def flow(d, s, boxes, y, x=LEFT + 0.1, w=2.05, gap=0.37, h=0.95, size=18, shaded=None):
    """A row of boxes joined by arrows."""
    for i, text in enumerate(boxes):
        left = x + i * (w + gap)
        d.label(s, text, left, y, w, h, fill=TINT if shaded is None or i in shaded else None, size=size)
        if i:
            arrow(s, left - gap + 0.04, y + h / 2, left - 0.04, y + h / 2)


# ----------------------------------------------------------------------------------------------- the task

def setup_slide(d):
    s = d.slide("The setup: record and shine light through one window")
    x0, w = LEFT + 1.0, 4.6
    d.label(s, "LED array: 4 × 4, red light", x0, 1.6, w, 0.85, fill=TINT, size=18)
    d.label(s, "Air gap and cover glass", x0, 2.45, w, 0.55, size=16)
    d.label(s, "See-through ECoG: 32 electrodes", x0, 3.0, w, 0.85, fill=TINT, size=18)
    d.label(s, "Cortex: left PPC, Jaws in part of it", x0, 3.85, w, 1.2, size=18)
    arrow(s, x0 - 0.3, 1.7, x0 - 0.3, 4.95)
    d.text(s, "light", 3.05, 0.4, x=LEFT - 0.05, w=0.9, size=16, bullet=False, color=MUTED)
    arrow(s, x0 + w, 3.42, x0 + w + 0.75, 3.42)
    d.text(s, "LFP, 1 kHz", 2.95, 0.4, x=x0 + w - 0.02, w=1.3, size=15, bullet=False, color=MUTED)
    d.text(s, "Not to scale", 5.1, 0.4, x=x0, w=w, size=14, bullet=False, color=MUTED)
    x = 7.55
    d.text(s, ["25 mm window over left posterior parietal cortex (PPC), stable for years",
               "Jaws, an inhibitory opsin, covers 92 mm² in H and 76 mm² in L",
               "Light: 900 ms from target on, on a random half of stim-day trials",
               "The window stays clear for 10 to 14 weeks, then tissue is removed"], 1.5, 4.0, x=x, w=13.333 - x - 0.3, size=20)
    d.text(s, "Griggs, Stanis et al., Nat Commun 2026. Stanis et al., EMBC 2025.", 6.5, 0.4, size=16, bullet=False, color=MUTED)
    notes(s, "Monkeys H and L: adult male rhesus macaques. Areas 7a and 7b near the intraparietal sulcus (IPS). Vector AAV8-hSyn-Jaws-GFP by convection-enhanced delivery, up to 50 uL per site. "
             "Red LEDs 634 nm, cortical temperature change under 1 C. Expression faded after about 2 years, and reinjection restored the response. "
             "The two papers give different numbers for light intensity (about 12 mW/mm2 vs 1.2 mW/cm2) and spacing (EMBC: 800 um pitch, while 32 electrodes over 314 mm2 would average about 3 mm), so they're left off the slide. Worth asking Noah.")


def findings_slide(d):
    s = d.slide("What the papers found on this task")
    steps(d, s, [("Behaviour", "Light slows reaches by tens of ms, most clearly downward. Paths look similar"),
                 ("Planning", "Theta power rises near the IPS and in areas 7a and 7b"),
                 ("Light", "ECoG power goes up, not down, and more over Jaws cortex"),
                 ("Controls", "No such effect in an untreated monkey, with blue light, or once expression fades"),
                 ("Likely why", "Upper layers are inhibited, which releases layer 5. A simulation agrees"),
                 ("Network", "Over Jaws cortex, gamma coherence and theta-gamma coupling fall")], y=1.5, gap=0.8)
    d.text(s, "Griggs, Stanis et al., Nat Commun 2026. Stanis et al., EMBC 2025.", 6.5, 0.4, size=16, bullet=False, color=MUTED)
    notes(s, "Reach time, stim vs no stim: H p = 0.04, L p = 0.009. Downward: H p = 0.03, L p = 0.0001. Path length is longer overall in L (p = 4.2e-6). Controls: untreated Monkey C p = 0.53; blue light and tissue growth show no Jaws vs no-Jaws difference "
             "(p = 0.90, 0.47); H after about 2 years p = 0.76. Theta here is 4-10 Hz, planning vs the 500 ms before the target. Coherence and coupling come from one electrode per region (EMBC). "
             "Check on our no-stim days, 4-10 Hz, plan vs hold: the all-channel mean is flat (+0.11 dB on 11-12, -0.25 on 08-23), but channels split. On 11-12, 8 rise (A6 +2.1, A5 +1.5 dB) and 7 fall; on 08-23, 5 rise (B2 +1.9) and 8 fall. "
             "A local rise fits the papers, but we can't place these channels without the channel map.")


def task_slides(d):
    s = d.slide("The task: a 4-target center-out reach")
    native(s, "ios_reach", y=1.25, bottom=5.5,
           panel_flags={"Trial timeline": dict(no_xaxis=True, no_yaxis=True),
                        "Single trial": dict(no_yaxis=True), "Trial average": dict(no_yaxis=True)})
    d.text(s, ["Monkeys H and L reach with the right hand to 1 of 4 targets, 80 mm from center",
               "Noah's curated set: H 10 days, 3,865 trials. L 6 days, 5,116 trials",
               "This deck: H 2021-11-12, no stim, 320 trials, 22 channels, LFP at 333 Hz"], 5.5, 1.6, size=20)
    notes(s, "Head-fixed, in a dark rig. The middle finger is tracked by motion capture; targets are 25 mm across on a monitor within reach, and a tone is the go cue. Juice reward. "
             "The papers analyse more trials than the curated set: Nat Commun H 4,522 over 8 sessions, L 11,101 over 13; EMBC H 3,904, L 8,588.")

    setup_slide(d)
    findings_slide(d)

    s = d.slide("A trial has four phases")
    x0, w, top = LEFT + 0.1, (WIDTH - 0.2) / 4, 2.25
    rows = [("hold", "400 ms", "end of a 1 s hold at center"),
            ("plan", "median 400 ms", "target shown, waiting for go"),
            ("move", "median 408 ms", "go cue to touch, reaction included"),
            ("post", "500 ms", "after the touch")]
    for i, (name, length, what) in enumerate(rows):
        d.label(s, name, x0 + i * w, top, w, 0.8, fill=TINT, size=26, bold=True)
        d.label(s, length, x0 + i * w, top + 0.8, w, 0.6, size=18)
        d.label(s, what, x0 + i * w, top + 1.4, w, 0.9, size=18)
    for i, event in enumerate(("target on", "go cue", "touch")):
        box = d.text(s, event, top - 0.5, 0.4, x=x0 + (i + 1) * w - 1.0, w=2.0, size=18, bullet=False, color=MUTED)
        box.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    d.text(s, ["The delay is random: 50 to 700 ms in the task, 245 to 610 ms on this day",
               "On stim days the light runs 900 ms from target on, through plan and most reaches"], 4.85, 1.4, size=22)
    notes(s, "Medians over 320 trials on 2021-11-12; move runs 95 to 1768 ms. Timing from Nat Commun 2026, Fig. 7. The segment file has no separate reaction time, and the papers' reach time is the same go-to-touch interval. "
             "Each segment starts 500 ms before the target, so hold here means the last 400 ms of the 1 s hold.")

    figure(d, "Spectrum by phase: low frequencies fall in movement", "spectra_phase",
           "Move vs hold: alpha −7.9 dB, beta −5.5 dB, high gamma +1.2 dB. 200 ms Hann windows, evoked response left in.")
    notes(d.p.slides[len(d.p.slides) - 1],
          "Bands: theta 4-8, alpha 8-13, beta 13-30, gamma 30-60, high gamma 65-115 Hz. Evoked response left in. 200 ms Hann windows every "
          "50 ms inside each phase, demeaned; per channel the trial-mean power, then dB, then the mean over channels. Resolution about 10 Hz. "
          "23 trials have a move shorter than 200 ms and add nothing to move; 18 add nothing to post. Bootstrap 95% over trials, move vs hold: "
          "alpha -8.37 to -7.55, beta -5.84 to -5.23, high gamma +1.00 to +1.46. Plan vs hold: beta -1.04. Raw lfp gives the same picture: "
          "change from hold differs by under 0.7 dB everywhere from 3 to 100 Hz. The high gamma band runs to 115 Hz, past the plot. The dip near 60 Hz is the notch filter: EMBC 2025 lists notches at 60, 120 and 180 Hz, "
          "and both lfp and noart dip about 9 dB at 60 Hz. Log: cache/spectra_run.log.")

    figure(d, "Another no-stim day shows the same pattern", "spectra_phase_0823",
           "H 2021-08-23, 404 trials, 16 channels. Move vs hold: alpha −8.2 dB, beta −7.1 dB, high gamma +1.8 dB.")


# ----------------------------------------------------------------------------------------------- the paper

def paper_slides(d):
    s = d.slide("The TP paper: from a known model to a polytope")
    add_equation(s, r"$S(p)=\sum_{r}w_r(p)\,S_r,\qquad w_r\geq 0,\qquad\sum_{r}w_r=1$", LEFT + 0.1, 1.45, size=28)
    flow(d, s, ["Known model S(p)", "Sample on a grid", "HOSVD", "Drop small singular values", "Make weights convex"], 2.6)
    d.text(s, ["Baranyi 2004, Method 4. It rewrites a model you already have, so LMI design can use it",
               "No data, noise or likelihood: its examples start from known equations",
               "So the paper has no system identification step. With data, we have to add one"], 4.1, 2.4, size=24)

    s = d.slide("The paper, steps 1 and 2: sample, then HOSVD")
    y = steps(d, s, [("1  Sample", r"$S_g=S(p_g),\quad g=1,\ …,\ P$"),
                     ("2  HOSVD", r"$[\,\mathrm{vec}\,S_1;\ \cdots;\ \mathrm{vec}\,S_P\,]=U\,D\,V^{\top}$"),
                     ("   Keep r", r"$S(p_g)\ ≈\ \sum_{j=1}^{r}U_{gj}\,S_j$"),
                     ("   Error", r"$\mathrm{error}\ ≤\ \sum_{j>r}σ_j^{2}$")], y=1.5)
    d.text(s, ["Each column of U is one weight function on the grid, joined by straight lines",
               "Rank is read from singular values that are exactly zero"], y + 0.1, 1.3, size=22)
    notes(s, "With several parameters, HOSVD runs on each parameter dimension and the vertex counts multiply; the system matrix dimensions "
             "aren't compressed. With one parameter it is this one SVD. Error bound: the paper's eq. 37.")

    s = d.slide("The paper, step 3: make the weights convex")
    y = steps(d, s, [("Free choice", r"$U'=U\,T,\qquad S'=T^{-1}S\qquad\mathrm{same\ }S(p)\mathrm{\ for\ any\ invertible\ }T$"),
                     ("Sum to one", "Transformation 1. It may add one weight column"),
                     ("Nonnegative", "Transformation 2. Shrink all weights toward uniform")], y=1.4, gap=0.72, size=22)
    d.table(s, [["S(p) = p S₀", "SVD weight", "After 1", "After 2", "Tightest"],
                ["p = 1", "0.371", "0.621, 0.379", "0.540, 0.460", "1, 0"],
                ["p = 1.5", "0.557", "0.931, 0.069", "0.644, 0.356", "0.5, 0.5"],
                ["p = 2", "0.743", "1.241, −0.241", "0.747, 0.253", "0, 1"]], y + 0.05, [2.2, 1.8, 2.3, 2.3, 1.8], row_h=0.42, size=18)
    d.text(s, "Valid but loose: vertices 3.22 S₀ and −1.61 S₀ for gains that run from 1 to 2.", y + 1.85, 0.5, size=20, bullet=False)
    notes(s, "The worked example is mine, not the paper's. As printed, the paper's Z in Transformation 2 doesn't normalise (row sums about 0.44 here). "
             "I used Z = (bI + 11')/(n + b), with b = 1 unless the most negative entry is below -1, and checked it: exact reconstruction, weights in [0, 1]. "
             "Remark 3 says discarding only zero singular values never adds a basis function, but this example does add one. "
             "A loose polytope makes the LMI harder to solve, because it must hold at every vertex.")


# ----------------------------------------------------------------------------------------------- from data

def bridge_slide(d):
    s = d.slide("From data to vertices: what we add to the paper")
    rows = [("TP paper", ["Known model S(p)", "Sample on a grid", "HOSVD", "Convex weights"], (0, 1, 2, 3)),
            ("Data, HOSVD", ["Recordings", "GDAR fit in each bin", "HOSVD", "Convex weights"], (2, 3)),
            ("Data, joint fit", ["Recordings", "Statistics in each bin", "Joint fit", "Convex weights"], (3,))]
    y = 1.6
    for label, boxes, shaded in rows:
        d.text(s, label, y + 0.28, 0.5, x=LEFT, w=2.1, size=20, bullet=False)
        flow(d, s, boxes, y, x=LEFT + 2.1, w=2.2, gap=0.33, shaded=shaded)
        y += 1.3
    d.text(s, ["Shaded: taken from the paper. White: what data needs",
               "The fit in each bin plays the part of sampling S(p), but with noise"], y + 0.05, 1.2, size=22)
    notes(s, "The data step is ours, using Felix's GDAR model and estimator in each bin. The joint fit keeps the paper's model form and its idea of convex "
             "weights, but it replaces both the per-bin fits and HOSVD.")


def model_slides(d):
    s = d.slide("Felix's GDAR: one operator from data")
    y = add_equation(s, r"$x(t+1)=\sum_{k=1}^{K}A_k\,x(t-k+1)+e(t),\qquad A_k=A_k^{\top},\quad A_k\ \mathrm{zero\ off\ the\ graph}$", LEFT + 0.1, 1.25, size=24)
    native(s, "gdar_fit", y=y + 0.05, bottom=6.6, panel_flags={"Edges, all lags": SMALL_NODES})
    edge_key(s)
    d.text(s, "Schwock et al. 2024. Here (22 nodes + 50 edges) × 16 lags = 1,152 numbers, one operator for the whole day.", 6.65, 0.5, size=18, bullet=False)

    s = d.slide("TP-GDAR: a few GDARs, mixed across the trial")
    y = add_equation(s, r"$\Theta(p)=\sum_{j=1}^{r}w_j(p)\,\Theta_j,\qquad \Theta=[\,A_1\ \cdots\ A_K\,]$", LEFT + 0.1, 1.25, size=24)
    native(s, "tp_mix", y=y + 0.05, bottom=6.6, panel_flags={"Vertex 1": SMALL_NODES, "Vertex 2": SMALL_NODES})
    edge_key(s)
    d.text(s, "Rank 2: 2,332 numbers. 16 separate fits: 18,432.", 6.65, 0.5, size=18, bullet=False)

    s = d.slide("p is trial phase, warped per trial")
    x0, wbin, top = LEFT + 0.1, (WIDTH - 0.2) / 16, 2.2
    for ph, name in enumerate(["hold", "plan", "move", "post"]):
        d.label(s, name, x0 + ph * 4 * wbin, top, 4 * wbin, 0.7, fill=TINT, size=24, bold=True)
        for b in range(4):
            d.label(s, f"{b + 1}/4", x0 + (ph * 4 + b) * wbin, top + 0.7, wbin, 0.55, size=16)
    for i, event in enumerate(["-400 ms", "target on", "go cue", "touch", "touch + 500 ms"]):
        cx = x0 + i * 4 * wbin
        tx = min(max(cx - 1.2, x0 - 0.1), x0 + 16 * wbin + 0.1 - 2.4)
        box = d.text(s, event, top + 1.35, 0.4, x=tx, w=2.4, size=18, bullet=False, color=MUTED)
        box.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT if i == 4 else PP_ALIGN.CENTER
    d.text(s, ["16 bins: each phase is cut into 4 equal parts of its own length on that trial",
               "Go and touch are known only afterwards, so this p can't be used online"], 4.7, 1.2, size=22)


def math_slides(d):
    s = d.slide("Data step: a GDAR in each bin")
    steps(d, s, [("Model in bin g", r"$x(t+1)=\Theta_g\,z(t)+e(t),\qquad z(t)=[\,x(t);\ \cdots;\ x(t-K+1)\,]$"),
                 ("Statistics", r"$G_g=\sum_{t\in g}z(t)\,z(t)^{\top},\qquad C_g=\sum_{t\in g}z(t)\,x(t+1)^{\top}$"),
                 ("Free numbers", r"$\mathrm{vec}\,\Theta=R\,\phi$"),
                 ("Noise", "Σ from one pooled fit, then fixed"),
                 ("In the metric", r"$H_g=R^{\top}(G_g\otimes Σ^{-1})R,\qquad h_g=R^{\top}\mathrm{vec}(Σ^{-1}C_g^{\top})$"),
                 ("Fit in bin g", r"$\phi_g^{\mathrm{fit}}=H_g^{-1}h_g$")], y=1.45, gap=0.87)
    notes(s, "R maps the free numbers to the full coefficient matrix: symmetric, zero off the graph. The pooled fit with G and C summed over bins is Felix's estimator; "
             "it matches CVARModel.fit_gdar to 1e-11. About 10,800 samples per bin on this day.")

    s = d.slide("HOSVD on the fits: every coefficient counts the same")
    y = steps(d, s, [("Stack the fits", r"$[\,\mathrm{vec}\,\Theta_1^{\mathrm{fit}};\ \cdots;\ \mathrm{vec}\,\Theta_{16}^{\mathrm{fit}}\,]=U\,D\,V^{\top}$"),
                     ("Keep r", r"$\Theta(p_g)\ ≈\ \sum_{j=1}^{r}U_{gj}\,\Theta_j$"),
                     ("It minimises", r"$\sum_g\|\,\Theta(p_g)-\Theta_g^{\mathrm{fit}}\,\|_F^{2}$")], y=1.5)
    d.text(s, ["Exact samples make this the right loss. With noisy fits it treats a badly estimated coefficient like a good one",
               "No exact zeros to read a rank from: singular values 1, 0.018, 0.011, 0.010, ..."], y + 0.15, 1.7, size=22)

    s = d.slide("Joint fit: each bin scored on its own data")
    y = steps(d, s, [("Loss in bin g", r"$L_g(\phi)=\sum_{t\in g}\|\,x(t+1)-\Theta(\phi)\,z(t)\,\|_{Σ⁻¹}^{2}$"),
                     ("Same as", r"$L_g(\phi)=\|\,\phi-\phi_g^{\mathrm{fit}}\,\|^{2}\ \mathrm{in\ the\ metric\ }H_g\ +\ \mathrm{const}$"),
                     ("HOSVD", r"$\mathrm{min}\ \sum_g\|\,\Theta(p_g)-\Theta_g^{\mathrm{fit}}\,\|_F^{2}$"),
                     ("Joint", r"$\mathrm{min}\ \sum_g\|\,\phi(p_g)-\phi_g^{\mathrm{fit}}\,\|^{2}\ \mathrm{in\ the\ metric\ }H_g$")], y=1.5)
    d.text(s, "Directions that a bin's data pin down count more: the fit in bin g has covariance H⁻¹ in that bin.", y + 0.15, 0.6, size=22, bullet=False)
    notes(s, "Cov of phi-hat is the inverse of H_g given the regressors, with Gaussian noise and known Sigma. So the joint fit is the maximum-likelihood low-rank fit for "
             "that fixed Sigma. HOSVD would be the same only if every H_g were proportional to the identity. This is my derivation, not the paper's.")

    s = d.slide("Joint fit: alternate two exact solves")
    y = steps(d, s, [("Objective", r"$J(W,\Phi)=\sum_g L_g(\sum_{j}W_{gj}\,\phi_j)$"),
                     ("Start", r"$\phi_1=\mathrm{pooled\ fit},\qquad W_{g1}=1$"),
                     ("W step", r"$(\Phi\,H_g\,\Phi^{\top})\,w_g=\Phi\,h_g$"),
                     ("Φ step", r"$(\sum_g w_g w_g^{\top}\otimes H_g)\,\mathrm{vec}\,\Phi=\sum_g w_g\otimes h_g$"),
                     ("Stop", r"$|\,\Delta J\,|\ ≤\ 10⁻⁷\ |\,J\,|$")], y=1.45, gap=0.85)
    d.text(s, "J never rises, and real fits took 5 to 7 sweeps. No closed form: it finds a local optimum.", y + 0.1, 0.6, size=22, bullet=False)
    notes(s, "Start: the other operators come from an SVD of the per-bin fits around the pooled fit, in the pooled data metric. After each sweep the weights are rescaled so each "
             "column has norm sqrt(P), which changes nothing in the model. Ranks aren't nested: the rank 3 fit needn't contain the rank 2 one. Limits: 25 sweeps in folds, 40 on the full day. Jitter 1e-10.")


def gain_table_slide(d):
    s = d.slide("On real data, HOSVD loses most of the gain at ranks 1 and 2")
    rows = [["Rank", "Evoked left in", "", "", "Evoked removed", "", ""],
            ["", "HOSVD", "Metric SVD", "Joint", "HOSVD", "Metric SVD", "Joint"],
            ["1", "−0.094", "+0.084", "+0.107", "−3.234", "+0.081", "+0.133"],
            ["2", "+0.016", "+0.180", "+0.190", "+0.003", "+0.211", "+0.248"],
            ["3", "+0.123", "+0.212", "+0.222", "+0.239", "+0.304", "+0.323"],
            ["16 separate fits", "+0.211", "", "", "+0.392", "", ""]]
    d.table(s, rows, 1.5, [2.5, 1.55, 1.75, 1.45, 1.55, 1.75, 1.45], row_h=0.52, size=20)
    table = s.shapes[-1].table
    table.cell(0, 1).merge(table.cell(0, 3))
    table.cell(0, 4).merge(table.cell(0, 6))
    for j in (1, 4):
        table.cell(0, j).text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    d.text(s, ["Held-out gain over GDAR, nats per sample, 5-fold over trials, H 2021-11-12",
               "Metric SVD: the same SVD in the pooled data metric. It gets most of the gain, so the metric matters more than the iterations",
               "Separate fits match or beat rank 3 here. Low rank buys fewer numbers and unseen task times"], 4.8, 2.2, size=20)
    notes(s, "From scripts/12d_hosvd_vs_joint.py, L-2026-11-04-l. One day only, with the target-onset step unmasked. HOSVD also gives a different time course: at rank 2, the cosine of the largest "
             "principal angle between HOSVD and joint weight spans is 0.50 with evoked left in, 0.89 removed.")


def vertex_slides(d):
    s = d.slide("From the joint fit to convex weights")
    y = steps(d, s, [("Sum to one", r"$W_a=W+(1-W\,a)\,a^{\top}/(a^{\top}a),\qquad W\,a\ ≈\ 1$"),
                     ("Data metric", r"$Y=W_a\,L_r,\qquad L_r\,L_r^{\top}=\Phi\,H_{\mathrm{pool}}\,\Phi^{\top}$"),
                     ("Vertices", "Rank 2: the two extreme bins. Rank 3: the smallest enclosing triangle"),
                     ("Result", r"$\Theta(p)=\sum_{v}\lambda_v(p)\,\Theta_v,\qquad\lambda_v\geq 0,\qquad\sum_v\lambda_v=1$")], y=1.5)
    d.text(s, ["Same freedom as the paper, a different T. No added vertex, and rank-2 weights span the full 0 to 1",
               "Cost: sum-to-one misses by 0.017 to 0.037 in the weights. Its likelihood cost isn't measured",
               "Only Θ(p) is unique. The vertices are one choice of coordinates"], y + 0.1, 2.0, size=20)
    notes(s, "This step refits nothing. Held-out gains are measured before it, so they're unaffected, but the vertex maps and weight plots aren't exact. Rank 3 corners may never be visited by the data.")

    s = figure(d, "TP beats one GDAR on held-out trials and unseen task times", "tp_vs_gdar",
               "Gain = held-out one-step likelihood over GDAR, 5-fold. Left: p = 4 phases. Middle, right: 12 time bins. So numbers differ from the 16-bin table.")
    notes(s, "Each model gets its own noise covariance when scored, Sigma = 0.99 Sigma-hat + 0.01 diag(Sigma-hat), so the fit doesn't quite optimise the score. Script 12.")

    s = figure(d, "Rank 2, evoked removed: a switch across the trial", "vertices_r2_induced",
               panel_flags={"Vertex 1": SMALL_NODES, "Vertex 2": SMALL_NODES})
    edge_key(s)
    notes(s, "By the end of post the weights return only part way toward vertex 1. Weights: jackknife band over the 5 training splits.")

    s = figure(d, "Rank-2 fit vs data: similar shape, weaker in plan and move", "spectra_model",
               "Correlation with the data, 3 to 100 Hz: r = 0.81 to 0.97. Beta (13-30 Hz) in move: data −3.6 dB, model −1.8 dB. In post the sign differs: +0.7 vs −0.3 dB.")
    notes(s, "Model: the rank-2 fit with evoked left in, its AR spectrum frozen at each of the 16 bins, averaged in dB within a phase, with the same 200 ms window applied in frequency. "
             "Data and model are each shown relative to the mean of their four phases. A frozen spectrum isn't the true spectrum of a changing system, and the model doesn't include the evoked response as an input. "
             "No model was fitted for this slide. Plan: data +0.9, model +0.5 dB; hold: +1.9, +1.6 dB; High gamma (65-115 Hz) agrees within 0.25 dB in every phase.")


def limits_slide(d):
    s = d.slide("Limits")
    d.text(s, ["Trial phase uses go and touch, so it isn't known online",
               "Weights describe the average trial, not each trial",
               "One monkey. The HOSVD table and the data vs model spectrum are from one day",
               "The fits shown have no light, so they say nothing about the Jaws effects",
               "Σ is fixed during the fit, and the fit is a local optimum",
               "The cleaned LFP has steps at 0 and 900 ms, not masked here. Masking moved the gains little",
               "Which channels sit over 7a, 7b and Jaws cortex still needs Noah"], 1.6, 5.3, size=24)
    notes(s, "The steps are likely from the artefact cleaning. The Nat Commun methods split each trial at light on and off, zero each epoch at the split, and rejoin them by matching 50 ms means, and no-stim trials go through the same processing. "
             "On 11-12 the median jump is 23.5 uV at 0 ms and 13.9 at 900 ms in noart, against 2.5 typical, and raw lfp has none. Smaller steps at 10 and 910 ms aren't explained by the paper. The paper itself drops the 50 ms after light on and off. To confirm with Noah.")


def build():
    d = Deck()
    title_slide(d, "TP-GDAR on IOS Reach", "The task, its spectrum by phase, and the math")
    task_slides(d)
    paper_slides(d)
    bridge_slide(d)
    model_slides(d)
    math_slides(d)
    gain_table_slide(d)
    vertex_slides(d)
    limits_slide(d)
    for slide in d.p.slides:        # connectors inherit a soft shadow from the default theme, which blurs thin lines
        for ref in slide._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef"):
            ref.set("idx", "0")
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
