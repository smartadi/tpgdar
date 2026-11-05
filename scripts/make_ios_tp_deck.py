r"""Simple deck: the IOS Reach task, its spectrum by phase, the TP paper's pipeline, GDAR, the TP-GDAR math at rank 3, and a control example.

Run:  .venv\Scripts\python scripts/make_ios_tp_deck.py
Writes TP-GDAR_IOS_Reach.pptx in the repo.  The copy to edit online lives in OneDrive\Talks; this script never touches it,
so a rebuild can't overwrite edits made there.  Edits Aditya made online on 2026-11-05 (slides on the task, the setup and
the TP paper, and removing the Limits slide; then, by 10:42, removing four control slides and nudging slide 8's state-space
equation up 0.03 in) are written into this script.  Not carried over: a text box holding only "s" over the consensus movie.

Text and figures are ordinary text and shapes (scripts/deck_native.py), which PowerPoint for the web can edit.  Plotted
numbers aren't linked to the shapes.  Equations are LaTeX pictures (scripts/deck_latex.py) with their source as alt text.

Figure data: scripts/make_deck_figure_spectra.py, make_deck_figure_ios.py, make_deck_figures_real.py, make_deck_figures_r3.py,
make_deck_figures_nine.py.  Movie: 17_consensus_demo.py --movie, then --mp4.
Numbers: cache/spectra_run.log (spectra), cache/13_run.log (rank 3 fits), cache/15_run.log and 17_run.log (control).
Paper: docs/TP_model_transformation_as_a_way_to_LMI-based_controller_design.pdf (Baranyi 2004), Method 3 and 4.
Felix's GDAR: docs/2024.02.26.582177v1.full.pdf (Schwock et al. 2024).
Dataset: Griggs, Stanis et al., Nat Commun 2026, 17:3128 (s41467-026-69448-3), and Stanis et al., EMBC 2025 (10.1109/EMBC58623.2025.11254470).
"""

import os
import sys

from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_latex import add_latex  # noqa: E402
from make_mpc_tp_deck import MP4, POSTER, arrow, edge_key, figure, native  # noqa: E402
from make_tp_math_deck import LEFT, MUTED, ROOT, TINT, WIDTH, Deck, title_slide  # noqa: E402

OUT = os.path.join(ROOT, "TP-GDAR_IOS_Reach.pptx")
LABEL_W = 2.6
SMALL_NODES = {"node_radius": 0.07}
FELIX = "Schwock et al., bioRxiv 2024, doi 10.1101/2024.02.26.582177."
SPECTRUM_TITLE = "Rank 3 fit vs data: similar shape, weaker in plan and move"
SPECTRUM_LINE = "Correlation with the data, 3 to 100 Hz: r = 0.88 to 0.98. Beta (13-30 Hz) in move: data −3.6 dB, model −1.9 dB. In post the sign differs: +0.7 vs −0.1 dB."
SPECTRUM_NOTES = ("Model: the rank 3 fit with evoked left in, its AR spectrum frozen at each of the 16 bins, averaged in dB within a phase, with the same 200 ms window applied in frequency. "
                  "Data and model are each shown relative to the mean of their four phases. A frozen spectrum isn't the true spectrum of a changing system, and the model doesn't include the evoked response as an input. "
                  "No model was fitted for this slide. Beta, data vs model: hold +1.9, +1.6 dB; plan +0.9, +0.5 dB. High gamma (65-115 Hz) agrees within 0.25 dB in every phase. Log: cache/spectra_run.log.")


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def steps(d, s, rows, y=1.5, size=24, gap=0.9, x=LEFT, label_w=LABEL_W, right=LEFT + WIDTH, tag_w=0.0):
    """Rows of a grey label and an equation.  A row whose second item starts with a colon is plain text, and a third item is
    a grey tag at the right."""
    for row in rows:
        label, body = row[0], row[1]
        d.text(s, label, y + gap / 2 - 0.25, 0.5, x=x, w=label_w, size=18, bullet=False, color=MUTED)
        if body.startswith(":"):
            d.text(s, body[1:], y + gap / 2 - 0.27, 0.5, x=x + label_w, w=right - x - label_w, size=20, bullet=False)
        else:
            add_latex(s, body, x + label_w + 0.1, y, size=size, max_w=right - x - label_w - 0.1 - tag_w, h_row=gap)
        if len(row) > 2:
            d.text(s, row[2], y + gap / 2 - 0.22, 0.45, x=right - tag_w, w=tag_w, size=14, bullet=False, color=MUTED)
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
    x = 7.55
    d.text(s, ["25 mm window over left posterior parietal cortex (PPC), stable for years",
               "Jaws, an inhibitory opsin, covers 92 mm² area",
               "Light: 900 ms from target on, on a random half of stim-day trials",
               "The window stays clear for 10 to 14 weeks, then tissue is removed"], 1.5, 3.09, x=x, w=13.333 - x - 0.3, size=20)
    d.text(s, "Griggs, Stanis et al., Nat Commun 2026. Stanis et al., EMBC 2025.", 6.5, 0.4, size=16, bullet=False, color=MUTED)
    notes(s, "Monkeys H and L: adult male rhesus macaques. Jaws covers 92 mm2 in H and 76 mm2 in L. Areas 7a and 7b near the intraparietal sulcus (IPS). Vector AAV8-hSyn-Jaws-GFP by convection-enhanced delivery, up to 50 uL per site. "
             "Red LEDs 634 nm, cortical temperature change under 1 C. Expression faded after about 2 years, and reinjection restored the response. "
             "The two papers give different numbers for light intensity (about 12 mW/mm2 vs 1.2 mW/cm2) and spacing (EMBC: 800 um pitch, while 32 electrodes over 314 mm2 would average about 3 mm), so they're left off the slide. Worth asking Noah.")


def findings_slide(d):
    s = d.slide("What the papers found on this task")
    steps(d, s, [("Behaviour", ":Light slows reaches by tens of ms, most clearly downward. Paths look similar"),
                 ("Planning", ":Theta power rises near the IPS and in areas 7a and 7b"),
                 ("Light", ":ECoG power goes up, not down, and more over Jaws cortex"),
                 ("Controls", ":No such effect in an untreated monkey, with blue light, or once expression fades"),
                 ("Likely why", ":Upper layers are inhibited, which releases layer 5. A simulation agrees"),
                 ("Network", ":Over Jaws cortex, gamma coherence and theta-gamma coupling fall")], y=1.5, gap=0.8)
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
    d.text(s, ["Monkeys reach with the right hand to 1 of 4 targets, 80 mm from center",
               "This deck: H 2021-11-12, no stim, 320 trials, 22 channels, LFP at 333 Hz"], 5.5, 0.87, size=20)
    notes(s, "Head-fixed, in a dark rig. The middle finger is tracked by motion capture; targets are 25 mm across on a monitor within reach, and a tone is the go cue. Juice reward. "
             "Noah's curated set: H 10 days, 3,865 trials; L 6 days, 5,116 trials. "
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
    steps(d, s, [("State space", r"\begin{bmatrix}\dot x(t)\\ y(t)\end{bmatrix}=S(p(t))\begin{bmatrix}x(t)\\ u(t)\end{bmatrix},\qquad S(p)=\begin{bmatrix}A(p)&B(p)\\ C(p)&D(p)\end{bmatrix}"),
                 ("Polytope", r"S(p)=\sum_{r}w_r(p)\,S_r,\qquad w_r\geq 0,\qquad\sum_{r}w_r=1")], y=1.25, size=24, gap=0.95, label_w=1.9)
    next(sh for sh in s.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE).top -= 29688     # Aditya's online nudge, 0.03 in up
    flow(d, s, ["Known model", "Sample on a grid", "HOSVD", "Drop small singular values", "Make weights convex"], 3.3)
    d.text(s, "Baranyi 2004, Method 4. It rewrites a model you obtain from sysID, so LMI design can use it.", 4.6, 0.6, size=24)
    add_latex(s, r"\mathrm{SysID:}\quad Y(t)=C\,[A,\ B]\,[\,x(t);\ u(t)\,]", LEFT + 0.1, 5.7, size=26)
    notes(s, "State-space form and S(p): the paper's eqs. 1 and 2. For a discrete-time system the left side reads x(t+1) instead of the time derivative.")

    s = d.slide("The paper, steps 1 and 2: sample, then HOSVD")
    y = steps(d, s, [("1  Sample", r"S_g=S(p_g),\quad g=1,\ \dots,\ P"),
                     ("2  HOSVD", r"[\,\mathrm{vec}\,S_1;\ \cdots;\ \mathrm{vec}\,S_P\,]=U\,D\,V^{\top}"),
                     ("   Keep r", r"S(p_g)\approx\sum_{j=1}^{r}U_{gj}\,S_j"),
                     ("   Error", r"\mathrm{error}\leq\sum_{j>r}\sigma_j^{2}")], y=1.3, gap=1.0)
    d.text(s, ["Each kept column is one weight function on the grid, joined by straight lines",
               "Rank is read from singular values that are exactly zero"], y + 0.1, 1.3, size=22)
    notes(s, "With several parameters, HOSVD runs on each parameter dimension and the vertex counts multiply; the system matrix dimensions "
             "aren't compressed. With one parameter it is this one SVD. Error bound: the paper's eq. 37.")

    s = d.slide("The paper, step 3: make the weights convex")
    y = steps(d, s, [("Free choice", r"U'=U\,T,\qquad S'=T^{-1}S\qquad\text{same }S(p)\text{ for any invertible }T"),
                     ("1  Sum to one", ":Transformation 1. It may add one weight column; here it doesn't"),
                     ("2  Nonnegative", ":Transformation 2. Shrink all weights toward uniform")], y=1.25, gap=0.62, size=22)
    add_latex(s, r"\text{Rank 3 example:}\quad S(p)=S_0+p\,S_1+p^{2}S_2,\quad 1\leq p\leq 2", LEFT + 0.1, y + 0.05, size=20)
    native(s, "convex_r3", y=y + 0.45, bottom=6.45)
    d.text(s, "Valid but loose: after step 2 no weight passes 0.61, and the vertices' triangle has about 300 times the tightest area.", 6.5, 0.5, size=20, bullet=False)
    notes(s, "The worked example is mine, not the paper's. S_0, S_1, S_2 are orthonormal, on 41 points of p. Singular values relative to the first: 1, 0.121, 0.0066, then zero, so rank 3. "
             "The constant is in the span of the weights, so Transformation 1 adds no column; U' = U diag(U'1). After it the weights run from -0.44 to 1.42. "
             "As printed, the paper's Z in Transformation 2 doesn't normalise, so I used Z = (bI + 11')/(n + b), with b = 1 unless the most negative entry is below -1. Here b = 1. Weights then run from 0.14 to 0.61. "
             "All steps reconstruct S(p) exactly. Tightest is the smallest enclosing triangle, the same code as the real fits: weights reach 0.75, 0.93, 0.93. Triangle area in vertex space: 59.2 after step 2, 0.19 tightest. "
             "A loose polytope makes the LMI harder to solve, because it must hold at every vertex.")


# ----------------------------------------------------------------------------------------------- bins and GDAR

def bins_slide(d):
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


def gdar_slides(d):
    figure(d, "An electrode array is a graph", "graph_array", "One number per electrode. Edges join neighbours.")

    s = d.slide("Activity spreads along edges: graph diffusion")
    y = steps(d, s, [("Heat diffusion", r"\dot{x}=w\,\Delta x", "Eq. 3"),
                     ("On the graph", r"x(t+1)=(I-w\,L)\,x(t)+e(t),\qquad L=BB^{\top}", "Eq. 4")],
              y=1.2, size=24, gap=0.68, label_w=2.2, tag_w=0.8)
    native(s, "diffusion", y=y + 0.05, bottom=6.65)
    d.text(s, FELIX, 6.72, 0.4, size=16, bullet=False, color=MUTED)
    notes(s, "Their notation: s for activity, p for model order, u for noise. Here x, K and e, because p is trial phase and u is the input in this deck. B is the node-to-edge incidence matrix, "
             "so L is the graph Laplacian; the continuous Laplacian is approximated by minus L.")

    s = d.slide("Felix's GDAR: diffusion with learned weights and memory")
    y = steps(d, s, [("At node i", r"x_i(t+1)=\sum_{k=1}^{K}m_{k,i}\,x_i(t-k+1)+\sum_{j\in N(i)}f_{ij}(t+1)", "Eq. 1"),
                     ("Flow on edge ij", r"f_{ij}(t+1)=\sum_{k=1}^{K}w_{k,ij}\,\bigl(x_j(t-k+1)-x_i(t-k+1)\bigr)", "Eq. 2"),
                     ("All nodes", r"x(t+1)=\sum_{k=1}^{K}\bigl(M_k-B\,W_k\,B^{\top}\bigr)\,x(t-k+1)+e(t)", "Eq. 5"),
                     ("As a VAR", r"x(t+1)=\sum_{k=1}^{K}A_k\,x(t-k+1)+e(t),\qquad A_k=A_k^{\top},\ \text{ zero off the graph}", "Eq. 7")],
              y=1.2, size=22, gap=1.03, label_w=2.2, tag_w=0.8)
    d.text(s, ["Voltage difference × conductivity = flow on each edge. Net flow into a node changes its activity",
               "Fit: least squares on the free numbers in closed form (Eq. 8 to 11)"], y + 0.05, 1.05, size=20)
    d.text(s, FELIX + " Their s, p, u are our x, K, e.", 6.72, 0.4, size=16, bullet=False, color=MUTED)
    notes(s, "M_k = diag(m_k) holds the node terms and W_k = diag(w_k) the edge terms, at lag k. A_k off-diagonal equals the edge weight, and the diagonal is the node term minus the sum of the neighbours' A_k entries. "
             "The GDAR flow is f(t+1) = sum_k W_k B' x(t-k+1), their eq. 6. Fit: they first solve least squares without the noise covariance (eq. 10), estimate the covariance from its residual (eq. 11), and then solve generalized least squares (eq. 9).")

    s = d.slide("One GDAR fitted to this day")
    native(s, "gdar_fit", y=1.25, bottom=6.6, panel_flags={"Edges, all lags": SMALL_NODES})
    edge_key(s)
    d.text(s, "Here (22 nodes + 50 edges) × 16 lags = 1,152 numbers, one operator for the whole day.", 6.65, 0.5, size=18, bullet=False)


def tp_gdar_slide(d):
    s = d.slide("TP-GDAR: three GDARs, mixed across the trial")
    y = add_latex(s, r"\Theta(p)=\sum_{j=1}^{3}w_j(p)\,\Theta_j,\qquad \Theta=[\,A_1\ \cdots\ A_K\,]", LEFT + 0.1, 1.3, size=24)
    native(s, "tp_mix_r3", y=y + 0.1, bottom=6.6, panel_flags={"Vertex 1": SMALL_NODES, "Vertex 2": SMALL_NODES, "Vertex 3": SMALL_NODES})
    edge_key(s)
    d.text(s, "Rank 3: 3,495 numbers. 16 separate fits: 18,432. Evoked response left in.", 6.65, 0.5, size=18, bullet=False)
    notes(s, "3,495 = 3 x 1,152 operator numbers plus the free weights, 16 x 3 less 9. The corners of the triangle are one choice of coordinates. Vertex 2 lies 1.01 of the data's spread from the nearest bin, so the data never reach it.")


def bridge_slide(d):
    s = d.slide("From data to vertices: what we add to the paper")
    rows = [("TP paper", ["Known model", "Sample on a grid", "HOSVD", "Convex weights"], (0, 1, 2, 3)),
            ("Data, HOSVD", ["Recordings", "GDAR fit in each bin", "HOSVD", "Convex weights"], (2, 3)),
            ("Data, joint fit", ["Recordings", "Statistics in each bin", "Joint fit", "Convex weights"], (3,))]
    y = 1.6
    for label, boxes, shaded in rows:
        d.text(s, label, y + 0.28, 0.5, x=LEFT, w=2.1, size=20, bullet=False)
        flow(d, s, boxes, y, x=LEFT + 2.1, w=2.2, gap=0.33, shaded=shaded)
        y += 1.3
    d.text(s, ["Shaded: taken from the paper. White: what data needs",
               "The fit in each bin plays the part of sampling the model, but with noise"], y + 0.05, 1.2, size=22)
    notes(s, "The data step is ours, using Felix's GDAR model and estimator in each bin. The joint fit keeps the paper's model form and its idea of convex "
             "weights, but it replaces both the per-bin fits and HOSVD.")


# ----------------------------------------------------------------------------------------------- the math

def math_slides(d):
    s = d.slide("Data step: a GDAR in each bin")
    y = steps(d, s, [("Model in bin g", r"x(t+1)=\Theta_g\,z(t)+e(t)"),
                     ("Regressor", r"z(t)=[\,x(t);\ \cdots;\ x(t-K+1)\,]"),
                     ("Objective", r"\Theta_g^{\mathrm{fit}}=\arg\min_{\Theta}\ \sum_{t\in g}\|\,x(t+1)-\Theta\,z(t)\,\|^{2}"),
                     ("Constraint", ":Each lag operator is symmetric and zero off the graph, as in GDAR")], y=1.5, gap=1.0, size=26)
    d.text(s, "The same fit as Felix's GDAR, run once per bin. About 10,800 samples per bin on this day.", y + 0.1, 0.6, size=22, bullet=False)
    notes(s, "Left off the slide to keep it simple: the norm in the real fit is weighted by the inverse noise covariance, Sigma from one pooled fit, then fixed (Felix's generalized least squares, his eq. 9). "
             "The pooled fit matches CVARModel.fit_gdar to 1e-11. The weighted form is written out on the joint-fit slide, which needs it.")

    s = d.slide("HOSVD on the fits: every coefficient counts the same")
    y = steps(d, s, [("Stack the fits", r"[\,\mathrm{vec}\,\Theta_1^{\mathrm{fit}};\ \cdots;\ \mathrm{vec}\,\Theta_{16}^{\mathrm{fit}}\,]=U\,D\,V^{\top}"),
                     ("Keep 3", r"\Theta(p_g)\approx\sum_{j=1}^{3}W_{gj}\,\Theta_j,\qquad W=\text{first 3 columns of }U"),
                     ("It minimises", r"\sum_g\|\,\Theta(p_g)-\Theta_g^{\mathrm{fit}}\,\|_F^{2}")], y=1.4, gap=1.05)
    d.text(s, ["Exact samples make this the right loss. With noisy fits it treats a badly estimated coefficient like a good one",
               "No exact zeros to read a rank from: singular values 1, 0.018, 0.011, 0.010, ..."], y + 0.15, 1.7, size=22)


def convex_slide(d):
    s = d.slide("From rank 3 weights to convex weights")
    y = steps(d, s, [("Sum to one", r"W_a=W+(\mathbf{1}-W a)\,a^{\top}/(a^{\top}a),\qquad W a\approx\mathbf{1}"),
                     ("Data metric", r"Y=W_a\,L,\qquad L\,L^{\top}=\Phi\,H_{\mathrm{pool}}\,\Phi^{\top}"),
                     ("Vertices", ":The smallest triangle that encloses the weights of all bins"),
                     ("Result", r"\Theta(p)=\sum_{v=1}^{3}\lambda_v(p)\,\Theta_v,\qquad\lambda_v\geq 0,\qquad\sum_v\lambda_v=1")], y=1.4, gap=0.85)
    d.text(s, ["Same freedom as the paper, a different transform. No added vertex",
               "The metric says how well all the data pin down each direction",
               "We ran this on the joint fit, which comes next. Sum-to-one misses by 0.017 to 0.019; the likelihood cost of that isn't measured",
               "Corners needn't be visited by the data. Only the mix is unique, not the vertices"], y + 0.05, 2.5, size=20)
    notes(s, "Phi stacks the 3 operators in free numbers. H_pool is the metric H_g of the joint-fit slide, summed over bins. This step refits nothing. Held-out gains are measured before it, so they're unaffected, but the vertex maps and weight plots aren't exact. "
             "Nearest bins, as a fraction of the spread: evoked left in 0, 1.01, 0.16; evoked removed 0.12, 0, 0.35. The step needs only rank 3 weights and a metric; it has been run on the joint fit, not on the HOSVD weights.")


def joint_slides(d):
    s = d.slide("Joint fit: score each bin on its own data")
    steps(d, s, [("Loss in bin g", r"L_g(\phi)=\sum_{t\in g}\|\,x(t+1)-\Theta(\phi)\,z(t)\,\|_{\Sigma^{-1}}^{2}"),
                 ("Noise", r"\Sigma\ \text{ from one pooled fit, then fixed}"),
                 ("Free numbers", r"\mathrm{vec}\,\Theta=R\,\phi"),
                 ("Statistics", r"G_g=\sum_{t\in g}z(t)\,z(t)^{\top},\qquad C_g=\sum_{t\in g}z(t)\,x(t+1)^{\top}"),
                 ("Metric", r"H_g=R^{\top}(G_g\otimes\Sigma^{-1})R,\qquad h_g=R^{\top}\mathrm{vec}(\Sigma^{-1}C_g^{\top})"),
                 ("Fit in bin g", r"\phi_g^{\mathrm{fit}}=H_g^{-1}h_g")], y=1.4, gap=0.9)
    notes(s, "R maps the free numbers to the full coefficient matrix: symmetric, zero off the graph. The pooled fit with G and C summed over bins is Felix's estimator; "
             "it matches CVARModel.fit_gdar to 1e-11. H_pool on the convex-weights slide is H_g summed over bins.")

    s = d.slide("Joint vs HOSVD: same low-rank form, different metric")
    y = steps(d, s, [("Loss, rewritten", r"L_g(\phi)=\|\,\phi-\phi_g^{\mathrm{fit}}\,\|_{H_g}^{2}+\mathrm{const}"),
                     ("HOSVD", r"\min\ \sum_g\|\,\Theta(p_g)-\Theta_g^{\mathrm{fit}}\,\|_F^{2}"),
                     ("Joint", r"\min\ \sum_g\|\,\phi(p_g)-\phi_g^{\mathrm{fit}}\,\|_{H_g}^{2}")], y=1.5, gap=1.05, size=26)
    d.text(s, "Directions that a bin's data pin down count more, because the metric is the inverse covariance of that bin's fit.", y + 0.15, 1.0, size=22, bullet=False)
    notes(s, "Cov of phi-hat is the inverse of H_g given the regressors, with Gaussian noise and known Sigma. So the joint fit is the maximum-likelihood low-rank fit for "
             "that fixed Sigma. HOSVD would be the same only if every H_g were proportional to the identity. This is my derivation, not the paper's. "
             "Held-out gain over GDAR at rank 3, nats per sample, 5-fold, 16 bins, H 2021-11-12, script 12d: evoked left in HOSVD +0.123, SVD in the pooled metric +0.212, joint +0.222, separate fits +0.211; "
             "evoked removed +0.239, +0.304, +0.323, +0.392. So the metric does most of the work. One day only, with the target-onset step unmasked.")

    s = d.slide("Joint fit: alternate two exact solves")
    y = steps(d, s, [("Objective", r"J(W,\Phi)=\sum_g L_g\Bigl(\sum_{j}W_{gj}\,\phi_j\Bigr)"),
                     ("Start", r"\phi_1=\text{pooled fit},\qquad W_{g1}=1"),
                     ("Weight step", r"(\Phi\,H_g\,\Phi^{\top})\,w_g=\Phi\,h_g"),
                     ("Operator step", r"\Bigl(\sum_g w_g w_g^{\top}\otimes H_g\Bigr)\,\mathrm{vec}\,\Phi=\sum_g w_g\otimes h_g"),
                     ("Stop", r"|\Delta J|\leq 10^{-7}\,|J|")], y=1.4, gap=0.88)
    d.text(s, "The objective never rises, and these fits took 5 to 6 sweeps. No closed form: it finds a local optimum. Then the same convex step.", y + 0.05, 0.9, size=20, bullet=False)
    notes(s, "Start: the other operators come from an SVD of the per-bin fits around the pooled fit, in the pooled data metric. After each sweep the weights are rescaled so each "
             "column has norm sqrt(P), which changes nothing in the model. Limits: 25 sweeps in folds, 40 on the full day. Jitter 1e-10.")


def result_slides(d):
    s = figure(d, "TP beats one GDAR on held-out trials and unseen task times", "tp_vs_gdar_r3",
               "Gain = held-out one-step likelihood over GDAR, 5-fold. Separate fits match or beat rank 3, except on 12 bins with evoked left in.")
    notes(s, "Each model gets its own noise covariance when scored, Sigma = 0.99 Sigma-hat + 0.01 diag(Sigma-hat), so the fit doesn't quite optimise the score. Script 12. "
             "4 phases: evoked left in rank 3 +0.167, separate +0.180; evoked removed +0.259, +0.297. 12 time bins: left in +0.137, +0.098; removed +0.168, +0.173. Fold SE 0.008 to 0.013. "
             "Numbers on 12 bins: GDAR 1,152, rank 3 3,483, separate 13,824. "
             "Task time left out (evoked left in): each inner bin is left out of the fit in turn. Rank 3 reads its weights off its neighbours; 'fit on neighbour' is a separate fit on the nearest bin; 'fit on that bin' is a separate fit that did see that bin. "
             "All are scored with GDAR's noise covariance, since a left-out bin has no residuals of its own. Over those bins, evoked left in / removed: rank 3 +0.122 / +0.110, neighbour +0.044 / +0.004, that bin +0.077 / +0.121; "
             "rank 3 minus neighbour, per trial, +0.079 +/- 0.007 / +0.105 +/- 0.008. Given its own bin's covariance, which the others don't get, the neighbour fit reaches +0.089 / +0.082: below zero before go, +0.23 / +0.27 after it. "
             "Corrected 2026-11-05 (L-2026-11-05-d): before, each fit had its own covariance and the neighbour fit looked worse than GDAR. "
             "One monkey. No light in these fits, so they say nothing about the Jaws effects. "
             "The cleaned LFP has steps at 0 and 900 ms, not masked here; masking moved the gains little. They likely come from the artefact cleaning, which rejoins epochs at light on and off even on no-stim trials. "
             "On 11-12 the median jump is 23.5 uV at 0 ms and 13.9 at 900 ms in noart, against 2.5 typical, and raw lfp has none. To confirm with Noah.")

    s = figure(d, "Rank 3, evoked removed: hold, plan, move", "vertices_r3_induced",
               panel_flags={"Vertex 1": SMALL_NODES, "Vertex 2": SMALL_NODES, "Vertex 3": SMALL_NODES})
    edge_key(s)
    notes(s, "Vertex 1 leads through hold (0.67 to 0.94). Vertex 2 leads from plan 1/4 to 3/4 (1.00, 0.79, 0.49), and again at post 3/4 and 4/4 (0.65, 0.72). Vertex 3 leads from plan 4/4 to post 2/4 (0.54 to 0.81). So post heads back toward the early-plan vertex, not the hold one. "
             "Weights: jackknife band over the 5 training splits. Corners are one choice of coordinates: nearest bins are 0.12, 0 and 0.35 of the spread away.")

    s = figure(d, SPECTRUM_TITLE, "spectra_model", SPECTRUM_LINE)
    notes(s, SPECTRUM_NOTES)


# ----------------------------------------------------------------------------------------------- control

def system_slide(d):
    s = d.slide("Control example: the system")
    add_latex(s, r"x(t+1)=\sum_{k=1}^{2}A_k(p)\,x(t-k+1)+B\,u(t)+e(t),\qquad [\,A_1(p)\ \ A_2(p)\ \ B\,]=(1-p)\,\Theta_0+p\,\Theta_1", LEFT + 0.1, 1.3, size=20, max_w=WIDTH - 0.2)
    # the graph, with an input arrow on each node
    x0, y0, step = LEFT + 0.55, 3.0, 1.25
    xy = [(x0 + j * step, y0 + i * step) for i in range(3) for j in range(3)]
    for i in range(3):
        for j in range(3):
            a = xy[i * 3 + j]
            for q in ([xy[i * 3 + j + 1]] if j < 2 else []) + ([xy[(i + 1) * 3 + j]] if i < 2 else []):
                ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(a[0]), Inches(a[1]), Inches(q[0]), Inches(q[1]))
                ln.line.color.rgb, ln.line.width = RGBColor(0x99, 0x99, 0x99), Pt(2.5)
    r = 0.25
    for x, y in xy:
        arrow(s, x + 0.5, y - 0.5, x + r * 0.72, y - r * 0.72)
        c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - r), Inches(y - r), Inches(2 * r), Inches(2 * r))
        c.fill.solid()
        c.fill.fore_color.rgb = TINT
        c.line.color.rgb, c.line.width = RGBColor(0x44, 0x44, 0x44), Pt(1.25)
    for k in (0, 8):
        x, y = xy[k]
        add_latex(s, rf"x_{k + 1}", 0, y - r, size=16, center_x=x, h_row=2 * r)
        add_latex(s, rf"u_{k + 1}", x + 0.42, y - 0.84, size=18)
    y = 2.15
    x = 5.45
    for sym, text in ((r"x(t)", "Activity at 9 nodes on a 3 × 3 grid, joined by 12 edges"),
                      (r"u(t)", "9 inputs, one per node, like stimulation at each site"),
                      (r"B", "Input gain at each node, fixed, 0.5 to 1.5"),
                      (r"A_k(p)", "How activity spreads over the last 2 steps"),
                      (r"p", "Moves from 0 to 1 and back. The controller never sees it"),
                      (r"e(t)", "Noise at each node, standard deviation 0.3")):
        add_latex(s, sym, x, y, size=20, h_row=0.56)
        d.text(s, text, y + 0.03, 0.5, x=x + 1.0, w=13.333 - x - 1.3, size=19, bullet=False)
        y += 0.56
    d.text(s, ["Synthetic, so the truth is known. Without control activity grows: largest eigenvalue 1.03 at the unstable end",
               "Goal: keep the network stable for every p, then drive all 9 nodes to one target"], y + 0.1, 1.3, x=x, w=13.333 - x - 0.3, size=18)
    notes(s, "Numbers per operator: 2 x (9 + 12) + 9 = 51. 20 random plants in each of two designed cases: p = 0 stable (0.90) and p = 1 mildly unstable (1.03), or both ends at 1.03. The instability is by design and says nothing about cortex. "
             "The truth is a mix of two operators, so this example has 2 vertices. It is the one place in the deck with 2.")


def control_slides(d):
    system_slide(d)

    s = d.slide("Polytopic control: the math")
    y = steps(d, s, [("Feedback", r"u=F\,\xi,\quad \xi(t)=[\,x(t);\ x(t-1)\,],\qquad \xi(t+1)=\Bigl(\sum_v\lambda_v(p)\,A_v+B_cF\Bigr)\,\xi(t)"),
                     ("At each vertex", r"(A_v+B_cF)^{\top}\,P\,(A_v+B_cF)\ \prec\ \alpha^{2}P"),
                     ("As an LMI", r"(A_vQ+B_cY)^{\top}\,(\alpha Q)^{-1}\,(A_vQ+B_cY)\ \prec\ \alpha Q,\qquad Q=P^{-1},\ \ Y=FQ")],
              y=1.45, gap=1.05, size=22)
    d.text(s, ["If it holds at the vertices it holds at every mix of them: every p, and p moving at any speed",
               "The rate sets a guaranteed decay. The fitted certificate needs a margin to hold on the true plant",
               "The loop is closed on the true plant, never on the fitted model"], y + 0.15, 2.0, size=20)
    notes(s, "At rate 1 the fitted certificate held on the true plant in 0 of 20 plants, because the solution sits on the LMI's boundary. At rate 0.9 it held in 20 of 20, in both cases. "
             "A_v here is the companion form of vertex v and B_c the input matrix in that form.")

    s = d.slide("Consensus: all 9 nodes follow one target")
    y = steps(d, s, [("Target", r"r(t)\ \text{ steps through } 0,\ +3,\ -3,\ +3,\ 0\text{, the same at every node}"),
                     ("Running error", r"z(t+1)=z(t)+r(t)-x(t)"),
                     ("Input", r"u(t)=F_x\,\xi(t)+F_z\,z(t),\qquad \xi(t)=[\,x(t);\ x(t-1)\,]")], y=1.45, gap=0.95, size=24)
    d.text(s, ["All 9 nodes get an input at every step, each worked out from all nodes' recent activity and error",
               "Consensus needs that: with fewer inputs, only that many nodes can be held at a chosen value",
               "The same LMI on the fitted vertices gives one fixed gain. It doesn't see p, which keeps moving",
               "The loop runs on the true plant, here with both ends of p unstable"], y + 0.15, 2.5, size=20)
    notes(s, "State xi and error z together still form a polytope, so the same LMI applies, at rate 0.95. Script 17.")

    s = d.slide("Consensus: the movie")
    iw, ih = Image.open(POSTER).size
    h = 4.75
    w = h * iw / ih
    s.shapes.add_movie(MP4, Inches((13.333 - w) / 2), Inches(1.2), Inches(w), Inches(h), poster_frame_image=POSTER, mime_type="video/mp4")
    d.text(s, ["Dot = node activity. Ring = target. Line width = edge strength at that moment's p. Traces: all 9 nodes, target dashed, p below",
               "Left, no inputs: activity runs away. Right, one fixed gain: every node reaches the target. Held error about 0.33, set by the noise"],
           6.0, 1.3, size=16)
    notes(s, "One plant with both ends unstable, polytopic gain. Medians over 20 plants in the held windows: no control, stable at all p in 0 of 20, rms error 24.5. Polytopic, 20 of 20, rms error 0.338, rms input 1.97. Pooled LQR does this too, 20 of 20, rms error 0.333, rms input 1.94.")


def build():
    d = Deck()
    title_slide(d, "TP-GDAR on IOS Reach", "The task, its spectrum by phase, the math, and a control example")
    task_slides(d)
    paper_slides(d)
    bins_slide(d)
    gdar_slides(d)
    tp_gdar_slide(d)
    bridge_slide(d)
    math_slides(d)
    convex_slide(d)
    joint_slides(d)
    result_slides(d)
    control_slides(d)
    for slide in d.p.slides:        # connectors inherit a soft shadow from the default theme, which blurs thin lines
        for ref in slide._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef"):
            ref.set("idx", "0")
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
