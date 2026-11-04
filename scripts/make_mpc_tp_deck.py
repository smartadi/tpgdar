r"""Talk deck, low on text and editable: the MPC replay, then graphs, GDAR and TP-GDAR, then a control demo.

Run:  .venv\Scripts\python scripts/make_mpc_tp_deck.py
Writes GDAR-TP_MPC_and_TP-GDAR.pptx.  The fuller write-up of the model and training stays in
GDAR-TP_model_training.pptx (scripts/make_tp_math_deck.py).

Built for PowerPoint for the web: text, equations and figures are ordinary text and shapes
(scripts/deck_native.py), and the movie is an embedded MP4.  Pictures remain only for the paper's MPC
panels, from Figure 6.  The plotted numbers aren't linked to the shapes: change the script and rebuild.

Figure data comes first, from scripts/make_deck_figures_real.py, make_deck_figures_nine.py,
make_deck_figure_ios.py (figures/deck/*.json) and 17_consensus_demo.py --movie, then --mp4.
Part 1 numbers: the NC draft (Closedloop_nc/results.tex, 5a955c7), checked against
brain_paper/controller-analysis/data/mpc_direct_*_noCL.mat and mpc_varx_supp_*.mat.  The command
limit numbers are the JN correction of 2026-11-03 to the NC sentence (brain_paper RESEARCH.md,
stats_v2/poor_trials.py).  Figure: Closedloop_nc/images/Figure6.pdf.
Part 2 numbers: scripts 11 to 14, facts on the data from yazdanlab/iosreach/README.md.
Part 3 numbers: scripts 15 and 17 (cache/15_run.log, 17_run.log).
"""

import os
import subprocess
import sys

from lxml import etree
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_native import add_edge_key, add_equation, render_figure  # noqa: E402
from make_tp_math_deck import EQ, LEFT, MUTED, ROOT, TINT, WIDTH, Deck, title_slide  # noqa: E402

OUT = os.path.join(ROOT, "GDAR-TP_MPC_and_TP-GDAR.pptx")
DECK = os.path.join(ROOT, "figures", "deck")
MP4 = os.path.join(ROOT, "figures", "17_consensus.mp4")
POSTER = os.path.join(ROOT, "figures", "17_consensus_poster.png")
FIG6 = os.path.join(os.path.dirname(ROOT), "Closedloop_nc", "images", "Figure6.pdf")
TOP, BOTTOM = 1.3, 7.25
NOTE_SIZE = 18
EQ_X = LEFT + 0.1


def figure6_crops():
    """Figure 6 at 300 dpi, the constant-target and sine rows, without panels a, b and m."""
    png = os.path.join(EQ, "figure6")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", FIG6, png], check=True)
    im = Image.open(png + ".png").convert("RGB")
    W, H = im.size
    out = {}
    for name, box in (("constant", (0.0, 0.205, 1.0, 0.725)), ("sine", (0.0, 0.745, 0.80, 1.0))):
        path = os.path.join(EQ, f"figure6_{name}.png")
        crop = im.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
        if name == "constant":      # panels c-h only: blank what is left of panel b above e and f
            crop.paste((255, 255, 255), (int(0.5 * W), 0, W, int((0.262 - box[1]) * H)))
        crop.save(path)
        out[name] = path
    return out


def fit_picture(s, path, x, w, top=TOP, bottom=BOTTOM):
    iw, ih = Image.open(path).size
    h = bottom - top
    if w / h > iw / ih:
        pw, ph = h * iw / ih, h
    else:
        pw, ph = w, w * ih / iw
    s.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(top), width=Inches(pw), height=Inches(ph))
    return pw, ph


def native(s, name, x=LEFT, y=TOP, w=WIDTH, h=None, bottom=BOTTOM, **kw):
    """Draw figures/deck/<name>.json as shapes."""
    render_figure(s, os.path.join(DECK, name + ".json"), x, y, w, h if h is not None else bottom - y, **kw)


def figure(d, title, name, note=None, **kw):
    """A title and a figure.  At most one short line under it."""
    s = d.slide(title)
    native(s, name, bottom=6.55 if note else BOTTOM, **kw)
    if note:
        d.text(s, note, 6.6, 0.5, size=NOTE_SIZE, bullet=False)
    return s


def edge_key(s, label="edge weight"):
    """The colour key for edge maps, in the title row at the right."""
    add_edge_key(s, 13.333 - 0.25 - (1.52 + 0.09 * len(label)), 0.62, label=label)


def arrow(s, x0, y0, x1, y1):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    ln.line.color.rgb, ln.line.width = RGBColor(0x44, 0x44, 0x44), Pt(1.75)
    a = ln.line._get_or_add_ln()
    etree.SubElement(a, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", type="triangle", w="med", len="med")


def mpc_slides(d):
    crops = figure6_crops()

    s = d.slide("The MPC: predict, plan the laser, apply the first step")
    add_equation(s, r"$\mathrm{min}_{u}\ \sum_{h}\left(\hat{y}_{t+h}-r\right)^{2}+\lambda\sum_{h}\left(\Delta u_{t+h}\right)^{2}$", EQ_X, 1.45, size=26)
    bx, by = 6.6, 2.75
    d.label(s, "MPC", bx, by, 3.3, 0.5, fill=TINT, size=18, bold=True)
    d.label(s, "VARX model: predicts activity", bx, by + 0.5, 3.3, 0.7, size=15)
    d.label(s, "predict, optimise, apply first step", bx, by + 1.2, 3.3, 0.7, size=15)
    d.label(s, "cortex", bx + 4.4, by + 0.5, 1.6, 0.9, fill=TINT, size=18, bold=True)
    arrow(s, bx + 1.65, by - 0.5, bx + 1.65, by)
    d.text(s, "target r", by - 0.6, 0.4, x=bx + 1.75, w=1.6, size=15, bullet=False, color=MUTED)
    arrow(s, bx + 3.3, by + 0.65, bx + 4.4, by + 0.65)
    d.text(s, "laser", by + 0.2, 0.4, x=bx + 3.4, w=1.0, size=15, bullet=False, color=MUTED)
    arrow(s, bx + 4.4, by + 1.2, bx + 3.3, by + 1.2)
    d.text(s, "activity", by + 1.2, 0.4, x=bx + 3.35, w=1.1, size=15, bullet=False, color=MUTED)
    d.text(s, ["Replans every frame",
               "200 ms prediction R²: 0.41, last value 0.10",
               "A replay on recorded data, not a live test"], 2.7, 2.6, w=5.6, size=22)

    s = d.slide("Constant target: MPC lowers the error PI leaves")
    fit_picture(s, crops["constant"], LEFT, WIDTH, top=1.3, bottom=6.4)
    d.text(s, "RMSE: PI 1.46, MPC 0.88, perfect prediction 0.27. Lower on 107 of 108 trials.", 6.5, 0.5,
           size=NOTE_SIZE, bullet=False)

    s = d.slide("Sinusoidal target: a smaller gain over feedback with preview")
    fit_picture(s, crops["sine"], LEFT, WIDTH, top=1.5, bottom=6.2)
    d.text(s, "RMSE: CL+p 3.00, MPC 2.67, perfect prediction 2.35. Lower on 56 of 57 trials.", 6.4, 0.5,
           size=NOTE_SIZE, bullet=False)

    s = d.slide("What the MPC replay says")
    d.text(s, ["Prediction lowers the error and variability that feedback leaves",
               "Constant target: about 40% lower, about 80% with perfect prediction (ratios of medians)",
               "Most remaining error is prediction error, except the worst quarter of trials: "
               "there 89% of it occurs at a command limit",
               "Simulation on recorded data, one session per target type"], 1.7, 4.8, size=24)


def ios_slide(d):
    s = d.slide("The data: IOS Reach")
    native(s, "ios_reach", y=1.25, bottom=5.45,
           panel_flags={"Trial timeline": dict(no_xaxis=True, no_yaxis=True),
                        "Single trial": dict(no_yaxis=True), "Trial average": dict(no_yaxis=True)})
    d.text(s, ["Macaque H, 4-target centre-out reach. Recorded by Noah Stanis",
               "Two 16-channel ECoG arrays, LFP at 1 kHz, 333 Hz after decimation",
               "3 days: 320, 404 and 610 trials. On the stim day, 307 trials have a 900 ms light pulse at target on"],
           5.5, 1.7, size=17)


def graph_slides(d):
    figure(d, "An electrode array is a graph", "graph_array", "One number per electrode. Edges join neighbours.")

    s = d.slide("Activity spreads along edges")
    y = add_equation(s, r"$x(t+1)=(I-h\,L)\,x(t),\qquad L=D-A$", EQ_X, 1.3, size=28)
    native(s, "diffusion", y=y + 0.1, bottom=7.1)

    s = d.slide("Felix's GDAR: learn the spread from data")
    y = add_equation(s, r"$x(t+1)=\sum_{k=1}^{K}A_k\,x(t-k+1)+e(t),\qquad A_k=A_k^{\top},\quad A_k\ \mathrm{zero\ off\ the\ graph}$", EQ_X, 1.25, size=24)
    native(s, "gdar_fit", y=y + 0.05, bottom=6.6)
    edge_key(s)
    d.text(s, "One number per edge per lag: 1,152 numbers, and one operator for the whole day.", 6.65, 0.5,
           size=NOTE_SIZE, bullet=False)


def tp_slides(d):
    s = d.slide("TP-GDAR: a few GDARs, mixed across the trial")
    y = add_equation(s, r"$\Theta(p)=\sum_{j=1}^{r}w_j(p)\,\Theta_j,\qquad \Theta=[\,A_1\ \cdots\ A_K\,]$", EQ_X, 1.25, size=24)
    native(s, "tp_mix", y=y + 0.05, bottom=6.6)
    edge_key(s)
    d.text(s, "Rank 2: 2,332 numbers. 16 separate fits: 18,432.", 6.65, 0.5, size=NOTE_SIZE, bullet=False)

    s = d.slide("p is trial phase, warped per trial")
    x0, wbin, top = LEFT + 0.1, (WIDTH - 0.2) / 16, 2.2
    for ph, nm in enumerate(["hold", "plan", "move", "post"]):
        d.label(s, nm, x0 + ph * 4 * wbin, top, 4 * wbin, 0.7, fill=TINT, size=24, bold=True)
        for b in range(4):
            d.label(s, f"{b + 1}/4", x0 + (ph * 4 + b) * wbin, top + 0.7, wbin, 0.55, size=16)
    for i, ev in enumerate(["-400 ms", "target on", "go cue", "touch", "touch + 500 ms"]):
        w = 2.4
        cx = x0 + i * 4 * wbin
        tx = min(max(cx - w / 2, x0 - 0.1), x0 + 16 * wbin + 0.1 - w)
        box = d.text(s, ev, top + 1.35, 0.4, x=tx, w=w, size=18, bullet=False, color=MUTED)
        box.text_frame.paragraphs[0].alignment = (PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT if i == 4 else PP_ALIGN.CENTER)
    d.text(s, "Go and touch are known only afterwards, so this p can't be used online.", 5.0, 0.6, size=22, bullet=False)


def training_slides(d):
    s = d.slide("Training 1: statistics per bin, then the baseline")
    y = add_equation(s, r"$G_g=\sum_{t\in g}z(t)\,z(t)^{\top},\qquad C_g=\sum_{t\in g}z(t)\,x(t+1)^{\top}$", EQ_X, 1.6, size=28)
    labels = ["Symmetric OLS, all bins", "Σ from its residuals, fixed", "Refit weighted by Σ⁻¹"]
    bw = 3.85
    for i, t in enumerate(labels):
        d.label(s, t, LEFT + 0.1 + i * (bw + 0.19), y + 0.5, bw, 0.9, fill=TINT, size=20)
    d.text(s, "Same estimator as Felix's fit_gdar: matches to 1e-11.", y + 1.8, 0.5, size=22, bullet=False)

    s = d.slide("Training 2: the objective is the data loss")
    y = add_equation(s, r"$L_g(\phi)=\phi^{\top}H_g\,\phi-2\,\phi^{\top}h_g,\qquad H_g=R^{\top}(G_g\otimes\Sigma^{-1})R$", EQ_X, 1.6, size=28)
    y = add_equation(s, r"$J(W,\Phi)=\sum_{g=1}^{P}L_g(\sum_{j=1}^{r}W_{gj}\,\phi_j)$", EQ_X, y + 0.3, size=28)
    d.text(s, "Directions the data excite count more.", y + 0.4, 0.5, size=22, bullet=False)

    s = d.slide("Training 3: alternate two exact solves")
    y = add_equation(s, r"$\mathrm{W\ step:}\quad(\Phi\,H_g\,\Phi^{\top})\,w_g=\Phi\,h_g$", EQ_X, 1.6, size=28)
    y = add_equation(s, r"$\Phi\ \mathrm{step:}\quad(\sum_g w_g w_g^{\top}\otimes H_g)\,\mathrm{vec}\,\Phi=\sum_g w_g\otimes h_g$", EQ_X, y + 0.3, size=28)
    d.text(s, "J never increases. Real fits took 5 to 7 sweeps.", y + 0.4, 0.5, size=22, bullet=False)


def hosvd_slide(d):
    s = d.slide("HOSVD vs the joint fit: same vectors, different metric")
    y = add_equation(s, r"$\mathrm{HOSVD\ minimises:}\quad\sum_g\ \|\,\Theta(p_g)-\Theta_g^{\mathrm{bin}}\,\|_F^{2}$", EQ_X, 1.6, size=26)
    y = add_equation(s, r"$\mathrm{The\ data\ want:}\quad\sum_g\ \|\,(\Theta(p_g)-\Theta_g^{\mathrm{bin}})\,G_g^{1/2}\,\|_F^{2}$", EQ_X, y + 0.25, size=26)
    d.table(s, [["OS-Pulse, held-out gain (nats/sample)", ""], ["HOSVD", "-45 to -7.5"], ["Joint fit", "-0.11 to +0.52"]],
            y + 0.5, [5.2, 2.4], row_h=0.5, size=18)
    d.text(s, "Felix's GDAR has no HOSVD: it fits one operator. HOSVD was our first route to TP.", y + 2.3, 0.5,
           size=NOTE_SIZE, bullet=False)


def vertex_eval_slides(d):
    s = d.slide("From components to vertices")
    y = add_equation(s, r"$\Theta(p)=\sum_{v}\lambda_v(p)\,\Theta_v,\qquad\lambda_v\geq 0,\qquad\sum_v\lambda_v=1$", EQ_X, 1.7, size=30)
    d.text(s, ["Rank 2: the two extreme bins",
               "Rank 3: the smallest triangle around all bins, so corners may never be visited",
               "One choice of coordinates: only Θ(p) is unique"], y + 0.5, 3.0, size=24)

    s = d.slide("Evaluation: held-out one-step likelihood")
    y = add_equation(s, r"$\mathrm{gain}=\ell_{\mathrm{TP}}-\ell_{\mathrm{GDAR}}\quad(\mathrm{nats/sample})$", EQ_X, 1.7, size=30)
    d.text(s, ["5-fold over trials, refit on each split",
               "Each model has its own noise covariance"], y + 0.5, 2.0, size=24)


def result_slides(d):
    figure(d, "TP beats one GDAR on held-out trials and unseen task times", "tp_vs_gdar")
    s = figure(d, "Rank 3, evoked left in: vertex 2 is an extrapolation", "vertices_r3_felix")
    edge_key(s)
    s = figure(d, "Rank 2, evoked removed: the switch returns only part way", "vertices_r2_induced")
    edge_key(s)
    figure(d, "Other days: timing repeats, edge maps mostly don't", "other_days")


def limitations_slide(d):
    s = d.slide("Limitations")
    d.text(s, ["Trial phase uses go and touch, so it isn't known online",
               "Weights describe the average trial, not each trial",
               "One monkey, three days, and the channel map isn't confirmed with Noah"], 1.8, 4.0, size=26)


def control_slides(d):
    title_slide(d, "Part 3: a control demo", "A sparse 9-node graph learned with TP-GDAR, with an input at each node")

    s = d.slide("The example: 9 nodes, known truth, an input at each node")
    y = add_equation(s, r"$x(t+1)=\sum_{k=1}^{2}A_k(p)\,x(t-k+1)+B\,u(t)+e(t)$", EQ_X, 1.5, size=26)
    y = add_equation(s, r"$\Theta(p)=[\,A_1(p)\ \ A_2(p)\ \ B\,]=(1-p)\,\Theta_0+p\,\Theta_1$", EQ_X, y + 0.15, size=26)
    d.text(s, ["3 × 3 grid, 12 edges: 51 numbers per operator",
               "20 plants, unstable at one end of p or both (the second case added after seeing the first)",
               "Instability by design: this says nothing about cortex"], y + 0.4, 3.0, size=22)

    s = figure(d, "TP-GDAR recovers operators and input gains", "nine_learning")
    edge_key(s, "edge; node = input gain")

    s = d.slide("One gain for every p: check the corners")
    native(s, "polytopic_explainer_hold", y=1.3, w=7.9, h=3.75, panel_flags={
        "Energy over time": dict(no_xaxis=True, no_yaxis=True, title="Energy over time, p held at 1"),
        "Energy change per step": dict(text_y={"vertex": 0.7, "with the gain": 0.57})})
    y = add_equation(s, r"$u=F\,\xi,\quad V=\xi^{\top}P\,\xi,\quad V(t+1)<\alpha^{2}\,V(t)\ \mathrm{at\ every\ vertex}$", EQ_X, 5.1, w=7.9, size=20)
    d.text(s, "No margin (α = 1): the proof held on the true plant in 0 of 20 plants. α = 0.9: 20 of 20.", y + 0.05, 0.5,
           w=7.9, size=16, bullet=False)
    x = LEFT + 8.1
    d.text(s, ["Feedback: input = F × recent activity",
               "Energy V: how far the network is from rest",
               "If energy shrinks at each vertex, it shrinks at every mix: any p, moving at any speed",
               "F needs the fitted vertices, not p",
               "Finding F and P is a convex problem (an LMI)",
               "Tested on the true plant, never the fitted model"], 1.45, 5.5, x=x, w=13.333 - x - 0.3, size=17)

    figure(d, "Both gains keep the network stable", "nine_control", panel_flags={"As p moves": dict(no_xaxis=True)})

    s = d.slide("Consensus: every node reaches the target")
    iw, ih = Image.open(POSTER).size
    h = 5.3
    w = h * iw / ih
    s.shapes.add_movie(MP4, Inches((13.333 - w) / 2), Inches(1.25), Inches(w), Inches(h), poster_frame_image=POSTER, mime_type="video/mp4")
    d.text(s, "Outer ring = target. Pooled LQR does this too: held error about 0.33 for both, set by the noise.", 6.65, 0.5,
           size=NOTE_SIZE, bullet=False)

    s = d.slide("Here the vertices matter more than the LMI")
    native(s, "nine_effort", y=1.3, bottom=5.15)
    d.text(s, ["With an input at every node, pooled LQR stabilises everything too",
               "The fitted vertices can certify either gain",
               "The LMI reaches tighter rates (0.6: 40 of 40 plants, LQR 10 of 40) for less input"], 5.2, 2.0, size=20)


def build():
    d = Deck()
    title_slide(d, "MPC replay and TP-GDAR",
                "Part 1: model-predictive control, brain paper\n"
                "Part 2: graphs, GDAR and TP-GDAR on IOS Reach\n"
                "Part 3: a control demo")
    mpc_slides(d)
    title_slide(d, "Part 2: TP-GDAR", "Graphs, GDAR, then a model that changes across the trial")
    ios_slide(d)
    graph_slides(d)
    tp_slides(d)
    training_slides(d)
    hosvd_slide(d)
    vertex_eval_slides(d)
    result_slides(d)
    limitations_slide(d)
    control_slides(d)
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
