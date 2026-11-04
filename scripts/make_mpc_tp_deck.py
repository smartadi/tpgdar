r"""Plain deck: the brain-paper MPC replay, then TP-GDAR with its comparison plots.

Run:  .venv\Scripts\python scripts/make_mpc_tp_deck.py
Writes GDAR-TP_MPC_and_TP-GDAR.pptx.

Part 1 numbers: the NC draft (Closedloop_nc/results.tex, 5a955c7), checked against
brain_paper/controller-analysis/data/mpc_direct_*_noCL.mat and mpc_varx_supp_*.mat.  The command
limit numbers are the JN correction of 2026-11-03 to the NC sentence (brain_paper RESEARCH.md,
stats_v2/poor_trials.py).
Figure: Closedloop_nc/images/Figure6.pdf.  Part 2 numbers: scripts 11 to 14.
Part 3 numbers: script 15 (cache/15_run.log).
"""

import os
import subprocess
import sys

from PIL import Image
from pptx.util import Inches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tp_math_deck import (EQ, LEFT, MUTED, ROOT, WIDTH, Deck, limitations_slide,  # noqa: E402
                               model_slides, title_slide, training_slides, vertex_eval_slides)

OUT = os.path.join(ROOT, "GDAR-TP_MPC_and_TP-GDAR.pptx")
FIG = os.path.join(ROOT, "figures")
FIG6 = os.path.join(os.path.dirname(ROOT), "Closedloop_nc", "images", "Figure6.pdf")
TOP, BOTTOM = 1.35, 7.25

# (title, figure, side notes or None); the replication slide is appended only if its file exists
RESULT_SLIDES = [
    ("TP beats one GDAR on held-out trials and unseen task times", "12_tp_vs_gdar_prediction.png", None),
    ("The gain holds with more lags; hold is a large share", "12_tp_vs_gdar_operators.png", None),
    ("Rank 2, evoked left in: a switch before the go cue", "13_vertices_felix_r2.png",
     ["Held-out gain +0.190 nats/sample; 16 separate fits +0.211",
      "One vertex through hold and early plan, the other from late plan through move and early post",
      "Vertices differ by about 8 dB near 15 Hz in the model, not yet checked against data"]),
    ("Rank 3, evoked left in: vertex 2 is an extrapolation", "13_vertices_felix_r3.png",
     ["Held-out gain +0.222 nats/sample, above the 16 separate fits",
      "The triangle encloses all bins, so corners need not be visited by the data",
      "Vertex 2 is a full spread from any bin: don't read its strong coupling as a real state"]),
    ("Rank 2, evoked removed: the switch returns only part way", "13_vertices_induced_r2.png",
     ["Held-out gain +0.248 nats/sample; 16 separate fits +0.392",
      "Same switch in late plan, about half way back after touch",
      "Hold vertex is slower: spectral radius 0.947 vs 0.929"]),
    ("Rank 3, evoked removed: plan 1/4 is an unexplained jump", "13_vertices_induced_r3.png",
     ["Held-out gain +0.323 nats/sample; separate fits still win at +0.392",
      "Weights jump right after target onset, and that bin becomes a vertex",
      "This could be left-over evoked response; not investigated yet"]),
]



def figure6_crops():
    """Figure 6 at 300 dpi, split into the constant-target and sine rows, without panel m."""
    png = os.path.join(EQ, "figure6")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", FIG6, png], check=True)
    im = Image.open(png + ".png").convert("RGB")
    W, H = im.size
    out = {}
    for name, box in (("diagram", (0.0, 0.0, 0.365, 0.19)), ("constant", (0.0, 0.205, 1.0, 0.725)),
                      ("sine", (0.0, 0.745, 0.80, 1.0))):
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
    return pw


def mpc_slides(d):
    crops = figure6_crops()

    s = d.slide("The MPC: predict, plan the laser, apply the first step")
    d.eq(s, r"$\min_{u}\ \sum_{h}\left(\hat{y}_{t+h}-r\right)^{2}+\lambda\sum_{h}\left(\Delta u_{t+h}\right)^{2}$", 1.45, fs=24)
    d.text(s, ["Every frame: plan the laser over the next 1 s, apply the first command, replan",
               "Model: a VARX on the past second of the controlled site, 100 cortical sites and the laser",
               "Held-out 200 ms prediction R²: VARX 0.41, site + laser 0.37, last value 0.10",
               "A replay, not a live experiment: a VARX of the whole session stands in for cortex, driven by each "
               "trial's recorded residuals, and the MPC's own model never saw closed-loop trials"],
           2.45, 3.9, w=6.0, size=18)
    fit_picture(s, crops["diagram"], 6.9, 5.85, top=2.3, bottom=5.6)

    s = d.slide("Constant target: MPC lowers the error PI leaves")
    fit_picture(s, crops["constant"], LEFT, WIDTH, top=1.3, bottom=6.1)
    d.text(s, ["Median RMSE, %ΔF/F: recorded PI 1.46, MPC 0.88, perfect prediction 0.27 (Mouse 1e, steady-state window 1-3 s)",
               "Lower than PI on 107 of 108 trials; across-trial variance down 37%"], 6.15, 1.0, size=16)

    s = d.slide("Sinusoidal target: a smaller gain over feedback with preview")
    fit_picture(s, crops["sine"], LEFT + 0.1, WIDTH - 0.2, top=1.35, bottom=4.75)
    d.table(s, [["Median RMSE, %ΔF/F", ""], ["Recorded CL+p", "3.00"], ["MPC", "2.67"], ["Perfect prediction", "2.35"]],
            4.95, [2.75, 0.95], row_h=0.42, size=14)
    d.text(s, ["Lower than CL+p on 56 of 57 trials",
               "Across-trial variance 0.71 times that of CL+p",
               "Mouse 3b, 4 s window; open loop lags by 71°, CL+p and MPC stay near zero"],
           4.9, 2.1, x=4.75, w=8.0, size=16)

    s = d.slide("What the MPC replay says, and how far it goes")
    d.text(s, ["Predicting activity lowers the error and the variability that feedback leaves",
               "Constant target: error down about 40% with a fitted model, about 80% with perfect prediction "
               "(ratios of medians; the median per-trial MPC/PI ratio is 0.68)",
               "Most remaining error is prediction error, except on the worst quarter of trials (n = 27): with perfect prediction "
               "their median RMSE stays at 1.97 (0.21 for the rest), and 89% of that squared error occurs with the command at a limit",
               "Descriptive simulation on recorded data, not a live test and no statistical tests",
               "One session per target type (Mouse 1e, Mouse 3b)"], 1.5, 5.2, size=20)


def figure_slide(d, title, name, notes):
    s = d.slide(title)
    path = os.path.join(FIG, name)
    if notes is None:
        fit_picture(s, path, LEFT, WIDTH)
        return
    pw = fit_picture(s, path, LEFT, 8.5)
    x = LEFT + pw + 0.2
    d.text(s, notes, 1.6, 5.0, x=x, w=13.333 - x - 0.45, size=16)


def hosvd_slide(d):
    s = d.slide("HOSVD vs the joint fit: same vectors, different metric")
    y = d.eq(s, r"$M=[\,\mathrm{vec}\,\hat{\Theta}_1;\ \dots;\ \mathrm{vec}\,\hat{\Theta}_P\,]=U\,S\,V^{\top},\qquad W=U_r,\quad \Theta_j=S_{jj}\,V_j$", 1.45, fs=24)
    y = d.eq(s, r"$\mathrm{HOSVD\ minimises:}\quad \sum_g\ ||\,\Theta(p_g)-\hat{\Theta}_g\,||_F^{2}$", y + 0.2, fs=24)
    y = d.eq(s, r"$\mathrm{The\ data\ want:}\quad \sum_g\ ||\,(\Theta(p_g)-\hat{\Theta}_g)\,G_g^{1/2}\,||_F^{2}$", y + 0.1, fs=24)
    d.text(s, ["Felix's GDAR has no HOSVD: it fits one operator. HOSVD comes from TP model transformation, and was our first route",
               "Each 22 × 352 operator is flattened along p (M is 16 × 7,744) and reshaped back, so lags, channels and support survive",
               "The SVD treats all entries alike, so rank goes to poorly determined coefficients",
               "Real OS-Pulse, held-out trials: HOSVD -45 to -7.5 nats/sample, joint fit -0.11 to +0.52",
               "HOSVD was not run on IOS Reach; these fits use the joint route"], y + 0.2, 3.0, size=17)


def replication_slides(d):
    figure_slide(d, "Other days: timing repeats, edge maps mostly don't", "14_days_comparison.png", None)
    s = d.slide("Same gain on other days, different edges")
    d.table(s, [["Hydrogen, day", "Trials", "Rank 2", "Separate fits", "Edges, r with 11-12"],
                ["11-12 no stim", "320", "+0.190 / +0.248", "+0.211 / +0.392", "reference"],
                ["08-23 no stim", "404", "+0.221 / +0.266", "+0.278 / +0.384", "0.15 / 0.47 (16 edges)"],
                ["11-11 no light", "303", "+0.193 / +0.277", "+0.210 / +0.389", "0.12 / 0.15 (40 edges)"],
                ["11-11 light", "307", "+0.242 / +0.400", "+0.292 / +0.610", "0.21 / 0.13 (40 edges)"]],
            1.4, [2.5, 1.0, 2.55, 2.55, 3.3], row_h=0.44, size=15)
    d.text(s, "Held-out gain over GDAR, nats/sample. Each cell: evoked left in / evoked removed.", 3.65, 0.4, size=14,
           bullet=False, color=MUTED)
    d.text(s, ["The gain and the timing of the weights repeat across days",
               "Move-minus-rest edge maps mostly do not repeat, so don't read one day's map as a fixed anatomy of the switch",
               "Weights run 0 to 1 by the choice of extreme bins, so overlap in range is partly by construction",
               "Lit trials: the model has no input term, so these vertices can absorb the light (the jump in panel D)"],
           4.15, 3.0, size=18)


def nine_node_slides(d):
    title_slide(d, "Part 3: a control example",
                "A sparse 9-node graph learned with TP-GDAR, an input at each node, one polytopic gain")

    s = d.slide("The example: 9 nodes, known truth, an input at each node")
    y = d.eq(s, r"$x(t+1)=\sum_{k=0}^{1} A_k(p)\,x(t-k)+B\,u(t)+e(t),\qquad B=\mathrm{diag}(b_1,\dots,b_9)$", 1.45, fs=24)
    y = d.eq(s, r"$\Theta(p)=[\,A_0(p)\ \ A_1(p)\ \ B\,]=(1-p)\,\Theta_0+p\,\Theta_1$", y + 0.15, fs=24)
    d.text(s, ["3 × 3 grid, 12 edges, 2 lags, symmetric: 51 numbers per operator, of which 9 are input gains",
               "Data: 30 trials of 200 samples at each of 7 values of p, with random input at every node",
               "20 random plants in each of two cases. One unstable: spectral radius 0.90 at p = 0, 1.03 at p = 1. Two unstable: 1.03 at both",
               "The second case was added after seeing the first, as a stress test",
               "Instability is put in by design so a controller has something to fix. This says nothing about cortex"],
           y + 0.3, 3.4, size=19)

    figure_slide(d, "TP-GDAR recovers the operators and the input gains", "15_nine_node_learning.png", None)

    s = d.slide("Polytopic control: one gain, checked at the vertices")
    y = d.eq(s, r"$u=F\,\xi,\quad \xi(t)=[\,x(t);\ x(t-1)\,],\qquad \xi(t+1)=\left(\sum_v\,\lambda_v(p)\,A_v+B_cF\right)\xi(t)$", 1.45, fs=24)
    y = d.eq(s, r"$(A_v+B_cF)^{\top}\,P\,(A_v+B_cF)\ \prec\ \alpha^{2}P\qquad \mathrm{at\ every\ vertex}\ v$", y + 0.15, fs=24)
    y = d.eq(s, r"$\mathrm{Solved\ as\ an\ LMI\ in}\ \ Q=P^{-1},\ Y=FQ:\quad (A_vQ+B_cY)^{\top}\,(\alpha Q)^{-1}\,(A_vQ+B_cY)\ \prec\ \alpha Q$", y + 0.15, fs=22)
    d.text(s, ["If it holds at the vertices it holds at every mix of them: every p, and p moving at any speed",
               "The gain does not need to know p, only the fitted vertices and B",
               "α sets a guaranteed decay rate. At α = 1 the fitted certificate held on the true plant in 0 of 20 plants, because the "
               "solution sits on the LMI's boundary. At α = 0.9 it held in 20 of 20, in both cases",
               "The loop is closed on the true plant, never on the fitted model"], y + 0.3, 3.3, size=18)

    figure_slide(d, "Both gains stabilise every plant", "15_nine_node_control.png", None)

    s = d.slide("Here the vertices matter more than the LMI")
    d.table(s, [["Median over 20 plants", "Stable at all p", "Worst true ρ", "rms x", "rms input"],
                ["One unstable: open loop", "0 / 20", "1.030", "9.76", "0"],
                ["Pooled LQR, R = 1", "20 / 20", "0.534", "0.311", "0.126"],
                ["Polytopic, α = 0.9", "20 / 20", "0.501", "0.307", "0.171"],
                ["Two unstable: open loop", "0 / 20", "1.030", "14.2", "0"],
                ["Pooled LQR, R = 1", "20 / 20", "0.567", "0.314", "0.169"],
                ["Polytopic, α = 0.9", "20 / 20", "0.506", "0.308", "0.184"]],
            1.35, [4.3, 1.9, 1.9, 1.3, 1.5], row_h=0.36, size=14)
    d.text(s, ["Pooled LQR stabilises every plant too: with an input at every node the gain can nearly cancel the mean operator",
               "With the LMI set to minimise the gain (panels C, F), the same worst-case ρ takes less input: 15 to 35% less with two "
               "unstable ends; with one, even down to ρ = 0.8 and 10 to 17% less below that",
               "Noise suppression, rms x per unit of input, is about even",
               "The fitted vertices can certify either gain: pooled LQR passes at rates 0.9 and 0.8 (19 or 20 of 20 on the true plant). "
               "A rate-0.6 certificate exists for it in only 10 and 0 of 20; the LMI reaches 0.6 in all 40",
               "Synthetic, linear, full state, no delay or actuator limits. A few inputs on real data is the real test"],
           3.95, 3.4, size=16)


def build():
    d = Deck()
    title_slide(d, "MPC replay and TP-GDAR",
                "Part 1: model-predictive control, brain paper\n"
                "Part 2: TP-GDAR vs Felix Schwock's GDAR, Hydrogen, IOS Reach\n"
                "Part 3: polytopic control on a synthetic 9-node example")
    mpc_slides(d)
    title_slide(d, "Part 2: TP-GDAR", "The model, how it is trained, and what it finds on IOS Reach")
    model_slides(d)
    training_slides(d)
    hosvd_slide(d)
    vertex_eval_slides(d)
    for title, name, notes in RESULT_SLIDES:
        figure_slide(d, title, name, notes)
    replication_slides(d)
    limitations_slide(d)
    nine_node_slides(d)
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
