r"""Explainer version of TP-GDAR_stable_vertices.pptx for Azadeh (23 November 2026): results and parameters first, few
equations, and a plain sentence under each one.  Aditya asked for it: no "since 5 November" slide, no subtitle on the title
slide, and a simple explainer for every equation.

Run:  .venv\Scripts\python scripts/make_deck_figures_hull.py       (figures, from the caches of scripts 13 and 18)
      .venv\Scripts\python scripts/make_azadeh_explainer_deck.py
Writes TP-GDAR_for_Azadeh.pptx in the repo.  TP-GDAR_stable_vertices.pptx is not touched.

Numbers: tables read the caches (cache/13_vertices_H_2021-11-12.pkl, cache/14_days.pkl); typed numbers carry a src comment
(cache/13_run.log, 18_run.log, the figure script's printout, the segment files for monkey L).  Slide text is Claude's draft and
every slide's notes say so.
"""

import os
import pickle
import sys

from pptx.enum.text import PP_ALIGN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the caches pickle gdar_tp objects
from deck_latex import add_latex  # noqa: E402
from make_mpc_tp_deck import edge_key, fit_picture, native  # noqa: E402
from make_tp_math_deck import LEFT, MUTED, ROOT, TINT, WIDTH, Deck, title_slide  # noqa: E402

OUT = os.path.join(ROOT, "TP-GDAR_for_Azadeh.pptx")
DECK = os.path.join(ROOT, "figures", "deck")
DRAFT = "Draft: slide text written by Claude, not yet checked by Aditya."
C13 = os.path.join(ROOT, "cache", "13_vertices_H_2021-11-12.pkl")
C14 = os.path.join(ROOT, "cache", "14_days.pkl")
SMALL_NODES = {"node_radius": 0.07}


def notes(s, text):
    s.notes_slide.notes_text_frame.text = DRAFT + ("\n\n" + text if text else "")


def caption(d, s, text, y=6.62):
    d.text(s, text, y, 0.6, size=17, bullet=False, color=MUTED)


def table(d, s, rows, y, widths, size=16, row_h=0.44, left_cols=1):
    d.table(s, rows, y, widths, row_h=row_h, size=size)
    t = s.shapes[-1].table
    for i in range(len(rows)):
        for j in range(left_cols):
            for para in t.cell(i, j).text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT


def explained(d, s, rows, y, x=LEFT, label_w=2.2, size=24, gap=0.3, width=WIDTH):
    """Rows of a grey label, an equation, and one plain sentence under the equation."""
    for label, tex, plain in rows:
        d.text(s, label, y + 0.12, 0.5, x=x, w=label_w, size=18, bullet=False, color=MUTED)
        below = add_latex(s, tex, x + label_w, y, size=size, max_w=width - label_w - 0.1)
        d.text(s, plain, below + 0.02, 0.8, x=x + label_w, w=width - label_w, size=18, bullet=False, color=MUTED)
        y = below + 0.02 + 0.33 * (1 + len(plain) // 105) + gap
    return y


def load():
    with open(C13, "rb") as fh:
        c13 = pickle.load(fh)
    with open(C14, "rb") as fh:
        c14 = pickle.load(fh)
    return c13, c14


# ----------------------------------------------------------------------------------------------- the model
def title(d):
    title_slide(d, "TP-GDAR on IOS Reach: four stable network states", " ")
    s = d.p.slides[-1]
    for ph in list(s.placeholders):        # no subtitle (Aditya, 23 November)
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    notes(s, "Monkey H, 2021-11-12, no stimulation. 23 November 2026.")


def model_slide(d):
    s = d.slide("The model in three lines")
    explained(d, s, [
        ("One network", r"x(t+1)=\sum_{k=1}^{16}A_k\,x(t-k+1)+\text{noise}",
         "Each channel's next value is a weighted sum of the last 48 ms on its own channel and its graph neighbours. "
         "The weights are the network (Felix's GDAR)."),
        ("A changing network", r"A(p)=\sum_{j=1}^{r}w_j(p)\,A_j",
         "Across the trial the network changes. We write it as a mix of r fixed networks, with mixing weights w that "
         "depend on where we are in the trial, p."),
        ("A blend of states", r"A(p)=\sum_{v=1}^{r+1}\lambda_v(p)\,A_v,\qquad \lambda_v\geq 0,\quad \sum_v\lambda_v=1",
         "The same model, rewritten exactly: at every point of the trial the network is a blend of r + 1 states, in "
         "proportions that add up to 1. This is the form a controller needs."),
    ], 1.35)
    notes(s, "x(t): the 22 channels at sample t, 333 Hz. A_k: 22 x 22, symmetric, nonzero only on the diagonal and the 50 graph "
             "edges. Each A_j and each state A_v is a complete 16-lag network. MATH §4, §5, §9.6.")


def bins_slide(d):
    s = d.slide("The trial clock: 16 bins, stretched to each trial")
    x0, wbin, top = LEFT + 0.1, (WIDTH - 0.2) / 16, 2.0
    for ph, name in enumerate(["hold", "plan", "move", "post"]):
        d.label(s, name, x0 + ph * 4 * wbin, top, 4 * wbin, 0.7, fill=TINT, size=24, bold=True)
        for b in range(4):
            d.label(s, f"{b + 1}/4", x0 + (ph * 4 + b) * wbin, top + 0.7, wbin, 0.55, size=16)
    for i, event in enumerate(["−400 ms", "target on", "go cue", "touch", "touch + 500 ms"]):
        cx = x0 + i * 4 * wbin
        tx = min(max(cx - 1.2, x0 - 0.1), x0 + 16 * wbin + 0.1 - 2.4)
        box = d.text(s, event, top + 1.35, 0.4, x=tx, w=2.4, size=18, bullet=False, color=MUTED)
        box.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT if i == 4 else PP_ALIGN.CENTER
    d.text(s, ["Each phase is cut into 4 equal parts of its own length on that trial, so bins line up across trials",
               "p, the position in the trial, is the bin. Go and touch are known only afterwards"], 4.5, 1.2, size=22)
    notes(s, "MATH §2.3. Hold is -400 to 0 ms, plan from target to go, move from go to touch, post 500 ms after touch.")


def parameters_slide(d):
    s = d.slide("Parameters")
    rows = [["Parameter", "Value", "What it means"],
            ["Data", "monkey H, 2021-11-12, 320 trials", "reach task, no stimulation"],
            ["Channels, edges", "22, 50", "an edge joins two electrodes at most 1.5 apart on the array layout"],
            ["Sampling, history", "333 Hz, 16 samples", "each step predicts 3 ms ahead from the last 48 ms"],
            ["Numbers per network", "1,152", "16 lags × (22 self + 50 edge weights)"],
            ["Trial bins", "16", "4 phases × 4 quarters"],
            ["Rank r", "2 or 3", "fixed networks mixed across the trial; r + 1 = 3 or 4 states"],
            ["Testing", "5 folds over trials", "fit on 4/5 of trials, score the held-out 1/5"],
            ["Score", "nats per sample", "held-out prediction, gain over one fixed network (GDAR)"],
            ["State rule", "all states stable", "smallest blend set whose states all have spectral radius < 1"]]
    table(d, s, rows, 1.35, [3.0, 3.4, 5.8], size=16, row_h=0.46, left_cols=3)
    notes(s, "Channel B14 is dropped (22 of 23). Graph: electrode positions from Felix's 32ch_mmad file, edge if the distance is "
             "at most 1.5 (MATH §1). 1,152 free numbers per network and rank 3's 3,495: 5 November deck. Folds are stratified by "
             "reach direction (MATH §8). State rule: MATH §9.6, approved 2026-11-22.")


# ----------------------------------------------------------------------------------------------- results
def gain_slide(d):
    s = d.slide("Result 1: a changing network predicts unseen trials better")
    table(d, s, [["Gain over one fixed network (nats/sample)", "evoked left in", "evoked removed"],
                 ["Rank 2 (3 states)", "+0.190 (0.009)", "+0.248 (0.012)"],      # src: cache/13_run.log
                 ["Rank 3 (4 states)", "+0.222 (0.009)", "+0.323 (0.013)"],      # src: cache/13_run.log
                 ["16 unrelated networks, one per bin", "+0.211", "+0.392"]],    # src: cache/13_run.log
          1.4, [5.6, 2.8, 2.8], size=18, row_h=0.52)
    d.text(s, ["Higher is better: how much more likely the held-out trials are under the model, per 3 ms sample",
               "Evoked removed: the average response to target, go and touch is subtracted first, so only trial-to-trial "
               "activity is left to explain",
               "Rank 3 does as well as 16 unrelated networks with evoked left in, with far fewer numbers"], 3.85, 2.6, size=20)
    caption(d, s, "Monkey H, 2021-11-12, 320 trials, 5 folds; fold SE in brackets.")
    notes(s, "Numbers per model: rank 3 about 3,495 against 18,432 for 16 separate networks (5 November deck). With evoked removed "
             "the 16 separate networks do better (+0.392 against +0.323). Source: cache/13_run.log. MATH §6, §8.")


def states_slide(d):
    s = d.slide("Result 2: the fit is an exact blend of stable states")
    fit_picture(s, os.path.join(DECK, "hull_r2_geometry.png"), LEFT, 5.4, top=1.3, bottom=6.3)
    d.text(s, ["Each grey dot is one trial bin, placed by its fitted network (rank 2: a triangle, 3 states)",
               "The corners are the states. Each bin is a blend of the corners, and the blend rebuilds the fit with no loss",
               "Many triangles contain the dots. We keep the smallest whose corners are all stable (solid); "
               "dashed is a looser one"], 1.4, 3.3, x=6.3, w=WIDTH - 5.7, size=20)
    explained(d, s, [("Stable", r"\rho_v<1",
                      "After a push, activity in that state dies away instead of growing")],
              4.85, x=6.3, label_w=1.2, size=24, width=WIDTH - 5.7)
    notes(s, "Dots: the 16 bins' weights in the pooled data metric (MATH §9.6). Kept triangle: tight (tau = 1), rotation 30.5 degrees, "
             "corner spectral radii 0.965, 0.964, 0.968; dashed: tau = 1.5. rho_v: the largest eigenvalue modulus of state v's "
             "companion matrix. Approved by Aditya 2026-11-22. Source: figure script printout, cache/13_vertices_H_2021-11-12.pkl.")


def stability_slide(d, c13):
    s = d.slide("Requiring stable states needs a slightly larger set")
    rows = [["Rank 3, 4 states", "Size of the set", "Spectral radius of each state"]]
    old = {"felix": ("0.968", "1.012", "0.966", "0.965"),     # src: cache/13_run.log line 10
           "induced": ("0.964", "1.037", "0.943", "0.941")}   # src: cache/13_run.log line 37
    for v, lab in (("felix", "evoked left in"), ("induced", "evoked removed")):
        f = c13[v]["full"][3]
        sm, p = f["smallest"], f["poly"]
        rows.append([f"{lab}: smallest set", f"{sm['h']:.3f}", ", ".join(old[v])])
        rows.append([f"{lab}: smallest stable set", f"{p['h']:.3f}", ", ".join(f"{q:.3f}" for q in p["rho"])])
    table(d, s, rows, 1.4, [4.6, 2.4, 5.2], size=18, row_h=0.52)
    d.text(s, ["The smallest set had one state above 1, a network that would amplify. The rule picks the next smallest set "
               "where all four are below 1",
               "Prediction doesn't change: the blend still rebuilds the same fit",
               "Rank 2 needed no change: its smallest triangle was already stable"], 4.3, 2.2, size=20)
    notes(s, "Size: centre-to-face distance of the regular simplex in the data metric (MATH §9.6). Candidates: 2,000 random rotations "
             "at rank 3, tried from the smallest up. Approved by Aditya 2026-11-22.")


def weights_slide(d):
    s = d.slide("Result 3: one state for hold, one for move")
    native(s, "hull_weights_r3", y=1.3, bottom=6.5)
    caption(d, s, "Share of each state across the trial (vertex = state). Hold: state 3, 0.92 at hold 1/4. "
                  "Move: state 4, 0.72 to 0.74. States 1 and 2 share plan and late post.")   # src: figure script printout
    notes(s, "Evoked removed: state 3 hold (0.97), state 4 move (0.78 to 0.82). State numbers are arbitrary, and the two panels are "
             "separate fits. Bands: jackknife SE over 5 training splits. Source: cache/13_vertices_H_2021-11-12.pkl.")


def maps_slide(d):
    s = d.slide("What each state looks like: network and rhythm")
    native(s, "hull_maps_r3", y=1.25, bottom=6.5, panel_flags={f"Vertex {v}, ρ = {r}": SMALL_NODES for v, r in
                                                                  ((1, "0.969"), (2, "0.997"), (3, "0.969"), (4, "0.965"))})
    edge_key(s)
    caption(d, s, "Top: each state's network, summed over lags. Bottom: its rhythm relative to one fixed network. "
                  "State 2 adds a 10 Hz peak (+20.8 dB) but sits outside what the data visit.")   # src: figure script printout
    notes(s, "Evoked left in. State 2 is 1.14 data spreads from the nearest bin (post 3/4), so its map exaggerates the direction "
             "in which the bins change rather than showing a state the network reaches. State 4 (move) is lowest from 10 to 20 Hz, "
             "consistent with the beta drop during movement. MATH §11.2, §11.4.")


def days_slide(d, c13, c14):
    s = d.slide("Result 4: stable states on 14 of 16 fits, over 4 data sets")
    rows = [["Data set", "Evoked", "Rank 2: largest ρ", "Rank 3: largest ρ"]]
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
    assert (n_fits, n_stable) == (16, 14), (n_fits, n_stable)
    table(d, s, rows, 1.35, [3.2, 2.0, 3.0, 3.0], size=17)
    d.text(s, ["Not stable: the light trials of 11-11 with evoked removed. The model has no input for the light, "
               "which may be why; that isn't tested"], 1.35 + 0.44 * 9 + 0.25, 1.0, size=19)
    notes(s, "Monkey H. 11-11 is a stimulation day, split into trials with and without light. Largest spectral radius over the kept "
             "states. Source: cache/14_days.pkl, cache/13_vertices_H_2021-11-12.pkl.")


def single_slides(d):
    s = d.slide("Result 5: the blend can be read off 60 ms of one trial")
    native(s, "single_trial_r3", y=1.3, bottom=6.45)
    caption(d, s, "Left: one held-out trial, blend read from its own data (solid) and from the clock (dashed). "
                  "Right: the move state rises at the same time after target, early or late go.")
    notes(s, "Each trial's blend is the one whose states best predict that trial's last 60 ms, with the states fixed and no go or "
          "touch times used (MATH §10, approved 2026-11-05). Example: trial 247, go 399 ms, touch 823 ms. Go-time thirds: 107, "
          "106, 107 trials, mean go 302, 401, 517 ms. Evoked left in.")   # src: figure script printout

    s = d.slide("...and predicts the next sample better than the clock")
    native(s, "single_trial_gain", x=LEFT, y=1.3, w=6.2, bottom=6.5)
    d.text(s, ["Clock: blend from the trial's own go and touch times. Data: blend from the last 30, 60 or 120 ms only",
               "Go time can't be read off the blend: r = −0.02 to +0.05",           # src: cache/18_run.log
               "Touch partly can: r = +0.35 to +0.44",                               # src: cache/18_run.log
               "Caveat: a blend that adapts within a trial may gain from any slow change, not only task states"],
           1.4, 5.0, x=7.0, w=13.333 - 7.0 - 0.4, size=19)
    notes(s, "Gain over one fixed network, nats/sample, fold SE: evoked left in clock +0.1618 (0.0077), 30 ms +0.1669 (0.0080), "
             "60 ms +0.1939 (0.0079), 120 ms +0.2013 (0.0078); removed +0.2150, +0.2212, +0.2409, +0.2413. Source: cache/18_run.log.")


# ----------------------------------------------------------------------------------------------- close
def open_slide(d):
    s = d.slide("Open questions")
    d.text(s, ["Is part of the state change the evoked response, which the model has no input for? Checks running today",
               "11-11 light trials: no stable set. Add the light as an input?",
               "One monkey so far"], 1.5, 3.0, size=24)
    notes(s, "Checks of 23 November: overlapping bins, a mask for cleaning steps in noart at 0 and 900 ms, and a null test with a "
             "fixed network plus the real evoked wave (second deck). MATH §14 lists the open method questions.")


def next_slide(d):
    s = d.slide("Next: monkey L, and three more H sessions")
    table(d, s, [["Session", "Channels", "Trials", "Light trials", "Median move (ms)"],
                 ["L 2022-08-10", "21", "841", "440", "106"],
                 ["H 2021-11-16", "15", "663", "337", "266"],
                 ["H 2021-08-18", "22", "211", "0", "338"],
                 ["H 2021-07-28", "32", "90", "0", "588"],
                 ["H 2021-11-12, for reference", "23", "321", "0", "409"]],
          1.35, [4.2, 1.6, 1.4, 1.8, 2.4], size=17)   # src: segment files, read 2026-11-22
    d.text(s, ["L's other sessions have 3 to 7 channels, too few for a network. 08-10 is the one L session to fit",
               "L moves fast (median 106 ms), so its move bins hold few samples"], 1.35 + 0.44 * 6 + 0.3, 1.4, size=20)
    notes(s, "Channel counts before dropping bad channels. The run plan is docs/PLAN_OTHER_MONKEYS.md.")


def build():
    c13, c14 = load()
    d = Deck()
    title(d)
    model_slide(d)
    bins_slide(d)
    parameters_slide(d)
    gain_slide(d)
    states_slide(d)
    stability_slide(d, c13)
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
