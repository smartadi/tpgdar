r"""Second deck of 23 November 2026: the evoked response, why it matters for TP-GDAR, the noart steps, overlapping bins (MATH §2.4),
and the null test (script 20), with results.  One message per slide; details sit in the speaker notes.

Run:  .venv\Scripts\python scripts/make_deck_figures_evoked.py      (figures, from the caches of scripts 13, 18 and 20)
      .venv\Scripts\python scripts/make_evoked_overlap_deck.py
Writes TP-GDAR_evoked_overlap.pptx in the repo.  Same building blocks as TP-GDAR_stable_vertices.pptx: shapes PowerPoint for
the web can edit, equations as LaTeX pictures with their source as alt text.  Slide text is Claude's draft and every slide's notes
say so.  Numbers in text are read from the caches here, or carry a src comment (the figure script's printout, cache/20_run.log).
"""

import importlib
import os
import sys

from pptx.enum.text import PP_ALIGN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the caches pickle gdar_tp objects
from make_ios_tp_deck import flow, steps  # noqa: E402
from make_mpc_tp_deck import native  # noqa: E402
from make_tp_math_deck import MUTED, ROOT, Deck, title_slide  # noqa: E402

ev = importlib.import_module("make_deck_figures_evoked")
OUT = os.path.join(ROOT, "TP-GDAR_evoked_overlap.pptx")
DRAFT = "Draft: slide text written by Claude, not yet checked by Aditya."
SETUPS = (("", "No overlap (as before)"), ("_overlap0.5", "Overlap 0.5 bin"), ("_overlap1", "Overlap 1 bin"),
          ("_overlap2", "Overlap 2 bins"), ("_mask", "No overlap, steps masked"), ("_overlap1_mask", "Overlap 1 bin, steps masked"))


def notes(s, text):
    s.notes_slide.notes_text_frame.text = DRAFT + "\n\n" + text


def caption(d, s, text, y=6.65):
    d.text(s, text, y, 0.6, size=17, bullet=False, color=MUTED)


def table(d, s, rows, y, widths, size=16, row_h=0.44, left_cols=1):
    d.table(s, rows, y, widths, row_h=row_h, size=size)
    t = s.shapes[-1].table
    for i in range(len(rows)):
        for j in range(left_cols):
            for para in t.cell(i, j).text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT


def gain(suffix, v, r):
    return ev.gain(ev.c13(suffix), v, r)[0]


def gtxt(suffix, v, r):
    x, se = ev.gain(ev.c13(suffix), v, r)
    return f"{x:+.3f} ({se:.3f})" if r else f"{x:+.3f}"


# ----------------------------------------------------------------------------------------------- opening
def summary_slide(d):
    s = d.slide("Summary")
    rec, nul = gain("", "felix", 3), gain("_null", "felix", 3)
    d.text(s, ["Two of the three biggest peaks of the trial-mean LFP are mostly cleaning steps in noart, not in the raw LFP",
               f"A fixed network plus the real evoked wave gives a rank 3 gain of {nul:+.3f}, against {rec:+.3f} on the "
               "recordings (evoked left in). It does give hold and move vertices",
               f"Overlapping bins and the step mask change the held-out gain little: rank 3 {gain('_overlap1_mask', 'felix', 3):+.3f} "
               "with both",
               "Next: choose the overlap, and whether the mask becomes the default"], 1.5, 4.5, size=24)
    notes(s, "H 2021-11-12, 320 trials, 22 channels. Gains: held-out one-step log-likelihood over GDAR, nats per sample, 5 folds. "
             "Read from the caches of script 13: no suffix, _null, _overlap1_mask.")


def pipeline_slide(d):
    s = d.slide("The chain, and what changed today")
    flow(d, s, ["Data", "16 bins", "Evoked removal (optional)", "GDAR", "Bowls"], 1.7, shaded={0, 1})
    flow(d, s, ["Joint fit", "Held-out score", "Vertices", "Readouts", "Single trials"], 3.3, shaded=set())
    d.text(s, ["Shaded: changed today. Data: the noart steps can be masked. Bins: they can overlap",
               "New check on the whole chain: a null test with no state change"], 4.9, 1.4, size=20)
    notes(s, "MATH sections 1.1, 2.3 and 2.4, 3, 4, 6.1, 6.2 to 6.6, 8, 9.6, 11, 10. Approved: overlapping bins, step mask and "
             "null test (23 Nov), the vertex hull and its rotation rule (22 Nov), single-trial weights (5 Nov), the Phi-step "
             "numerics, GDAR's Sigma for leave-one-bin-out, the data spectrum. The rest is written up but not approved (MATH §13).")


def bowl_slide(d):
    s = d.slide("Bins and bowls: each bin scores every possible operator")
    y = steps(d, s, [("One number", r"E_g(a)=\sum_{t\in g}\bigl(x(t+1)-a\,x(t)\bigr)^{2},\qquad\text{lowest at }\hat a_g")],
              y=1.25, gap=0.85, size=24, label_w=1.9)
    native(s, "ev_bowls", y=y + 0.05, bottom=6.5)
    caption(d, s, "One coefficient, three bins. Right: with a 1-bin overlap the bins share samples, so their bottoms move together.")
    notes(s, "With all 1,152 numbers the bowl is L_g(phi) = ||phi - phi_g||^2 in bin g's data metric H_g: bottom = the bin's own "
             "best operator, steepness = how firmly its data pin each direction down. The joint fit puts the 16 operators as low "
             "as possible in their own bowls, summed. Plotted: 0.5 (a - a_g)^2 H_g[i,i] / n_g for the lag-1 self term of the "
             "channel whose hold 4/4 and plan 1/4 bottoms are furthest apart. With the overlap, hold 4/4's bottom moves from 1.88 "
             "to 1.95, close to plan 1/4's 1.97: the samples just after target pull hard.")   # src: figure script printout, bowls


# ----------------------------------------------------------------------------------------------- the evoked response
def wave_slide(d):
    s = d.slide("The evoked response: a wave that repeats on every trial")
    native(s, "ev_wave", y=1.25, bottom=6.45)
    caption(d, s, "Dashed: target, median go, median touch. Peaks at 19 and 916 ms: 29.5 and 17.3 µV in noart, 7.8 and 3.1 µV "
                  "in the raw LFP. The one at 166 ms is in both.")  # src: figure script printout, wave
    notes(s, "Left: trial mean of the three channels with the largest trial mean (solid) and the fitted wave (dashed, MATH §3). "
             "The channels look alike, so the wave is common to the array. Right: RMS over channels of the trial mean, peaks in "
             "0-60, 100-250 and 880-960 ms: noart 29.5 (19 ms), 25.0 (166 ms), 17.3 (916 ms); raw lfp prepared the same way 7.8 "
             "(16 ms), 23.6 (166 ms), 3.1 (889 ms). So the early and late peaks are mostly the noart steps, and the 166 ms one "
             "is in the raw signal too. Evoked removed leaves 7.5 uV at 13 ms (a 9 ms kernel can't follow a 1 ms step); its median "
             "is 0.39 uV against 2.32. The null data keep the wave. Median go 400 ms, median touch 785 ms.")   # src: figure script printout, wave


def why_slide(d):
    s = d.slide("Why it matters: the bins are cut where the wave is")
    y = steps(d, s, [("Data", r"x(t+1)=\sum_{k}A_k\,x(t-k+1)+B\,u(t)+e(t)"),
                     ("Model", r"x(t+1)=\sum_{k}A_k(p)\,x(t-k+1)+e(t)")],
              y=1.3, gap=0.9, size=24, label_w=1.6)
    d.text(s, ["The model has no input u, so the operators have to make the wave",
               "The bins start at target, go and touch, where the wave is",
               "So a gain over GDAR could be a network that changes, or a fixed network with an input",
               "Evoked removed: fit the wave with event kernels and subtract it first"], y + 0.3, 3.2, size=22)
    notes(s, "u(t): target, go, touch. A wave that reaches many channels at once makes channel j's past predict channel i's future, "
             "which looks like coupling in an edge map. Held-out scoring can't tell: the wave repeats in test trials too. Evoked "
             "removed (MATH §3): one kernel per event and reach direction in 9 ms bins, fitted on training trials and subtracted. "
             "Costs: part of the network's own response goes too; the removed data carry each trial's go and touch times; no kernel "
             "for the light. MATH §3 is not yet approved.")


def steps_slide(d):
    s = d.slide("noart has steps at 0 and 900 ms that the raw LFP doesn't")
    native(s, "ev_steps", y=1.25, bottom=6.45)
    caption(d, s, "The mask drops every target that touches a step: 44 of 667 target times.")   # src: figure script printout, steps
    notes(s, "Median change per ms at 1 kHz: 23.5 uV at 0->1 ms, 7.9 at 10->11, 13.9 at 900->901, 8.6 at 910->911, against a "
             "typical 2.5 uV (raw lfp 2.4, no step). Dropped targets: -2 to 61 and 898 to 961 ms, i.e. a target whose sample or "
             "any of its 16 lags falls in -3..15 or 897..915 ms (dashed lines). Where in the cleaning the steps come from isn't "
             "known (script 12c).")


# ----------------------------------------------------------------------------------------------- the changes
def overlap_slide(d):
    s = d.slide("Overlapping bins share samples with their neighbours")
    y = steps(d, s, [("Weight", r"a_g(\tau)=\text{share of a window of }\Delta\text{ bins around }\tau\text{ that falls in bin }g")],
              y=1.25, gap=0.85, size=24, label_w=1.6)
    native(s, "ev_weights_a", y=y + 0.05, bottom=6.5)
    caption(d, s, "Dashed: target onset. Weights sum to 1, so each sample counts once. Held-out samples are still scored by their own bin.")
    notes(s, "MATH §2.4, approved 23 November. tau: warped time in bins (4 per phase). a_g(tau) = |[tau - D/2, tau + D/2] ∩ [g, g + 1)| "
             "/ |[tau - D/2, tau + D/2] ∩ [0, 16)|. D = 0 is today's bins. Bin g reaches D/2 bins into each neighbour, across "
             "target, go and touch. The bowls take each sample's terms with these weights; the joint fit, per-bin fits and hull "
             "run unchanged. Runs at D = 0, 0.5, 1, 2.")


def null_design_slide(d):
    s = d.slide("The null test: one fixed network plus each trial's real wave")
    y = steps(d, s, [("Null data", r"\tilde x_n(t)=\underbrace{y_n(t)}_{\text{fixed GDAR, noise}}+\underbrace{m_n(t)}_{\text{trial }n\text{'s wave}}")],
              y=1.3, gap=1.2, size=26, label_w=1.9)
    d.text(s, ["No state change anywhere, so any TP-GDAR gain comes from the wave",
               "Fitted exactly like the recordings"], y + 0.2, 1.2, size=22)
    rows = [["Check", "Value"],
            ["Fixed network stable (spectral radius)", "0.927"],                         # src: cache/20_run.log
            ["Trial mean of null data against the wave (correlation)", "0.972"],
            ["Variance: recordings / null data / wave alone", "955 / 773 / 53 µV²"]]
    table(d, s, rows, y + 1.6, [8.0, 3.0], size=17)
    notes(s, "y: GDAR fitted to all 320 trials with the wave removed, driven by Gaussian noise with its Sigma, 2,000 burn-in samples, "
             "seed 0. m_n: trial n's fitted wave (recordings less evoked removed), with its own go, touch and direction. Each trial "
             "mean-centred as the recordings are. A second set, y alone, checks that no wave gives no gain. Script 20, approved "
             "23 November.")


# ----------------------------------------------------------------------------------------------- results
def gain_slide(d):
    s = d.slide("Held-out gain: overlap and the mask change it little")
    native(s, "ev_gain", y=1.25, bottom=6.45)
    caption(d, s, "Solid: overlap of 0 to 2 bins. Dashed: the same with the steps masked.")
    rows = "; ".join(f"{name}: left in {gtxt(suf, 'felix', 2)} / {gtxt(suf, 'felix', 3)}, removed {gtxt(suf, 'induced', 2)} / "
                     f"{gtxt(suf, 'induced', 3)}, per bin {gtxt(suf, 'felix', None)} / {gtxt(suf, 'induced', None)}"
                     for suf, name in SETUPS)
    notes(s, "Gain over GDAR, nats per sample, rank 2 / rank 3 (fold SE): " + rows + ". Source: caches of script 13.")


def weights_slide(d):
    s = d.slide("Rank 3 weights with overlap and the mask")
    native(s, "ev_weights", y=1.2, bottom=6.5)
    caption(d, s, "Evoked left in. Colours by where each vertex peaks.")
    notes(s, "Vertex numbers are arbitrary in each fit, so the vertices are coloured by role: most weight at hold 1/4, then most "
             "mean weight over move, then most at post 3/4, then the last. Bands: jackknife SE over 5 training splits. The figure "
             "script prints each vertex's spectral radius, distance from the data and weights.")


def null_gain_slide(d):
    s = d.slide("Null test: how much of the gain the wave alone gives")
    native(s, "ev_null_gain", y=1.25, bottom=6.4)
    caption(d, s, f"Left in: the wave alone gives {gain('_null', 'felix', 3):+.3f} of the recordings' {gain('', 'felix', 3):+.3f}. "
                  f"Removed: {gain('_nullnowave', 'induced', 3):+.3f} on the null even with no wave, which points to the removal "
                  "step itself.", y=6.5)
    notes(s, "Rank 3, gain over GDAR on each data set's own held-out trials, nats per sample. Recordings: " +
          "; ".join(f"{name} {gtxt(suf, 'felix', 3)} / {gtxt(suf, 'induced', 3)}" for suf, name in
                    (("", "no overlap"), ("_overlap1_mask", "overlap 1, masked"))) +
          ". Null: " + "; ".join(f"{name} {gtxt(suf, 'felix', 3)} / {gtxt(suf, 'induced', 3)}" for suf, name in
                                (("_null", "no overlap"), ("_nullnowave", "no overlap, no wave"),
                                 ("_overlap1_mask_null", "overlap 1, masked"))) + " (left in / removed).")


def null_weights_slide(d):
    s = d.slide("Null test: the weights the wave alone produces")
    native(s, "ev_null_weights", y=1.2, bottom=6.4)
    h_rec, h_nul = (ev.c13(suf)["felix"]["full"][3]["poly"]["h"] for suf in ("", "_null"))
    caption(d, s, "Evoked left in. The wave alone also gives a hold vertex and a move vertex, on a smaller simplex (size "
                  f"{h_nul:.3f} against {h_rec:.3f}). The weight plot by itself doesn't show a changing network; the size of the gain is the better test.",
            y=6.45)
    notes(s, "Null data: one fixed network plus each trial's fitted wave. Coloured by role as before; roles are given in order (hold, "
             "move, post, the rest), so in the null panels the vertex labelled 'peaks in post' peaks at plan 1/4, just after target "
             "(1.00 without overlap). Size: the simplex's centre-to-face distance in the data metric (MATH §9.6).")   # src: figure script printout, null weights


def takeaways_slide(d):
    s = d.slide("What this says, and what to decide")
    S0, S1, SM = ev.c18("")["felix"], ev.c18("_overlap1_mask")["felix"], ev.c18("_mask")["felix"]
    d.text(s, [f"Most of the gain is not the wave: a fixed network plus the wave gives {gain('_null', 'felix', 3):+.3f} of "
               f"{gain('', 'felix', 3):+.3f}",
               "But the wave alone also makes a hold and a move vertex, so weight plots alone don't show a changing network",
               f"Masking the steps raises the single-trial gain from the last 30 ms: {S0['gain'][10]:+.3f} to {SM['gain'][10]:+.3f} "
               "(fewer samples scored)",
               f"With overlap and the mask the single-trial fits agree across folds: lowest match {min(S1['match']):.2f}, "
               f"against {min(S0['match']):.2f} before",
               "Decide: the overlap window (0.5, 1 or 2 bins), and whether the step mask becomes the default",
               "Then: the same checks on monkey L"], 1.4, 5.0, size=22)
    notes(s, "Single-trial gains over GDAR, evoked left in, clock / past 60 ms: " +
          "; ".join(f"{name} {ev.c18(suf)['felix']['gain']['clock']:+.3f} / {ev.c18(suf)['felix']['gain'][20]:+.3f}"
                    for suf, name in SETUPS) + ". Fold match: mean correlation of a fold fit's 16-bin weights with the full-data "
          "fit's. Masked runs score 153,212 samples against 167,932 (cache/18_run_mask.log): targets next to a step are dropped, "
          "and a window with no unmasked sample isn't scored. Source: caches of script 18.")


def build():
    d = Deck()
    title_slide(d, "TP-GDAR: the evoked response, overlapping bins and a null test",
                "Monkey H, 2021-11-12. 23 November 2026")
    notes(d.p.slides[0], "")
    for make in (summary_slide, pipeline_slide, bowl_slide, wave_slide, why_slide, steps_slide, overlap_slide,
                 null_design_slide, gain_slide, weights_slide, null_gain_slide, null_weights_slide, takeaways_slide):
        try:
            make(d)
        except FileNotFoundError as err:          # a run still in progress
            print("skipped", make.__name__, "-", os.path.basename(err.filename or str(err)))
    for slide in d.p.slides:        # connectors inherit a soft shadow from the default theme, which blurs thin lines
        for ref in slide._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef"):
            ref.set("idx", "0")
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
