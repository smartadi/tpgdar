"""Plain deck on the TP-GDAR model and how it is trained, equations rendered as images.

Run:  .venv\\Scripts\\python scripts/make_tp_math_deck.py
Writes GDAR-TP_model_training.pptx.  Numbers are from scripts 12 and 13 (H 2021-11-12).
"""

import hashlib
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ = os.path.join(ROOT, "figures", "eq_tp")
OUT = os.path.join(ROOT, "GDAR-TP_model_training.pptx")
DPI = 300
INK = RGBColor(0, 0, 0)
MUTED = RGBColor(0x55, 0x55, 0x55)
TINT = RGBColor(0xEE, 0xEE, 0xEE)
LEFT = 0.6
WIDTH = 12.13

plt.rcParams["mathtext.fontset"] = "cm"
os.makedirs(EQ, exist_ok=True)


def render(tex, fs=26):
    tex = re.sub(r"(\\sum_(\{[^}]*\}|\w)(\^\{[^}]*\})?)", r"\1\\,", tex).replace(r"^{\top}", r"^{\mathrm{T}}")
    path = os.path.join(EQ, f"{hashlib.md5(f'{tex}{fs}'.encode()).hexdigest()[:12]}.png")
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=fs, color="black", ha="left", va="bottom")
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


class Deck:
    def __init__(self):
        self.p = Presentation()
        self.p.slide_width, self.p.slide_height = Inches(13.333), Inches(7.5)
        for layout in self.p.slide_layouts:
            for ph in list(layout.placeholders):
                if ph.placeholder_format.type in (13, 15, 16):      # slide number, footer, date
                    ph._element.getparent().remove(ph._element)
        self.title_only = self.p.slide_layouts[5]
        self.title_slide = self.p.slide_layouts[0]
        for ph in self.title_only.placeholders:
            ph.left, ph.top, ph.width, ph.height = Inches(LEFT), Inches(0.4), Inches(WIDTH), Inches(0.9)
        for ph in self.title_slide.placeholders:
            top = 2.3 if ph.placeholder_format.idx == 0 else 3.8
            ph.left, ph.top, ph.width, ph.height = Inches(LEFT), Inches(top), Inches(WIDTH), Inches(1.4)

    def slide(self, title):
        s = self.p.slides.add_slide(self.title_only)
        s.shapes.title.text = title
        tf = s.shapes.title.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for para in tf.paragraphs:
            para.alignment = PP_ALIGN.LEFT
            for r in para.runs:
                r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(32), True, "Calibri", INK
        return s

    def eq(self, s, tex, y, x=LEFT + 0.2, fs=26, scale=1.0):
        path = render(tex, fs)
        w, h = Image.open(path).size
        pic = s.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w / DPI * scale))
        pic._element.nvPicPr.cNvPr.set("descr", tex)
        return y + h / DPI * scale

    def text(self, s, items, y, h, x=LEFT, w=WIDTH, size=18, bullet=True, color=INK):
        box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        for i, item in enumerate([items] if isinstance(items, str) else items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = para.add_run()
            run.text = item
            run.font.size, run.font.name, run.font.color.rgb = Pt(size), "Calibri", color
            para.space_after = Pt(7)
            if bullet:
                pPr = para._p.get_or_add_pPr()
                pPr.set("marL", "228600")
                pPr.set("indent", "-228600")
                bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
                pPr.append(bu)
        return box

    def table(self, s, rows, y, widths, x=LEFT + 0.1, row_h=0.46, size=16):
        t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(widths)),
                               Inches(row_h * len(rows))).table
        tblPr = t._tbl.tblPr
        for child in list(tblPr):
            tblPr.remove(child)
        tblPr.set("bandRow", "0")
        for j, wd in enumerate(widths):
            t.columns[j].width = Inches(wd)
        for i, row in enumerate(rows):
            t.rows[i].height = Inches(row_h)
            for j, val in enumerate(row):
                c = t.cell(i, j)
                c.text = val
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                c.fill.solid()
                c.fill.fore_color.rgb = TINT if i == 0 else RGBColor(255, 255, 255)
                for para in c.text_frame.paragraphs:
                    para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
                    for r in para.runs:
                        r.font.size, r.font.name, r.font.color.rgb = Pt(size), "Calibri", INK
                        r.font.bold = i == 0

    def label(self, s, text, x, y, w, h, fill=None, size=14, bold=False):
        shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.shadow.inherit = False
        shp.line.color.rgb = RGBColor(0x88, 0x88, 0x88)
        shp.line.width = Pt(0.75)
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill or RGBColor(255, 255, 255)
        tf = shp.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.03)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = tf.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        r = para.add_run()
        r.text = text
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), bold, "Calibri", INK


def title_slide(d, title, subtitle):
    s = d.p.slides.add_slide(d.title_slide)
    s.shapes.title.text = title
    s.placeholders[1].text = subtitle
    for para in s.shapes.title.text_frame.paragraphs:
        para.alignment = PP_ALIGN.LEFT
        for r in para.runs:
            r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(40), True, "Calibri", INK
    for para in s.placeholders[1].text_frame.paragraphs:
        para.alignment = PP_ALIGN.LEFT
        for r in para.runs:
            r.font.size, r.font.name, r.font.color.rgb = Pt(20), "Calibri", MUTED


def model_slides(d):
    s = d.slide("Felix's GDAR: one operator for the whole day")
    y = d.eq(s, r"$z(t)=\left[\,x(t);\ x(t-1);\ \dots;\ x(t-K+1)\,\right]$", 1.5)
    y = d.eq(s, r"$x(t+1)=\Theta\,z(t)+e(t),\qquad e(t)\sim\mathcal{N}(0,\Sigma),\qquad \Theta=[\,A_1\ \cdots\ A_K\,]$", y + 0.2)
    y = d.eq(s, r"$A_k=A_k^{\top},\qquad (A_k)_{ij}=0\ \ \mathrm{unless}\ \ i=j\ \ \mathrm{or}\ \ (i,j)\in E$", y + 0.2)
    y = d.eq(s, r"$\mathrm{vec}\,\Theta=R\,\phi,\qquad \dim\phi=K\,(N+E)$", y + 0.2)
    d.text(s, ["Each lag matrix is symmetric and non-zero only on graph edges and the diagonal",
               "Here N = 22 channels, E = 50 edges and K = 16 lags (48 ms), so 1,152 free numbers",
               "R is a fixed sparse matrix that copies each edge coefficient into (i, j) and (j, i)"], y + 0.3, 1.5)

    s = d.slide("The TP structure: a low-rank mix over p")
    y = d.eq(s, r"$\Theta(p)=\sum_{j=1}^{r}\,w_j(p)\,\Theta_j\qquad\Longleftrightarrow\qquad \phi(p_g)=\sum_{j=1}^{r}W_{gj}\,\phi_j$", 1.5, fs=28)
    d.text(s, ["Each Θⱼ is a complete symmetric GDAR operator with all 16 lags",
               "Only the p direction is compressed; channels and lags are not",
               "The weights W are free per bin, so nothing forces them to be smooth",
               "W and the Θⱼ can trade an invertible r × r matrix, so they are not unique: only Θ(p) is"], y + 0.3, 2.3)
    d.table(s, [["Model", "Parameters"], ["Felix's GDAR", "1,152"], ["TP-GDAR rank 2", "2,332"],
                ["TP-GDAR rank 3", "3,495"], ["16 separate fits", "18,432"]], 4.75, [4.0, 2.0], row_h=0.4)
    d.eq(s, r"$n_{\mathrm{params}}=r\,K(N+E)+Pr-r^2$", 5.45, x=7.3)

    s = d.slide("p is trial phase, warped per trial")
    x0, wbin, top = LEFT + 0.1, (WIDTH - 0.2) / 16, 1.6
    events = ["hold", "plan", "move", "post"]
    for ph, nm in enumerate(events):
        d.label(s, nm, x0 + ph * 4 * wbin, top, 4 * wbin, 0.5, fill=TINT, size=18, bold=True)
        for b in range(4):
            d.label(s, f"{b + 1}/4", x0 + (ph * 4 + b) * wbin, top + 0.5, wbin, 0.45, size=14)
    for i, ev in enumerate(["-400 ms", "target on", "go cue", "touch", "touch + 500 ms (max 1400 ms)"]):
        w = 3.0 if i == 4 else 1.8
        cx = x0 + i * 4 * wbin
        tx = min(max(cx - w / 2, x0 - 0.1), x0 + 16 * wbin + 0.1 - w)
        box = d.text(s, ev, top + 1.0, 0.4, x=tx, w=w, size=14, bullet=False, color=MUTED)
        box.text_frame.paragraphs[0].alignment = (PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT if i == 4 else PP_ALIGN.CENTER)
    y = d.eq(s, r"$g(t)=4\,\mathrm{phase}(t)+\left\lfloor\,4\ \frac{t-t_{\mathrm{start}}}{t_{\mathrm{end}}-t_{\mathrm{start}}}\,\right\rfloor,\qquad g\in\{0,\dots,15\}$", 3.3)
    d.text(s, ["Each phase is cut by its own duration on that trial, so all trials line up on P = 16 bins",
               "A sample belongs to the bin of its target x(t+1); its lags can reach into the previous bin",
               "Go and touch times are known only afterwards, so this warped p cannot be used online"], y + 0.3, 1.7)


def training_slides(d):
    s = d.slide("Training 1: statistics per bin, then the baseline")
    y = d.eq(s, r"$G_g=\sum_{t\in g} z(t)\,z(t)^{\top},\qquad C_g=\sum_{t\in g} z(t)\,x(t+1)^{\top},\qquad S_g=\sum_{t\in g} x(t+1)\,x(t+1)^{\top}$", 1.5)
    d.text(s, "Summed within trials, never across a trial boundary. The rest of the fit uses only these sums.",
           y + 0.1, 0.5, bullet=False)
    y += 0.8
    labels = ["1. Symmetric OLS on all bins pooled", "2. Σ from its residuals, then fixed", "3. Refit weighted by Σ⁻¹: gives φ₀"]
    bw = 3.85
    for i, t in enumerate(labels):
        d.label(s, t, LEFT + 0.1 + i * (bw + 0.19), y, bw, 0.8, fill=TINT, size=16)
    y = d.eq(s, r"$\hat{\Sigma}=\frac{1}{n}\sum_{g}\sum_{t\in g}\ \left(x(t+1)-\Theta_{\mathrm{OLS}}\,z(t)\right)\left(x(t+1)-\Theta_{\mathrm{OLS}}\,z(t)\right)^{\top}$", y + 1.05)
    d.text(s, ["This is Felix's estimator; on one trial it matches Felix's fit_gdar to a relative difference of 1e-11",
               "Preprocessing first: decimate by 3 to 333 Hz, demean each trial, optionally remove evoked kernels"], y + 0.2, 1.1)

    s = d.slide("Training 2: the objective is the data loss")
    d.text(s, "For one bin the GLS loss is quadratic in φ:", 1.4, 0.45, bullet=False)
    y = d.eq(s, r"$L_g(\phi)=\phi^{\top}H_g\,\phi-2\,\phi^{\top}h_g+\mathrm{const}$", 1.9, fs=28)
    y = d.eq(s, r"$H_g=R^{\top}\left(G_g\otimes\hat{\Sigma}^{-1}\right)R,\qquad h_g=R^{\top}\,\mathrm{vec}\left(\hat{\Sigma}^{-1}C_g^{\top}\right)$", y + 0.2, fs=28)
    d.text(s, "The whole model minimises the loss summed over bins:", y + 0.3, 0.45, bullet=False)
    y = d.eq(s, r"$J(W,\Phi)=\sum_{g=1}^{P}L_g\left(\sum_{j=1}^{r}W_{gj}\,\phi_j\right)$", y + 0.8, fs=28)
    d.text(s, "Data decide the metric: well-excited directions count more. An unweighted SVD of per-bin fits, "
              "the older two-stage route, treats all coefficients equally.", y + 0.25, 1.0, bullet=False)

    s = d.slide("Training 3: initialise, then alternate")
    d.text(s, "Start from the pooled model plus the leading per-bin deviations:", 1.4, 0.45, bullet=False)
    y = d.eq(s, r"$\phi_1=\phi_0,\quad W_{g1}=1,\quad W_{g,j>1}=0$", 1.85, fs=22)
    y = d.eq(s, r"$\phi_{j>1}=\mathrm{leading\ directions\ of\ }(\hat{\phi}_g-\phi_0)\ \mathrm{in\ the\ metric\ }\bar{H}=\sum_g H_g$", y + 0.08, fs=22)
    d.text(s, "Each sweep solves one block exactly, so J can never increase:", y + 0.12, 0.45, bullet=False)
    y = d.eq(s, r"$\mathrm{W\ step,\ per\ bin:}\qquad \left(\Phi\,H_g\,\Phi^{\top}\right)w_g=\Phi\,h_g$", y + 0.6, fs=26)
    y = d.eq(s, r"$\Phi\ \mathrm{step,\ all\ bins:}\qquad \left(\sum_g w_g w_g^{\top}\otimes H_g\right)\mathrm{vec}\,\Phi=\sum_g w_g\otimes h_g$", y + 0.15, fs=26)
    d.text(s, ["After each sweep, rescale weight columns to unit RMS; operators take the inverse scale",
               "Stop at a relative change in J below 1e-7, or 40 sweeps (25 on training splits); real fits took 5 to 7",
               "No penalty beyond numerical jitter of 1e-10, matching Felix's unpenalised fit"], y + 0.2, 1.5)


def vertex_eval_slides(d):
    s = d.slide("From components to vertices")
    y = d.eq(s, r"$\Theta(p)=\sum_{v}\lambda_v(p)\,\Theta_v,\qquad \lambda_v\geq 0,\qquad \sum_v\lambda_v=1$", 1.5, fs=28)
    d.text(s, ["Find a with Wa ≈ 1 and project W onto that plane (worst error 0.033 to 0.037 at rank 2, about 0.02 at rank 3)",
               "Rank 2 vertices: the two extreme bins, in the pooled data metric",
               "Rank 3 vertices: the smallest triangle on convex-hull edges that encloses all bins; corners need not be visited by the data",
               "λ comes from non-negative least squares",
               "Sum-to-one makes the companion form, which the LMI controller needs, mix linearly across vertices",
               "This is post-processing: the held-out gains come from the fitted W and Φ"], y + 0.3, 3.8)

    s = d.slide("Evaluation: held-out one-step likelihood")
    y = d.eq(s, r"$\ell=-\frac{1}{2}\left[\frac{1}{n}\,\mathrm{tr}\left(\hat{\Sigma}_m^{-1}\sum_{t}\epsilon_m(t)\,\epsilon_m(t)^{\top}\right)+N\log 2\pi+\log|\hat{\Sigma}_m|\right]$", 1.5, fs=26)
    y = d.eq(s, r"$\epsilon_m(t)=x(t+1)-\Theta_m(p_{g(t)})\,z(t),\qquad \mathrm{gain}=\ell_{\mathrm{TP}}-\ell_{\mathrm{GDAR}}\quad(\mathrm{nats/sample})$", y + 0.2, fs=26)
    d.text(s, ["5-fold over trials, stratified by reach direction, refitting everything on each training split",
               "Each model m has one Σ, from its own training residuals, with 1% shrinkage toward the diagonal",
               "Weight uncertainty: express each split's fit in the full-data vertices and take the jackknife SE",
               "K = 16 was picked on the baseline only; the Felix likelihood is still rising at K = 32"], y + 0.3, 2.4)


def result_slide(d):
    s = d.slide("Result: rank 2 acts as a switch across the trial")
    s.shapes.add_picture(os.path.join(ROOT, "figures", "13_vertices_felix_r2.png"), Inches(LEFT), Inches(1.4), height=Inches(5.7))
    d.table(s, [["Gain over GDAR", "Evoked in", "Removed"], ["Rank 2", "+0.190", "+0.248"], ["Rank 3", "+0.222", "+0.323"],
                ["16 separate fits", "+0.211", "+0.392"]], 1.55, [2.0, 1.05, 1.0], x=8.75, row_h=0.42, size=14)
    d.text(s, ["Nats/sample on held-out trials, K = 16",
               "Figure: rank 2, evoked left in",
               "One vertex through hold and early plan, the other from late plan through move and early post"],
           3.5, 2.6, x=8.75, w=4.05, size=15)


def limitations_slide(d):
    s = d.slide("Limitations of the procedure")
    d.text(s, ["Phase is not known in real time: bins use go and touch times, and the model does not infer phase from neural activity",
               "Weights describe the average trial; they depend only on the bin, not on trial-to-trial state",
               "Bins can only be interpolated if they are ordered; the 4-phase grid has nothing to interpolate",
               "Σ is shared and fixed during the alternation, so this is not a full maximum-likelihood fit",
               "Vertices are one choice of coordinates for the fitted family",
               "One monkey, three days: the timing of the weights repeats, but the edge maps mostly do not. "
               "Noah has not confirmed the channel map"], 1.5, 5.2, size=20)


def build():
    d = Deck()
    title_slide(d, "TP-GDAR: the model and how it is trained",
                "Symmetric TP-GDAR compared with Felix Schwock's GDAR\n"
                "Worked example: Hydrogen 2021-11-12, IOS Reach, no stim")
    model_slides(d)
    training_slides(d)
    vertex_eval_slides(d)
    result_slide(d)
    limitations_slide(d)
    d.p.save(OUT)
    print("wrote", OUT, len(d.p.slides), "slides")


if __name__ == "__main__":
    build()
