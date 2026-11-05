r"""LaTeX equations as pictures for the decks: latex + dvipng, cached in figures/eq_tex by a hash of the source.

Pictures are black on transparent, so they sit on white or tinted shapes, and carry their LaTeX source as alt text, so
an equation can be recovered from the slide.  They aren't editable in PowerPoint; change them here and rebuild.
"""

import hashlib
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image
from pptx.util import Inches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "figures", "eq_tex")
DPI = 1200          # at 10 pt, so about 500 dpi on the slide at 24 pt
DOC = r"""\documentclass[10pt]{article}
\usepackage{amsmath,amssymb}
\pagestyle{empty}
\setlength{\textwidth}{30in}
\setlength{\paperwidth}{32in}
\begin{document}
\noindent\mbox{$\displaystyle %s$}
\end{document}
"""


def render(tex):
    """Path of the PNG for one display-style equation, without the surrounding dollars."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, hashlib.md5((DOC + tex).encode()).hexdigest()[:14] + ".png")
    if os.path.exists(path):
        return path
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "e.tex"), "w", encoding="utf-8") as fh:
            fh.write(DOC % tex)
        run = subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "e.tex"], cwd=tmp, capture_output=True,
                             text=True, timeout=120)
        if run.returncode:
            errors = [ln for ln in run.stdout.splitlines() if ln.startswith("!")]
            raise RuntimeError(f"latex failed on {tex!r}: {errors}")
        out = os.path.join(tmp, "e.png")
        subprocess.run(["dvipng", "-D", str(DPI), "-T", "tight", "-bg", "White", "-fg", "Black", "-o", out, "e.dvi"],
                       cwd=tmp, capture_output=True, text=True, timeout=120, check=True)
        grey = np.asarray(Image.open(out).convert("L"))
    rgba = np.zeros(grey.shape + (4,), dtype=np.uint8)
    rgba[..., 3] = 255 - grey
    Image.fromarray(rgba).save(path, optimize=True)
    return path


def size_of(tex, size):
    """(width, height) in inches at font size `size` points."""
    w, h = Image.open(render(tex)).size
    k = size / 10 / DPI
    return w * k, h * k


def add_latex(slide, tex, x, y, size=24, max_w=None, h_row=None, center_x=None):
    """Picture of `tex` with its left edge at x (or centered on center_x), its top at y (or centered in a row of height
    h_row starting at y).  Shrinks to fit max_w and says so.  Returns the y below it."""
    path = render(tex)
    w, h = size_of(tex, size)
    if max_w is not None and w > max_w:
        print(f"  equation shrunk to {size * max_w / w:.1f} pt to fit: {tex[:60]}")
        h, w = h * max_w / w, max_w
    top = y if h_row is None else y + (h_row - h) / 2
    left = x if center_x is None else center_x - w / 2
    pic = slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(w), Inches(h))
    pic.name = "Equation"
    pic._element.nvPicPr.cNvPr.set("descr", tex)
    return max(top + h, y + (h_row or 0))
