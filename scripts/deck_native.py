r"""Editable pieces for the talk deck, built for PowerPoint for the web.

Microsoft's pages say the web version displays charts and equations made on the desktop, can't insert
them, and doesn't promise to edit them; it can't play animated GIFs at all.  So everything here is what
it can certainly edit, shapes and text:

  add_equation    a small LaTeX subset as ordinary text with subscripts and superscripts
  render_figure   a JSON panel spec (figures/deck/<name>.json) as grouped lines, freeforms and text boxes;
                  panel kinds xy, line, bar, strip, network, matrix

The data isn't linked to these shapes: to change a number, change the script and rebuild.
"""

from __future__ import annotations

import json
import math
import re

from matplotlib.ticker import MaxNLocator
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

INK, MUTED, GRID, AXIS = "222222", "555555", "E6E4DE", "9A9A9A"
FONT = "Calibri"
BASE_PT = 13
EMU = 914400

# ---------------------------------------------------------------- equations as text

SYMBOLS = {
    "Delta": "Δ", "alpha": "α", "lambda": "λ", "phi": "φ", "Phi": "Φ", "Theta": "Θ", "Sigma": "Σ", "xi": "ξ", "ell": "ℓ",
    "otimes": "⊗", "top": "T", "geq": "≥", "prec": "≺", "in": "∈", "cdots": "⋯", "sum": "Σ", "times": "×",
    "qquad": "  ", "quad": " ", ",": " ", ";": " ", " ": " ", "|": "‖",
}


def _eq_runs(tex):
    """(text, baseline, italic) runs from a small LaTeX subset.  Nested scripts stay one level down."""
    toks = re.findall(r"\\[A-Za-z]+|\\.|[{}_^]|[^\\{}_^]", tex.strip().strip("$"))
    i = 0
    runs = []

    def emit(text, base, italic=False):
        if runs and runs[-1][1] == base and runs[-1][2] == italic:
            runs[-1][0] += text
        else:
            runs.append([text, base, italic])

    def item(base):
        nonlocal i
        t = toks[i]
        i += 1
        if t == "{":
            while toks[i] != "}":
                item(base)
            i += 1
        elif t in ("_", "^"):
            shift = -25000 if t == "_" else 30000
            item(shift)
        elif t == r"\hat":
            at = len(runs)
            item(base)
            runs[at if at < len(runs) else -1][0] += "̂"
        elif t == r"\mathrm":
            i += 1
            words = []
            while toks[i] != "}":
                words.append(toks[i].lstrip("\\"))
                i += 1
            i += 1
            emit("".join(words), base)
        elif t in (r"\left", r"\right"):
            emit(toks[i].replace("\\", ""), base)
            i += 1
        elif t.startswith("\\"):
            name = t[1:]
            if name not in SYMBOLS:
                raise ValueError(f"unknown command {t}")
            emit(SYMBOLS[name], base)
        else:
            if t in "=+-<" and base == 0:
                t = f" {chr(0x2212) if t == '-' else t} "
            emit("−" if t == "-" else t, base, italic=t.isalpha() and t.isascii())

    while i < len(toks):
        item(0)
    return runs


def add_equation(slide, tex, x, y, w=12.1, size=24, h=None):
    """Editable text.  Returns the y below it."""
    h = h or size / 72 * 1.75
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = "Equation"
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = tf.paragraphs[0]
    for text, base, italic in _eq_runs(tex):
        r = para.add_run()
        r.text = text
        r.font.size, r.font.name, r.font.italic = Pt(size), "Cambria", italic
        r.font.color.rgb = RGBColor.from_string("000000")
        if base:
            r.font._element.set("baseline", str(base))
    return y + h


# ---------------------------------------------------------------- drawing primitives


def _rgb(c):
    return RGBColor.from_string(c.lstrip("#").upper())


def _text(group, text, x, y, w, h, size=BASE_PT, color=MUTED, bold=False, align="center", anchor="middle", rotate=0):
    tb = group.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    tf.vertical_anchor = {"middle": MSO_ANCHOR.MIDDLE, "top": MSO_ANCHOR.TOP}[anchor]
    para = tf.paragraphs[0]
    para.alignment = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT, "left": PP_ALIGN.LEFT}[align]
    r = para.add_run()
    r.text = text
    r.font.size, r.font.name, r.font.bold = Pt(size), FONT, bold
    r.font.color.rgb = _rgb(color)
    if rotate:
        tb.rotation = rotate
    return tb


def _seg(group, x0, y0, x1, y1, color=AXIS, width=0.75, dash=False):
    ln = group.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    ln.line.color.rgb, ln.line.width = _rgb(color), Pt(width)
    if dash:
        ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return ln


def _dot(group, cx, cy, r, fill, line=None):
    s = group.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    s.shadow.inherit = False
    s.fill.solid()
    s.fill.fore_color.rgb = _rgb(fill)
    if line:
        s.line.color.rgb, s.line.width = _rgb(line), Pt(0.75)
    else:
        s.line.fill.background()
    return s


def _poly(group, pts, color, width, dash=False):
    """Open polyline as one freeform, in inches."""
    if len(pts) < 2:
        return None
    start = pts[0]
    b = group.shapes.build_freeform(int(start[0] * EMU), int(start[1] * EMU), scale=1.0)
    b.add_line_segments([(int(a * EMU), int(c * EMU)) for a, c in pts[1:]], close=False)
    s = b.convert_to_shape()
    s.shadow.inherit = False
    s.fill.background()
    s.line.color.rgb, s.line.width = _rgb(color), Pt(width)
    if dash:
        s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return s


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _tick_label(t):
    """0.01 stays 0.01, and 1e-08 reads 10⁻⁸."""
    if t > 0 and (t >= 1e4 or t < 1e-3):
        e = math.log10(t)
        if abs(e - round(e)) < 1e-9:
            return "10" + str(round(e)).translate(_SUP)
    return f"{t:g}"


def _mix(c0, c1, a):
    a = max(0.0, min(1.0, a))
    v = [int(int(c0[i:i + 2], 16) * (1 - a) + int(c1[i:i + 2], 16) * a) for i in (0, 2, 4)]
    return "%02X%02X%02X" % tuple(v)


def _diverging(v, vmax):
    a = v / vmax if vmax else 0.0
    return _mix("EFEEEA", "E34948", a) if a >= 0 else _mix("EFEEEA", "2A78D6", -a)


# ---------------------------------------------------------------- axes


class _Axes:
    """Data to inches for one panel, with axes, grid, ticks, labels, title and legend drawn as shapes."""

    def __init__(self, group, spec, x, y, w, h, xlim, ylim, xticks=None, xticklabels=None):
        self.g, self.spec = group, spec
        self.xlog, self.ylog = bool(spec.get("xlog")), bool(spec.get("ylog"))
        self.xlim, self.ylim = xlim, ylim
        no_x, no_y = spec.get("no_xaxis"), spec.get("no_yaxis")
        left = 0.1 if no_y else 0.62 + (0.28 if spec.get("ylabel") else 0)
        bottom = 0.05 if no_x else 0.3 + (0.28 if spec.get("xlabel") else 0)
        top = (0.38 if spec.get("title") else 0.08) + (0.3 if spec.get("legend") else 0.0)
        self.x0, self.x1 = x + left, x + w - 0.12
        self.y0, self.y1 = max(y + h - bottom, y + top + 0.1), y + top   # a panel never turns inside out
        if spec.get("title"):
            _text(group, spec["title"], x, y, w, 0.32, size=15, color=INK, bold=True)
        ticks = [] if no_y else self._ticks(ylim, self.ylog, self.y0 - self.y1)
        for t in ticks:
            yy = self.py(t)
            _seg(group, self.x0, yy, self.x1, yy, GRID, 0.75)
            _text(group, _tick_label(t), self.x0 - 0.6, yy - 0.12, 0.54, 0.24, align="right")
        if not no_x:
            _seg(group, self.x0, self.y0, self.x1, self.y0)
        if not no_y:
            _seg(group, self.x0, self.y0, self.x0, self.y1)
        xt = [] if no_x else xticks if xticks is not None else self._ticks(xlim, self.xlog, self.x1 - self.x0)
        labels = xticklabels if xticklabels is not None else [_tick_label(t) for t in xt]
        for t, lab in zip(xt, labels):
            xx = self.px(t)
            _seg(group, xx, self.y0, xx, self.y0 + 0.05)
            _text(group, lab, xx - 0.5, self.y0 + 0.05, 1.0, 0.24)
        if spec.get("xlabel") and not no_x:
            _text(group, spec["xlabel"], self.x0, self.y0 + 0.28, self.x1 - self.x0, 0.26)
        if spec.get("ylabel") and not no_y:
            ph = self.y0 - self.y1
            cx, cy = x + 0.13, (self.y0 + self.y1) / 2
            _text(group, spec["ylabel"], cx - ph / 2, cy - 0.13, ph, 0.26, rotate=-90)

    @staticmethod
    def _ticks(lim, log, room=4.0):
        """Tick values for an axis ``room`` inches long, sparse enough that labels never touch."""
        most = max(2, int(room / 0.42))
        if log:
            lo, hi = math.ceil(math.log10(lim[0]) - 1e-9), math.floor(math.log10(lim[1]) + 1e-9)
            decades = list(range(lo, hi + 1))
            if len(decades) < 3:    # under two decades: 2 and 5 as well
                vals = [m * 10.0 ** e for e in range(lo - 1, hi + 1) for m in (1, 2, 5)]
                vals = [t for t in vals if lim[0] - 1e-9 <= t <= lim[1] * (1 + 1e-9)]
                return vals if len(vals) <= most else [10.0 ** e for e in decades]
            step = math.ceil(len(decades) / most)
            return [10.0 ** e for e in decades if e % step == 0]
        vals = [t for t in MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(*lim) if lim[0] - 1e-9 <= t <= lim[1] + 1e-9]
        return vals if len(vals) <= most else [vals[0], vals[-1]]

    def label(self, t):
        """A centre-anchored direct label.  One that names a line gets a short swatch of that line's colour."""
        cx, cy = self.px(t["x"]), self.py(t["y"])
        _text(self.g, t["text"], cx - 1.0, cy - 0.13, 2.0, 0.26, color=INK)
        if t.get("color"):
            edge = cx - 0.044 * len(t["text"]) - 0.06
            _seg(self.g, edge - 0.26, cy, edge, cy, t["color"], 2.0)

    def px(self, v):
        a, b = self.xlim
        f = (math.log10(v) - math.log10(a)) / (math.log10(b) - math.log10(a)) if self.xlog else (v - a) / (b - a)
        return self.x0 + f * (self.x1 - self.x0)

    def py(self, v):
        a, b = self.ylim
        f = (math.log10(v) - math.log10(a)) / (math.log10(b) - math.log10(a)) if self.ylog else (v - a) / (b - a)
        return self.y0 + f * (self.y1 - self.y0)

    def clipped(self, xs, ys):
        """Pieces of the polyline inside the y range, in inches.  Missing points break the line."""
        pieces, cur = [], []
        lo, hi = self.y0, self.y1
        prev = None
        for a, b in zip(xs, ys):
            bad = a is None or b is None or (self.xlog and a <= 0) or (self.ylog and b <= 0)
            if bad:
                if len(cur) > 1:
                    pieces.append(cur)
                cur, prev = [], None
                continue
            p = (self.px(a), self.py(b))
            if prev is not None:
                (x0, y0), (x1, y1) = prev, p
                in0, in1 = hi <= y0 <= lo, hi <= y1 <= lo
                if in0 and in1:
                    cur.append(p)
                elif in0 or in1 or (y0 - lo) * (y1 - lo) < 0 or (y0 - hi) * (y1 - hi) < 0:
                    ts = sorted(t for t in ((lo - y0) / (y1 - y0) if y1 != y0 else 2, (hi - y0) / (y1 - y0) if y1 != y0 else 2) if 0 <= t <= 1) or [1.0 if in0 else 0.0]
                    if in0:
                        cur.append((x0 + ts[0] * (x1 - x0), y0 + ts[0] * (y1 - y0)))
                        if len(cur) > 1:
                            pieces.append(cur)
                        cur = []
                    elif in1:
                        cur = [(x0 + ts[-1] * (x1 - x0), y0 + ts[-1] * (y1 - y0)), p]
                    elif len(ts) == 2:
                        pieces.append([(x0 + t * (x1 - x0), y0 + t * (y1 - y0)) for t in ts])
                else:
                    if len(cur) > 1:
                        pieces.append(cur)
                    cur = []
            elif hi <= p[1] <= lo:
                cur = [p]
            prev = p
        if len(cur) > 1:
            pieces.append(cur)
        return pieces

    def series(self, s, xs=None):
        xs = s["x"] if xs is None else xs
        color, width = s.get("color") or "#2A78D6", s.get("width") or 2.0
        if s.get("line", "solid") != "none":
            for piece in self.clipped(xs, s["y"]):
                _poly(self.g, piece, color, width, dash=s.get("line") == "dash")
        for a, b, e in zip(xs, s["y"], s.get("err") or []):   # half-widths, drawn as whiskers
            if a is None or b is None or not e:
                continue
            top, low = self.py(min(b + e, self.ylim[1])), self.py(max(b - e, self.ylim[0]))
            _seg(self.g, self.px(a), low, self.px(a), top, color, 1.0)
        if s.get("marker"):
            for a, b in zip(xs, s["y"]):
                if a is None or b is None or (self.ylog and b <= 0):
                    continue
                if self.y1 - 1e-6 <= self.py(b) <= self.y0 + 1e-6:
                    _dot(self.g, self.px(a), self.py(b), 0.05, color)

    def legend(self, entries):
        """Short line and name for each entry, right-aligned above the plot."""
        cx = self.x1
        for name, color, dash in reversed(entries):
            tw = 0.085 * len(name) + 0.08
            _text(self.g, name, cx - tw, self.y1 - 0.33, tw, 0.26, align="left")
            _seg(self.g, cx - tw - 0.32, self.y1 - 0.2, cx - tw - 0.06, self.y1 - 0.2, color, 2.0, dash=dash)
            cx -= tw + 0.44


def _nice_limits(values, log=False):
    v = [a for a in values if a is not None and (not log or a > 0)]
    lo, hi = min(v), max(v)
    if log:
        return 10.0 ** math.floor(math.log10(lo)), 10.0 ** math.ceil(math.log10(hi))
    if lo == hi:
        return lo - 1, hi + 1
    ticks = MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return float(ticks[0]), float(ticks[-1])


def _limits(spec_lim, values, log):
    auto = None
    lim = list(spec_lim) if spec_lim else [None, None]
    if lim[0] is None or lim[1] is None:
        auto = _nice_limits(values, log)
    return (auto[0] if lim[0] is None else lim[0], auto[1] if lim[1] is None else lim[1])


def _xy(group, spec, x, y, w, h):
    series = spec["series"]
    xlim = spec.get("xlim")
    if not xlim or None in xlim:
        vals = [v for s in series for v in s["x"] if v is not None]
        xlim = _limits(spec.get("xlim"), vals, spec.get("xlog"))
        if not spec.get("xlog") and not spec.get("xlim"):
            xlim = (min(vals), max(vals))
    ylim = _limits(spec.get("ylim"), [v for s in series for v in s["y"]] + [h_["y"] for h_ in spec.get("hlines") or []], spec.get("ylog"))
    ax = _Axes(group, spec, x, y, w, h, xlim, ylim)
    for hl in spec.get("hlines") or []:
        yy = ax.py(hl["y"])
        _seg(group, ax.x0, yy, ax.x1, yy, hl.get("color", "#222222"), 1.25, dash=hl.get("dash"))
    for s in series:
        ax.series(s)
    for t in spec.get("texts") or []:
        ax.label(t)
    if spec.get("legend"):
        ax.legend([(s["name"], s.get("color") or "#2A78D6", s.get("line") == "dash") for s in series if s.get("name")])


def _line(group, spec, x, y, w, h):
    groups = spec.get("cat_groups")
    if groups:
        n = sum(int(g[1]) for g in groups)
        ticks, labels, edge = [], [], 0
        for name, k in groups:
            ticks.append(edge + (int(k) - 1) / 2)
            labels.append(name)
            edge += int(k)
    else:
        n = len(spec["categories"])
        ticks, labels = list(range(n)), spec["categories"]
    ylim = _limits(spec.get("ylim"), [v for s in spec["series"] for v in s["y"]], spec.get("ylog"))
    ax = _Axes(group, spec, x, y, w, h, (-0.5, n - 0.5), ylim, xticks=ticks, xticklabels=labels)
    if groups:
        edge = 0
        for _, k in groups[:-1]:
            edge += int(k)
            _seg(group, ax.px(edge - 0.5), ax.y0, ax.px(edge - 0.5), ax.y1, AXIS, 0.75, dash=True)
    for s in spec["series"]:
        ax.series(s, xs=list(range(n)))
    for t in spec.get("texts") or []:
        ax.label(t)
    if spec.get("legend"):
        ax.legend([(s["name"], s.get("color") or "#2A78D6", s.get("line") == "dash") for s in spec["series"] if s.get("name")])


def _bar(group, spec, x, y, w, h):
    cats, series = spec["categories"], spec["series"]
    vals = [v for s in series for v in s["values"]]
    lim = spec.get("vlim") or (min(0, min(vals)), max(vals))
    n, m = len(cats), len(series)
    if spec.get("horizontal"):
        sp = dict(spec, xlabel=spec.get("vlabel"), ylabel=None)
        left = 0.3 + 0.085 * max(len(c) for c in cats)
        ax = _Axes(group, sp, x + left - 0.62, y, w - left + 0.62, h, tuple(lim), (0.0, float(n)))
        for tick_group in list(group.shapes):      # the category axis carries names, not numbers
            if tick_group.has_text_frame and tick_group.text_frame.text in ("0", "1", "2", "3", "4", "5", "6"):
                if abs(tick_group.left / EMU + 0.54 - ax.x0) < 0.2:
                    tick_group._element.getparent().remove(tick_group._element)
        band = (ax.y0 - ax.y1) / n
        bar_h = 0.7 * band / m
        for i, c in enumerate(cats):
            top = ax.y1 + i * band + 0.15 * band
            _text(group, c, x, ax.y1 + i * band, ax.x0 - x - 0.08, band, align="right")
            for k, s in enumerate(series):
                v = s["values"][i]
                r = group.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(ax.x0), Inches(top + k * bar_h), Inches(max(ax.px(v) - ax.x0, 0.01)), Inches(bar_h))
                r.shadow.inherit = False
                r.line.fill.background()
                r.fill.solid()
                r.fill.fore_color.rgb = _rgb(s["color"])
                if spec.get("show_values"):
                    _text(group, f"{v:.2f}", ax.px(v) + 0.05, top + k * bar_h, 0.6, bar_h, align="left", color=INK)
    else:
        ax = _Axes(group, spec, x, y, w, h, (0.0, float(n)), tuple(lim), xticks=[i + 0.5 for i in range(n)], xticklabels=cats)
        band = (ax.x1 - ax.x0) / n
        bar_w = 0.7 * band / m
        for i in range(n):
            left = ax.x0 + i * band + 0.15 * band
            for k, s in enumerate(series):
                v = s["values"][i]
                r = group.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left + k * bar_w), Inches(ax.py(v)), Inches(bar_w), Inches(max(ax.y0 - ax.py(v), 0.01)))
                r.shadow.inherit = False
                r.line.fill.background()
                r.fill.solid()
                r.fill.fore_color.rgb = _rgb(s["color"])
                if spec.get("show_values"):
                    _text(group, f"{v:.2f}", left + k * bar_w - 0.2, ax.py(v) - 0.26, bar_w + 0.4, 0.24, color=INK)
    if spec.get("legend"):
        ax.legend([(s["name"], s["color"], False) for s in series])


def _strip(group, spec, x, y, w, h):
    cats, values = spec["categories"], spec["values"]
    ylim = _limits(spec.get("ylim"), [v for vs in values for v in vs], spec.get("ylog"))
    ax = _Axes(group, spec, x, y, w, h, (-0.5, len(cats) - 0.5), ylim, xticks=list(range(len(cats))), xticklabels=cats)
    color = spec.get("color", "#2A78D6")
    for i, vs in enumerate(values):
        n = len(vs)
        for k, v in enumerate(sorted(vs)):
            jitter = 0.16 * ((k * 0.618) % 1 - 0.5) * 2
            _dot(group, ax.px(i + jitter), ax.py(v), 0.04, color)
        if spec.get("median"):
            med = sorted(vs)[n // 2] if n % 2 else 0.5 * (sorted(vs)[n // 2 - 1] + sorted(vs)[n // 2])
            _seg(group, ax.px(i - 0.28), ax.py(med), ax.px(i + 0.28), ax.py(med), INK, 2.25)


def _network(group, spec, x, y, w, h):
    head = 0.38 if spec.get("title") else 0.0
    if spec.get("title"):
        _text(group, spec["title"], x, y, w, 0.32, size=15, color=INK, bold=True)
    pos, edges = spec["pos"], spec["edges"]
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    sx, sy = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
    best = min(((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2) ** 0.5 for i in range(len(pos)) for j in range(i))
    margin = 0.15
    scale = min((w - 2 * margin) / sx, (h - head - 2 * margin) / sy)
    ox, oy = x + (w - scale * sx) / 2, y + head + ((h - head) - scale * sy) / 2

    def px(p):
        return ox + (p[0] - min(xs)) * scale, oy + (max(ys) - p[1]) * scale
    vals, vmax = spec.get("edge_values"), spec.get("vmax") or 1.0
    for k, (i, j) in enumerate(edges):
        a, b = px(pos[i]), px(pos[j])
        if vals is None:
            _seg(group, *a, *b, "B5B3AD", 1.5)
        else:
            _seg(group, *a, *b, _diverging(vals[k], vmax), 1.0 + 5.0 * abs(vals[k]) / vmax)
    radius = spec.get("node_radius") or min(0.15, 0.3 * scale * best)
    nv, nmax = spec.get("node_values"), spec.get("node_vmax") or 1.0
    for k, p in enumerate(pos):
        if nv is None:
            fill = "FFFFFF"
        elif spec.get("node_cmap") == "div":
            fill = _diverging(nv[k], nmax)
        else:
            fill = _mix("FFFFFF", "2A78D6", max(nv[k] / nmax, 0.0) ** (spec.get("node_gamma") or 1.0))
        _dot(group, *px(p), radius, fill, line="444444")


def _matrix(group, spec, x, y, w, h):
    head = 0.38 if spec.get("title") else 0.0
    if spec.get("title"):
        _text(group, spec["title"], x, y, w, 0.32, size=15, color=INK, bold=True)
    m = spec["values"]
    n = len(m)
    size = min(w, h - head) * 0.94
    ox, oy = x + (w - size) / 2, y + head + ((h - head) - size) / 2
    cell = size / n
    frame = group.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(ox), Inches(oy), Inches(size), Inches(size))
    frame.shadow.inherit = False
    frame.fill.solid()
    frame.fill.fore_color.rgb = _rgb("FAFAF7")
    frame.line.color.rgb, frame.line.width = _rgb(AXIS), Pt(0.75)
    vmax = spec.get("vmax") or 1.0
    for i in range(n):
        for j in range(n):
            v = m[i][j]
            if not v:
                continue
            sq = group.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(ox + j * cell), Inches(oy + i * cell), Inches(cell), Inches(cell))
            sq.shadow.inherit = False
            sq.line.fill.background()
            sq.fill.solid()
            sq.fill.fore_color.rgb = _rgb("2A78D6" if spec.get("binary") else _diverging(v, vmax))


def add_edge_key(slide, x, y, label="edge weight"):
    """A small − 0 + colour key, as one group of shapes.  x and y are the top left, in inches."""
    group = slide.shapes.add_group_shape()
    _text(group, "−", x, y - 0.03, 0.2, 0.24)
    for k, v in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
        box = group.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.2 + k * 0.2), Inches(y + 0.03), Inches(0.2), Inches(0.13))
        box.shadow.inherit = False
        box.fill.solid()
        box.fill.fore_color.rgb = _rgb(_diverging(v, 1.0))
        box.line.fill.background()
    _text(group, "+", x + 1.2, y - 0.03, 0.2, 0.24)
    _text(group, label, x + 1.42, y - 0.03, 0.09 * len(label) + 0.1, 0.24, align="left")
    return group


def render_figure(slide, path, x, y, w, h, panel_flags=None):
    """Draw figures/deck/<name>.json into the box (inches).  Each panel is one group.

    ``panel_flags`` maps a panel title to extra keys for it, such as no_yaxis.
    """
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    for panel in spec["panels"]:
        flags = dict((panel_flags or {}).get(panel.get("title"), {}))
        moved = flags.pop("text_y", {})     # move a direct label up or down, by its text
        panel.update(flags)
        for t in panel.get("texts") or []:
            t["y"] = moved.get(t["text"], t["y"])
    rows, cols = spec["grid"]
    rh = spec.get("row_heights") or [1] * rows
    cw = spec.get("col_widths") or [1] * cols
    ry = [sum(rh[:i]) / sum(rh) * h for i in range(rows + 1)]
    cx = [sum(cw[:i]) / sum(cw) * w for i in range(cols + 1)]
    draw = {"xy": _xy, "line": _line, "bar": _bar, "strip": _strip, "network": _network, "matrix": _matrix}
    for panel in spec["panels"]:
        r, c, rs, cs = panel["cell"]
        group = slide.shapes.add_group_shape()
        group.name = panel.get("title") or panel["kind"]
        draw[panel["kind"]](group, panel, x + cx[c], y + ry[r], cx[c + cs] - cx[c], ry[r + rs] - ry[r])
