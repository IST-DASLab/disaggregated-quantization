"""Shared style and drawing primitives for the paper's vector schematics.

Two conventions hold across every figure built on this module.

COLOUR is the one in notebooks/plots.ipynb -- the same hex constants, so a schematic
sits next to bars_disag.pdf without a palette change and "red = NVFP4" carries over from
the bar charts. The assignment is by ROLE here rather than by method, because a
schematic names parts (storage, prefill, decode) where the result figures name arms.

GEOMETRY is authored in one abstract unit system (x right, y up, arbitrary units) and
converted to inches exactly once, by canvas(). Layout code never mentions physical
sizes; changing the target width rescales the whole drawing, and only the FONT SIZES --
which are in points, i.e. absolute -- have to be rechecked. Author at the width the
figure is actually typeset at -- half of \\linewidth for a wrapfigure -- so that no
LaTeX-side scaling shrinks the type, and keep the point sizes above 5.5pt.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Palette -- verbatim from notebooks/plots.ipynb
# ---------------------------------------------------------------------------
RED = "#cd001a"
ORANGE = "#ef6a00"
YELLOW = "#f2cd00"
GREEN = "#79c300"
BLUE = "#1961ae"
PURPLE = "#61007d"
PINK = "#e5006d"
BROWN = "#8a5a2b"
OLIVE = "#6b8f00"
TEAL = "#00a19a"
FOREST = "#1f7a33"
NAVY = "#0b2d6b"
GREY = "#6e6e6e"
SKY = "#3fa9f5"
CYAN = "#00b8d4"
INDIGO = "#534bae"

# Two colour axes, and they are independent.
#
# PHASE -- who owns a part of the layer. Applies to pills, storage, matmul nodes and the
# arrows between them, so "what is still shared" is readable at a glance: green thins out
# as a format disaggregates.
C_PREFILL = RED     # prefill only
C_DECODE = BLUE     # decode only
C_SHARED = GREEN    # both: one storage feeding both phases, or one pathway serving both
#
# A quantized format chip takes the colour of the PHASE that runs it, so NVFP4 is red
# wherever prefill owns it and green where both phases share one pathway. Only floats are
# exempt: unquantized is grey in every phase.
C_FLOAT = GREY      # BF16 / FP32

INK = "#1a1a1a"     # primary text / arrows
MUTED = "#6f6f6f"   # secondary text
HAIR = "#d8d8d8"    # panel outlines
PANEL_BG = "#f6f6f6"

# Font sizes in POINTS at the authored figure width (see module docstring).
FS_TITLE = 8.5
FS_PILL = 8.0
FS_CHIP = 9.0
FS_LABEL = 8.0
FS_NOTE = 7.5

FONT_FAMILY = "sans-serif"   # plots.ipynb leaves matplotlib's default (DejaVu Sans)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def tint(color: str, amount: float) -> tuple:
    """Mix `color` with white. amount=0 -> color, amount=1 -> white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def shade(color: str, amount: float) -> tuple:
    """Mix `color` with black. amount=0 -> color, amount=1 -> black."""
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def readable(color: str, on="white") -> tuple:
    """Darken `color` until bold text in `on` stands off it.

    GREEN is light enough that white type on it is barely legible, so a pill filled with
    the raw palette value would be unreadable at 8pt while RED and BLUE are fine.
    """
    r, g, b = to_rgb(color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return shade(color, 0.34) if (on == "white" and lum > 0.35) else to_rgb(color)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
def use_style() -> None:
    mpl.rcParams.update({
        # Type 42 embeds TrueType outlines; matplotlib's Type 3 default is rejected by
        # some camera-ready checkers and cannot be searched or copied out of the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": FONT_FAMILY,
    })


def canvas(xlim, ylim, width_in: float):
    """Axes spanning the whole figure, in abstract units, with a 1:1 aspect."""
    use_style()
    (x0, x1), (y0, y1) = xlim, ylim
    units_per_inch = (x1 - x0) / width_in
    fig = plt.figure(figsize=(width_in, (y1 - y0) / units_per_inch))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:                      # non-Agg backend
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        return FigureCanvasAgg(fig).get_renderer()


def text_extent(ax, artist):
    """Bounding box of a drawn text, in DATA units -- measured, not estimated."""
    bb = artist.get_window_extent(renderer=_renderer(ax.figure))
    return bb.transformed(ax.transData.inverted())


# pad_inches is explicit because bbox_inches="tight" ADDS 0.1in per side by default, which
# grows the page past the width the figure was authored for -- LaTeX then scales it back
# down to \linewidth and every point size shrinks with it.
CROP = dict(bbox_inches="tight", pad_inches=0.02)


def save(fig, path) -> None:
    """Write the figure as vector PDF -- the only artifact the paper consumes."""
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, transparent=True, **CROP)
    w, h = fig.get_size_inches()
    print(f"wrote {p} ({w:.2f} x {h:.2f} in authored)")


def snapshot(fig, dpi: int = 300):
    """Raster of the same figure, IN MEMORY, for assembling a preview contact sheet.

    Nothing rasterized is written next to the PDFs: a stray PNG beside a figure is one
    more thing that can go stale or get \includegraphics'd by accident.
    """
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", **CROP)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def panel(ax, x0, y0, x1, y1, fc=PANEL_BG, ec=HAIR, lw=0.6, r=1.6, zorder=1, **kw):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder, **kw)
    ax.add_patch(p)
    return p


def label(ax, x, y, s, fs=FS_LABEL, color=MUTED, ha="center", va="center",
          weight="normal", style="normal", zorder=6, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=zorder, **kw)


def chip(ax, cx, cy, text, color, h, fs=FS_CHIP, pad_x=1.5, zorder=4):
    """Rounded format tag centred on (cx, cy). Returns (left, right, width)."""
    t = ax.text(cx, cy, text, fontsize=fs, color=shade(color, 0.5), ha="center",
                va="center", fontweight="bold", zorder=zorder + 1)
    w = text_extent(ax, t).width + 2 * pad_x
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.3}",
        facecolor=tint(color, 0.86), edgecolor=color, linewidth=0.8, zorder=zorder))
    return cx - w / 2, cx + w / 2, w


def pill(ax, x, y, text, color, fs=FS_PILL, pad_x=2.6, pad_y=1.3, zorder=4):
    """Section tag, anchored by its LEFT edge at x. Returns (left, right).

    Both dimensions come from the MEASURED text: a fixed height set in layout units
    silently clips the caps as soon as the point size changes.
    """
    t = ax.text(x + pad_x, y, text, fontsize=fs, color="white", ha="left", va="center",
                fontweight="bold", zorder=zorder + 1)
    bb = text_extent(ax, t)
    w, h = bb.width + 2 * pad_x, bb.height + 2 * pad_y
    ax.add_patch(FancyBboxPatch(
        (x, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h * 0.4}",
        facecolor=readable(color), edgecolor="none", zorder=zorder))
    return x, x + w


def arrow(ax, p0, p1, color=INK, lw=0.9, ls="-", head=4.5, zorder=5, alpha=1.0,
          astyle="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=astyle, mutation_scale=head, linewidth=lw,
                        linestyle=ls, color=color, shrinkA=0, shrinkB=0, zorder=zorder,
                        alpha=alpha, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def polyarrow(ax, pts, color=INK, lw=0.9, ls="-", head=4.5, zorder=5, alpha=1.0,
              astyle="-|>"):
    """Arrow along a polyline (sharp corners), head on the last segment.

    astyle="-" draws it headless, which is what a branch merging INTO another line wants:
    the head belongs to the line the merged flow leaves by, not to the tributary.
    """
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    a = FancyArrowPatch(path=Path(pts, codes), arrowstyle=astyle, mutation_scale=head,
                        linewidth=lw, linestyle=ls, color=color, zorder=zorder,
                        alpha=alpha, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a
