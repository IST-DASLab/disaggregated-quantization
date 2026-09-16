# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "pillow"]
# ///
"""Quantization-aware distillation WITH disaggregation, as one training step.

    uv run notebooks/schematics/fig_qadd_training.py     # writes qadd_training.pdf

An SFT sequence already carries the split the deployment will make -- the prompt is what a
prefill engine is fed, the reply is what a decode engine generates -- so the phase of every
position is read straight off the label mask. The student then runs ONE forward over the
whole sequence: inside every linear the sequence SPLITS by that phase, prefill positions
up and decode positions down, each side through its own pathway, and the results MERGE back
into one sequence. The teacher sees the same batch, frozen and undivided.

The figure is deliberately format-free. Which formats the two pathways carry is the subject
of fig_dq_linear.py; here the only claim is that the sequence is routed by phase and
rejoined, which is true of every DQ format. Token cells are the one recurring object: a
token keeps its colour from the batch, through the pathway that serves it, into the merged
output, so neither the split nor the merge needs a label.

LAYOUT. Wide and short, for the top of a page at full \\linewidth. The batch takes a row of
its own across the top: ten tokens set large enough to read are ~145 units wide, and in a
left-hand column they would have left the pathways no room. So the sequence spans the top,
the model sits under it, and the teacher and the loss share the right-hand column.

The linear's left and right edges are TORN, the mark fig_odp_timeline.py uses for a cut
axis: the layer is one of many, with more stack on either side. The backward pass ends at
the left tear -- earlier layers lie past it, and the batch, which takes no gradient at all,
past them.

Palette and grammar are the other schematics' -- red is prefill, blue is decode, grey is
unquantized.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import (C_DECODE, C_FLOAT, C_PREFILL, FS_LABEL, FS_NOTE, FS_PILL,
                   INK, MUTED, arrow, canvas, label, panel, pill, polyarrow,
                   readable, save, shade, snapshot, text_extent, tint)

OUT = Path(__file__).resolve().parents[1] / "figures" / "qadd_training.pdf"
PREVIEW = Path(__file__).resolve().parent / "preview_qadd.png"

# ---------------------------------------------------------------------------
# Metrics. 258 units to the 5.5in, where the linear-layer schematics use ~53/in. Point
# sizes are ABSOLUTE, so drawing the geometry in a smaller unit space is what makes the
# type larger relative to the figure: everything measured off text (pills, word cells)
# keeps its physical size and simply occupies more units.
# ---------------------------------------------------------------------------
FIG_W_IN = 5.5
# Two units of margin on every side. Patches are clipped to the axes, so a panel whose
# border sits ON the limit loses half its stroke -- the batch's left edge, the student's
# bottom and the right column's right edge all do.
XLIM, YLIM = (-2, 260), (-2, 82)

# A banner carries its type smaller than a standing figure does: 30% off the shared sizes,
# and the geometry follows, which is where most of the height went. Everything that sizes
# itself from measured text -- pills, word cells, the batch panel -- follows for free.
TYPE_SCALE = 0.7
FS_LABEL, FS_NOTE, FS_PILL = (f * TYPE_SCALE for f in (FS_LABEL, FS_NOTE, FS_PILL))

PAD = 4
PILL_H, PILL_PAD_X = 6.9, 2.0
WORD_H, WORD_PAD, WORD_GAP = 9.0, 1.3, 0.6       # a token of the example sequence
CELL_W, CELL_H = 2.6, 5.8                        # the same token, abstract, in the layer

PROMPT = ["What", "is", "the", "capital", "of", "France", "?"]
REPLY = ["It's", "Paris", "."]

# The batch panel is sized from the MEASURED sequence, not typed: the example sentence is
# edited far more often than the layout is, and a fixed width leaves it half empty.
BATCH_X0, BATCH_Y0, BATCH_Y1 = 0, 58, 80
BATCH_MARGIN = 5
BOX = (20, 0, 176, 54)                # the student
ZOOM = (24, 3, 172, 44)               # one linear inside it
# One column for the two boxes on the right: same width, same centre, so they read as a
# pair. The width is the loss's longest line plus its padding -- everything else in the
# column is narrower, and the student was given what was left.
TEACH = (184, 37, 258, 54)
LOSS = (184, 14.5, 258, 32.5)

# The sequence arrives WHOLE and leaves whole: it parts and rejoins inside the linear, so
# both corridors sit within the layer and the split is the layer's own doing.
#
# The two lanes sit as close as their boxes allow: 6 units between them and 7 to the
# layer's edges, which is enough for the merge corridor to read as a corridor and for the
# layer not to look shrink-wrapped around its pathways.
PRE_Y, DEC_Y, OUT_Y = 32.5, 14.5, 23.5   # the two lanes, and what the merge emits
PBOX = (92, PRE_Y - 6.5, 126, PRE_Y + 6.5)
DBOX = (92, DEC_Y - 6.5, 126, DEC_Y + 6.5)
TRUNK_X = 10                          # the batch's way down to the layer
# The split needs room to BE a split: a long run from the tear before the corridor -- which
# is also what "Forward"/"Backward" are set over -- and 6 more before the tokens it hands
# over, or the turn, the arrowheads and the first cell all land on top of each other.
SPLIT_X, MERGE_X = 56, 134            # the corridors the sequence parts and rejoins in
LANE_X, MERGED_X = 62, 139

# Every flow is drawn as a PAIR straddling its centre line: forward solid, backward dotted.
# Shifting a whole axis-aligned path by (+H, +H) raises its horizontals by H and moves its
# verticals right by H, and (-H, -H) does the mirror, so the two paths stay parallel
# through every corner without any per-segment normal arithmetic.
H = 1.2
HEAD = 8.0                            # arrowhead scale, in points: absolute, like type
STUB = 3.2                            # solid tip carried on a dotted shaft
DOT = (0, (1.2, 1.3))
# The backward pass ends at the layer's torn left edge. Past it lie the earlier layers,
# which this figure does not draw, and past THEM the batch, which takes no gradient at all.
GRAD_STOP_X = 26


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def cells(ax, x0, y, colors, w=CELL_W, h=CELL_H):
    """A run of token cells, one per position, coloured by the phase that serves it."""
    for i, c in enumerate(colors):
        panel(ax, x0 + i * w, y - h / 2, x0 + (i + 1) * w, y + h / 2,
              fc=tint(c, 0.72), ec=c, lw=0.6, r=0.7, zorder=3)
    return x0 + len(colors) * w


def word_widths(ax, items):
    """Each token's cell width, measured -- a token is as wide as the word it holds."""
    out = []
    for w in items:
        t = ax.text(0, 0, w, fontsize=FS_NOTE)
        out.append(text_extent(ax, t).width + 2 * WORD_PAD)
        t.remove()
    return out


def sequence_width(ax, items):
    return sum(word_widths(ax, items)) + WORD_GAP * (len(items) - 1)


def words(ax, x0, y, items, color, h=WORD_H):
    """The same cells, carrying the example sequence."""
    x = x0
    for w, cw in zip(items, word_widths(ax, items)):
        panel(ax, x, y - h / 2, x + cw, y + h / 2, fc=tint(color, 0.78), ec=color,
              lw=0.6, r=0.9, zorder=3)
        label(ax, x + cw / 2, y, w, fs=FS_NOTE, color=shade(color, 0.45), zorder=5)
        x += cw + WORD_GAP
    return x - WORD_GAP


def span(ax, x0, x1, y, text, color):
    """A phase's extent over the batch, named UNDER it."""
    ax.plot([x0, x1], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=4)
    label(ax, (x0 + x1) / 2, y - 3.9, text, fs=FS_NOTE - 0.7, color=shade(color, 0.3),
          weight="bold")


def wavy_cut(ax, p0, p1, amp=1.3, waves=7, n=240, color="#3c3c3c", lw=1.0):
    """A torn edge -- the same mark fig_odp_timeline.py puts on its broken axes.

    What crosses a torn edge reads as continuing past it, which a straight border does not.
    """
    import math
    (x0, y0), (x1, y1) = p0, p1
    ts = [i / (n - 1) for i in range(n)]
    off = [amp * math.sin(2 * math.pi * waves * t) for t in ts]
    xs = [x0 + (x1 - x0) * t for t in ts]
    ys = [y0 + (y1 - y0) * t for t in ts]
    if x0 == x1:
        xs = [x0 + o for o in off]
    else:
        ys = [y0 + o for o in off]
    ax.plot(xs, ys, color=color, lw=lw, zorder=2, solid_capstyle="butt")


def torn_box(ax, x0, y0, x1, y1, ec="#3c3c3c", lw=1.0):
    """Straight top and bottom, torn left and right: more model on either side."""
    panel(ax, x0, y0, x1, y1, fc="white", ec="none", r=0, zorder=1)
    for y in (y0, y1):
        ax.plot([x0, x1], [y, y], color=ec, lw=lw, zorder=2, solid_capstyle="butt")
    wavy_cut(ax, (x0, y0), (x0, y1), color=ec, lw=lw)
    wavy_cut(ax, (x1, y0), (x1, y1), color=ec, lw=lw)


def shift(pts, d):
    """Parallel copy of an axis-aligned path, offset to one side, ENDS pulled back.

    Translating the whole path by (d, d) is what keeps it parallel through every corner,
    but it also slides the two terminal points ALONG their own segments -- which lands an
    arrowhead d inside the box it points at, and starts a tail d past where it should.
    Removing the along-travel component at each end fixes both tips and disturbs nothing
    in between.
    """
    out = [(x + d, y + d) for x, y in pts]
    for end, other in ((0, 1), (-1, -2)):
        (x0, y0), (x1, y1) = pts[other], pts[end]
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / n, (y1 - y0) / n
        along = d * (ux + uy)
        out[end] = (out[end][0] - ux * along, out[end][1] - uy * along)
    return out


def pair(ax, pts, color, fwd="-|>", bwd="-|>", back=None):
    """A flow and its gradient, one on each side of the path the layout defines.

    `back` overrides the return path where it is not simply the forward one reversed --
    the only case is the trunk, whose gradient stops at the tear.
    """
    polyarrow(ax, shift(pts, H), color=color, astyle=fwd, head=HEAD)
    dotted(ax, shift(list(reversed(pts)) if back is None else back, -H), color, bwd)


def dotted(ax, pts, color, astyle, lw=0.9):
    """A dotted shaft ending in a SOLID head.

    matplotlib strokes an arrow's head outline with the line's dash pattern, so a dotted
    arrow drawn in one piece arrives as a handful of flecks rather than a tip. The shaft
    stops a stub short and a solid arrow covers the last of it.
    """
    if astyle == "-":                                   # a headless tributary
        polyarrow(ax, pts, color=color, astyle="-", ls=DOT, lw=lw)
        return
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    n = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / n, (y1 - y0) / n
    knee = (x1 - ux * STUB, y1 - uy * STUB)
    polyarrow(ax, list(pts[:-1]) + [knee], color=color, astyle="-", ls=DOT, lw=lw)
    arrow(ax, knee, (x1, y1), color=color, lw=lw, head=HEAD)


def centred_pill(ax, cx, y, name, color):
    """pill() anchors by its LEFT edge, so centring it means measuring the text first --
    the same rule as everywhere else here: measure, never estimate."""
    probe = ax.text(0, 0, name, fontsize=FS_PILL, fontweight="bold")
    w = text_extent(ax, probe).width + 2 * PILL_PAD_X
    probe.remove()
    # fs is passed EXPLICITLY: pill()'s default binds style's size at import, so a pill
    # left to it would be drawn at a size this module never measured -- and a pill wider
    # than the width it was centred on hangs off to the right.
    return pill(ax, cx - w / 2, y, name, color, fs=FS_PILL, pad_x=PILL_PAD_X)


def runs(ax, cx, y, parts, fs, gap=1.5, **kw):
    """One line built from differently coloured runs, the whole line centred on cx.

    Used where a single word has to carry a colour the rest of the line does not -- the
    loss is scored on the REPLY, and that word is the same blue as the reply everywhere
    else in the figure.
    """
    widths = []
    for s, _ in parts:
        t = ax.text(0, 0, s, fontsize=fs)
        widths.append(text_extent(ax, t).width)
        t.remove()
    x = cx - (sum(widths) + gap * (len(parts) - 1)) / 2
    for (s, colour), w in zip(parts, widths):
        label(ax, x, y, s, fs=fs, color=colour, ha="left", **kw)
        x += w + gap


def pathway(ax, rect, color, name, descriptor):
    """One phase's pathway, named and nothing more -- the formats are another figure."""
    panel(ax, *rect, fc=tint(color, 0.9), ec=tint(color, 0.4), lw=0.9, r=2.0, zorder=2)
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    centred_pill(ax, cx, cy + 2.9, name, color)
    label(ax, cx, cy - 3.7, descriptor, fs=FS_NOTE, color=shade(color, 0.25))


def build():
    fig, ax = canvas(XLIM, YLIM, FIG_W_IN)

    # ---- The batch: the split is already in it ----------------------------------
    by = BATCH_Y1 - PAD - WORD_H / 2
    bx = BATCH_X0 + BATCH_MARGIN
    seq_w = sequence_width(ax, PROMPT + REPLY)
    batch = (BATCH_X0, BATCH_Y0, bx + seq_w + BATCH_MARGIN, BATCH_Y1)
    panel(ax, *batch, fc="white", ec="#3c3c3c", lw=1.0, r=2.4)
    mid = words(ax, bx, by, PROMPT, C_PREFILL)
    end = words(ax, mid + WORD_GAP, by, REPLY, C_DECODE)
    span(ax, bx, mid, by - 6.2, "Prompt", C_PREFILL)
    span(ax, mid + WORD_GAP, end, by - 6.2, "Reply", C_DECODE)

    # ---- The student: one forward, split by phase and rejoined --------------------
    # Same border as every other box in the figure: the nesting is already carried by the
    # inner box's torn edges, so a second weight of grey only looked like an accident.
    panel(ax, *BOX, fc="white", ec="#3c3c3c", lw=1.0, r=2.4, zorder=0)
    torn_box(ax, *ZOOM)
    head_y = (ZOOM[3] + BOX[3]) / 2
    pill(ax, BOX[0] + 4, head_y, "STUDENT", C_FLOAT, fs=FS_PILL, pad_x=PILL_PAD_X)
    # The layer below is ONE linear, and the student holds many: said here, in the strip
    # directly above the box it qualifies, and flush with that box's right edge.
    label(ax, ZOOM[2], head_y, "Every transformer linear", fs=FS_NOTE, color=MUTED,
          ha="right")
    # A point smaller than the other headings, and set tight: in a banner-height layer the
    # only clear space is the wedge above the merged sequence.
    label(ax, ZOOM[2] - 4, ZOOM[3] - 2, "Disaggregated\nQuantized\nLinear",
          fs=FS_LABEL, color=INK, weight="bold", ha="right", va="top",
          linespacing=1.2)
    pathway(ax, PBOX, C_PREFILL, "PREFILL", "pathway")
    pathway(ax, DBOX, C_DECODE, "DECODE", "pathway")

    # Split: the whole sequence comes down into the layer, then each phase's own tokens
    # turn off to its lane. The cells are what turn, so the arrows carry no labels.
    # Backward, the same corridors run the other way, and stop at the tear.
    pair(ax, [(TRUNK_X, batch[1]), (TRUNK_X, OUT_Y), (SPLIT_X, OUT_Y)], INK, fwd="-",
         back=[(SPLIT_X, OUT_Y), (GRAD_STOP_X, OUT_Y)])
    # Named once, on the one stretch where the two directions run side by side and alone.
    lane_cx = (ZOOM[0] + 4 + SPLIT_X) / 2
    label(ax, lane_cx, OUT_Y + H + 3.4, "Forward", fs=FS_NOTE, color=MUTED)
    label(ax, lane_cx, OUT_Y - H - 3.4, "Backward", fs=FS_NOTE, color=MUTED)
    for y, c, n in ((PRE_Y, C_PREFILL, len(PROMPT)), (DEC_Y, C_DECODE, len(REPLY))):
        right = cells(ax, LANE_X, y, [c] * n)
        # Forward this is a split, so the heads are on the lanes; backward it is a merge
        # into the trunk, so the tributaries are headless.
        pair(ax, [(SPLIT_X, OUT_Y), (SPLIT_X, y), (LANE_X - 2, y)], readable(c), bwd="-")
        pair(ax, [(right + 2, y), (PBOX[0], y)], readable(c))

    # Merge: both lanes run into one corridor headless, and a single arrow leaves it --
    # the head belongs to the flow that continues, not to either tributary. Backward the
    # roles swap: one flow arrives and each pathway gets its own head.
    for rect, c in ((PBOX, C_PREFILL), (DBOX, C_DECODE)):
        y = (rect[1] + rect[3]) / 2
        pair(ax, [(rect[2], y), (MERGE_X, y), (MERGE_X, OUT_Y)], readable(c), fwd="-")
    pair(ax, [(MERGE_X, OUT_Y), (MERGED_X, OUT_Y)], INK)
    out = cells(ax, MERGED_X, OUT_Y,
                [C_PREFILL] * len(PROMPT) + [C_DECODE] * len(REPLY))

    # ---- Teacher: the same batch, frozen, undivided -------------------------------
    panel(ax, *TEACH, fc="white", ec="#3c3c3c", lw=1.0, r=2.4)
    # Both lines hang off the box's CENTRE, like the loss's do, rather than off its top
    # edge: measured from the top, the pill took its padding and the line under it got
    # whatever was left, which was almost nothing.
    tcx, tcy = (TEACH[0] + TEACH[2]) / 2, (TEACH[1] + TEACH[3]) / 2
    centred_pill(ax, tcx, tcy + 3.0, "TEACHER", C_FLOAT)
    label(ax, tcx, tcy - 4.2, "Frozen BF16", fs=FS_NOTE, color=MUTED)
    # The teacher now sits UNDER the batch rather than beside it, so the same batch reaches
    # it by stepping out of the panel's right end and dropping into its top.
    tfx = (TEACH[0] + TEACH[2]) / 2
    polyarrow(ax, [(batch[2], by), (tfx, by), (tfx, TEACH[3])], color=INK, head=HEAD)

    # ---- Loss ---------------------------------------------------------------------
    panel(ax, *LOSS, fc="white", ec="#3c3c3c", lw=1.0, r=2.4)
    lx, lcy = (LOSS[0] + LOSS[2]) / 2, (LOSS[1] + LOSS[3]) / 2
    # Half a point under the other headings: at 8pt this line runs to within 2 units of
    # the box on both sides, and it sets the column's width for everything else.
    label(ax, lx, lcy + 4.3, "KL(teacher‖student)", fs=FS_LABEL, color=INK,
          weight="bold")
    # Blue on one word: the positions the loss is scored on are exactly the ones the
    # decode pathway serves, and they are blue everywhere else in the figure.
    runs(ax, lx, lcy - 4.3, [("Reply", shade(C_DECODE, 0.15)), ("tokens only", MUTED)],
         FS_NOTE, gap=1.1)
    pair(ax, [(out + 2, OUT_Y), (LOSS[0], OUT_Y)], INK)
    arrow(ax, (lx, TEACH[1]), (lx, LOSS[3]), color=INK, head=HEAD)
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, OUT)
    from PIL import Image
    Image.open(snapshot(fig)).convert("RGB").save(PREVIEW)
    print(f"wrote {PREVIEW}")
