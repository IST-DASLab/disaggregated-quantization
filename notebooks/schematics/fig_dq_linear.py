# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "pillow"]
# ///
"""The three linear layers of Section 'Arbitrary disaggregated formats', as one series.

    uv run notebooks/schematics/fig_dq_linear.py        # writes all three

  quantized_linear.pdf     One stored weight, ONE pathway. Everything is autocast to
                           NVFP4, decode included, because there is only one pathway to
                           serve it with. The baseline the other two are read against.
  dq_linear.pdf            One stored weight, TWO pathways. Decode stops at BF16 and
                           skips the autocast; prefill keeps it. Nothing is stored twice.
  full_disag_linear.pdf    TWO layers. Each phase gets its own master, its own storage
                           and its own pathway, so the formats no longer have to agree --
                           drawn as two separate boxes because at serving time they are
                           two engines on two devices, each with its own tokens.

Read left to right in all three: what arrives, what it is converted to, what the matmul
consumes. The differences between the figures are structural, never stylistic -- same
columns, same band heights, same colours -- so a reader can diff them by eye.

All three are authored at half of ICLR's \\linewidth (2.75in) and share point sizes, so
they can sit in a row or a column without rescaling. The run also stitches the PNGs into
`notebooks/schematics/preview.png` -- read that to see the series in one look.

Formats follow qad/quantizers/{grids,blocked,dual}.py: LUT3 is LLOYD43_SIGNED_3BIT with
signed block scaling in groups of 16 (3 bit/weight + one E4M3 scale per 16 = 3.5), NVFP4
is E2M1 with an E4M3 block scale over the same groups (4 + 0.5 = 4.5).

Layout rules that hold everywhere here, and why (see .claude/skills/paper-schematics):
  - A pathway's WEIGHT row faces its storage, and its pill sits on the far side, so a
    weight feed never has to cross the row the activations arrive on.
  - Row centres are DERIVED from band height, never typed.
  - Nothing is drawn with an opaque background; the output is transparent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import (C_DECODE, C_FLOAT, C_PREFILL, C_SHARED, FS_CHIP, FS_LABEL,
                   FS_NOTE, INK, MUTED, arrow, canvas, chip, label, panel, pill,
                   polyarrow, readable, save, shade, snapshot, tint)

OUT = Path(__file__).resolve().parents[1] / "figures"      # alongside the result figures
PREVIEW = Path(__file__).resolve().parent / "preview.png"
DASH = (0, (2.6, 1.8))

# ---------------------------------------------------------------------------
# Metrics shared by every figure in the series. All x are relative to a box's left edge.
# ---------------------------------------------------------------------------
BOX_W = 114
PAD = 4                              # box edge -> panel edge
BAND_H = 38                          # pathway band: pill row + two chip rows
STO_H = 20                           # storage band: pill row + one text row
CHIP_H = 10.0
PILL_H, TEXT_H = 8.4, 5.5            # rendered ink heights at FS_PILL / FS_LABEL
GAP = (BAND_H - PILL_H - 2 * CHIP_H) / 4
STO_GAP = (STO_H - PILL_H - TEXT_H) / 3

COL_A, COL_B = 27, 62                # chip columns: as it arrives / as the matmul takes it
MM = (82, BOX_W - PAD - GAP)         # matmul node, inset from the panel like everything else
FEED = 10                            # where a weight leaves storage for its pathway
LEAD = 14                            # how far x and y run outside a box
STEM = -6                            # corridor a master uses when it cannot drop straight
MASTER_W, MASTER_H = 42, 16

LUT3 = "LUT3,  3.5 bit / weight"
NVFP4 = "NVFP4,  4.5 bit / weight"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def layer_box(ax, x0, y0, h):
    panel(ax, x0, y0, x0 + BOX_W, y0 + h, fc="white", ec="#3c3c3c", lw=1.0, r=2.4,
          zorder=0)


def storage(ax, x0, y0, text, color, name, pill_top=True):
    """Storage band with its bottom-left at (x0 + PAD, y0). Returns (bottom, top).

    `color` is the PHASE that owns this weight -- shared when both phases read it.
    """
    band = (x0 + PAD, y0, x0 + BOX_W - PAD, y0 + STO_H)
    panel(ax, *band)
    hi, lo = y0 + STO_H - STO_GAP, y0 + STO_GAP
    py, ty = (hi - PILL_H / 2, lo + TEXT_H / 2) if pill_top else (lo + PILL_H / 2,
                                                                 hi - TEXT_H / 2)
    pill(ax, band[0] + 2, py, name, color)
    label(ax, band[0] + 2, ty, text, fs=FS_LABEL, color=MUTED, ha="left")
    return y0, y0 + STO_H


COL = {"BF16": COL_A, "NVFP4": COL_B}


def chain(ax, x0, y, names, color):
    """A row's format chips, in fixed columns, ending in an arrow to the matmul.

    A chip earns its place only where something HAPPENS to the tensor -- a dequantize or a
    cast. Activations arriving in BF16 are not a step, so that row starts at whatever it
    is cast to, or, if nothing converts, with an EMPTY chain: then what feeds the row runs
    straight into the matmul. Columns are keyed by FORMAT rather than by position, so a
    format lands in the same column in every row of every figure even when the row above
    has more steps than the row below.

    A quantized chip is drawn in the PATHWAY's colour -- the phase that runs it owns it --
    so the same NVFP4 reads red under prefill and green where one pathway serves both.
    """
    entry = cursor = None
    for name in names:
        left, right, _ = chip(ax, x0 + COL[name], y, name,
                              C_FLOAT if name == "BF16" else color, CHIP_H)
        if entry is None:
            entry = left
        else:
            arrow(ax, (cursor, y), (left, y))
        cursor = right
    if entry is None:
        return x0 + MM[0]
    arrow(ax, (cursor, y), (x0 + MM[0], y))
    return entry


def pathway(ax, x0, y0, *, color, name, descriptor, weights, acts, op, storage_above):
    """One compute pathway. Weight row faces storage; pill goes on the far side.

    Returns (x_row, w_row, x_entry, w_entry) -- the caller owns everything that leaves the
    band (activations in, output out, the feed from storage), because those differ per
    figure.
    """
    band = (x0 + PAD, y0, x0 + BOX_W - PAD, y0 + BAND_H)
    panel(ax, *band)
    if storage_above:                                   # weight row on top, pill at foot
        w_row = band[3] - GAP - CHIP_H / 2
        x_row = w_row - CHIP_H - GAP
        pill_y = y0 + GAP + PILL_H / 2
    else:
        pill_y = band[3] - GAP - PILL_H / 2
        x_row = band[3] - 2 * GAP - PILL_H - CHIP_H / 2
        w_row = x_row - CHIP_H - GAP

    _, right = pill(ax, band[0] + 2, pill_y, name, color)
    label(ax, right + 3, pill_y, descriptor, fs=FS_LABEL, color=MUTED, ha="left")

    w_entry = chain(ax, x0, w_row, weights, color)
    x_entry = chain(ax, x0, x_row, acts, color)

    panel(ax, x0 + MM[0], min(x_row, w_row) - CHIP_H / 2, x0 + MM[1],
          max(x_row, w_row) + CHIP_H / 2, fc=tint(color, 0.9), ec=tint(color, 0.45),
          lw=0.8, r=1.8, zorder=2)
    label(ax, x0 + (MM[0] + MM[1]) / 2, (x_row + w_row) / 2, op, fs=FS_CHIP,
          color=shade(color, 0.4), weight="bold")
    return x_row, w_row, x_entry, w_entry


def feed(ax, x0, y_from, w_row, chip_left, color):
    """Storage hands its weight to a pathway: out of the band, along, into the chip."""
    polyarrow(ax, [(x0 + FEED, y_from), (x0 + FEED, w_row), (chip_left, w_row)],
              color=readable(color))


def master_card(ax, left, y0):
    """Dashed FP32 master card with its left edge at `left`. Returns (bottom, top, cx)."""
    rect = (left, y0, left + MASTER_W, y0 + MASTER_H)
    panel(ax, *rect, ec="#bcbcbc", lw=0.8, r=2.0, linestyle=DASH)
    gap = (MASTER_H - 2 * TEXT_H) / 3
    mx = (rect[0] + rect[2]) / 2
    label(ax, mx, rect[3] - gap - TEXT_H / 2, "FP32 master", fs=FS_NOTE, color=INK,
          weight="bold")
    label(ax, mx, rect[1] + gap + TEXT_H / 2, "Training only", fs=FS_NOTE, color=MUTED,
          style="italic")
    return rect[1], rect[3], mx


def master(ax, x0, y0, storage_edge):
    """Master card wired straight to the storage edge it sits over (or under)."""
    lo, hi, mx = master_card(ax, x0 + PAD, y0)
    arrow(ax, (mx, lo if lo > storage_edge else hi), (mx, storage_edge), color=MUTED,
          lw=0.9, ls=DASH, astyle="<|-|>")


def stream_in(ax, x0, chip_left, y):
    """Activations arriving, named OUTSIDE the layer."""
    arrow(ax, (x0 - LEAD, y), (chip_left, y))
    label(ax, x0 - LEAD / 2, y + 5.5, "x", fs=FS_CHIP, color=INK, style="italic",
          weight="bold")


def stream_out(ax, x0, y):
    """The output leaving, at the level the activations came in on."""
    arrow(ax, (x0 + MM[1], y), (x0 + BOX_W + LEAD, y))
    label(ax, x0 + BOX_W + LEAD / 2, y + 5.5, "y", fs=FS_CHIP, color=INK, style="italic",
          weight="bold")


# ---------------------------------------------------------------------------
# 1. Quantized linear -- one pathway, so everything is autocast
# ---------------------------------------------------------------------------
def build_quantized():
    h = PAD + BAND_H + PAD + STO_H + PAD
    fig, ax = canvas((-LEAD, BOX_W + LEAD), (0, h + 2 + 5 + MASTER_H + 2), 2.75)
    layer_box(ax, 0, 2, h)
    sto_lo, sto_hi = storage(ax, 0, 2 + PAD + BAND_H + PAD, LUT3, C_SHARED, "PREFILL+DECODE WEIGHTS")
    x_row, w_row, x_in, w_in = pathway(
        ax, 0, 2 + PAD, color=C_SHARED, name="PREFILL+DECODE PATH", descriptor="",
        weights=("BF16", "NVFP4"), acts=("NVFP4",), op="GEMM", storage_above=True)
    feed(ax, 0, sto_lo, w_row, w_in, C_SHARED)
    master(ax, 0, 2 + h + 5, sto_hi)
    stream_in(ax, 0, x_in, x_row)
    stream_out(ax, 0, x_row)
    return fig


# ---------------------------------------------------------------------------
# 2. Disaggregated -- one storage between two pathways
# ---------------------------------------------------------------------------
def build_disaggregated():
    dec_y = 2 + PAD
    sto_y = dec_y + BAND_H + PAD
    pre_y = sto_y + STO_H + PAD
    h = PAD + BAND_H + PAD + STO_H + PAD + BAND_H + PAD
    fig, ax = canvas((-LEAD, BOX_W + LEAD), (0, 2 + h + 5 + MASTER_H + 2), 2.75)
    layer_box(ax, 0, 2, h)
    sto_lo, sto_hi = storage(ax, 0, sto_y, LUT3, C_SHARED, "PREFILL+DECODE WEIGHTS")
    px, pw, pchip, pw_in = pathway(
        ax, 0, pre_y, color=C_PREFILL, name="PREFILL PATH", descriptor="Compute-bound",
        weights=("BF16", "NVFP4"), acts=("NVFP4",), op="GEMM",
        storage_above=False)
    dx, dw, dchip, dw_in = pathway(
        ax, 0, dec_y, color=C_DECODE, name="DECODE PATH", descriptor="Memory-bound",
        weights=("BF16",), acts=(), op="GEMV", storage_above=True)
    feed(ax, 0, sto_hi, pw, pw_in, C_SHARED)
    feed(ax, 0, sto_lo, dw, dw_in, C_SHARED)

    # The master cannot reach a storage that sits BETWEEN the pathways without meeting
    # something: x fans out down one margin and y fans in down the other. Crossing the x
    # trunk with a dashed training-only line, out in open margin, is the mildest option.
    my = 2 + h + 5 + MASTER_H / 2
    panel(ax, PAD, 2 + h + 5, PAD + MASTER_W, 2 + h + 5 + MASTER_H, ec="#bcbcbc", lw=0.8,
          r=2.0, linestyle=DASH)
    gap = (MASTER_H - 2 * TEXT_H) / 3
    mx = PAD + MASTER_W / 2
    label(ax, mx, 2 + h + 5 + MASTER_H - gap - TEXT_H / 2, "FP32 master", fs=FS_NOTE,
          color=INK, weight="bold")
    label(ax, mx, 2 + h + 5 + gap + TEXT_H / 2, "Training only", fs=FS_NOTE, color=MUTED,
          style="italic")
    polyarrow(ax, [(PAD, my), (STEM, my), (STEM, (sto_lo + sto_hi) / 2),
                   (PAD, (sto_lo + sto_hi) / 2)], color=MUTED, lw=0.8, ls=DASH,
              astyle="<|-|>")

    # One activation stream in, one output stream out: both fan at right angles, and both
    # are named once, on the single arrow.
    tail, split, merge = -LEAD, -10, BOX_W + 4
    arrow(ax, (tail, px), (pchip, px))
    polyarrow(ax, [(split, px), (split, dx), (dchip, dx)])
    label(ax, (tail + split) / 2, px + 5.5, "x", fs=FS_CHIP, color=INK, style="italic",
          weight="bold")
    polyarrow(ax, [(MM[1], (dx + dw) / 2), (merge, (dx + dw) / 2), (merge, px)],
              astyle="-")
    arrow(ax, (MM[1], px), (BOX_W + LEAD, px))
    label(ax, (merge + BOX_W + LEAD) / 2, px + 5.5, "y", fs=FS_CHIP, color=INK,
          style="italic", weight="bold")
    return fig


# ---------------------------------------------------------------------------
# 3. Fully disaggregated -- two layers, two masters, two storages
# ---------------------------------------------------------------------------
def build_full_disag():
    h = PAD + BAND_H + PAD + STO_H + PAD
    dec_y = 0
    pre_y = dec_y + h + 6
    mas_y = pre_y + h + 8
    fig, ax = canvas((-LEAD, BOX_W + LEAD), (-2, mas_y + MASTER_H + 4), 2.75)

    # Prefill: its own master, its own storage, in the format ITS matmul wants. The weight
    # is STORED as NVFP4, so nothing converts on the way in -- no dequantize, no
    # re-quantize. Only the activations still have to be cast.
    layer_box(ax, 0, pre_y, h)
    sto_lo, sto_hi = storage(ax, 0, pre_y + PAD + BAND_H + PAD, NVFP4, C_PREFILL, "PREFILL WEIGHTS")
    px, pw, px_in, pw_in = pathway(
        ax, 0, pre_y + PAD, color=C_PREFILL, name="PREFILL PATH", descriptor="Compute-bound",
        weights=(), acts=("NVFP4",), op="GEMM", storage_above=True)
    feed(ax, 0, sto_lo, pw, pw_in, C_PREFILL)
    stream_in(ax, 0, px_in, px)
    stream_out(ax, 0, px)

    # Decode: the same layer again, sharing nothing with it -- not the master, not the
    # stored weight, not even the bit width. That is the whole point of the figure.
    layer_box(ax, 0, dec_y, h)
    dsto_lo, dsto_hi = storage(ax, 0, dec_y + PAD + BAND_H + PAD, LUT3, C_DECODE, "DECODE WEIGHTS")
    dx, dw, dx_in, dw_in = pathway(
        ax, 0, dec_y + PAD, color=C_DECODE, name="DECODE PATH", descriptor="Memory-bound",
        weights=("BF16",), acts=(), op="GEMV", storage_above=True)
    feed(ax, 0, dsto_lo, dw, dw_in, C_DECODE)
    stream_in(ax, 0, dx_in, dx)
    stream_out(ax, 0, dx)

    # Both masters sit above, side by side. Prefill's drops straight into the layer under
    # it; decode's has to loop down the margin past that whole layer, so it takes the left
    # card -- leaving from the left edge keeps the loop clear of the other card.
    dlo, dhi, _ = master_card(ax, PAD, mas_y)
    polyarrow(ax, [(PAD, (dlo + dhi) / 2), (STEM, (dlo + dhi) / 2),
                   (STEM, (dsto_lo + dsto_hi) / 2), (PAD, (dsto_lo + dsto_hi) / 2)],
              color=MUTED, lw=0.8, ls=DASH, astyle="<|-|>")
    plo, _, pmx = master_card(ax, BOX_W - PAD - MASTER_W, mas_y)
    arrow(ax, (pmx, plo), (pmx, sto_hi), color=MUTED, lw=0.9, ls=DASH, astyle="<|-|>")
    return fig


def stitch(shots, out, gutter=26):
    """Contact sheet of the family, for previewing the series in one look.

    Not a paper artifact -- it lives outside figures/ so that directory stays exactly the
    set of PDFs the paper includes. The rasters come from snapshot() at one dpi, so
    pasting them at native size keeps their relative scale honest.
    """
    from PIL import Image
    ims = [Image.open(b).convert("RGB") for b in shots]
    sheet = Image.new("RGB", (sum(i.width for i in ims) + gutter * (len(ims) + 1),
                              max(i.height for i in ims) + 2 * gutter), "white")
    x = gutter
    for im in ims:
        sheet.paste(im, (x, gutter))                   # top-aligned, like a figure row
        x += im.width + gutter
    sheet.save(out)
    print(f"wrote {out} ({sheet.width}x{sheet.height})")


SERIES = [(build_quantized, "quantized_linear"),
          (build_disaggregated, "dq_linear"),
          (build_full_disag, "full_disag_linear")]

if __name__ == "__main__":
    shots = []
    for builder, name in SERIES:
        fig = builder()
        save(fig, OUT / f"{name}.pdf")
        shots.append(snapshot(fig))
    stitch(shots, PREVIEW)
