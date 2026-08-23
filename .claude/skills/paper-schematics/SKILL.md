---
name: paper-schematics
description: "Builds and edits the paper's vector schematics — the DRAWN figures (box-and-arrow dataflow, format/pipeline diagrams), not the plotted results. Scripts and PDFs live in notebooks/schematics/. Use when asked to make, restyle, or re-lay-out a diagram for the paper, e.g. 'illustrate the upcasted format', 'make that box green', 'the arrows overlap'."
---

# Paper schematics

Result figures come from `notebooks/plots.ipynb`. DRAWN figures come from here:
`notebooks/schematics/fig_*.py` on top of `style.py` (palette + primitives). Both write
their PDFs to the SAME place, `notebooks/figures/`, which is what the paper includes.

Only PDFs land there. Rasters exist solely to be looked at during iteration: `snapshot()`
renders one in memory and `stitch()` assembles them into `notebooks/schematics/preview.png`,
so no PNG is ever written beside a PDF where it could go stale or be \includegraphics'd
by accident.

    uv run notebooks/schematics/fig_dq_linear.py     # PDFs + preview.png, prints sizes

One script per FAMILY, not per figure: `fig_dq_linear.py` emits all three linear-layer
diagrams from shared components (`layer_box`, `storage`, `pathway`, `feed`, `master`,
`stream_in/out`) and shared metrics. A series that a reader is meant to diff by eye has
to keep identical columns, band heights and colours, and that only survives edits if
there is exactly one definition of each. Add a figure to a family by writing another
`build_*()`, not by copying the script.

Everything below was paid for in review round-trips. The rules are ordered by how much
they cost.

## 1. Ask the width first, then cut text to fit it

The figure's typeset width is the single most load-bearing decision, because point sizes
are ABSOLUTE: at half of `\linewidth` (2.75in) the type has to sit near 8-9pt to read,
and that leaves room for roughly a dozen short strings. Full width buys ~2x that.

So: author at the final size (`FIG_W_IN`), never larger — a figure that LaTeX scales down
to `\linewidth` has every label shrunk by the same factor, which is exactly the "text is
too small" complaint. Then delete every word the drawing already says. A `BF16 -> NVFP4`
arrow does not need "quantize" on it; a matmul node does not need a name under it. Format
details (E4M3 scales, group size, grid names) belong in the caption, not the figure.

Cutting text is not cosmetic — it is what frees the space that larger type needs.

## 2. Settle the arrow topology before any coordinates

Draw the graph on paper first and check it is PLANAR with the fixed nodes where they must
go (inputs left, outputs right, anything "outside the layer" outside the box). Crossings
are not a rendering detail you can fix later; they are a property of the arrangement, and
discovering one after placing 40 coordinates means re-deriving all of them.

What this cost, and what fixed it, in `fig_dq_linear.py`:

- Storage feeding two pathways from the TOP forces a trunk down one margin, which then
  crosses whatever enters from that side. Putting STORAGE in the MIDDLE band, between the
  two pathways, makes both weight feeds short right-angle elbows and removes the trunk.
- Each pathway must then put its WEIGHT row on the side FACING storage, so the feed never
  has to pass the row the activations arrive on.
- A single input that splits and a single output that merges put a vertical in BOTH
  margins. With the master weight also outside, one crossing becomes unavoidable — the
  planar embedding does not exist. Pick which crossing you want (dashed "training-only"
  line over a solid inference arrow, out in open margin, is the least confusing) and place
  it deliberately rather than discovering it.

A branch merging into another line is drawn headless (`astyle="-"`); the head belongs to
the line the merged flow leaves by.

## 3. Derive geometry; hand-placed numbers rot

Put every coordinate in one named block at the top. Compute row centres from the band
height, pill height and chip height with equal gaps — do not type row centres. Bands get
retuned constantly, and hand-placed rows drift into 2.0-top / 0.8-bottom asymmetries that
are invisible until someone calls them ugly.

    GAP = ((PRE[3] - PRE[1]) - PILL_H - 2 * CHIP_H) / 4
    PRE_PILL = PRE[3] - GAP - PILL_H / 2
    X_ROW    = PRE[3] - 2 * GAP - PILL_H - CHIP_H / 2

Inset nested boxes by that same `GAP` (`MM = (82, BX1 - GAP)`); a matmul node flush
against its panel edge reads as a mistake.

## 4. Iterate: edit -> render -> LOOK -> audit

Never report a figure done without reading its PNG. Batch several edits per render; each
cycle is one `uv run` plus one `Read`. A family script stitches its PNGs into
`notebooks/schematics/preview.png` (see `stitch()`), so the whole series is ONE image to
read rather than three -- and side by side it exposes inconsistencies between figures
that none of them shows alone.

For anything about spacing, do not eyeball it and do not parse the PNG (panel detection
is fragile, and `bbox_inches` moves the origin). Ask matplotlib where things landed:

    uv run .claude/skills/paper-schematics/scripts/audit_layout.py \
        notebooks/schematics/fig_dq_linear.py

It runs every `build*()` in the script and prints each panel's top/bottom padding and
internal gaps in LAYOUT UNITS.

Before saying done: read the PNG, run the audit, and check the PDF page size
(`pypdf`) against `FIG_W_IN`.

## 5. Rules that each cost one round-trip

- **Nothing opaque.** The PDF is saved transparent, so a white `bbox` behind a label is
  the wrong colour on every background but the one you checked it against. Give each label
  empty space in the layout instead. This is a layout constraint, not a styling one.
- **`bbox_inches="tight"` ADDS `pad_inches=0.1` per side** — 0.2in of growth, which undoes
  rule 1. `save()` pins `pad_inches=0.02` and prints the authored size; keep both.
- **Measure text, never estimate it.** `chip()`/`pill()` size themselves from
  `text_extent()`. A pill with a hardcoded height clips its caps the moment the point size
  changes; a chip guessed at 0.5em/char overflows in bold.
- **Light fills need dark type.** `pill()` runs `readable()`, which darkens fills whose
  luminance is too high — white on raw `GREEN` (#79c300) is ~2:1 and unreadable at 8pt.
- **Put a row label on the side away from the other row**, or it reads as labelling that
  one. In mirrored bands this means a per-band sign, not a constant offset.
- **Position child text relative to its parent box** (`MASTER[3] - 4`, not `123`), or it
  silently ends up outside the box the next time the box moves.

## 6. Style contract

`style.py` holds the palette verbatim from `notebooks/plots.ipynb`, so schematics sit next
to `bars_disag.pdf` unchanged. Colour is assigned by ROLE here (storage / prefill / decode
/ unquantized), not by method as in the result figures, but prefill keeps `METHODS["nvfp4"]`
red so the bar charts carry over. `pdf.fonttype = 42` is set: matplotlib's Type 3 default
is rejected by some camera-ready checkers.

Reuse the primitives — `panel`, `chip`, `pill`, `label`, `arrow`, `polyarrow`, `tint`,
`shade`. Add to `style.py` only what a second figure would also want; delete primitives
nothing draws with any more.
