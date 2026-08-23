# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Vertical-rhythm audit for a schematic, in LAYOUT UNITS.

    uv run .claude/skills/paper-schematics/scripts/audit_layout.py notebooks/schematics/fig_x.py

Asks matplotlib where every artist actually landed instead of probing the PNG: pixel
probing needs a unit calibration that `bbox_inches="tight"` invalidates, and detecting
panels by fill colour breaks as soon as a chip shares that colour.

Runs every `build*()` in the script. For each grey panel it finds, it reports the padding
above the topmost ink, the gaps between rows, and the padding below the bottom-most ink.
Even numbers mean an even rhythm; that is the whole check.
"""

import importlib.util
import sys
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba


def load(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(items, tol=0.5):
    """Merge vertically-overlapping items into rows, top-down."""
    merged = []
    for lo, hi in sorted(items, key=lambda i: -i[1]):
        if merged and hi > merged[-1][0] - tol:          # still inside the open row
            merged[-1][0] = min(merged[-1][0], lo)
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def audit(fig, panel_rgba):
    ax = fig.axes[0]
    rend = FigureCanvasAgg(fig).get_renderer()
    inv = ax.transData.inverted()
    boxes = [(a, a.get_window_extent(rend).transformed(inv))
             for a in list(ax.texts) + list(ax.patches)]
    ink = [(bb.y0, bb.y1, bb.x0, bb.x1) for _, bb in boxes if bb.height > 0]

    bands = [bb for a, bb in boxes
             if hasattr(a, "get_facecolor") and a.get_facecolor() == panel_rgba]
    print(f"  {fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in, "
          f"{len(bands)} panels")
    for bb in sorted(bands, key=lambda b: -b.y1):
        # The panel's OWN patch is not ink: it fills the band by definition and would
        # report every padding as zero.
        inside = [(lo, hi) for lo, hi, x0, x1 in ink
                  if lo > bb.y0 - 0.1 and hi < bb.y1 + 0.1 and x0 > bb.x0 - 0.1
                  and x1 < bb.x1 + 0.1
                  and not (hi - lo > bb.height - 1 and x1 - x0 > bb.width - 1)]
        if len(inside) < 2:
            continue
        r = rows(inside)
        gaps = [round(r[i][0] - r[i + 1][1], 2) for i in range(len(r) - 1)]
        print(f"    band {bb.y0:6.1f}..{bb.y1:6.1f}  top {bb.y1 - r[0][1]:5.2f}  "
              f"gaps {gaps}  bottom {r[-1][0] - bb.y0:5.2f}")


def main(path):
    mod = load(path)
    import style                                        # same dir, put on sys.path above
    panel_rgba = to_rgba(style.PANEL_BG)
    builders = [(n, f) for n, f in sorted(vars(mod).items())
                if n.startswith("build") and callable(f)]
    for name, fn in builders:
        print(f"{Path(path).name}::{name}")
        audit(fn(), panel_rgba)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "notebooks/schematics/fig_dq_linear.py")
