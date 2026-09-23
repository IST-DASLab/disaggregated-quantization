# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "pillow"]
# ///
"""Render the paper's QADD layout in the blog's NVIDIA Dark / Focus style.

    uv run notebooks/schematics/fig_qadd_blog.py

Reuses the measured layout without changing the paper figure or its exports.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgba
from matplotlib.text import Text

import fig_qadd_training as scheme
import style

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "notebooks/blogpost/media/qadd_training_blog.png"
BACKGROUND = "#0B0F19"
INK = "#FFFFFF"
MUTED = "#F7F7F7"
EDGE = "#757575"
PANEL = "#313131"
PREFILL = "#76B900"
DECODE = "#757575"


def build():
    for suffix in ("Rg", "Md"):
        font_manager.fontManager.addfont(
            ROOT / f"01_BRAND_ASSETS/fonts/NVIDIASans_{suffix}.ttf"
        )
    # Set the font before measuring words and pills, not after positioning them.
    style.FONT_FAMILY = "NVIDIA Sans"
    fig = scheme.build()
    fig.set_facecolor(BACKGROUND)
    ax = fig.axes[0]
    ax.set_facecolor(BACKGROUND)

    colors = {
        to_hex(style.INK): INK,
        to_hex(style.MUTED): MUTED,
        "#3c3c3c": EDGE,
        "#ffffff": INK,
    }
    for old, new in ((scheme.C_PREFILL, PREFILL),
                     (scheme.C_DECODE, DECODE), (scheme.C_FLOAT, EDGE)):
        colors[to_hex(old)] = new
        colors[to_hex(style.readable(old))] = new
        for amount in (0.4, 0.72, 0.78, 0.9):
            # The paper's light token fills become restrained dark fills.
            colors[to_hex(style.tint(old, amount))] = style.shade(new, amount)
        for amount in (0.15, 0.25, 0.3, 0.45):
            colors[to_hex(style.shade(old, amount))] = (
                style.tint(new, 0.35) if old == scheme.C_PREFILL else MUTED
            )

    def recolor(color):
        rgba = to_rgba(color)
        return to_rgba(colors.get(to_hex(rgba), to_hex(rgba)), rgba[3])

    for line in ax.lines:
        line.set_color(recolor(line.get_color()))
    for patch in ax.patches:
        face = patch.get_facecolor()
        # Large panels and the torn layer interior stay black; solid pills
        # become charcoal with bright outlines for readable white headings.
        if to_hex(face) == "#ffffff":
            patch.set_facecolor(to_rgba(BACKGROUND, face[3]))
        elif to_hex(face) in (scheme.C_PREFILL, scheme.C_DECODE, scheme.C_FLOAT):
            patch.set_facecolor(to_rgba(PANEL, face[3]))
        else:
            patch.set_facecolor(recolor(face))
        patch.set_edgecolor(recolor(patch.get_edgecolor()))
    for text in fig.findobj(Text):
        text.set_color(recolor(text.get_color()))
        if text.get_fontweight() == "bold":
            text.set_fontweight("medium")
        if text.get_text() in ("STUDENT", "TEACHER", "PREFILL", "DECODE"):
            text.set_text(text.get_text().title())
        # NVIDIA Sans has no double-vertical-line glyph.
        if "‖" in text.get_text():
            text.set_text(text.get_text().replace("‖", " || "))
    return fig


if __name__ == "__main__":
    fig = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=1920 / scheme.FIG_W_IN, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")
