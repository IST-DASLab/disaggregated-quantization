"""Paper figure: prefill latency, blocks streamed off the SSD vs resident on GPU.

    python plot_for_paper.py

Same data and same claims as plot_prefill.py, restyled for print: a 2x2 grid on a 6x4 inch
canvas so it sits beside the decode bar chart at the same size, the house palette, no figure
title (the caption carries it), and the legend inside the last panel rather than stealing a
strip of the figure. Type is sized for that canvas -- 8-10 pt, i.e. roughly a paper's body
size once the figure is placed at column width; a 1x4 strip's larger type would be absurd here.

Panels share BOTH axes: over 2k-16k every model sits in the same latency band, so a common
y makes the four panels directly comparable and shows the models separating with size, and
sharing x lets the inner tick labels go so only the outer rim carries them.

The sweep covers 128-32k; this figure shows 2k-16k. Outside that window the curves carry no
information for the paper -- below 2k the offloaded curve is a flat line pinned to the drive,
above 16k every curve has converged -- and including them compresses the crossover, which is
the part being argued about.

BF16 SSD is measured and in the CSV but not drawn. It is drive-bound across this entire
window (parity only at 16k, and only just), so it is a flat line that adds nothing but ink;
the BF16 resident curve is the baseline that matters and the NVFP4 pair carries the claim.

Encoding: colour is the weight format, line style is where the weights live.
"""

import argparse
import re
import pathlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Original six, unchanged.
RED    = "#cd001a"
ORANGE = "#ef6a00"
YELLOW = "#f2cd00"
GREEN  = "#79c300"
BLUE   = "#1961ae"
PURPLE = "#61007d"
# Extensions, same flat saturated register.
PINK   = "#e5006d"
BROWN  = "#8a5a2b"
OLIVE  = "#6b8f00"
TEAL   = "#00a19a"
FOREST = "#1f7a33"
NAVY   = "#0b2d6b"
GREY   = "#6e6e6e"
SKY    = "#3fa9f5"   # 2-bit family: distinct from BLUE (lloyd21) and NAVY
CYAN   = "#00b8d4"   # 2-bit NVR decode, upcast family
INDIGO = "#534bae"   # 2-bit NVR decode, split family

HERE = Path(__file__).parent
# Figures land in the repo's notebooks/figures, next to the notebook that uses them, rather
# than beside the code that made them.
FIGDIR = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CSV = HERE / "offload_prefill.csv"
# One figure per model family, each a 2x2 at the same size, so they can sit side by side
# or stack in a paper. `{family}_prefill_offload_paper.{pdf,png}`.
OUT_FMT = str(FIGDIR / "{family}_prefill_offload_paper")

# Load-only floors, measured per (model, quant) by load_floor.py at that configuration's
# real block size. NOT derived from one throughput figure: the drive gives 2.63 GB/s on
# 0.6B's 8 MiB NVFP4 blocks and 5.92 GB/s on 8B's 368 MiB bf16 ones, so a single rate is
# ~2.3x too optimistic at the small end.
FLOOR_CSV = HERE / "load_floor.csv"

# bf16 is the baseline, so it takes the cool hue; NVFP4 is the intervention and takes the
# warm one. RED against BLUE is the strongest separation the palette offers, and survives
# greyscale printing because the two differ in luminance as well as hue.
QUANT = ("bf16", "nvfp4")
QUANT_LABEL = {"bf16": "BF16", "nvfp4": "NVFP4"}
STYLE = {
    "resident": dict(ls="-", marker="o", lw=2.0, ms=5.0),
    "ssd": dict(ls="--", marker="s", lw=2.0, ms=5.0),
}
MODE_LABEL = {"resident": " prefill", "ssd": " ODP"}
MODE_COLOR = {"resident": RED, "ssd": BLUE}
MODES = ("resident", "ssd")
# Measured, in the CSV, deliberately not drawn -- see the module docstring.
SKIP = {
    ("bf16", "ssd"),
    ("bf16", "resident"),
}


SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bmBM])(?![a-zA-Z])")


def params_b(model: str) -> float:
    """Parameter count in billions, so panels order by size rather than alphabetically.

    Handles both naming schemes in the CSV: 'Qwen/Qwen3-1.7B' -> 1.7, and Gemma's, which
    breaks a naive rsplit('-') twice -- 'google/gemma-3-1b-it' ends in the instruction-tuned
    suffix rather than the size, and 'google/gemma-3-270m' is in millions, so parsed as a
    bare number it would sort as the LARGEST model instead of the smallest.
    """
    hits = SIZE_RE.findall(model)
    if not hits:
        return float("inf")
    n, unit = hits[-1]
    return float(n) / (1000.0 if unit in "mM" else 1.0)


def family_of(model: str) -> str:
    """'qwen3' / 'gemma3', used to split the CSV into one figure per family."""
    return "gemma3" if "gemma" in model.lower() else "qwen3"


def short_name(model: str) -> str:
    """'google/gemma-3-1b-it' -> 'Gemma3-1B'; 'Qwen/Qwen3-8B' -> 'Qwen3-8B'."""
    n, unit = SIZE_RE.findall(model)[-1]
    stem = "Gemma3" if family_of(model) == "gemma3" else "Qwen3"
    return f"{stem}-{n}{unit.upper()}"


def plot_family(df, load_floor, family: str):
    df = df[df["model"].map(family_of) == family]
    if df.empty:
        print(f"no rows for {family}, skipping")
        return
    models = sorted(df["model"].unique(), key=params_b)

    # 2x2 at 6x4 inches, so this sits beside the decode figure at the same size in a paper.
    # A 1x4 strip reads better in isolation but cannot be paired with a bar chart.
    ncol = 2 if len(models) > 2 else len(models)
    nrow = -(-len(models) // ncol)
    fig, axes_grid = plt.subplots(nrow, ncol, figsize=(6, 4), sharex=True, sharey=False)
    axes = list(np.atleast_1d(axes_grid).ravel())
    for extra in axes[len(models):]:          # unused cell if the count is odd
        extra.set_visible(False)

    seq_len_rule = (df["seq_len"] >= 2048) & (df["seq_len"] <= 32768)
    # seq_len_rule = (df["seq_len"] >= 0)

    floors = {}
    for ax, model in zip(axes, models):
        sub_m = df[df["model"] == model]
        n_blocks = int(sub_m["n_blocks"].iloc[0])

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        for quant in QUANT:
            sub = sub_m[sub_m["quant"] == quant]
            if sub.empty:
                continue
            # Drive bound: every forward re-reads the whole model off the SSD.
            floor = load_floor.get((model, quant))
            for mode in MODES:
                if (quant, mode) in SKIP:
                    continue
                cur = sub[sub["mode"] == mode].sort_values("seq_len")
                cur = cur[seq_len_rule]
                if cur.empty:
                    continue
                ax.plot(cur["seq_len"], cur["latency_ms"], color=MODE_COLOR[mode],
                        label=f"{QUANT_LABEL[quant]}{MODE_LABEL[mode]}", **STYLE[mode])
            if quant == "nvfp4" and floor is not None:
                # The floor the NVFP4 offload curve rides on: n_blocks x block / 5.7 GB/s.
                # Only NVFP4's is drawn, since BF16's offload curve is not on the figure.
                ax.axhline(floor, color=MODE_COLOR[mode], ls=(0, (1, 3)), lw=1.2, alpha=0.85)
                floors[model] = floor

        # Powers of ten only. Under about 1.5 decades matplotlib starts LABELLING the
        # minor ticks as well (2x10^2, 3x10^2, ...), so the 8B panel carried nine y labels
        # while every other panel carried two, and the four panels stopped looking like one
        # figure. The minor ticks themselves stay -- they carry the grid that makes a log
        # axis readable between decades -- only their labels go.
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_title(short_name(model), fontsize=10, pad=3)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", color="#dddddd", lw=0.7)
        ax.grid(True, which="minor", color="#f0f0f0", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    seq = sorted(df[seq_len_rule]["seq_len"].unique())
    axes[0].set_xticks(seq)
    axes[0].set_xticklabels([f"{s // 1024}k" if s >= 1024 else str(s) for s in seq])
    # Outer labels only: at 6x4 a label per panel is most of the canvas.
    for i, ax in enumerate(axes[:len(models)]):
        if i // ncol == nrow - 1:
            ax.set_xlabel("Sequence length, tokens", fontsize=12)
        if i % ncol == 0:
            ax.set_ylabel("Prefill latency, ms", fontsize=12)

    # Label the NVFP4 drive floor once -- it is the same construction on every panel and
    # only needs explaining once. On the leftmost panel, right-aligned at its right edge:
    # that corner is empty there, whereas on the 8B panel the label runs into the NVFP4 SSD
    # curve, which sits just above its own floor. Below the line rather than above, for the
    # same reason.
    ax_first = axes[0]
    if models[0] in floors:
        ax_first.annotate(fr"SSD$\rightarrow$DRAM latency",
                          xy=(1.00, floors[models[0]]),
                          xycoords=ax_first.get_yaxis_transform(),
                          xytext=(0, -6), textcoords="offset points",
                          ha="right", va="top", fontsize=8, color=BLUE)

    # Inside the rightmost panel: a figure-level legend would cost a strip of the canvas,
    # and at this aspect ratio that is a large fraction of it.
    axes[-1].legend(loc="lower right", frameon=False, fontsize=10, handlelength=2.0,
                    borderaxespad=0.4, labelspacing=0.3)

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.5)
    out = OUT_FMT.format(family=family)
    for ext in ("pdf",):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", transparent=True)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", nargs="+", default=["all"],
                    help="qwen3, gemma3, or all (default: every family in the CSV)")
    args = ap.parse_args()

    df = pd.read_csv(CSV).dropna(subset=["latency_ms"])
    load_floor = (pd.read_csv(FLOOR_CSV).set_index(["model", "quant"])["load_ms"].to_dict()
                  if FLOOR_CSV.exists() else {})
    fams = (sorted({family_of(m) for m in df["model"].unique()})
            if "all" in args.family else args.family)
    for fam in fams:
        plot_family(df, load_floor, fam)


if __name__ == "__main__":
    main()
