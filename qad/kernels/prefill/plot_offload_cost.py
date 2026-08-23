"""What fraction of an offloaded prefill is the drive, per model.

    python plot_offload_cost.py

The companion to plot_for_paper.py. That figure plots latency; this one plots the quantity
the latency curves are really about:

    offload cost = (offloaded - resident) / offloaded

the share of the offloaded forward NOT hidden behind compute. It is bounded in [0, 1),
which the raw SSD/resident ratio is not -- that ratio runs to 30-40x at short context and
squashes the whole interesting region against 1.0. 0% means the fetch is completely hidden
and offloading is free.

Reading the figure: every model starts at ~97%, because a short prefill IS a model load
with a forward attached, and that number is a property of the drive rather than the model
-- which is why all eight curves start on top of each other. They separate through the
2k-8k knee and land at a few percent by 16k.

Gemma sits above Qwen at the 8k knee (16-26% against 5-12%). Same cause as everywhere else
in this directory: Gemma 3's 5:1 sliding window makes its per-block compute cheaper, so
there is less compute to hide the same fetch behind and its crossover arrives later.

Rows are model families and columns are size, so a column is roughly a size class and a row
is an architecture. Axes are shared, so panels are directly comparable -- the point of the
figure is that the curves are the SAME SHAPE at different offsets.
"""

import re
import pathlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

RED    = "#cd001a"
ORANGE = "#ef6a00"
BLUE   = "#1961ae"
GREY   = "#6e6e6e"

HERE = Path(__file__).parent
# Figures land in the repo's notebooks/figures, next to the notebook that uses them, rather
# than beside the code that made them.
FIGDIR = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CSV = HERE / "offload_prefill.csv"
OUT = FIGDIR / "offload_cost"

PLOT_BF16 = False

# Same encoding as the latency figures: colour is the weight format.
QUANT = {"bf16": BLUE, "nvfp4": RED}
QUANT_LABEL = {"bf16": "BF16", "nvfp4": "NVFP4"}
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bmBM])(?![a-zA-Z])")
# Same window as plot_for_paper.py, so the two figures line up column for column.
SEQ_LO, SEQ_HI = 256, 32768


def params_b(model: str) -> float:
    hits = SIZE_RE.findall(model)
    if not hits:
        return float("inf")
    n, unit = hits[-1]
    return float(n) / (1000.0 if unit in "mM" else 1.0)


def family_of(model: str) -> str:
    return "gemma3" if "gemma" in model.lower() else "qwen3"


def short_name(model: str) -> str:
    n, unit = SIZE_RE.findall(model)[-1]
    stem = "Gemma3" if family_of(model) == "gemma3" else "Qwen3"
    return f"{stem}-{n}{unit.upper()}"


def main():
    df = pd.read_csv(CSV).dropna(subset=["latency_ms"])
    df = df[(df["seq_len"] >= SEQ_LO) & (df["seq_len"] <= SEQ_HI)]

    # Qwen on top, Gemma below -- one row per architecture.
    rows = [f for f in ("qwen3", "gemma3") if (df["model"].map(family_of) == f).any()]
    by_family = {f: sorted(df[df["model"].map(family_of) == f]["model"].unique(),
                           key=params_b) for f in rows}
    ncol = max(len(v) for v in by_family.values())

    fig, axes = plt.subplots(len(rows), ncol, figsize=(18, 4), sharex=True, sharey=True,
                             squeeze=False)

    for r, fam in enumerate(rows):
        models = by_family[fam]
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(models):
                ax.set_visible(False)
                continue
            model = models[c]
            sub = df[df["model"] == model]
            for quant, colour in QUANT.items():
                if quant == "bf16" and not PLOT_BF16:
                    continue
                q = sub[sub["quant"] == quant]
                wide = q.pivot_table(index="seq_len", columns="mode",
                                     values="latency_ms")
                if not {"resident", "ssd"} <= set(wide.columns):
                    continue
                pct = 100 * (wide["ssd"] - wide["resident"]) / wide["ssd"]
                ax.plot(pct.index, pct.values, color=colour, marker="o", ms=4.0, lw=2.0,
                        label=QUANT_LABEL[quant])

            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 100)
            ax.set_title(short_name(model), fontsize=12, pad=3)
            ax.tick_params(labelsize=8)
            ax.grid(True, which="major", color="#dddddd", lw=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    seq = sorted(df["seq_len"].unique())
    axes[0][0].set_xticks(seq)
    axes[0][0].set_xticklabels([f"{s // 1024}k" if s >= 1024 else str(s) for s in seq])
    for c in range(ncol):
        axes[-1][c].set_xlabel("Sequence length, tokens", fontsize=12)
    for r in range(len(rows)):
        axes[r][0].set_ylabel("Offload cost, %", fontsize=12)

    # In the panel with the most headroom: by 16k every curve is on the floor, so the top
    # right of the last panel is empty on every model. Only when there is something to
    # disambiguate -- a one-entry legend is ink explaining what the caption already says.
    if PLOT_BF16:
        axes[0][-1].legend(loc="upper right", frameon=False, fontsize=10, handlelength=1.6,
                           borderaxespad=0.3, labelspacing=0.3)

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
    for ext in ("pdf",):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", transparent=True)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
