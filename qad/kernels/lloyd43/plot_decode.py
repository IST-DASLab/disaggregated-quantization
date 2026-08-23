"""Paper figure: decode speedup of the linear layers, 3-bit and 4-bit weight formats.

    python vllm_serve.py --model Qwen/Qwen3-8B --quant none lloyd43 lloyd21 \
        --out benchmarks/vllm_decode.csv                   # measure, per model
    python plot_decode.py                                  # every family -> PDF + PNG
    python plot_decode.py --family gemma3                  # just one

REAL vLLM END-TO-END, not the synthetic per-shape estimate that used to live in bench.py.
That timed each projection in isolation and weighted it by how often a model contains it;
excludes attention, norms, the lm_head and every launch overhead, and it is sensitive to a
choice the served model has already made for you -- weighting by unfused q/k/v rather than
the merged QKV vLLM actually runs moves the answer by up to 0.4x. Keep that estimate for
attributing a result to the kernel; plot this for what a user gets.

One figure per model family, same axes and same style, so Qwen3 and Gemma 3 can be read
against each other. Gemma 3 is the harder case for a weight-only format: head_dim is 256
with as few as ONE kv head, so its k/v projections are 256 rows tall -- small enough that
launch overhead, not bandwidth, sets the floor.

Styled to match ../prefill/plot_for_paper.py: house palette, larger type, initial-caps
axes, in-axes legend, no figure title (the caption carries it), PDF and PNG.

WHAT IS BEING COMPARED
----------------------
Every bar is a weight-only or weight+activation format measured against dense bf16 on the
SAME weights -- lloyd43 packs them, both NVFP4 variants quantize them, none of it comes
from a third-party checkpoint. So the chart is about the format and its kernel, not about
whose calibration was better.

At batch 1 this is a GEMV and bandwidth-bound, so bytes per weight is very nearly the whole
story:

    bf16      2.0
    NVFP4     4/8 + 1/16 = 0.5625      (fp4 + one fp8 scale per 16)
    lloyd43   3/8 + 1/16 = 0.4375      (3-bit index + one fp8 scale per 16)

which puts lloyd43 1.29x ahead of NVFP4 on traffic alone. Anything short of that is kernel
efficiency, and anything past it wants explaining.

W4A4 versus W4A16 is the other axis. Decode has no arithmetic intensity to speak of, so the
fp4 tensor cores have nothing to do, while W4A4 still pays to quantize the activation on
every call. W4A16 should therefore win at batch 1 even though W4A4 wins at prefill -- the
two are not competing for the same job.

There is no roofline series: the memory ceiling is a bound on lloyd43's traffic
specifically (HANDOFF 5.1) and is not a quantity the NVFP4 bars can be read against.
"""

import argparse
import re
import pathlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

HERE = Path(__file__).parent
# Figures land in the repo's notebooks/figures, next to the notebook that uses them, rather
# than beside the code that made them.
FIGDIR = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# REAL vLLM end-to-end decode, from `vllm_serve.py --out benchmarks/vllm_decode.csv`.
# Not the synthetic per-shape estimate in {family}_model_speedup.csv: that one times each
# projection in isolation and weights it by how often the model contains it, which excludes
# attention, norms, the lm_head and every launch overhead, and which weighting to use
# (fused q/k/v or not) changes the answer by up to 0.4x. This is whatever the served model
# actually does.
CSV = HERE / "benchmarks" / "vllm_decode.csv"
OUT_FMT = str(FIGDIR / "{family}_decode_speedup")
FAMILY_LABEL = {"qwen3": "Qwen3", "gemma3": "Gemma 3"}
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bmBM])(?![a-zA-Z])")


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
    return f"{n}{unit.upper()}"

# lloyd43 is the subject and takes the cool hue; the two NVFP4 variants take warm ones.
# RED and ORANGE are close, which is deliberate -- they are the same format at different
# activation widths. Close is not indistinguishable: the pair separates by dE 14.4 under
# deuteranopia and 16.4 under normal vision, comfortably clear of the dE 8 floor, so the
# bars stay readable without value labels on them.
SERIES = [
    ("nvfp4", "NVFP4", RED),
    ("nvfp4a16", "NVFP4A16", ORANGE),
    ("lloyd43", "LUT3", BLUE),
    ("lloyd21", "LUT2", NAVY),
]


def _placeholder(family: str, reason: str) -> None:
    """Overwrite the figure with a visible WIP card.

    Leaving the previous render in place is worse than writing nothing: the file keeps its
    name, its timestamp moves only when someone looks, and it silently answers with data
    from a different measurement. This one cannot be mistaken for a result.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    ax.text(0.5, 0.58, "WIP", ha="center", va="center", fontsize=42, color=GREY,
            weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.40, f"{FAMILY_LABEL.get(family, family)} decode", ha="center",
            va="center", fontsize=12, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.30, reason, ha="center", va="center", fontsize=8, color=GREY,
            transform=ax.transAxes)
    out = OUT_FMT.format(family=family)
    for ext in ("pdf", ):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {out}.(pdf|png)  [WIP: {reason}]")


def plot_family(family: str):
    if not CSV.exists():
        _placeholder(family, f"no {CSV.name}; run vllm_serve.py --out {CSV.name}")
        return
    raw = pd.read_csv(CSV)
    raw = raw[raw["model"].map(family_of) == family]
    if raw.empty:
        _placeholder(family, f"no {family} rows in {CSV.name} yet")
        return
    tok = raw.pivot_table(index="model", columns="quant", values="decode_tok_s")
    if "none" not in tok:
        _placeholder(family, "no bf16 baseline arm measured")
        return
    # Speedup over the bf16 arm of the SAME model, measured in the same way.
    df = pd.DataFrame({f"{q}_speedup": tok[q] / tok["none"]
                       for q in tok.columns if q != "none"})
    df = df.reindex(sorted(df.index, key=params_b))
    names = [short_name(m) for m in df.index]
    xs = np.arange(len(names))
    width = 0.20

    # 6x4 inches, matching ../prefill/plot_for_paper.py, so the two figures can be placed
    # side by side at the same size.
    fig, ax = plt.subplots(figsize=(4, 4))

    bars = []
    for i, (key, label, colour) in enumerate(SERIES):
        col = f"{key}_speedup"
        vals = df[col].tolist() if col in df else [float("nan")] * len(names)
        off = (i - (len(SERIES) - 1) / 2) * (width + 0.012)
        bars.append((ax.bar(xs + off, [v if v == v else 0 for v in vals], width,
                            label=label, color=colour, zorder=3), vals))

    # Bars carry no value labels: fifteen of them on a 6-inch canvas is ~0.26 in per bar,
    # which forces ~7 pt type, and the reader is being asked to compare heights anyway.
    # A missing series still gets a marker, since a gap and a zero look identical.
    for bs, vals in bars:
        for r, v in zip(bs, vals):
            if v != v:                 # NaN: that format did not run for this model
                ax.annotate("n/a", xy=(r.get_x() + r.get_width() / 2, 0.04),
                            ha="center", va="bottom", fontsize=7, color=GREY)

    # BF16 parity. Not a series -- a reference, so it is grey and dashed.
    # ax.axhline(1.0, color=GREY, lw=1.3, ls=(0, (4, 3)), zorder=2)

    ax.set_xticks(xs)
    # Just the size: every model is a Qwen3, and the axis label says Model.
    ax.set_xticklabels(names)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("Decode speedup over BF16", fontsize=12)
    ax.set_xlabel(f"{FAMILY_LABEL.get(family, family)} model", fontsize=12)
    finite = [v for _, vals in bars for v in vals if v == v]
    # Headroom for the legend, which sits inside the axes above the bars.
    ax.set_ylim(1.0, 4.5) # (max(finite) if finite else 1.0) * 1.18)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=12,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0.4)

    fig.tight_layout()
    out = OUT_FMT.format(family=family)
    for ext in ("pdf", ):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", transparent=True)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", nargs="+", default=["all"],
                    help="qwen3, gemma3, or all (default: every family with a CSV)")
    args = ap.parse_args()
    fams = sorted(FAMILY_LABEL) if "all" in args.family else args.family
    for fam in fams:
        plot_family(fam)


if __name__ == "__main__":
    main()
