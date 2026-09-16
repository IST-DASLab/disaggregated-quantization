"""Generate tab:nvfp4-breakdown from profiler CSV and resident stack timings."""
from common import KERNELS, csv_index, number, prefill_data, latency, row
from decimal import Decimal, ROUND_HALF_UP

COMPONENTS = [
    ("qkv", "qkv"), ("o", "o"), ("gate_up", r"gate\_up"), ("down", "down"),
    ("MLP non-linearity", "MLP non-linearity"),
    ("activation quant (standalone)", r"Activation quant.\ (standalone)"),
    ("norms + RoPE + residual", "Norms + RoPE + residual"), ("attention", "Attention"),
]
MODELS = [
    ("google/gemma-3-12b-it", 48, "Gemma-3-12B: 48 layers, 40 sliding attention / 8 global attention"),
    ("Qwen/Qwen3-8B", 36, "Qwen-3-8B: 36 layers, all global attention"),
]


def render():
    data = csv_index(KERNELS / "nvfp4_breakdown.csv", ("model", "seq_len", "component"))
    prefill = prefill_data()
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             row(["Component", "BF16 (ms)", "NVFP4 (ms)", "Speedup", r"\% NVFP4"])]
    for model, layers, name in MODELS:
        lines += [r"\midrule", row([r"\multicolumn{5}{l}{" + name + "}"]), r"\midrule"]
        values = [(number(data[model, "16384", key], "bf16_ms", False),
                   number(data[model, "16384", key], "nvfp4_ms", False)) for key, _ in COMPONENTS]
        totals = [sum(v[i] for v in values) for i in (0, 1)]
        for (_, label), (bf, nv) in zip(COMPONENTS, values):
            # Preserve decimal rounding of the recorded three-decimal CSV values.
            fixed = lambda value: str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
            lines.append(row([label, fixed(bf) if bf else "-", fixed(nv),
                              f"{bf/nv:.2f}" + r"$\times$" if bf and nv else "-",
                              f"{100*nv/totals[1]:.0f}"]))
        wall = [number(data[model, "16384", "LAYER wall clock"], c) for c in ("bf16_ms", "nvfp4_ms")]
        lines += [r"\midrule", row(["Device busy (sum of kernels)", *[f"{v:.2f}" for v in totals],
                                    f"{totals[0]/totals[1]:.2f}" + r"$\times$", "100"]),
                  row(["Launch/idle gap", *[f"{w-v:.2f}" for w, v in zip(wall, totals)], "-", "-"]),
                  row(["Layer (wall clock)", *[f"{v:.2f}" for v in wall],
                       f"{wall[0]/wall[1]:.2f}" + r"$\times$", "-"])]
        stack = [latency(prefill, model, q, "resident", 16384) / layers for q in ("bf16", "nvfp4")]
        lines.append(row([r"\textbf{Full stack, per layer}", *[rf"\textbf{{{v:.2f}}}" for v in stack],
                          rf"\textbf{{{stack[0]/stack[1]:.2f}}}" + r"$\times$", "-"]))
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    print(render())
