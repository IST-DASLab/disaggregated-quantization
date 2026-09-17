"""Generate core Qwen 3/Gemma 3 stack timings; 27B uses separate llama.cpp measurements."""
from common import prefill_data, latency, row

MODELS = [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"), ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ("Qwen/Qwen3-4B", "Qwen3-4B"), ("Qwen/Qwen3-8B", "Qwen3-8B"),
    ("google/gemma-3-270m", "Gemma-3-270M"), ("google/gemma-3-1b-it", "Gemma-3-1B"),
    ("google/gemma-3-4b-it", "Gemma-3-4B"), ("google/gemma-3-12b-it", "Gemma-3-12B"),
]


def render():
    data = prefill_data()
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             row([r"\multirow{2}{*}{Model}", r"\multicolumn{3}{c}{$S{=}16384$}",
                  r"\multicolumn{3}{c}{$S{=}32768$}"]),
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             row(["", "BF16 (ms)", "NVFP4", "+ODP", "BF16 (ms)", "NVFP4", "+ODP"]),
             r"\midrule"]
    for model, name in MODELS:
        cells = [name]
        for length in (16384, 32768):
            baseline = latency(data, model, "bf16", "resident", length)
            cells.append(f"{baseline:.0f}")
            for mode in ("resident", "zero-ssd"):
                cells.append(f"{baseline / latency(data, model, 'nvfp4', mode, length):.2f}" + r"$\times$")
        lines.append(row(cells))
        if model == "Qwen/Qwen3-8B":
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    print(render())
