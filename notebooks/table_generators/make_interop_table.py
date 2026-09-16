"""Generate the step-980 prefill/decode grid using cross_grid's exported mode.

The paper requires full benchmark coverage across all twelve cells. Intersections
of incomplete runs remain useful in cross_grid's CLI but are not publication data.
"""
import sys

from common import ROOT, row

sys.path.insert(0, str(ROOT / "evals/drivers"))
import cross_grid as CG

DECODES = ("IQ1_S", "IQ1_M", "IQ2_XXS")
PREFILLS = ("RTN", *DECODES)
BENCHES = (("mmlu_pro", "MMLU-Pro", 12032), ("mmmu", "MMMU-Pro", 1730))


def grid(bench, expected):
    cells = CG.load_exported_cells(bench, "qwen3.8-27b", DECODES, PREFILLS, step=980)
    keys = CG.common_keys(cells)
    if len(keys) != expected or any(len(values) != expected for values in cells.values()):
        raise ValueError(f"{bench}: require {expected} items in every cell; "
                         f"only {len(keys)} common items")
    return cells, keys


def render():
    lines = [r"\begin{tabular}{l|rrrr}", r"\toprule"]
    for bench, title, expected in BENCHES:
        if bench != "mmlu_pro":
            lines.append(r"\midrule")
        lines += [row([rf"\multicolumn{{5}}{{c}}{{\textbf{{{title}}} ($n={expected}$)}}"]),
                  row([r"\multirow{2}{*}{Frozen decode}",
                       r"\multicolumn{4}{c}{Prefill checkpoint (all NVFP4)}"]),
                  row(["", "PTQ (RTN)", *["QADD " + fmt.replace("_", r"\_") for fmt in DECODES]]),
                  r"\midrule"]
        cells, keys = grid(bench, expected)
        for decode in DECODES:
            values = [decode.replace("_", r"\_")]
            for prefill in PREFILLS:
                accuracy = 100 * sum(cells[decode, prefill][key] for key in keys) / len(keys)
                value = f"{accuracy:.2f}"
                if prefill == decode:
                    value = rf"\textbf{{{value}}}"
                else:
                    _, _, p = CG.mcnemar(cells[decode, prefill], cells[decode, decode], keys, bool)
                    if p < 0.05:
                        value += r"$^{\star}$"
                values.append(value)
            lines.append(row(values))
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    print(render())
