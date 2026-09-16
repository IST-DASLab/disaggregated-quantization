"""Generate tab:qadd27b_accuracy from all eight final-step evaluations."""
from common import qadd27b_data, row


def render():
    data, labels = qadd27b_data()
    lines = [r"\begin{tabular}{l|rrr|rrr}", r"\toprule",
             row([r"\multirow{2}{*}{Decode format}", r"\multicolumn{3}{c|}{MMLU-Pro}",
                  r"\multicolumn{3}{c}{MMMU-Pro}"]),
             row(["", "Weight-only", "Full disagg.", r"$\Delta$",
                  "Weight-only", "Full disagg.", r"$\Delta$"]), r"\midrule",
             row(["BF16", f"{data['mmlu_pro']['bf16', False]['accuracy']:.2f}", "--", "--",
                  f"{data['mmmu']['bf16', False]['accuracy']:.2f}", "--", "--"]), r"\midrule"]
    for fmt, label in labels.items():
        cells = [label.replace("_", r"\_")]
        for bench in ("mmlu_pro", "mmmu"):
            wo = data[bench][fmt, False]["accuracy"]
            full = data[bench][fmt, True]["accuracy"]
            cells += [f"{wo:.2f}", f"{full:.2f}", f"{full-wo:+.2f}"]
        lines.append(row(cells))
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    print(render())
