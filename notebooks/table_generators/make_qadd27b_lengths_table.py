"""Final-step generation lengths and truncation, paired with the accuracy table.

Read exact per-item completion-token counts from the versioned length exports and
length-limit termination counts from the matching results summaries. Require all
eight step-980 pairs, complete item coverage and a single evaluation per arm. Do not
substitute earlier checkpoints or include the separate interoperability experiments.
"""
import json

import numpy as np

from common import ROOT, qadd27b_data, row

RESULTS = ROOT / "evals/results/qwen3.8-27b"
SCORES = ROOT / "evals/scores/qwen3.8-27b"
BENCHES = {"mmlu_pro": ("mmlu_pro", 12032), "mmmu": ("mmmu/vision_cot", 1730)}
BUDGET = 32768


def length_metrics(ids, tokens, expected, label):
    if ids.shape != (expected,) or len(set(ids.tolist())) != expected:
        raise ValueError(f"{label}: incomplete or duplicate item IDs")
    if tokens.shape != (expected, 1):
        raise ValueError(f"{label}: expected one complete evaluation, got {tokens.shape}")
    if not np.issubdtype(tokens.dtype, np.integer) or np.any((tokens < 0) | (tokens > BUDGET)):
        raise ValueError(f"{label}: invalid or missing completion-token counts")
    return {"ids": ids, "mean": float(tokens.mean()),
            "median": float(np.median(tokens)),
            "p95": float(np.quantile(tokens, 0.95, method="linear"))}


def read_metrics(bench, tag):
    folder, expected = BENCHES[bench]
    path = RESULTS / folder / tag / "summary.json"
    summary = json.loads(path.read_text())
    if summary.get("n_scored") != expected:
        raise ValueError(f"{path}: expected {expected} scored items")
    generation = summary["generation"] if bench == "mmlu_pro" else summary
    if bench == "mmlu_pro" and generation.get("n") != expected:
        raise ValueError(f"{path}: incomplete generation summary")
    truncated = generation["truncated"]
    if type(truncated) is not int or not 0 <= truncated <= expected:
        raise ValueError(f"{path}: invalid truncation count {truncated}")
    with np.load(SCORES / f"{bench}_lengths.npz", allow_pickle=False) as lengths:
        metrics = length_metrics(lengths[f"{tag}|ids"], lengths[f"{tag}|tokens"],
                                 expected, f"{bench}/{tag}")
    metrics["truncated"] = 100 * truncated / expected
    return metrics


def render():
    data, labels = qadd27b_data()  # Same arms as the accuracy table; requires step 980.
    lines = [r"\begin{tabular}{l|rr|rr|rr|rr}", r"\toprule"]
    for bench, title in (("mmlu_pro", "MMLU-Pro"), ("mmmu", "MMMU-Pro")):
        if bench != "mmlu_pro":
            lines.append(r"\midrule")
        lines += [row([rf"\multicolumn{{9}}{{c}}{{\textbf{{{title}}}}}"]),
                  row([r"\multirow{2}{*}{Decode format}", r"\multicolumn{2}{c|}{Mean tokens}",
                       r"\multicolumn{2}{c|}{Median tokens}", r"\multicolumn{2}{c|}{95th percentile}",
                       r"\multicolumn{2}{c}{Truncated (\%)}"]),
                  row([""] + ["WO", "Pref."] * 4), r"\midrule"]
        for fmt, label in labels.items():
            metrics = {}
            for disagg in (False, True):
                metrics[disagg] = read_metrics(bench, data[bench][fmt, disagg]["tag"])
            if set(metrics[False]["ids"]) != set(metrics[True]["ids"]):
                raise ValueError(f"{bench}/{fmt}: item sets differ between arms")
            cells = [label.replace("_", r"\_")]
            for metric in ("mean", "median", "p95", "truncated"):
                for disagg in (False, True):
                    value = metrics[disagg][metric]
                    cells.append(f"{value:.2f}" if metric == "truncated" else f"{value:.0f}")
            lines.append(row(cells))
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    print(render())
