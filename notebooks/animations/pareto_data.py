"""Read the exact notebook Pareto inputs without executing plotting cells."""

import ast
import contextlib
import csv
import io
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_results():
    notebook = json.loads((ROOT / "notebooks/plots.ipynb").read_text())
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]
             if "def load_gsq_rco(" in "".join(cell.get("source", []))]
    if len(cells) != 1:
        raise ValueError("Expected exactly one Pareto loader cell")
    tree = ast.parse(cells[0])
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Assign, ast.FunctionDef))]
    namespace = dict(Path=Path, np=np, json=json, re=re)
    exec(compile(tree, "Pareto notebook definitions", "exec"), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        scores = namespace["load_gsq_rco"](ROOT / "evals/results/qwen3.8-27b")
    formats = namespace["_gsq_rco_plot_formats"]()
    quality = {}
    for bench in ("mmlu_pro", "mmmu"):
        quality[bench] = {
            "baseline": [scores[bench][fmt, False]["accuracy"] for fmt in formats],
            "odp": [scores[bench][fmt, True]["accuracy"] for fmt in formats],
            "bf16": scores[bench]["bf16", False]["accuracy"],
        }
        for fmt in formats:
            tag = scores[bench][fmt, True]["tag"]
            if namespace["_gsq_rco_tag"](tag)[3] != 980:
                raise ValueError(f"Expected final-step prefill evaluation: {bench}/{tag}")
    ttft = namespace["load_gsq_rco_prefill"](
        ROOT / "notebooks/data/qwen3_8_27b_llama_cpp_ttft.csv")
    lengths = sorted(n for n in ttft["baseline"] if n >= 1024)
    with (ROOT / "qad/kernels/prefill/load_floor_zero_ssd.csv").open() as stream:
        floor = [float(row["load_ms"]) / 1000 for row in csv.DictReader(stream)
                 if row["model"] == "Qwen/Qwen3.8-27B" and row["quant"] == "nvfp4"]
    if len(floor) != 1:
        raise ValueError("Expected one SSD loading reference")
    return {
        "formats": [namespace["GSQ_RCO_LABELS"][fmt] for fmt in formats],
        "sizes": [namespace["GSQ_RCO_BYTES"][fmt] / 1e9 for fmt in formats],
        "quality": quality,
        "lengths": lengths,
        "latency": {mode: [ttft[mode][n][0] / 1000 for n in lengths]
                    for mode in ("baseline", "odp")},
        "floor": floor[0],
    }


if __name__ == "__main__":
    print(json.dumps(load_results(), indent=2))
