"""Shared paths, validated CSV inputs, and notebook-loader access (no plotting)."""
import ast
import contextlib
import csv
import io
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"
KERNELS = ROOT / "qad/kernels/prefill"


def csv_index(path, keys):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate measurement in {path}: {key}")
        result[key] = row
    return result


def number(row, column, positive=True):
    value = float(row[column])
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {column}: {row}")
    return value


def prefill_data():
    return csv_index(KERNELS / "offload_prefill.csv", ("model", "quant", "mode", "seq_len"))


def latency(data, model, quant, mode, length):
    return number(data[model, quant, mode, str(length)], "latency_ms")


def qadd27b_data():
    """Use the plot's selection policy, but require all final-step evaluations."""
    import numpy as np
    nb = json.loads((NOTEBOOKS / "plots.ipynb").read_text())
    cells = ["".join(c.get("source", [])) for c in nb["cells"]
             if "def load_gsq_rco(" in "".join(c.get("source", []))]
    if len(cells) != 1:
        raise ValueError("Expected one Qwen3.8 loader cell")
    tree = ast.parse(cells[0])
    # Definitions/constants only: never run the cell's plot calls or read cached outputs.
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    ns = dict(Path=Path, np=np, json=json, re=re)
    exec(compile(tree, "Qwen3.8 notebook loader", "exec"), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        data = ns["load_gsq_rco"](ROOT / "evals/results/qwen3.8-27b")
    for bench in ns["GSQ_RCO_BENCHES"]:
        data[bench]["bf16", False]
        for fmt in ns["GSQ_RCO_BYTES"]:
            data[bench][fmt, False]
            final = data[bench][fmt, True]
            if ns["_gsq_rco_tag"](final["tag"])[3] != 980:
                raise ValueError(f"{bench}/{fmt}: step 980 incomplete; selected {final['tag']}")
    return data, ns["GSQ_RCO_LABELS"]


def row(cells):
    return " & ".join(map(str, cells)) + r" \\"
