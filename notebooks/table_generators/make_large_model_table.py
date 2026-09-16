"""Emit the LaTeX results table: BF16 / W4A16 / W4A4 (None, Format, Delta) per benchmark.

    python notebooks/table_generators/make_large_model_table.py  # both benchmarks, stdout
    python notebooks/table_generators/make_large_model_table.py --benches mmlu_pro
    python notebooks/table_generators/make_large_model_table.py --out table.tex

WHY IT IMPORTS compare_arms RATHER THAN RE-READING results/
------------------------------------------------------------
Everything this needs already exists there and is subtle enough that a second
implementation would drift: repeats are collapsed by directory convention (`tag`,
`tag-r1`, ...), partial passes are DROPPED rather than averaged (a run in flight has
finished only its fast items, which skew easy), and the Delta significance is an exact
paired sign-flip test on per-item differences, not a comparison of two means. Reusing
the module keeps the quantized-arm means and significance tests consistent.

Qwen3.8-27B's BF16 reference uses four explicitly selected summary.json files:
MMLU-Pro bf16-r1 through r4; MMMU-Pro bf16 plus r1 through r3. Its per-item
export currently contains only the later r5 run. Summary means suffice for this
reference column; they are never substituted for per-item significance inputs.

WHAT THE COLUMNS ARE
--------------------
  BF16    the model's baseline arm -- NOT always bf16. qwen3.8-2.4t is baselined on fp8
          (bf16 is 4.89 TB and would need 8 nodes for the reference arm alone) and
          kimi-k3 ships no bf16 at all, so it prints "-".
  W4A16   weight-only 4-bit.
  None    W4A4 with no disaggregation: one engine, quantized activations throughout.
  Format  W4A4 with FORMAT disaggregation: quantized activations on prefill, weight-only
          on decode.
  Delta   Format - None. Bold when the sign-flip test rejects at alpha=0.05.

RED MEANS INCOMPLETE, AND IT IS LOAD-BEARING
---------------------------------------------
A cell is \\textcolor{red}{} when its arm has fewer than `expected_items(bench)` items
in some pass, or does not have four complete passes, unless overridden with --repeats.
Both states produce a
perfectly plausible-looking number, and the whole point of printing it red is that a
number which is still moving must not be read as a result. An arm with no data at all
prints "-" instead.
"""
import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals/drivers"))
import compare_arms as CA  # noqa: E402

# Four complete passes for both benchmarks; --repeats can override it explicitly.
REPEATS = 4

# Pin the intended reference cohort, excluding later r5 and hardware/control runs.
BASELINE_SUMMARY_TAGS = {
    ("qwen3.8-27b", "mmlu_pro"): ("bf16-r1", "bf16-r2", "bf16-r3", "bf16-r4"),
    ("qwen3.8-27b", "mmmu"): ("bf16", "bf16-r1", "bf16-r2", "bf16-r3"),
}

# (models.json key, display name, rule after this row)
ROWS = [
    ("qwen3.8-27b",                "Qwen3.8-27B",      False),
    ("gemma-4-31b-it",             "Gemma-4-31B",      False),
    ("muse-glimmer",               "Muse-Glimmer-30B", True),
    ("gemma-4-26b-a4b-it",         "Gemma-4-26B",      False),
    ("nemotron-3-super-120b",      "Nemotron-3-120B",  False),
    ("nemotron-3-ultra-550b",      "Nemotron-3-550B",  False),
    ("qwen3.8-2.4t",               r"Qwen3.8-2.4T$^{\dagger}$", False),
    ("kimi-k3",                    r"Kimi-K3-2.8T$^{\ddagger}$", False),
]

# Column -> candidate tags, first match wins. Kimi is MXFP4 rather than NVFP4, and its
# W4A16 arm doubles as its baseline, which is why "BF16" resolves to nothing for it.
COLS = {
    "W4A16":  ["nvfp4a16", "mxfp4a16"],
    "None":   ["nvfp4", "mxfp4"],
    "Format": ["nvfp4pd", "mxfp4pd"],
}


_REG = None


def _baseline(model):
    """The model's baseline tag from models.json, same source compare_arms uses."""
    global _REG
    if _REG is None:
        import json
        mj = os.path.join(os.path.dirname(os.path.abspath(CA.__file__)), os.pardir,
                          "models.json")
        with open(os.path.normpath(mj)) as f:
            _REG = json.load(f)
    return (_REG.get(model, {}).get("report", {}) or {}).get("baseline")


def _first_present(bench, cands):
    for t in cands:
        if CA.items(bench, t):
            return t
    return None


def _cell(bench, tag):
    """(score, n_items, n_repeats) or None when the arm has no data at all."""
    per = CA.items(bench, tag) if tag else None
    if not per:
        return None
    uids = sorted(per)
    scores, k = CA.per_repeat_scores(per, uids)
    if not scores:
        return None
    return (sum(scores) / len(scores), len(uids), k)


def _baseline_summary_cell(model, bench):
    """Read the pinned BF16 cohort; never silently substitute another repeat."""
    tags = BASELINE_SUMMARY_TAGS.get((model, bench))
    if tags is None:
        return None
    root = Path(CA.RESULTS_ROOT) / model / CA.BENCH_ROOT[bench]
    want = CA.expected_items(bench)
    scores = []
    for tag in tags:
        path = root / tag / "summary.json"
        summary = json.loads(path.read_text())
        if summary.get("n_scored") != want:
            raise ValueError(f"{path}: expected {want} scored items, "
                             f"got {summary.get('n_scored')}")
        accuracy = float(summary["accuracy"])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"{path}: invalid accuracy {accuracy}")
        scores.append(100 * accuracy)
    print(f"  {model}/{bench}: BF16 uses {len(tags)} complete summaries "
          f"({', '.join(tags)})", file=sys.stderr)
    return (sum(scores) / len(scores), want, len(scores))


def _fmt(val, incomplete, bold=False):
    if val is None:
        return "-"
    s = f"{val:.2f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if incomplete:
        s = f"\\textcolor{{red}}{{{s}}}"
    return s


def _row_for(model, bench):
    """The five cells for one model and one benchmark, plus an incomplete flag."""
    CA.RESULTS = os.path.join(CA.RESULTS_ROOT, model)
    CA._CACHE.clear()
    baseline = _baseline(model)

    tags = {"BF16": baseline if baseline and CA.items(bench, baseline) else None}
    for col, cands in COLS.items():
        tags[col] = _first_present(bench, cands)
    # The baseline can BE the W4A16 arm (kimi). Do not print it twice.
    if tags["BF16"] and tags["BF16"] == tags["W4A16"]:
        tags["BF16"] = None

    cells = {c: _cell(bench, t) for c, t in tags.items()}
    if baseline == "bf16":
        reference = _baseline_summary_cell(model, bench)
        if reference is not None:
            cells["BF16"] = reference
    want = CA.expected_items(bench)
    def bad(v):
        return v is not None and (v[1] != want or v[2] != REPEATS)

    out, any_bad = {}, False
    for c in ("BF16", "W4A16", "None", "Format"):
        v = cells[c]
        out[c] = _fmt(v[0] if v else None, bad(v))
        any_bad |= bad(v)

    # Delta = Format - None, significance from the paired sign-flip test rather than
    # from the two means: the arms share items, and the pairing is most of the power.
    d, dbold = None, False
    if cells["None"] and cells["Format"]:
        d = cells["Format"][0] - cells["None"][0]
        try:
            f = CA.signflip(bench, tags["Format"], tags["None"])
            dbold = bool(f and f["p"] < 0.05)
        except Exception:                                          # noqa: BLE001
            dbold = False
    out["Delta"] = ("-" if d is None else
                    _fmt(d, bad(cells["None"]) or bad(cells["Format"]), bold=dbold)
                    .replace(f"{d:.2f}", f"{d:+.2f}"))
    return out, any_bad


def row_for(model, bench):
    # Imported loaders print partial-pass diagnostics; keep stdout valid LaTeX.
    with contextlib.redirect_stdout(sys.stderr):
        return _row_for(model, bench)


def main():
    global REPEATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+", default=["mmlu_pro", "mmmu"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", nargs="+", default=None,
                    help="restrict to these models.json keys")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help="override required passes (default: 4 for both benchmarks)")
    args = ap.parse_args()
    REPEATS = args.repeats
    benches = [w for b in args.benches for w in b.split()]

    # No export step here, and none needed: run_eval.sh exports each arm's per-item
    # scores as the last thing it does, so evals/scores is written by the same job that
    # produced the measurement and cannot lag it. This reads live results/ when present
    # and the export otherwise; both are current by construction.
    live = os.path.isdir(CA.RESULTS_ROOT)

    nb = len(benches)
    spec = "l|" + "|".join(["cc | ccc"] * nb)
    L = [r"    \begin{tabular}{" + spec + "}", r"    \toprule"]
    head = r"    \multirow{3}{*}{Model}"
    for i, b in enumerate(benches):
        name = {"mmlu_pro": "MMLU-Pro", "mmmu": "MMMU-Pro"}.get(b, b)
        end = "|" if i < nb - 1 else ""
        head += r" & \multicolumn{5}{c" + end + "}{" + name + "}"
    L.append(head + r" \\")
    L.append("    " + "".join(rf"\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}"
                              for i in range(nb)))
    sub = "    "
    for i in range(nb):
        end = "|" if i < nb - 1 else ""
        sub += (r" & \multirow{2}{*}{BF16} & \multirow{2}{*}{W4A16}"
                r" & \multicolumn{3}{c" + end + r"}{W4A4 disaggregation}")
    L.append(sub + r" \\")
    L.append("    " + " & ".join([""] + ["", "", "None", "Format", r"$\Delta$"] * nb)
             + r" \\")
    L.append(r"    \midrule")

    want = set(w for m in (args.model or []) for w in m.split())
    rows = [(m, d, r) for m, d, r in ROWS if not want or m in want]
    # First pass: the most passes any model achieved on each benchmark.
    empty = {}
    for model, disp, rule in rows:
        cells = []
        for b in benches:
            if not args.quiet:
                print(f"  ... {model} / {b}", file=sys.stderr, flush=True)
            r, _ = row_for(model, b)
            if all(r[c] == "-" for c in ("BF16", "W4A16", "None", "Format")):
                empty.setdefault(model, []).append(b)
            cells += [r["BF16"], r["W4A16"], r["None"], r["Format"], r["Delta"]]
        L.append(f"    {disp:<18} & " + " & ".join(f"{c:<6}" for c in cells) +
                 r" \\" + (r"\midrule" if rule else ""))

    # A model missing ONE benchmark is normal and prints "-": qwen3.8-2.4t is text-only
    # so MMMU-Pro cannot be run on it at all, and the Nemotrons have not been measured
    # on it. A model missing EVERY benchmark means the source is gone, which is the
    # silent-wrong-answer case worth stopping for.
    dead = [m for m, bs in empty.items() if len(bs) == len(benches)]
    if dead:
        src = "results/" if live else f"the export at {CA.SCORES_ROOT}"
        raise SystemExit(
            "no data on any benchmark for: " + ", ".join(dead) +
            f"\n  Looked in {src}. A whole row of '-' means the SOURCE is missing, not "
            "that the arms were never run -- emitting it would be a silent wrong answer."
            "\n  On a clone: pull a fresh evals/scores. On the cluster: check "
            "evals/results/<model>/<bench>/.")
    L += [r"    \bottomrule", r"    \end{tabular}"]
    text = "\n".join(L)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
