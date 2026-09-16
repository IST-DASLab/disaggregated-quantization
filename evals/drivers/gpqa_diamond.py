"""GPQA-Diamond against a local vLLM endpoint: generate, then score.

    python3 gpqa_diamond.py --out results/gpqa/bf16 --repeats 4

WHY NOT lm-eval's gpqa task
---------------------------
lm-eval has `gpqa_diamond_cot_zeroshot`, but its prompt and its answer regex are its
own, and the published 83.5 was not produced with them. Every lab that reports a GPQA
number reports it against the simple-evals template -- a fixed instruction to end with
`Answer: $LETTER`, and extraction by that exact pattern -- so that is what is used here.
Swapping the extraction rule alone moves a reasoning model's score by several points,
which is larger than the effect we intend to measure, so matching the reporting
convention matters more than reusing an installed harness.

REPEATS. 198 questions means one item is worth 0.51 points, and the vendor sampling
settings are temperature 1.0 / top_p 0.95, not greedy. A single pass has a binomial
standard error near 2.6 points at 83% accuracy -- wide enough to swallow the difference
between two serving configurations entirely. Each repeat re-permutes the choices with
its own seed (so position bias is averaged out too, not just sampling noise) and gets
its own output file, and the report gives the mean with a standard error over repeats.

The dataset ships as a CSV of gated `Idavidrein/gpqa`, read directly from the HF cache
rather than through `load_dataset`, since the repo has no loader script and the CSV is
the whole artifact.
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_resumable, truncation_report  # noqa: E402

HF_HUB = os.path.join(os.environ.get(
    "HF_HOME", "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache"), "hub")
CSV_GLOB = os.environ.get(
    "GPQA_CSV",
    os.path.join(HF_HUB, "datasets--Idavidrein--gpqa/snapshots/*/gpqa_diamond.csv"))

TEMPLATE = """Answer the following multiple choice question. The last line of your \
response should be of the following format: 'Answer: $LETTER' (without quotes) where \
LETTER is one of ABCD. Think step by step before answering.

{question}

A) {a}
B) {b}
C) {c}
D) {d}"""

# The same pattern simple-evals scores with. `$` is allowed on either side because
# models that were shown a literal `$LETTER` in the instruction frequently echo it.
ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*\$?([A-D])\$?")


def load_questions():
    paths = sorted(glob.glob(CSV_GLOB))
    if not paths:
        raise SystemExit(f"GPQA csv not found: {CSV_GLOB}")
    rows = list(csv.DictReader(open(paths[-1])))
    out = []
    for r in rows:
        out.append({
            "record_id": r["Record ID"],
            "domain": r.get("High-level domain", ""),
            "question": r["Question"].strip(),
            "correct": r["Correct Answer"].strip(),
            "wrong": [r[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)],
        })
    return out


def build(item, repeat):
    """Permute the four choices deterministically, so a rerun reproduces the prompt."""
    rng = random.Random(f"{item['record_id']}:{repeat}")
    choices = [item["correct"]] + item["wrong"]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    gold = "ABCD"[order.index(0)]
    prompt = TEMPLATE.format(question=item["question"], a=shuffled[0], b=shuffled[1],
                             c=shuffled[2], d=shuffled[3])
    return prompt, gold


def extract(text):
    """Last match wins: the instruction asks for the answer on the LAST line, and a
    model that reconsiders mid-answer leaves an earlier, superseded letter behind."""
    m = ANSWER_RE.findall(text or "")
    return m[-1].upper() if m else None


def score_file(path):
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r["pred"] = extract(r.get("content"))
        r["correct_pred"] = (r["pred"] == r["gold"])
    n = len(rows)
    acc = sum(r["correct_pred"] for r in rows) / n if n else 0.0
    rep = truncation_report(rows)
    # Unparsed is tracked separately from wrong. A model that reasons correctly but
    # ignores the output format scores zero here, and that is a harness finding, not a
    # capability finding -- collapsing the two hides which one moved.
    rep["unparsed"] = sum(r["pred"] is None for r in rows)
    return acc, n, rep, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    items = load_questions()
    if args.limit:
        items = items[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    per_repeat = []
    for rep in range(args.repeats):
        path = os.path.join(args.out, f"rep{rep}.jsonl")
        if not args.score_only:
            expanded = []
            for it in items:
                prompt, gold = build(it, rep)
                expanded.append({"record_id": it["record_id"], "domain": it["domain"],
                                 "repeat": rep, "gold": gold, "prompt": prompt})
            run_resumable(
                expanded,
                uid_of=lambda x: f"{x['record_id']}:{x['repeat']}",
                request_of=lambda x: [{"role": "user", "content": x["prompt"]}],
                out_path=path, workers=args.workers, desc=f"gpqa rep{rep}",
                max_tokens=args.max_tokens, base_url=args.base_url)
        if os.path.exists(path):
            acc, n, diag, _ = score_file(path)
            per_repeat.append(acc)
            print(f"rep{rep}: acc={acc:.4f} n={n} {diag}", flush=True)

    if per_repeat:
        mean = statistics.mean(per_repeat)
        # stdev over repeats is undefined for one pass; report the binomial estimate
        # instead of nothing, and say which one it is.
        if len(per_repeat) > 1:
            err = statistics.stdev(per_repeat) / len(per_repeat) ** 0.5
            kind = "sem_over_repeats"
        else:
            err = (mean * (1 - mean) / len(items)) ** 0.5
            kind = "binomial_se_single_pass"
        summary = {"benchmark": "gpqa_diamond", "accuracy": mean, "error": err,
                   "error_kind": kind, "repeats": per_repeat, "n": len(items)}
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nGPQA-Diamond: {mean * 100:.2f} +/- {err * 100:.2f} "
              f"({kind}, {len(per_repeat)} repeats)")


if __name__ == "__main__":
    main()
