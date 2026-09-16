"""Collapse IFBench's two scored files into one summary.json, with diagnostics.

    python3 ifbench_report.py results/ifbench/bf16

IFBench's own `run_eval.py` prints a report and writes `*-eval_results_{strict,loose}.jsonl`,
but it prints rather than returns, so nothing downstream can read a score without
re-parsing stdout. This writes the numbers as JSON next to them.

It also joins the scored rows back against `raw.jsonl` to report how many responses were
truncated or empty. That join is the point: a truncated response fails essentially every
constraint, so a run with meaningful truncation produces a low, plausible-looking IFBench
score that says nothing about instruction following. Without this number there is no way
to tell that apart from a genuinely poor result.
"""
import collections
import glob
import json
import os
import sys


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    out_dir = sys.argv[1]
    summary = {"benchmark": "ifbench"}

    for kind in ("strict", "loose"):
        matches = glob.glob(os.path.join(out_dir, f"*eval_results_{kind}.jsonl"))
        if not matches:
            print(f"missing {kind} results in {out_dir}")
            continue
        rows = load(matches[0])
        n = len(rows)
        summary[f"{kind}_prompt_acc"] = sum(r["follow_all_instructions"] for r in rows) / n
        flat = [(iid, ok) for r in rows
                for iid, ok in zip(r["instruction_id_list"], r["follow_instruction_list"])]
        summary[f"{kind}_instruction_acc"] = sum(ok for _, ok in flat) / len(flat)
        summary[f"{kind}_n_prompts"] = n
        summary[f"{kind}_n_instructions"] = len(flat)
        # Per-category, so a single broken constraint type is visible rather than
        # averaged into the headline.
        per = collections.defaultdict(lambda: [0, 0])
        for iid, ok in flat:
            per[iid][0] += bool(ok)
            per[iid][1] += 1
        summary[f"{kind}_per_instruction"] = {
            k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(per.items())}

    raw = os.path.join(out_dir, "raw.jsonl")
    if os.path.exists(raw):
        rows = load(raw)
        trunc = sum(r.get("finish_reason") == "length" for r in rows)
        empty = sum(not (r.get("content") or "").strip() for r in rows)
        toks = sorted(r["completion_tokens"] for r in rows if r.get("completion_tokens"))
        summary["generation"] = {
            "n": len(rows), "truncated": trunc, "empty_content": empty,
            "tokens_p50": toks[len(toks) // 2] if toks else None,
            "tokens_p95": toks[int(len(toks) * 0.95)] if toks else None,
            "tokens_max": toks[-1] if toks else None,
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nIFBench  strict prompt-level : {summary.get('strict_prompt_acc', 0) * 100:.2f}")
    print(f"IFBench  loose  prompt-level : {summary.get('loose_prompt_acc', 0) * 100:.2f}")
    print(f"IFBench  strict instruction  : {summary.get('strict_instruction_acc', 0) * 100:.2f}")
    if "generation" in summary:
        g = summary["generation"]
        print(f"generation: {g['truncated']} truncated / {g['empty_content']} empty "
              f"of {g['n']}; tokens p50={g['tokens_p50']} p95={g['tokens_p95']} "
              f"max={g['tokens_max']}")


if __name__ == "__main__":
    main()
