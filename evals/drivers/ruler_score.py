"""Score RULER predictions with RULER's OWN metrics.

    python3 drivers/ruler_score.py RESULTS_DIR

The metric functions are imported from the clone, not reimplemented:
scripts/eval/synthetic/constants.py defines string_match_all (niah, vt, cwe, fwe) and
string_match_part (qa), and which task uses which. Copying those four lines here would
work today and would be wrong the first time upstream changes one.

postprocess_pred mirrors evaluate.py:50 exactly -- strip, then replace control
characters with newlines, then strip again. It is reproduced rather than imported only
because evaluate.py cannot be imported at all: its module-level
`from nemo.collections.asr...import read_manifest` pulls in an ASR stack to read JSONL.

REPORTED SCORE. RULER's headline number for a length is the unweighted mean over the 13
tasks, each scored 0-100 -- not a pooled per-document accuracy. Tasks have equal weight
despite equal doc counts making those coincide here; they stop coinciding the moment a
task is dropped or --limit is uneven, so the mean is taken over task scores.
"""
import argparse
import collections
import importlib.util
import json
import os
import re
import sys

_NP = re.compile(r"[\x00-\x1f]")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def postprocess_pred(predict_str):
    """evaluate.py:50, verbatim in behaviour."""
    return _NP.sub("\n", predict_str.strip()).strip()


def score(results_dir, ruler_root):
    import yaml
    with open(os.path.join(ruler_root, "scripts/synthetic.yaml")) as fh:
        by_name = yaml.safe_load(fh)
    metrics = _load_module(
        os.path.join(ruler_root, "scripts/eval/synthetic/constants.py"),
        "ruler_eval_constants").TASKS

    raw = os.path.join(results_dir, "raw.jsonl")
    rows = []
    with open(raw) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:                                   # noqa: BLE001
                continue        # torn final line: the file may be being appended to
    # De-duplicated on uid. Resume appends, so a re-run that overlaps an interrupted one
    # can leave the same doc twice; counting it twice would quietly reweight the task.
    seen, uniq = set(), []
    for r in rows:
        if r["uid"] in seen:
            continue
        seen.add(r["uid"])
        uniq.append(r)

    by = collections.defaultdict(list)
    for r in uniq:
        by[(r["seqlen"], r["task"])].append(r)

    per_len = collections.defaultdict(dict)
    empty = collections.Counter()
    trunc = collections.Counter()
    for (seqlen, task), rs in sorted(by.items()):
        metric_fn = metrics[by_name[task]["task"]]["metric_fn"]
        preds = [postprocess_pred(r.get("content") or "") for r in rs]
        refs = [r["outputs"] for r in rs]
        per_len[seqlen][task] = {"score": metric_fn(preds, refs), "n": len(rs)}
        empty[seqlen] += sum(1 for p in preds if not p)
        trunc[seqlen] += sum(1 for r in rs if r.get("finish_reason") == "length")

    out = {"benchmark": "ruler", "per_length": {}}
    for seqlen, tasks in sorted(per_len.items()):
        scores = [t["score"] for t in tasks.values()]
        out["per_length"][str(seqlen)] = {
            "score": round(sum(scores) / len(scores), 2),
            "n_tasks": len(scores),
            "n_docs": sum(t["n"] for t in tasks.values()),
            "empty_predictions": empty[seqlen],
            "truncated": trunc[seqlen],
            "per_task": tasks,
        }
    if out["per_length"]:
        out["average"] = round(
            sum(v["score"] for v in out["per_length"].values()) / len(out["per_length"]), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--ruler-root", default=os.environ.get("RULER_HARNESS", ""))
    args = ap.parse_args()
    if not args.ruler_root:
        sys.exit("ERROR: --ruler-root (or $RULER_HARNESS) is required")

    out = score(args.results_dir, args.ruler_root)
    with open(os.path.join(args.results_dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    for seqlen, v in out["per_length"].items():
        flags = []
        if v["empty_predictions"]:
            flags.append(f"{v['empty_predictions']} empty")
        if v["truncated"]:
            flags.append(f"{v['truncated']} truncated")
        print(f"  {seqlen:>7}: {v['score']:6.2f}  "
              f"({v['n_tasks']} tasks, {v['n_docs']} docs)"
              f"{'  [' + ', '.join(flags) + ']' if flags else ''}")
    if "average" in out:
        print(f"\nRULER average: {out['average']:.2f} -> {args.results_dir}")
    # Empty predictions are how a thinking model fails this benchmark -- it burns the
    # 30-128 token budget inside <think> and returns nothing. Silent zeros look like a
    # long-context result; they are a configuration error. See ruler_infer's --think.
    tot_empty = sum(v["empty_predictions"] for v in out["per_length"].values())
    tot_docs = sum(v["n_docs"] for v in out["per_length"].values())
    if tot_docs and tot_empty > 0.2 * tot_docs:
        print(f"\nWARNING: {tot_empty}/{tot_docs} predictions are EMPTY. If this model "
              f"reasons, it spent RULER's short generation budget inside <think>; "
              f"rerun without --think.")


if __name__ == "__main__":
    main()
