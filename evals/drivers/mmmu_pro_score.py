"""Score MMMU-Pro generations with the benchmark's own answer parser.

    python3 mmmu_pro_score.py results/mmmu/vision_cot/bf16 [more dirs ...]

`parse_multi_choice_response` is imported from the MMMU-Pro repo, unmodified: option
matching for this benchmark is finicky (it falls back from "(A)" to "A." to matching the
option TEXT, and the fallback order changes scores), so the parser has to be theirs for
the number to mean what the published number means.

Their `evaluate.py` is not used as the entry point for two reasons. It hard-codes
`./output` as the input directory, and -- more importantly -- it SKIPS any file whose
row count differs from the expected total:

    if len(results) != NUM:
        print(f"Error: ... has {len(results)} results, expected {NUM}"); continue

A run that was killed at 95% therefore scores nothing at all, and the only signal is one
printed line in a log. Here a partial file is scored and the coverage is reported, so a
partial run yields a partial answer plus the exact count it is based on.

It also rewrites its input file in place with the parsed fields added. Generations cost
GPU-hours; the scorer does not get to mutate them.
"""
import json
import os
import random
import sys

HARNESS = os.environ.get(
    "MMMU_HARNESS",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/MMMU/mmmu-pro")
sys.path.insert(0, HARNESS)

import ast  # noqa: E402
from collections import defaultdict  # noqa: E402

from evaluate import (  # noqa: E402 - the benchmark's own parser
    get_multi_choice_info, parse_multi_choice_response)


def score_dir(d):
    raw = os.path.join(d, "raw.jsonl")
    if not os.path.exists(raw):
        print(f"skip {d}: no raw.jsonl")
        return None
    rows = [json.loads(l) for l in open(raw)]
    # Seed the parser's random fallback. parse_multi_choice_response ends with
    # `pred_index = random.choice(all_choices)` when nothing matches, unseeded, so
    # re-scoring the same file twice gave different accuracies -- ~0.1 points, which is
    # small until it decides whether a difference is significant. Seeding makes scoring
    # a function of the generations, which is what a scorer is supposed to be.
    random.seed(20260825)
    per_sub = defaultdict(lambda: [0, 0])
    n_right = n_trunc = n_empty = n_unparsed = 0
    for r in rows:
        pred_text = r.get("content") or ""
        index2ans, all_choices = get_multi_choice_info(
            ast.literal_eval(str(r["options"])))
        parsed = parse_multi_choice_response(pred_text, all_choices, index2ans)
        right = (parsed == r["answer"])
        n_right += right
        n_trunc += (r.get("finish_reason") == "length")
        n_empty += not pred_text.strip()
        # The benchmark's parser never returns None -- it falls back to a random choice
        # when nothing matches, which is deliberate but means "could not parse" is
        # invisible in the score. An empty response is the case where that fallback is
        # pure noise, so it is counted separately.
        n_unparsed += not pred_text.strip()
        sub = r.get("subdomain") or "unknown"
        per_sub[sub][0] += right
        per_sub[sub][1] += 1

    n = len(rows)
    out = {
        "dir": d, "n_scored": n, "accuracy": n_right / n if n else 0.0,
        "truncated": n_trunc, "empty_content": n_empty, "unparsed": n_unparsed,
        "per_subdomain": {k: {"acc": v[0] / v[1], "n": v[1]}
                          for k, v in sorted(per_sub.items())},
    }
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"{d}: acc={out['accuracy'] * 100:.2f} on {n} scored "
          f"(truncated {n_trunc}, empty {n_empty})")
    return out


def main():
    dirs = sys.argv[1:]
    if not dirs:
        raise SystemExit(__doc__)
    results = [r for r in (score_dir(d) for d in dirs) if r]
    if len(results) > 1:
        # The published MMMU-Pro figure is normally the mean of standard(10 options)
        # and vision. Weighted by item count so a partially-finished split cannot
        # dominate the average.
        tot = sum(r["n_scored"] for r in results)
        avg = sum(r["accuracy"] * r["n_scored"] for r in results) / tot
        print(f"\ncombined (item-weighted over {len(results)} splits, {tot} items): "
              f"{avg * 100:.2f}")


if __name__ == "__main__":
    main()
