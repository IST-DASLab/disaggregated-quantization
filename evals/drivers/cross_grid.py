"""Prefill x decode accuracy grid, paired on the items every cell has finished.

    python3 drivers/cross_grid.py                      # the default 3x4 grid
    python3 drivers/cross_grid.py --bench mmmu --stats
    python3 drivers/cross_grid.py --bench mmmu --stats --exported

Answers one question: does a prefill trained against a particular decode checkpoint work
better with THAT decode than with another one? The matched pairs sit on the diagonal, so
specialization would show as the diagonal dominating its row.

PAIRED ON THE INTERSECTION OF EVERY COMBINATION. Cells finish at different rates -- a
cross run started hours after the matched one can be 20% done -- and comparing a partial
arm's mean against a complete arm's mean compares different item sets, not different
models. Every number is computed over the items present in EVERY cell of the grid, and n
is printed because the grid is meant to be read while runs are still in flight.

A MISSING CELL IS FATAL, not skipped. Dropping it would quietly widen the intersection
to the cells that happen to exist, so the printed grid would answer a different question
than the one asked -- and the numbers would shift as cells arrived, with nothing marking
why. --allow-missing opts into the narrower grid over whatever is present.

STRICT SCORING, content only. The reasoning-fallback rule was useful while a checkpoint
bug stranded answers in the reasoning channel; it flatters exactly the configurations
that fail to emit an answer, which is a property these grids exist to expose. --lenient
restores it for comparison.

TAG CONVENTION, matching what bin/run_eval.sh was invoked with:
    RTN prefill        nvfp4p-unsloth-<decode>-d
    matched (pf==dec)  <tag-base><step>-<decode>-d
    crossed            <tag-base><step>-pf<prefill>-dec<decode>-d
A cell whose directory is missing is reported as absent rather than silently dropped,
because a silently missing cell turns a grid into a different comparison than it claims.

--exported uses the versioned per-item scores with the manuscript's benchmark
scoring, rather than re-scoring raw outputs. It requires one pass per cell and still
pairs on the intersection of all cells. For MMMU this preserves the official
parser's per-pass seeded fallback; raw mode below instead re-seeds per item.
Empty-content and lenient-scoring views require raw outputs and are unavailable
in exported mode.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RESULTS = os.path.join(EVALS, "results")
SCORES = os.path.join(EVALS, "scores")


def load_exported_cells(bench, model, decodes, prefills, tag_base="reason8k", step=980,
                        allow_missing=False):
    """The same grid from single-pass per-item scores; never use summary means."""
    path = os.path.join(SCORES, model, f"{bench}.json")
    with open(path) as stream:
        arms = json.load(stream)["arms"]
    cells = {}
    for decode in decodes:
        for prefill in prefills:
            tag = tag_for(prefill, decode, tag_base, step)
            arm = arms.get(tag)
            if arm is None:
                if allow_missing:
                    print(f"MISSING CELL: dec {decode} x pf {prefill} -> {tag}", file=sys.stderr)
                    continue
                raise ValueError(f"{path}: missing cell {tag}")
            ids, scores = arm["ids"], arm["scores"]
            if len(ids) != len(scores) or not ids or len(set(ids)) != len(ids):
                raise ValueError(f"{path}/{tag}: invalid or duplicate item IDs")
            if arm.get("repeats") != 1 or any(len(values) != 1 or values[0] not in (0, 1)
                                             for values in scores):
                raise ValueError(f"{path}/{tag}: expected one binary score per item")
            cells[decode, prefill] = dict(zip(ids, (bool(values[0]) for values in scores)))
    return cells


def common_keys(cells):
    if not cells:
        raise ValueError("no cells found")
    keys = set.intersection(*(set(values) for values in cells.values()))
    if not keys:
        raise ValueError("no items common to all grid cells")
    return keys


def _low(a):
    return a.lower().replace("_", "")


def tag_for(prefill, decode, tag_base, step):
    if prefill.upper() == "RTN":
        return f"nvfp4p-unsloth-{_low(decode)}-d"
    if prefill == decode:
        return f"{tag_base}{step}-{_low(decode)}-d"
    return f"{tag_base}{step}-pf{_low(prefill)}-dec{_low(decode)}-d"


def load(bench, tag, model):
    root = "mmmu/vision_cot" if bench == "mmmu" else bench
    p = os.path.join(RESULTS, model, root, tag, "raw.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue        # torn final line: the file is being appended to
            k = r.get("uid", r.get("question_id"))
            out.setdefault(k, r)
    return out


def scorer(bench, lenient):
    if bench == "mmlu_pro":
        from mmlu_pro_infer import ANSWER_RE

        def fn(r):
            m = ANSWER_RE.search(r.get("content") or "")
            if not m and lenient:
                m = ANSWER_RE.search(r.get("reasoning") or "")
            return bool(m) and m.group(1) == r["gold"]
        return fn

    import ast
    import random
    os.environ.setdefault("MMMU_HARNESS", os.path.join(
        os.path.dirname(EVALS), "..", "harnesses", "MMMU", "mmmu-pro"))
    from mmmu_pro_score import get_multi_choice_info, parse_multi_choice_response

    def fn(r):
        random.seed(20260825)
        i2a, ch = get_multi_choice_info(ast.literal_eval(str(r["options"])))
        txt = r.get("content") or ""
        if lenient and not txt.strip():
            txt = r.get("reasoning") or ""
        return parse_multi_choice_response(txt, ch, i2a) == r["answer"]
    return fn


def mcnemar(A, B, keys, ok):
    w = sum(1 for u in keys if ok(B[u]) and not ok(A[u]))
    l = sum(1 for u in keys if ok(A[u]) and not ok(B[u]))
    n = w + l
    p = (min(1.0, sum(math.comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n * 2)
         if n else 1.0)
    return w, l, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # The three decode formats this ablation is built around, and the prefills trained
    # against them plus the untrained RTN control.
    ap.add_argument("--decodes", default="IQ1_S,IQ1_M,IQ2_XXS", help="comma-separated")
    ap.add_argument("--prefills", default="RTN,IQ1_S,IQ1_M,IQ2_XXS",
                    help="comma-separated; RTN is the untuned NVFP4 control")
    ap.add_argument("--tag-base", default="reason8k")
    ap.add_argument("--step", type=int, default=980)
    ap.add_argument("--bench", default="mmlu_pro", choices=["mmlu_pro", "mmmu"])
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lenient", action="store_true", help="fall back to the reasoning channel")
    ap.add_argument("--exported", action="store_true", help="use versioned single-pass per-item scores")
    ap.add_argument("--empty", action="store_true", help="also print empty-content %")
    ap.add_argument("--stats", action="store_true", help="also print diagonal-vs-row tests")
    ap.add_argument("--allow-missing", action="store_true",
                    help="grid over the cells that exist instead of refusing (see module docstring)")
    args = ap.parse_args()

    decs = [d for d in args.decodes.split(",") if d]
    pfs = [p for p in args.prefills.split(",") if p]
    if args.exported and (args.lenient or args.empty):
        ap.error("--lenient and --empty require raw outputs, not --exported")
    ok = bool if args.exported else scorer(args.bench, args.lenient)

    cells, missing = {}, []
    if args.exported:
        cells = load_exported_cells(args.bench, args.model, decs, pfs, args.tag_base,
                                    args.step, args.allow_missing)
    else:
        for d in decs:
            for p in pfs:
                t = tag_for(p, d, args.tag_base, args.step)
                rows = load(args.bench, t, args.model)
                if rows is None:
                    missing.append((d, p, t))
                else:
                    cells[(d, p)] = rows
    if missing:
        which = "excluded from" if args.allow_missing else "BLOCKING"
        print(f"MISSING CELLS ({which} the grid):")
        for d, p, t in missing:
            print(f"  dec {d} x pf {p}  ->  {t}")
        if not args.allow_missing:
            raise SystemExit(
                f"\n{len(missing)} of {len(decs) * len(pfs)} cells have no results yet. "
                f"The grid is paired across ALL combinations, so it cannot be computed "
                f"until they land; pass --allow-missing to compare over the rest.")
        print()
    if not cells:
        raise SystemExit("no cells found")

    keys = common_keys(cells)
    rule = ("exported (manuscript scoring)" if args.exported else
            "lenient (+reasoning)" if args.lenient else "strict (content only)")
    print(f"{args.bench}, {rule}, n={len(keys)} items common to all "
          f"{len(cells)} cells   [matched = bracketed]\n")

    def acc(c):
        return 100 * sum(ok(cells[c][u]) for u in keys) / len(keys)

    head = f"{'':14}" + "".join(f"{'pf ' + p:>14}" for p in pfs)
    print(head)
    for d in decs:
        row = f"  dec {d:9}"
        for p in pfs:
            if (d, p) not in cells:
                row += f"{'-':>14}"
                continue
            v = f"{acc((d, p)):.2f}"
            row += f"{('[' + v + ']') if p == d else v:>14}"
        print(row)

    if args.empty:
        print(f"\n{head}   empty-content %")
        for d in decs:
            row = f"  dec {d:9}"
            for p in pfs:
                if (d, p) not in cells:
                    row += f"{'-':>14}"
                    continue
                e = 100 * sum(1 for u in keys
                              if not (cells[(d, p)][u].get("content") or "").strip()) / len(keys)
                row += f"{e:14.1f}"
            print(row)

    if args.stats:
        print("\ndiagonal (matched) vs every other cell in its row:")
        for d in decs:
            if (d, d) not in cells:
                continue
            for p in pfs:
                if p == d or (d, p) not in cells:
                    continue
                w, l, pv = mcnemar(cells[(d, p)], cells[(d, d)], keys, ok)
                print(f"  dec {d:8} matched - pf {p:8} = {acc((d, d)) - acc((d, p)):+6.2f}"
                      f"  (+{w}/-{l}, p={pv:.4f})")


if __name__ == "__main__":
    main()
