"""Collect every summary.json under muse/results into one table.

    python3 report_all.py                    # all tags
    python3 report_all.py --tags bf16 disagg # compare two serving configurations

Prints the measured score beside the published BF16 figure and the gap. The published
column is a REFERENCE, not a target to tune toward: the point of the exercise is the
bf16-vs-disagg delta, and a reproduction that lands a couple of points off tells you the
protocol differs somewhere, not that the model is worse.

Coverage and truncation are printed on the same row on purpose. A score computed over
1200 of 1730 items, or one with 40 truncated responses in it, is not comparable to
either the published figure or another tag — and both of those look like perfectly
ordinary numbers if the row does not say so.
"""
import argparse
import glob
import json
import os
import re
import statistics

# Two-sided 95% => the 0.975 quantile of Student's t, by degrees of freedom. A LOOKUP,
# because numpy has no t quantile and scipy is not in the overlay. It matters at these
# sample sizes: with 4 repeats (df=3) the multiplier is 3.18, not the 1.96 a normal
# approximation would use -- a 70% wider interval. Using the normal here would quietly
# make every result look better resolved than it is.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df):
    if df < 1:
        return float("inf")
    if df in _T975:
        return _T975[df]
    below = [k for k in _T975 if k <= df]
    return _T975[max(below)] if df < 30 else 2.054      # normal limit

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

# From the vendor's published report. MMMU-Pro's 74.0 is attributed to `vision` ONLY.
# The report itself does not name a subset -- it says only "1730 multiple choice
# questions ... the answer choice space has been significantly expanded", which fits
# both candidates (vision and standard (10 options) are each 1730 items and both derive
# from the 10-option construction), and it sources the number from Artificial Analysis
# rather than measuring it. So this mapping is a judgement, not a quotation. What
# supports it is measurement: vision lands at 73.4 and standard10 at 72.0.
#
# standard10 is deliberately left OUT of this dict rather than given the same target.
# Printing 74.0 against both would show a gap for a subset the figure may not describe,
# and a gap invites explaining. standard10 is still run and still scored -- it is a
# second measurement of the same serving change, which is its actual value here.
# Keyed by MODEL then benchmark: a published figure belongs to one model, and a table
# that shares them across models would silently score Qwen against Muse-Glimmer's
# reference. Qwen3.8-27B publishes no MMMU or MMMU-Pro number at all, so its rows have
# no target -- which is a fact to display, not a gap to fill.
PUBLISHED = {
    "muse-glimmer": {"gpqa": 83.5, "ifbench": 77.0, "mmmu/vision_cot": 74.0},
    "qwen3.8-27b": {"gpqa": 89.2},
}

# Which field in each summary.json is THE score. IFBench writes four; strict
# prompt-level is the one usually reported, and the rest stay in the file.
SCORE_KEY = {"gpqa": "accuracy", "ifbench": "strict_prompt_acc", "mmmu": "accuracy",
             "ocrbench": "accuracy", "mmlu_pro": "accuracy"}

# ...and what UNIT that field is in. Everything here writes a fraction and is scaled to
# points below, except ocrbench: its scorer runs the benchmark's own eval.py unmodified,
# and that script reports points. Scaling it again printed Gemma's 63.46 as 6345.50 on
# every ocrbench row.
#
# Fixed here rather than in the scorer because summary.json is the scorer's OUTPUT
# CONTRACT and twenty-odd of them are already on disk. Changing the writer would leave
# old and new files in different units with nothing to tell them apart -- a far worse
# failure than an obviously absurd number, since a plausible-but-wrong 63.46 from a file
# that meant 0.6346 would never be questioned. compare_arms is unaffected either way: it
# scores from scored.jsonl per item and never reads this field.
SCORE_SCALE = {"ocrbench": 1.0}


# `bf16-r2` is a REPEAT of `bf16`, not a different configuration. GPQA gets its repeats
# from --repeats inside one job; IFBench does not, so its repeats are separate jobs under
# suffixed tags. Grouping them here is what turns "disagg scored 2 points lower" into a
# statement with an error bar -- at temperature 1.0 and 299 prompts a single pair of runs
# cannot distinguish a real 2-point effect from ordinary sampling spread.
REPEAT_SUFFIX = re.compile(r"-r\d+$")


def base_tag(tag):
    return REPEAT_SUFFIX.sub("", tag)


def rows(tags=None, models=None):
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "**", "summary.json"),
                                 recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), RESULTS)
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue                    # results/<model>/<bench>/[<setting>/]<tag>
        model, bench, tag = parts[0], parts[1], parts[-1]
        family = "/".join(parts[1:-1])
        if tags and tag not in tags:
            continue
        if models and model not in models:
            continue
        d = json.load(open(path))
        score = d.get(SCORE_KEY.get(bench, "accuracy"))
        scale = SCORE_SCALE.get(bench, 100.0)
        gen = d.get("generation", d)
        out.append({
            "model": model,
            # How many independent generation passes stand behind this score. GPQA
            # carries its 4 repeats INSIDE one summary (--repeats 4); IFBench and MMMU
            # get theirs from separate jobs under -rN tags and are counted by the
            # grouping below. Both are the same quantity and must be counted the same
            # way, or an arm looks un-averageable purely because of how its repeats
            # were scheduled.
            "repeats": len(d["repeats"]) if isinstance(d.get("repeats"), list) else 1,
            "bench": family, "tag": tag,
            "score": None if score is None else score * scale,
            "n": d.get("n") or d.get("n_scored") or gen.get("n")
                 or d.get("strict_n_prompts"),
            "truncated": gen.get("truncated"),
            "empty": gen.get("empty_content"),
            "err": (d.get("error") or 0) * scale if d.get("error") is not None else None,
            "published": PUBLISHED.get(model, {}).get(family),
        })
    return out


def macro(data):
    """Per-arm average across the benchmarks, with ONE noise model throughout.

    THE MODEL. The only variance that matters for ranking serving configurations is
    GENERATION NOISE ON A FIXED ITEM SET: the model samples at temperature 1.0, so
    repeating an eval on the same items gives a different score. Every arm sees the
    identical items, so per-item difficulty is common-mode and cancels in any
    comparison between arms. Each benchmark therefore contributes `sem = stdev(repeat
    scores) / sqrt(n_repeats)`, and because the benchmarks are separate runs,

        SE(average) = (1/k) * sqrt(sum_i sem_i^2)

    is exact rather than an approximation.

    WHAT THIS REPLACED, AND WHY IT WAS WRONG. The first version of this function mixed
    two incommensurable estimators: repeat-SEM for GPQA and IFBench, and a BINOMIAL
    standard error sqrt(p(1-p)/n) for MMMU, which had no repeats. Those measure
    different things -- the first is run-to-run generation noise, the second is
    uncertainty from treating the item set as a sample of some larger population -- so
    adding them in quadrature produced a number that estimates nothing. It was also
    badly scaled: GPQA's binomial estimate is 2.7 points against a MEASURED repeat
    spread of 0.5, because item-sampling noise is exactly the component that cancels
    when two arms are scored on the same items.

    The fix was to run MMMU repeats, not to pick a different formula. A benchmark with
    one pass is reported with no error bar and is EXCLUDED from the average, because a
    missing measurement should shrink the table rather than get imputed.

    The spread BETWEEN benchmarks is deliberately not reported here. It is ~4-5 points
    for every arm, it measures how far GPQA sits from MMMU, and printing it beside an
    uncertainty invites reading it as one.
    """
    import collections
    per_tag = collections.defaultdict(dict)
    for r in data:
        per_tag[r["tag"]][r["bench"]] = r
    benches = sorted({r["bench"] for r in data})

    def usable(per):
        return (len(per) == len(benches)
                and all(per[b]["err"] is not None and per[b].get("runs", 1) > 1
                        for b in benches))

    full = {t: v for t, v in per_tag.items() if usable(v)}
    dropped = sorted(set(per_tag) - set(full))
    print(f"\n=== average over {len(benches)} benchmarks "
          f"({', '.join(benches)}) ===")
    if not full:
        print("  no arm has repeats on every benchmark yet -- nothing averageable.")
        if dropped:
            for t in dropped:
                have = [f"{b.split('/')[0]}x{per_tag[t][b].get('runs', 1)}"
                        for b in sorted(per_tag[t])]
                print(f"    {t:<14} has {', '.join(have)}")
        return []

    print(f"{'tag':<14} {'avg':>7} {'95% CI':>18}   per-benchmark")
    rows = []
    for tag, per in sorted(full.items()):
        scores = [per[b]["score"] for b in benches]
        k = len(benches)
        # Contribution of each benchmark to the standard error of the average.
        cs = [per[b]["err"] / k for b in benches]
        dfs = [per[b].get("runs", 1) - 1 for b in benches]
        se = sum(c * c for c in cs) ** 0.5
        # Welch-Satterthwaite: the average combines three variances estimated from only
        # 4 points each, so it has no single obvious degrees of freedom. This is the
        # standard effective-df for exactly that situation, and it keeps the interval
        # honest when one benchmark is much noisier than the others (which is the case
        # here -- GPQA's repeat spread is 3x IFBench's on the quantized arms).
        denom = sum(c ** 4 / d for c, d in zip(cs, dfs) if d > 0)
        nu = (sum(c * c for c in cs) ** 2) / denom if denom > 0 else 1
        h = t975(int(nu)) * se
        avg = statistics.mean(scores)
        rows.append({"tag": tag, "avg": avg, "se": se, "eff_df": nu,
                     "ci96": [avg - h, avg + h],
                     "per_benchmark": {b: {"score": per[b]["score"],
                                           "sem": per[b]["err"],
                                           "repeats": per[b].get("runs")}
                                       for b in benches}})
        detail = "  ".join(f"{per[b]['score']:.1f}" for b in benches)
        print(f"{tag:<14} {avg:>7.2f} {f'[{avg - h:.2f}, {avg + h:.2f}]':>18}   {detail}")
    if dropped:
        print(f"  excluded (no repeats on every benchmark): {', '.join(dropped)}")
    with open(os.path.join(RESULTS, "macro_average.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    data = rows(args.tags, args.models)
    if not data:
        raise SystemExit(f"no summary.json under {RESULTS}")

    # Collapse repeat tags into one row per (benchmark, configuration).
    groups = {}
    for r in data:
        groups.setdefault((r["model"], r["bench"], base_tag(r["tag"])), []).append(r)
    data = []
    for (model, bench, tag), g in groups.items():
        scores = [x["score"] for x in g if x["score"] is not None]
        merged = dict(g[0], model=model, bench=bench, tag=tag)
        if scores:
            merged["score"] = statistics.mean(scores)
            if len(scores) > 1:
                merged["err"] = statistics.stdev(scores) / len(scores) ** 0.5
        merged["runs"] = sum(x.get("repeats", 1) for x in g)
        for k in ("n", "truncated", "empty"):
            vals = [x[k] for x in g if x[k] is not None]
            merged[k] = sum(vals) if vals else None
        data.append(merged)

    for r in data:
        r["bench"] = f"{r['model']}/{r['bench']}"
    w = max(len(r["bench"]) for r in data)
    print(f"{'benchmark':<{w}}  {'tag':<14} {'score':>7} {'95% CI':>18} "
          f"{'pub':>6} {'gap':>6} {'reps':>5} {'n':>6} {'trunc':>6} {'empty':>6}")
    for r in sorted(data, key=lambda x: (x["bench"], x["tag"])):
        sc = r["score"]
        gap = None if (sc is None or r["published"] is None) else sc - r["published"]
        f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        # A 96% interval from the repeats, via Student's t with df = reps-1. Reported
        # instead of a +/- standard error because a standard error is not an interval
        # and readers convert it to one with the wrong multiplier: at 4 repeats the
        # correct factor is 3.48, and eyeballing "+/- 2 sigma" understates by 70%.
        reps = r.get("runs", 1)
        if r["err"] is not None and reps > 1:
            h = t975(reps - 1) * r["err"]
            r["ci96"] = [sc - h, sc + h]
            ci = f"[{sc - h:.2f}, {sc + h:.2f}]"
        else:
            r["ci96"] = None
            ci = "single pass"
        print(f"{r['bench']:<{w}}  {r['tag']:<14} {f(sc):>7} {ci:>18} "
              f"{f(r['published'], 1):>6} {f(gap, 1):>6} {reps:>5} "
              f"{r['n'] if r['n'] is not None else '-':>6} "
              f"{r['truncated'] if r['truncated'] is not None else '-':>6} "
              f"{r['empty'] if r['empty'] is not None else '-':>6}")

    macro(data)

    with open(os.path.join(RESULTS, "all_summaries.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\n-> {os.path.join(RESULTS, 'all_summaries.json')}")


if __name__ == "__main__":
    main()
