"""Freeze per-item scores into a small, versioned artifact for offline analysis.

    python3 bin/export_scores.py                 # every model / benchmark on disk
    python3 bin/export_scores.py --model muse-glimmer
    python3 bin/export_scores.py --check         # exit 1 if the export is stale

WHY THIS EXISTS
---------------
The generations are 785 MB and are gitignored, correctly: they are reproducible output,
not measurements. But `summary.json` -- the only thing versioned beside them -- carries
aggregates ONLY, and every test in compare_arms.py is paired over items. A sign-flip test
needs the 0/1 vector per item per repeat; a mean cannot be un-averaged back into one. So
a clone of this repo could render no table and run no test, and the analysis was pinned
to one filesystem.

This writes the smallest thing that is sufficient: for each (model, benchmark, tag), the
per-item score vector across repeats. 22 MB of plain JSON for the whole tree -- about
2 MB once git compresses it -- against 785 MB of generations, and it is what plotting
and table generation actually consume.

IT CALLS compare_arms' OWN LOADERS. Not a reimplementation -- the same functions, so an
exported number is identical to a live one by construction rather than by review. That
matters most where the scoring is not a simple mean: ocrbench_items applies the
w_i = N/(K*n_c) category reweighting that makes a flat item mean reproduce the
benchmark's mean-of-category-means headline, and mmlu_pro_items replays the seeded
extraction in question_id order. Both are easy to get subtly wrong twice.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Model outputs. Not one token. Everything needed to compute a score is here; nothing
needed to read an answer is. That keeps the artifact small, keeps generations out of
git history where they can never be removed, and means this file can be shared without
shipping a model's text.

If a qualitative question comes up -- why did this arm fail that item -- it needs the
generations, and the answer is to go back to the cluster. That is the intended trade.

STALENESS IS CHECKED, NOT ASSUMED. Each arm records the SIZE of every raw file it was
built from, plus a FORMAT_VERSION bumped whenever a loader's scoring changes. `--check`
regenerates and compares, so a stale export is a broken build rather than a quietly
wrong plot. Run it in CI, or before trusting a table produced somewhere else.

Sizes rather than mtimes: a resumed run appends, so the size moves whenever the content
does, while an mtime also moves when nothing was added -- and a guard that fires on a
no-op is one people learn to ignore.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVALS, "drivers"))

RESULTS = os.path.join(EVALS, "results")
SCORES = os.path.join(EVALS, "scores")
# Bumped whenever a loader's scoring changes in a way that alters exported numbers.
# An export carrying an older version is stale even if every input file is untouched.
FORMAT_VERSION = 1


def _sources(bench_root, tag):
    """The raw files an arm was built from, with their sizes -- the staleness key."""
    import compare_arms as C
    out = {}
    for d in C._repeat_dirs(bench_root, tag):
        for p in sorted(glob.glob(os.path.join(d, "raw*.jsonl"))
                        + glob.glob(os.path.join(d, "rep*.jsonl"))
                        + glob.glob(os.path.join(d, "scored.jsonl"))
                        + glob.glob(os.path.join(d, "*eval_results_strict.jsonl"))):
            out[os.path.relpath(p, RESULTS)] = os.path.getsize(p)
    return out


def export_model(model, benches=None, check=False):
    import compare_arms as C
    C.RESULTS = os.path.join(RESULTS, model)
    C._CACHE.clear()
    reg = {}
    mj = os.path.join(EVALS, "models.json")
    if os.path.exists(mj):
        reg = json.load(open(mj)).get(model, {}).get("report", {})

    stale, written = [], []
    for bench in (benches or list(C.LOADERS)):
        root = os.path.join(C.RESULTS, C.BENCH_ROOT.get(bench, bench))
        if not os.path.isdir(root):
            continue
        tags = C.available_tags(bench)
        arms = {}
        for tag in tags:
            per = C.items(bench, tag)
            if not per:
                continue
            ids = sorted(per)
            k = min(len(per[i]) for i in ids)
            arms[tag] = {
                # Item ids are stored per arm rather than once per benchmark: arms do
                # not always cover the same items (a rerun, a partial pass), and a
                # shared id list would silently pair item 5 of one arm with item 5 of
                # another after any divergence.
                "ids": [str(i) for i in ids],
                "scores": [[round(float(v), 6) for v in per[i][:k]] for i in ids],
                "repeats": k,
                "sources": _sources(root, tag),
            }
        if not arms:
            continue
        doc = {"format_version": FORMAT_VERSION, "model": model, "bench": bench,
               "baseline": reg.get("baseline"), "arms": arms}
        dest = os.path.join(SCORES, model, f"{bench.replace('/', '_')}.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # The trailing newline is part of what gets written, so it has to be part of
        # what is compared. Without it `old == new` was never true and --check reported
        # every export stale immediately after writing it -- a guard that always fires
        # is one that gets ignored, which is worse than not having it.
        new = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        old = open(dest).read() if os.path.exists(dest) else None
        if old == new:
            continue
        if check:
            stale.append(os.path.relpath(dest, EVALS))
        else:
            with open(dest, "w") as f:
                f.write(new)
            written.append((os.path.relpath(dest, EVALS), len(new), len(arms)))
    return written, stale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=None)
    ap.add_argument("--benches", nargs="+", default=None)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any export is out of date")
    args = ap.parse_args()

    models = args.model or sorted(
        d for d in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, d)))
    total, all_stale = 0, []
    for m in models:
        written, stale = export_model(m, args.benches, args.check)
        all_stale += stale
        for path, size, n_arms in written:
            print(f"  {path:<52} {n_arms:>2} arms  {size / 1024:>7.1f} KiB")
            total += size
    if args.check:
        if all_stale:
            print(f"STALE: {len(all_stale)} export(s) do not match the results on disk")
            for p in all_stale:
                print(f"   {p}")
            raise SystemExit(1)
        print("exports are up to date")
        return
    print(f"\n{total / 1024:.1f} KiB written under {os.path.relpath(SCORES, EVALS)}/")


if __name__ == "__main__":
    main()
