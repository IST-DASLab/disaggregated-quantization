"""Freeze per-item generation lengths into a compact, versioned artifact.

    python3 bin/export_lengths.py                    # every model / benchmark on disk
    python3 bin/export_lengths.py --model qwen3.8-27b

Companion to bin/export_scores.py. That file versions WHETHER each item was answered
correctly; this one versions HOW LONG the answer was, which the paper's appendix needs
and which is otherwise unrecoverable: raw.jsonl is gitignored (a single MMLU-Pro pass is
119 MB), so once a results tree is cleaned the lengths are gone.

WHY THIS MATTERS AND NOT JUST THE SCORE. Response length is how you tell a real quality
gain from bought test-time compute. Measured here: a QAD prefill beat an RTN one by 15
points on IQ1_S while generating 42% FEWER tokens -- a claim that cannot be made or
checked from scores alone.

NPZ OF INT32, NOT JSON. Three arrays per arm at (items x repeats) is 12,032 x 5 numbers
per array; as JSON that is tens of MB per benchmark and unreadable in a diff either way.
Compressed npz is ~1% of that and loads in one call.

WHAT IS STORED, per arm, row-aligned with `<tag>|ids`:

    <tag>|tokens   completion_tokens, the EXACT total the server generated
    <tag>|rchars   len(reasoning), characters
    <tag>|cchars   len(content),   characters

Characters, not tokens, for the two components: the OpenAI response reports one total
token count and does not split it across the reasoning and content channels, so a
per-channel token count would have to be re-tokenized -- slow, and wrong the moment the
tokenizer differs from the served one. `tokens` is exact; the two char counts give the
split. Their ratio is stable enough to apportion `tokens` when that is wanted.

-1 means the item is missing from that repeat, which happens whenever arms cover
different item sets (a partial pass, a rerun). It is not zero, because zero is a real
and common value: an empty content channel is the failure mode these evals keep hitting.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVALS, "drivers"))
RESULTS = os.path.join(EVALS, "results")
SCORES = os.path.join(EVALS, "scores")
FORMAT_VERSION = 1

# Checked in order: benchmarks disagree on what an item id is called, and picking the
# wrong one would silently produce a per-row array that cannot be joined to the scores.
ID_FIELDS = ("uid", "question_id", "id", "index")


def _rows(path):
    with open(path) as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue        # torn final line: the file may be being appended to


def _sort_key(i):
    """Match bin/export_scores.py's ordering exactly.

    That exporter sorts the loaders' NATIVE keys, which are ints for MMLU-Pro, then
    stringifies. Sorting the stringified form instead puts '10' before '2' and silently
    misaligns every row against the scores export -- the one thing these arrays must
    join to.
    """
    return (0, int(i), "") if i.isdigit() else (1, 0, i)


def _id_of(row):
    for f in ID_FIELDS:
        if f in row:
            return str(row[f])
    return None


def arm_arrays(dirs):
    """{id: [(tokens, rchars, cchars), ...]} in repeat order, or None if nothing read."""
    per, nrep = {}, 0
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "raw*.jsonl")))
        if not files:
            continue
        seen = {}
        for p in files:
            for r in _rows(p):
                i = _id_of(r)
                if i is None or i in seen:
                    continue    # de-duplicated like the scores exporter: a resumed run
                seen[i] = (int(r.get("completion_tokens") or 0),
                           len(r.get("reasoning") or ""),
                           len(r.get("content") or ""))
        if not seen:
            continue
        for i, v in seen.items():
            per.setdefault(i, {})[nrep] = v
        nrep += 1
    return (per, nrep) if nrep else (None, 0)


def export_model(model, benches=None):
    import compare_arms as C
    C.RESULTS = os.path.join(RESULTS, model)
    written = []
    for bench in (benches or list(C.LOADERS)):
        root = os.path.join(C.RESULTS, C.BENCH_ROOT.get(bench, bench))
        if not os.path.isdir(root):
            continue
        out = {}
        for tag in C.available_tags(bench):
            per, nrep = arm_arrays(C._repeat_dirs(root, tag))
            if not per:
                continue
            ids = sorted(per, key=_sort_key)
            tok = np.full((len(ids), nrep), -1, dtype=np.int32)
            rch = np.full((len(ids), nrep), -1, dtype=np.int32)
            cch = np.full((len(ids), nrep), -1, dtype=np.int32)
            for r, i in enumerate(ids):
                for c, v in per[i].items():
                    tok[r, c], rch[r, c], cch[r, c] = v
            out[f"{tag}|ids"] = np.array(ids)
            out[f"{tag}|tokens"] = tok
            out[f"{tag}|rchars"] = rch
            out[f"{tag}|cchars"] = cch
        if not out:
            continue
        out["__format_version__"] = np.array([FORMAT_VERSION], dtype=np.int32)
        dest = os.path.join(SCORES, model, f"{bench.replace('/', '_')}_lengths.npz")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        np.savez_compressed(dest, **out)
        n_arms = sum(1 for k in out if k.endswith("|ids"))
        print(f"  {model}/{bench}: {n_arms} arms -> {dest} "
              f"({os.path.getsize(dest)/1e6:.1f} MB)")
        written.append(dest)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--bench", default=None)
    args = ap.parse_args()
    models = [args.model] if args.model else sorted(
        d for d in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, d)))
    for m in models:
        export_model(m, [args.bench] if args.bench else None)


if __name__ == "__main__":
    main()
