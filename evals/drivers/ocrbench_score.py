"""Score OCRBench v2 generations with the benchmark's own eval.py.

    python3 ocrbench_score.py results/<model>/ocrbench/<tag> [more dirs ...]

Writes summary.json beside the generations: the EN overall, every scoring category, and
every raw task type, plus the generation diagnostics.

WHY A PRIVATE COPY OF THE HARNESS PER RUN
------------------------------------------
`spotting_metric.spotting_evaluation` hardcodes RELATIVE paths and wipes them on every
call:

    submit_path = "./eval_scripts/spotting_eval/submit"
    ...
    shutil.rmtree(file_path); os.makedirs(file_path)

So scoring must run with cwd at the OCRBench_v2 root, and two scoring processes sharing
that root will delete each other's working files mid-run. With four arms times four
repeats that is not a hypothetical. Each invocation therefore gets its own copy of
`eval_scripts/` in a temp directory -- it is a few hundred KB -- and the runs become
independent. Serialising instead would work too, and would be slower for no benefit.

WHY NOT get_score.py
--------------------
It prints and does not return, so nothing downstream can read a number without parsing
stdout, and it `import ipdb` at module scope. The category mapping is reproduced here
from that file, which is the part that matters, and the result is written as JSON in the
same shape as every other benchmark in this repo.

THE HEADLINE IS AN UNWEIGHTED MEAN OF CATEGORY MEANS, which is the benchmark's own
convention and not a choice made here. It means a 200-item category counts exactly as
much as an 800-item one, so a category the model floors on (coordinate output it was
never trained to emit) drags the overall as hard as a well-populated one. The per-category
table is therefore the real result and the overall is a summary of it; both are written.
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

# The VQA metric scores with METEOR, which needs WordNet. nltk searches only a fixed
# list of system directories plus NLTK_DATA, and compute nodes have no network, so a
# missing corpus fails deep inside scoring -- after the expensive TEDS work is done.
#
# PREPEND, do not setdefault. run_eval.sh already exports NLTK_DATA pointing at
# IFBench's corpus directory, which has punkt and stopwords but NOT wordnet, so a
# setdefault is a silent no-op and scoring dies at the end of a 50-minute job. Keeping
# any existing value means IFBench's own scoring is unaffected.
_NLTK = ("/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/"
         "nltk_data")
os.environ["NLTK_DATA"] = os.pathsep.join(
    [_NLTK] + ([os.environ["NLTK_DATA"]] if os.environ.get("NLTK_DATA") else []))

HARNESS = os.environ.get(
    "OCRBENCH_HARNESS",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/"
    "MultimodalOCR/OCRBench_v2")

# Reproduced from the benchmark's get_score.py. The 8 English scoring categories, in the
# order that file builds them.
EN_CATEGORIES = {
    "text_recognition": ["text recognition en", "fine-grained text recognition en",
                         "full-page OCR en"],
    "text_detection": ["text grounding en", "VQA with position en"],
    "text_spotting": ["text spotting en"],
    "relationship_extraction": ["key information extraction en",
                                "key information mapping en"],
    "element_parsing": ["document parsing en", "chart parsing en", "table parsing en",
                        "formula recognition en"],
    "mathematical_calculation": ["math QA en", "text counting en"],
    "visual_text_understanding": ["document classification en", "cognition VQA en",
                                  "diagram QA en"],
    "knowledge_reasoning": ["reasoning VQA en", "science QA en", "APP agent en",
                            "ASCII art classification en"],
}
TYPE_TO_CATEGORY = {t: c for c, ts in EN_CATEGORIES.items() for t in ts}

def run_eval(rows, workdir):
    """Run the benchmark's eval.py over `rows`, returning them with `score` attached."""
    src = os.path.join(HARNESS, "eval_scripts")
    dst = os.path.join(workdir, "eval_scripts")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    # nltk >= 3.9 caps edit_distance inputs at MAX_DISTANCE_INPUT_LEN (2000) as a DoS
    # guard, and OCRBench predates it. `full-page OCR en` compares whole transcriptions
    # -- 5156 characters in the reference predictions -- so the guard aborts scoring 58%
    # of the way through a run:
    #   ValueError: edit_distance: input length 5156 exceeds MAX_DISTANCE_INPUT_LEN
    # Raising it is the right call and not merely the convenient one: the guard exists
    # for two UNTRUSTED strings, while these are our own generations against the
    # benchmark's own ground truth. Truncating instead would silently change the metric.
    #
    # Done with an explicit runner rather than sitecustomize: sitecustomize depends on
    # the interpreter's startup machinery finding it, which it did not here, and its
    # failure mode is silence -- the run just dies at 58% again with the same error.
    # runpy executes eval.py unmodified under __main__, after the patch is in place.
    runner = os.path.join(workdir, "run_patched.py")
    with open(runner, "w") as f:
        # importlib, not `import nltk.metrics.distance as _d`: nltk's __init__
        # re-exports nltk.translate.metrics under the name `metrics`, so the dotted
        # form resolves to the wrong module and dies with
        #   ImportError: cannot import name 'distance' from 'nltk.translate.metrics'
        # import_module addresses the real submodule by name and ignores the shadowing.
        f.write("import sys, runpy, importlib\n"
                "_d = importlib.import_module('nltk.metrics.distance')\n"
                "_d.MAX_DISTANCE_INPUT_LEN = 10 ** 9\n"
                # Running eval.py directly puts its own directory on sys.path;
                # runpy does not, and eval.py imports its metric modules by bare name
                # (`from vqa_metric import ...`).
                "sys.path.insert(0, 'eval_scripts')\n"
                "sys.argv = ['eval.py'] + sys.argv[1:]\n"
                "runpy.run_path('eval_scripts/eval.py', run_name='__main__')\n")
    env = dict(os.environ)

    inp = os.path.join(workdir, "pred.json")
    outp = os.path.join(workdir, "scored.json")
    with open(inp, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    # cwd=workdir: eval.py's spotting metric resolves "./eval_scripts/spotting_eval/..."
    # against the process cwd, which is why the copy has to be the working directory
    # rather than merely on sys.path.
    proc = subprocess.run(
        [sys.executable, "run_patched.py", "--input_path", inp, "--output_path", outp],
        cwd=workdir, env=env, capture_output=True, text=True)
    if not os.path.exists(outp):
        raise SystemExit(f"eval.py produced no output\n--- stdout ---\n{proc.stdout[-2000:]}"
                         f"\n--- stderr ---\n{proc.stderr[-3000:]}")
    with open(outp) as f:
        return json.load(f)


def score_dir(d):
    raw = os.path.join(d, "raw.jsonl")
    if not os.path.exists(raw):
        print(f"skip {d}: no raw.jsonl")
        return None
    gen = [json.loads(l) for l in open(raw)]

    # eval.py reads `predict`; every other field it needs is already here verbatim,
    # because the inference driver carried the benchmark's own records through
    # untouched -- including the ABSENCE of `eval`, `bbox`, `bbox_list` and `content`
    # on tasks that do not use them, which is what eval.py's key-presence branching
    # reads. Nothing is repaired here and nothing needs to be.
    #
    # That is only true because the generations come from the original OCRBench_v2.json.
    # Against a HuggingFace parquet mirror this function needed two workarounds: a
    # fixed schema cannot express an absent key (it becomes the string "None", and
    # every ordinary VQA item then enters a branch it should not), and `answers` typed
    # list<string> turns the dict answers of `chart parsing en` and
    # `key information extraction en` into strings that die in dict_to_html. Reading
    # the authors' file removes the class of problem rather than those two instances.
    drop = {"content", "reasoning", "finish_reason", "completion_tokens", "uid"}
    rows = []
    for g in gen:
        row = {k: v for k, v in g.items() if k not in drop}
        # Undo the collision rename from ocrbench_infer: `content` in a generated row is
        # the MODEL's output, while eval.py's spotting metric wants the benchmark's own
        # `content` (the ground-truth word list). One name, two meanings.
        if "gt_content" in row:
            row["content"] = row.pop("gt_content")
        row["predict"] = g.get("content") or ""
        rows.append(row)

    with tempfile.TemporaryDirectory(prefix="ocrbench_") as workdir:
        scored = run_eval(rows, workdir)

    per_type, ignored = {}, 0
    for r in scored:
        if "ignore" in r:
            ignored += 1
            continue
        if "score" not in r:
            continue
        per_type.setdefault(r["type"], []).append(float(r["score"]))

    type_means = {t: sum(v) / len(v) for t, v in sorted(per_type.items())}
    cat_scores = {}
    for cat, types in EN_CATEGORIES.items():
        vals = [s for t in types for s in per_type.get(t, [])]
        if vals:
            cat_scores[cat] = sum(vals) / len(vals)
    overall = sum(cat_scores.values()) / len(cat_scores) if cat_scores else 0.0

    trunc = sum(1 for g in gen if g.get("finish_reason") == "length")
    empty = sum(1 for g in gen if not (g.get("content") or "").strip())
    toks = sorted(g["completion_tokens"] for g in gen if g.get("completion_tokens"))
    out = {
        "benchmark": "ocrbench_v2_en",
        # Scaled to 0-100 to match every other benchmark's summary in this repo; the
        # benchmark's own scripts print the same quantity in 0-1.
        "accuracy": 100.0 * overall,
        "n_scored": sum(len(v) for v in per_type.values()),
        "n_ignored": ignored,
        "categories": {k: 100.0 * v for k, v in sorted(cat_scores.items())},
        "task_types": {k: 100.0 * v for k, v in type_means.items()},
        "generation": {
            "n": len(gen), "truncated": trunc, "empty_content": empty,
            "tokens_p50": toks[len(toks) // 2] if toks else None,
            "tokens_p95": toks[int(len(toks) * 0.95)] if toks else None,
            "tokens_max": toks[-1] if toks else None,
        },
    }
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Per-item scores, persisted. summary.json alone cannot support compare_arms: the
    # paired bootstrap needs each arm's per-item vector on the SAME items, and the
    # repeat-SEM needs a score per pass. Re-deriving these means re-running eval.py for
    # 15-20 minutes, so they are written once here, next to the generations.
    with open(os.path.join(d, "scored.jsonl"), "w") as f:
        for r in scored:
            if "ignore" in r or "score" not in r:
                continue
            f.write(json.dumps({"id": r["id"], "type": r["type"],
                                "score": float(r["score"])}) + "\n")

    print(f"\n{d}")
    print(f"  EN overall {out['accuracy']:.2f}  "
          f"({out['n_scored']} scored, {ignored} ignored, {trunc} truncated)")
    for k, v in out["categories"].items():
        n = sum(len(per_type.get(t, [])) for t in EN_CATEGORIES[k])
        print(f"    {k:<26} {v:6.2f}  (n={n})")
    missing = [c for c in EN_CATEGORIES if c not in cat_scores]
    if missing:
        print(f"    MISSING categories (not in the generations): {', '.join(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    for d in args.dirs:
        score_dir(d)


if __name__ == "__main__":
    main()
