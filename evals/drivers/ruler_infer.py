"""RULER (github.com/NVIDIA/RULER) against a local vLLM OpenAI endpoint.

    python3 drivers/ruler_infer.py --out RESULTS --data-dir DATA \
        --seqlens 8192,32768 --base-url http://127.0.0.1:8000/v1 --model NAME

WHAT IS UPSTREAM AND WHAT IS OURS. The two things that define the benchmark -- how the
haystacks are generated and how the answers are scored -- are RULER's own code, called
unmodified:

    data:    scripts/data/prepare.py        (driven by bin/run_eval.sh, not from here)
    metrics: scripts/eval/synthetic/constants.py  (used by ruler_score.py)

Only the INFERENCE layer is ours, and deliberately so. RULER ships its own clients in
scripts/pred/client_wrappers.py, but none of them fits this harness:

  * VLLMClient talks to scripts/pred/serve_vllm.py's bespoke /generate endpoint, not to
    an OpenAI-compatible server. Using it would mean standing up a second, different
    server and abandoning the disaggregated pair that run_eval.sh already builds --
    which is the entire thing these evals exist to measure.
  * OpenAIClient is aimed at api.openai.com: it requires OPENAI_API_KEY and sizes its
    generation budget with tiktoken, whose token counts are wrong for this tokenizer.

Going through common.chat instead means RULER inherits --disagg, --tp, resume, the
abort-detection in common.chat and the truncation reporting, all of which are already
proven here. The prompt is assembled exactly as call_api.py does it (line 305):
`input + answer_prefix`.

WHY NOT `nemo`. RULER's call_api.py and evaluate.py import read_manifest from
nemo.collections.asr -- an ASR dependency used purely as `[json.loads(l) for l in f]`.
Its data generators use the repo's own scripts/data/manifest_utils.py instead, and the
scoring metrics live in a dependency-free constants.py, so the canonical generation and
the canonical metrics are both reachable without installing NeMo. Nothing is
reimplemented to avoid it; the import is simply not on the path we take.

THINKING IS OFF BY DEFAULT, and this is a real choice rather than an oversight. RULER
sets a per-task generation budget of 30-128 tokens in its own task configs (128 for
niah, 30 for vt, 32 for qa). A thinking model spends that entire budget inside <think>
and returns empty content -- not a low score, a zero, for every task. Raising the budget
instead would measure a different benchmark than the one RULER specifies. `--think`
opts back in for models where that is the intended configuration.
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                   # noqa: E402
from common import run_resumable, truncation_report             # noqa: E402


# Persisted per row, alongside the response run_resumable appends. Everything else RULER
# generates -- `input`, `answer_prefix`, `index`, `token_position_answer`,
# `length_w_model_temp` -- is either the haystack itself or derivable from the corpus,
# and is dropped. `length` is kept because it is the ACTUAL token count of the doc, which
# is the one thing that cannot be recovered without re-tokenizing.
KEEP = {"task", "seqlen", "doc", "outputs", "length"}


def _load_module(path, name):
    """Import a file by path. RULER's constants live inside a non-package tree."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def task_budgets(ruler_root):
    """{task_name: tokens_to_generate}, straight from RULER's own configs.

    synthetic.yaml maps a task NAME (niah_single_1) to a task TYPE (niah); the budget is
    attached to the type in data/synthetic/constants.py. Duplicating either here would
    let this file drift from the benchmark it claims to run.
    """
    import yaml
    with open(os.path.join(ruler_root, "scripts/synthetic.yaml")) as fh:
        by_name = yaml.safe_load(fh)
    types = _load_module(
        os.path.join(ruler_root, "scripts/data/synthetic/constants.py"),
        "ruler_data_constants").TASKS
    return {name: types[cfg["task"]]["tokens_to_generate"]
            for name, cfg in by_name.items()}


def load_docs(data_dir, seqlen, task, limit=0):
    """RULER's prepare.py writes <data_dir>/<seqlen>/<task>/validation.jsonl."""
    path = os.path.join(data_dir, str(seqlen), task, "validation.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # The doc id is the LINE NUMBER, not d["index"]. In niah.py `index` is the
            # character offset of the answer within the input (niah.py:281), which is
            # neither unique nor stable across tasks -- using it as a resume key would
            # silently collapse distinct docs onto one uid and lose samples.
            d["doc"] = i
            d["task"] = task
            d["seqlen"] = seqlen
            out.append(d)
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", required=True, help="prepare.py's --save_dir root")
    ap.add_argument("--ruler-root",
                    default=os.environ.get("RULER_HARNESS", ""),
                    help="clone of github.com/NVIDIA/RULER")
    ap.add_argument("--seqlens", required=True, help="comma-separated, e.g. 8192,32768")
    ap.add_argument("--tasks", default="", help="comma-separated; default: all present")
    ap.add_argument("--limit", type=int, default=0, help="docs per (task, seqlen)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                    help="see the module docstring: on by mistake means every score is 0")
    args = ap.parse_args()

    if not args.ruler_root:
        sys.exit("ERROR: --ruler-root (or $RULER_HARNESS) is required")
    seqlens = [int(s) for s in args.seqlens.split(",") if s.strip()]
    budgets = task_budgets(args.ruler_root)
    tasks = ([t for t in args.tasks.split(",") if t.strip()]
             or sorted(budgets))

    # RULER's answer_prefix is a PREFILLED ASSISTANT TURN, not part of the user's
    # question. call_api.py sends `input + answer_prefix` to a COMPLETION endpoint
    # (call_api.py:305), so the model literally continues the sentence "...the top 10
    # words that appear most often in the list are:" and emits only the list. Appending
    # it to the user message here instead let the model open a fresh assistant turn and
    # preamble: measured on cwe, 406 of 500 responses burned the entire 120-token budget
    # on "To determine the 10 most common words, we count the frequency... 1. **stucco**:
    # Appears at indices 33, 52, 109..." and were cut off mid-list. That scored 48.6
    # where an 8B model scores 88.9 on the same task -- a harness artifact read as a
    # long-context result.
    #
    # continue_final_message reopens the final assistant message instead of starting a
    # new one; add_generation_prompt must be false, and vLLM rejects setting both.
    # Verified by rendering this model's template: the prompt ends
    #   <|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: The top 10 words ... are:
    # which is exactly RULER's completion semantics inside a chat turn.
    common.DEFAULT_EXTRA_BODY.setdefault("continue_final_message", True)
    common.DEFAULT_EXTRA_BODY.setdefault("add_generation_prompt", False)

    if not args.think:
        # Merged into the request body by common.chat. setdefault there means a
        # models.json request_extra_body that already pins chat_template_kwargs wins,
        # so a model with its own thinking convention is not overridden from here.
        common.DEFAULT_EXTRA_BODY.setdefault(
            "chat_template_kwargs", {"enable_thinking": False})

    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "raw.jsonl")

    # ONE line naming the whole sweep, because run_resumable's own header is per
    # (length, task) and RULER prints 65 of them into a single raw.jsonl. Anything
    # reading "the last total" would see one task's 500 next to a file holding every
    # task's rows and report e.g. 30000/500. drivers/progress.py keys off this line.
    planned = sum(len(load_docs(args.data_dir, sl, t, args.limit))
                  for sl in seqlens for t in tasks)
    print(f"ruler sweep: {len(tasks)} tasks x {len(seqlens)} lengths "
          f"= {planned} total -> {raw}", flush=True)

    n_err = 0
    for seqlen in seqlens:
        for task in tasks:
            docs = load_docs(args.data_dir, seqlen, task, args.limit)
            if not docs:
                print(f"  skip {seqlen}/{task}: no data", flush=True)
                continue
            # THE HAYSTACK IS NOT PERSISTED. run_resumable writes `dict(item)` plus the
            # response, so leaving `input` on the item would store the full prompt next
            # to a ~100-token answer: measured at 97% of every row, which at 64k puts one
            # sweep in the multi-GB range per arm, on a filesystem that has already hit
            # its quota once. Nothing reads it back -- ruler_score and progress.py use
            # outputs/content/task/seqlen -- and it regenerates exactly from the corpus,
            # which is cached and keyed by tokenizer content.
            #
            # The prompt is therefore held in memory for this task only and looked up by
            # uid, so request_of still sends the identical bytes call_api.py would
            # (input + answer_prefix, call_api.py:305).
            prompts = {f"{d['seqlen']}|{d['task']}|{d['doc']}":
                       (d["input"], d.get("answer_prefix", "")) for d in docs}
            slim = [{k: v for k, v in d.items() if k in KEEP} for d in docs]
            # Per TASK, because RULER's budget is per task and the whole point of the
            # non-thinking default is that these budgets are small and deliberate.
            n_err += run_resumable(
                slim,
                uid_of=lambda d: f"{d['seqlen']}|{d['task']}|{d['doc']}",
                request_of=lambda d: [
                    {"role": "user",
                     "content": prompts[f"{d['seqlen']}|{d['task']}|{d['doc']}"][0]},
                    # RULER's answer_prefix is a PREFILLED ASSISTANT CONTINUATION, not
                    # part of the question -- see the module docstring.
                    {"role": "assistant",
                     "content": prompts[f"{d['seqlen']}|{d['task']}|{d['doc']}"][1]}],
                out_path=raw,
                workers=args.workers,
                desc=f"{seqlen}/{task}",
                max_tokens=budgets[task],
                base_url=args.base_url,
                model=args.model,
            )

    rows = [json.loads(l) for l in open(raw)] if os.path.exists(raw) else []
    print(f"\n{len(rows)} rows -> {raw}", flush=True)
    if rows:
        print(f"truncation: {truncation_report(rows)}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
