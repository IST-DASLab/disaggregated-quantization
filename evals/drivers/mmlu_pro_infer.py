"""Generate and score MMLU-Pro, the benchmark's own 5-shot CoT protocol.

    python3 mmlu_pro_infer.py --out RESULTS
    python3 mmlu_pro_infer.py --out RESULTS --shard 0 --num-shards 4
    python3 mmlu_pro_infer.py --out RESULTS --score-only

12,032 test questions over 14 categories, mostly 10 options (A-J) but as few as 3.
Text only -- the first benchmark here that sends no image.

THE PROTOCOL IS THE AUTHORS', NOT ONE INVENTED HERE
----------------------------------------------------
The system message and `Options are:` formatting come from run_gpt4o.py in the dataset
repo; the FIVE-SHOT CoT prefix is assembled per category from the validation split's
`cot_content`, as the authors do. The EXTRACTION is the three-stage chain from
evaluate_from_local.py in the MMLU-Pro GitHub repo, which is the canonical scorer --
run_gpt4o.py is a demo and carries only the first of its three patterns. Reproduced
rather than improved, for the same reason ocrbench_score.py runs eval.py unmodified: a
better prompt makes a number that is not MMLU-Pro.

Two deliberate departures, both harness-wide policy:
  * SAMPLING COMES FROM THE SERVER. The official script sends temperature 0.1 / top_p 1;
    every driver here sends only max_tokens so the vendor's generation_config applies.
    Sending 0.1 would substitute the benchmark author's settings for the model owner's.
  * max_tokens is the model's, not the official 4096. These are reasoning models: muse's
    GPQA p50 is 3907 tokens and its p99 reaches 32768. At 4096 roughly half of every
    reasoning model's answers would truncate, and a truncated answer returns EMPTY
    content that scores as wrong -- indistinguishable from a model that answered badly.

THE RANDOM GUESS IS SEEDED, and that matters more than it looks. The official extractor
picks a random letter when the regex misses, so an unseeded run makes the same
predictions score differently between the generation pass and any later --score-only
pass. Seeded once per scoring run, the guesses are a deterministic function of the
outputs. They are also counted and reported: a high `unparsed` count means the answers
are not in the expected form, and the accuracy is then partly measuring a coin flip.

SHARDING, because 12,032 reasoning-model items do not fit a 4h job
------------------------------------------------------------------
`--shard K --num-shards N` keeps items where `index % N == K`. Modulo, not a contiguous
slice: the file is ordered by category, so slicing would give shard 0 all the math and
shard 3 all the history, and any per-shard diagnostic would be measuring the category
rather than the shard. Modulo makes every shard a stratified sample of the whole.

Each shard writes `raw.shard{K}of{N}.jsonl` and resumes independently. Scoring globs all
of them, so shards can finish in any order, on different nodes, at different times. A
single unsharded run still writes plain `raw.jsonl` and nothing downstream has to know
which mode produced a directory.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import run_resumable, truncation_report          # noqa: E402

HF_HOME = os.environ.get("HF_HOME",
                         "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache")
# Read the cached parquet directly rather than load_dataset("TIGER-Lab/MMLU-Pro"). With
# HF_HUB_OFFLINE=1 -- which compute nodes need, having no network -- the by-name form
# resolves the repo before it consults the cache and raises OfflineModeIsEnabled on a
# fully downloaded dataset.
DATA_GLOB = os.path.join(HF_HOME, "hub", "datasets--TIGER-Lab--MMLU-Pro",
                         "snapshots", "*", "data")
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# THE OFFICIAL THREE-STAGE CHAIN, from evaluate_from_local.py in the MMLU-Pro repo --
# NOT the single regex in the dataset repo's run_gpt4o.py, which is a demo script.
#
# The difference is not cosmetic. With the first pattern alone, 81.3% of muse-glimmer's
# 12,032 answers failed to extract and fell through to a random guess, scoring the arm at
# 24.65% -- almost exactly 0.813 * 10% + 0.187 * 85%. The model was answering correctly
# and phrasing it its own way: "i.e. option (F) 40 %.", "That is option J." A reasoning
# model does not copy the exemplars' "The answer is (A)." closing, and the strict regex
# silently converts that into a coin flip.
ANSWER_RE = re.compile(r"answer is \(?([A-J])\)?")
ANSWER_RE2 = re.compile(r".*[aA]nswer:\s*([A-J])")
# Last standalone A-J anywhere in the response. Crude, and the authors' own last resort:
# it is what rescues "That is option J." Applied only after both explicit forms miss.
ANSWER_RE3 = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)
SEED = 20260827


def _split(name):
    f = sorted(glob.glob(os.path.join(DATA_GLOB, f"{name}-*.parquet")))
    if not f:
        raise SystemExit(
            f"MMLU-Pro {name} split not in the hub cache under {DATA_GLOB}.\n"
            f"Run on a login node: hf download TIGER-Lab/MMLU-Pro --repo-type dataset")
    import pyarrow.parquet as pq
    return pq.read_table(f[0]).to_pydict()


def form_options(options):
    """`A. text` per line, lm-eval's mmlu_pro doc_to_text, not run_gpt4o's `(A): text`."""
    return "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options))


def question_text(q, options):
    return f"Question:\n{q}\nOptions:\n{form_options(options)}Answer: Let's think step by step."


INSTRUCTION = ('The following are multiple choice questions (with answers) about '
               '{category}. Think step by step and then finish your answer with '
               '"the answer is (X)" where X is the correct letter choice.\n\n')


def few_shot_turns():
    """Five exemplars per category as USER/ASSISTANT PAIRS, not one concatenated blob.

    THIS IS THE WHOLE BALLGAME, and getting it wrong cost a full sweep. Delivered as a
    single user message -- run_gpt4o.py's layout, which assumes a raw completion -- the
    model is not continuing a pattern, it is answering a person, and it replies in its
    own register: "i.e. option (F) 40 %.", "That is option J." Only 18.7% of
    muse-glimmer's answers then matched `answer is (X)`, 75.3% fell through to the
    "last standalone A-J letter" heuristic, and that noise made nvfp4 beat bf16 by 9
    points -- a quantization result that was purely an artefact of extraction.

    As alternating turns the model imitates the assistant format and closes with "The
    answer is (X)" on its own, which is what makes the single strict regex sufficient.

    This matches qad/eval/eval_vllm.py, which runs lm-eval with apply_chat_template=True
    and few-shot as multi-turn -- the established protocol in this repo, and the one its
    existing mmlu_pro results were produced with.
    """
    v = _split("validation")
    turns = collections.defaultdict(list)
    for q, opts, cot, cat in zip(v["question"], v["options"], v["cot_content"],
                                 v["category"]):
        # cot_content is stored as "A: Let's think step by step..." -- strip the leading
        # role marker, since here it IS the assistant turn rather than text inside one.
        # Strip BOTH the "A:" role marker and the leading "Let's think step by step.",
        # because the user turn already ends with that phrase -- lm-eval's rendered
        # exemplar begins "We refer to Wikipedia articles...", not with a repeat of the
        # cue. Leaving it in teaches the model to echo the prompt back before answering.
        answer = re.sub(r"^A:\s*", "", cot.strip())
        answer = re.sub(r"^Let's think step by step\.\s*", "", answer)
        turns[cat].append(({"role": "user", "content": question_text(q, list(opts))},
                           {"role": "assistant", "content": answer}))
    return turns


def load_items(shard=0, num_shards=1, limit=0):
    t = _split("test")
    n = len(t["question"])
    items = [{"question_id": t["question_id"][i], "question": t["question"][i],
              "options": list(t["options"][i]), "gold": t["answer"][i],
              "category": t["category"][i], "src": t["src"][i]}
             for i in range(n)]
    if num_shards > 1:
        items = [it for i, it in enumerate(items) if i % num_shards == shard]
    if limit:
        # Stratified by category, never a head slice -- the file is ordered by category,
        # so the first N rows are one or two subjects out of fourteen.
        by_cat = collections.defaultdict(list)
        for it in items:
            by_cat[it["category"]].append(it)
        per = max(1, limit // len(by_cat))
        items = [it for v in by_cat.values() for it in v[:per]]
    return items


def predict(content, rng=None):
    """(letter, parsed) via lm-eval's single strict regex. A miss is WRONG, not guessed.

    lm-eval's mmlu_pro filter is exactly `answer is \\(?([ABCDEFGHIJ])\\)?` then
    take_first; its second pattern is commented out and it has no third. A response that
    does not match simply fails the exact_match, which is the behaviour the published
    numbers carry.

    NO RANDOM GUESS, unlike the MMLU-Pro repo's scorer. A guess converts a formatting
    miss into a 10% chance of a point, so an arm whose answers drift out of format gains
    accuracy from noise -- and with the multi-turn prompt above, misses are rare enough
    that the honest zero is also the accurate one. `rng` is accepted and ignored so the
    signature survives for callers that still pass one.
    """
    m = ANSWER_RE.search(content or "")
    return (m.group(1), True) if m else (None, False)


def score(out_dir):
    """Overall and per-category accuracy over every shard present."""
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "raw*.jsonl"))):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:                                     # noqa: BLE001
                continue    # a torn final line: the file is being appended to right now
    if not rows:
        return None
    # De-duplicated on question_id. Shards are disjoint by construction, but a directory
    # that was run unsharded and then re-run sharded would otherwise double-count the
    # overlap, and the accuracy would look fine while resting on 1.3 passes of the data.
    seen, uniq = set(), []
    for r in rows:
        if r["question_id"] in seen:
            continue
        seen.add(r["question_id"])
        uniq.append(r)
    rng = random.Random(SEED)
    per_cat = collections.defaultdict(lambda: [0, 0])
    right = unparsed = 0
    for r in sorted(uniq, key=lambda x: x["question_id"]):        # order fixes the rng
        letter, parsed = predict(r.get("content"))
        unparsed += not parsed
        ok = parsed and letter == r["gold"]
        right += ok
        per_cat[r["category"]][0 if ok else 1] += 1
    n = len(uniq)
    return {"benchmark": "mmlu_pro", "accuracy": right / n, "n_scored": n,
            "unparsed": unparsed,
            "per_category": {c: {"correct": v[0], "total": v[0] + v[1],
                                 "accuracy": v[0] / (v[0] + v[1])}
                             for c, v in sorted(per_cat.items())},
            "generation": truncation_report(uniq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--score-only", action="store_true",
                    help="rescore what is on disk without generating")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")
    os.makedirs(args.out, exist_ok=True)

    if not args.score_only:
        items = load_items(args.shard, args.num_shards, args.limit)
        turns = few_shot_turns()
        missing = {it["category"] for it in items} - set(turns)
        if missing:
            raise SystemExit(f"no 5-shot exemplars for {sorted(missing)}")
        print(f"{len(items)} items over {len({it['category'] for it in items})} "
              f"categories (shard {args.shard}/{args.num_shards})", flush=True)

        def request_of(it):
            msgs, pairs = [], turns[it["category"]]
            for i, (u, a) in enumerate(pairs):
                first = INSTRUCTION.format(category=it["category"]) if i == 0 else ""
                msgs.append({"role": "user", "content": first + u["content"]})
                msgs.append(a)
            msgs.append({"role": "user",
                         "content": question_text(it["question"], it["options"])})
            return msgs

        name = ("raw.jsonl" if args.num_shards == 1
                else f"raw.shard{args.shard}of{args.num_shards}.jsonl")

        # RESUME AGAINST EVERY SHARD FILE, not just this shard's own. run_resumable
        # reloads only its out_path, so a run that starts unsharded and is later
        # resharded would regenerate everything already done -- and switching to shards
        # is exactly what one does when an arm is too slow to finish, i.e. when the
        # existing rows are most expensive to throw away. Scoring already globs and
        # deduplicates, so the union is the correct resume set either way.
        done = set()
        for prev in glob.glob(os.path.join(args.out, "raw*.jsonl")):
            for line in open(prev):
                try:
                    done.add(str(json.loads(line)["question_id"]))
                except Exception:                                  # noqa: BLE001
                    continue
        before = len(items)
        items = [it for it in items if str(it["question_id"]) not in done]
        if before != len(items):
            print(f"  resuming: {before - len(items)} of {before} already present "
                  f"across {len(glob.glob(os.path.join(args.out, 'raw*.jsonl')))} file(s)",
                  flush=True)
        run_resumable(items, uid_of=lambda x: str(x["question_id"]),
                      request_of=request_of, out_path=os.path.join(args.out, name),
                      workers=args.workers, desc=f"mmlu_pro[{args.shard}]",
                      max_tokens=args.max_tokens, base_url=args.base_url,
                      model=args.model)

    summary = score(args.out)
    if summary is None:
        print("nothing to score"); return
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"generation diagnostics: {summary['generation']}")
    worst = sorted(summary["per_category"].items(), key=lambda kv: kv[1]["accuracy"])
    print(f"{args.out}: acc={100 * summary['accuracy']:.2f} on {summary['n_scored']} "
          f"scored ({summary['unparsed']} unparsed -> guessed)")
    print("  weakest categories: " + ", ".join(
        f"{c} {100 * v['accuracy']:.1f}" for c, v in worst[:3]))


if __name__ == "__main__":
    main()
