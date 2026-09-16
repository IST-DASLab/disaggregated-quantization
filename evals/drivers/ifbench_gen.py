"""IFBench response generation against a local vLLM endpoint.

    python3 ifbench_gen.py --out results/ifbench/bf16

Writes `<out>/responses.jsonl` in exactly the schema IFBench's own `run_eval.py`
expects -- `{"prompt": ..., "response": ...}` keyed by prompt text -- so scoring is the
upstream scorer, unmodified:

    python3 -m run_eval --input_data=data/IFBench_test.jsonl \
        --input_response_data=<out>/responses.jsonl --output_dir=<out>

WHY NOT generate_responses.py
-----------------------------
1. It always sends `temperature` (default 0.6) and a fixed `seed`, which overrides the
   model's own generation_config on the server. The published number comes from the
   vendor's settings, so the client must send neither.

2. On any request exception it writes an EMPTY response for that item and keeps going.
   Since its resume keys on prompts already present, a transient 500 is then permanent:
   the item is "done" and scores as a total instruction-following failure. IFBench is
   only 299 items, so one blip is worth 0.33 points on every constraint that item
   carries. Here a failure is left unrecorded and retried on the next run.

3. Its incremental save rewrites the entire file every 10 results from an in-memory
   list, so a kill during that write truncates the file. Append + fsync per item does
   not have that window.

SCORING TARGET is `content` only. IFBench constraints are things like "exactly three
sentences" or "no word longer than seven letters", which apply to the answer the user
sees -- scoring the chain-of-thought instead would fail nearly every constraint while
saying nothing about the model. The reasoning text is still written to a side file so
a truncation or parser question can be answered without regenerating.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_done, run_resumable, truncation_report  # noqa: E402

IFBENCH_DIR = os.environ.get(
    "IFBENCH_DIR",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/IFBench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-file",
                    default=os.path.join(IFBENCH_DIR, "data/IFBench_test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.input_file)]
    if args.limit:
        items = items[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw.jsonl")

    run_resumable(items,
                  uid_of=lambda x: x["key"],
                  request_of=lambda x: [{"role": "user", "content": x["prompt"]}],
                  out_path=raw_path, workers=args.workers, desc="ifbench",
                  max_tokens=args.max_tokens, base_url=args.base_url)

    rows = [json.loads(l) for l in open(raw_path)]
    print(f"generation diagnostics: {truncation_report(rows)}")
    missing = len(items) - len(load_done(raw_path))
    if missing:
        print(f"WARNING: {missing} items still missing -- rerun before scoring")

    # The scorer's file: prompt/response only, nothing else.
    resp_path = os.path.join(args.out, "responses.jsonl")
    with open(resp_path, "w") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["content"]},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {resp_path}")
    print("score with:\n"
          f"  cd {IFBENCH_DIR} && python3 -m run_eval "
          f"--input_data={args.input_file} "
          f"--input_response_data={resp_path} --output_dir={args.out}")


if __name__ == "__main__":
    main()
