"""Greedy-complete a few fixed prompts through the Nixl 1P1D proxy and record them.

Deliberately tiny and dependency-light: it needs only `requests` and `transformers`,
both already in the container, so the driver does NOT need the lm_eval overlay on its
PYTHONPATH (that overlay ships huggingface-hub 1.24.0, which shadows the container's
and makes `vllm serve` refuse to start).

The prompts and the greedy settings match eval_disagg.py's probe() so outputs from
the two stacks are directly comparable.

Also dumps the proxy's /healthcheck counters. `kv_ok` vs `kv_missing` there is the
cheapest evidence of whether kv_transfer_params were actually produced by the prefill
engine; it does not by itself prove the blocks moved (see verify_kv_transfer.py for
the behavioural test that does).

    python nixl_probe.py --port 8595 --tokenizer Qwen/Qwen3-0.6B --out probe.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

PROBE_PROMPTS = [
    "Natalia sold clips to 48 friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "What is 17 multiplied by 23? Answer with just the number.",
    "A train travels 60 miles in 1.5 hours. What is its average speed in mph?",
    "If a shirt costs $25 and is discounted by 20%, what is the sale price?",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    base = f"http://127.0.0.1:{args.port}"
    rows, failures = [], 0
    for prompt in (PROBE_PROMPTS[:args.limit] if args.limit else PROBE_PROMPTS):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        t0 = time.time()
        try:
            r = requests.post(f"{base}/v1/completions",
                              json={"model": "model", "prompt": text,
                                    "max_tokens": args.max_tokens,
                                    "temperature": 0, "seed": 0},
                              timeout=args.timeout)
        except Exception as e:                              # noqa: BLE001
            failures += 1
            print(f"[probe] REQUEST FAILED after {time.time()-t0:.1f}s: "
                  f"{type(e).__name__}: {e}", flush=True)
            rows.append({"prompt": prompt, "error": f"{type(e).__name__}: {e}"})
            continue
        dt = time.time() - t0
        if r.status_code != 200:
            failures += 1
            print(f"[probe] HTTP {r.status_code} after {dt:.1f}s: {r.text[:500]}",
                  flush=True)
            rows.append({"prompt": prompt, "error": f"HTTP {r.status_code}: {r.text[:500]}"})
            continue
        completion = r.json()["choices"][0]["text"]
        rows.append({"prompt": prompt, "completion": completion})
        print(f"[probe] ok in {dt:.1f}s\n  Q: {prompt[:70]}\n  A: {completion!r}",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[probe] wrote {out}", flush=True)

    try:
        hc = requests.get(f"{base}/healthcheck", timeout=30).json()
        print(f"[probe] proxy healthcheck: {json.dumps(hc)}", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[probe] healthcheck failed: {type(e).__name__}: {e}", flush=True)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
