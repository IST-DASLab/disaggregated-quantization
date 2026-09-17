"""Prove, from the engine logs, that a disaggregated run really was disaggregated.

    python3 verify_disagg.py <log-dir> [--out results/gpqa/disagg]

THE FAILURE THIS EXISTS FOR
---------------------------
If the KV never crosses, the decode engine simply recomputes the prompt with its own
weights. Every request succeeds, every completion is fluent, and the number reported is
homogeneous-decode wearing a disaggregated label. Nothing in the logs says "I fell back".

For the quantized pairs the giveaway was behavioural: a W4A4
prefill feeding a W4A16 decode produces text that neither homogeneous stack produces, so
A != B and A != C settles it. That test is USELESS here. Both engines hold identical
BF16 weights, so local recompute and a successful transfer are distributionally
identical, and no amount of staring at completions can separate them.

What separates them is where the prefill WORK happened, which the engines report
directly:

    decode : Avg prompt throughput: 0.0 tokens/s ... External prefix cache hit rate: 100.0%
    prefill: Avg prompt throughput: 75.1 tokens/s ... Running: 0 reqs

The decode engine spending no time on prompts while reporting that all of its prefix came
from an EXTERNAL source is the transfer, observed rather than assumed. A fallback would
show the mirror image: prompt throughput on decode in the thousands and an external hit
rate of 0.

Exit code is non-zero when the evidence is absent, so a mislabelled run cannot end
quietly. Generations are already on disk and are not discarded -- rerunning after fixing
the stack resumes rather than restarts.
"""
import argparse
import json
import os
import re
import statistics
import sys

# vLLM prints one of these per logging interval per engine.
THROUGHPUT_RE = re.compile(r"Avg prompt throughput:\s*([\d.]+) tokens/s")
EXTERNAL_RE = re.compile(r"External prefix cache hit rate:\s*([\d.]+)%")


def parse(path):
    if not os.path.exists(path):
        return None
    prompt_tp, external = [], []
    with open(path, errors="replace") as f:
        for line in f:
            m = THROUGHPUT_RE.search(line)
            if m:
                prompt_tp.append(float(m.group(1)))
            m = EXTERNAL_RE.search(line)
            if m:
                external.append(float(m.group(1)))
    if not prompt_tp:
        return None
    # Idle intervals dominate both tails, so summarise over the ACTIVE ones: a mean
    # taken across a mostly-idle log says nothing about what happened under load.
    active = [t for t in prompt_tp if t > 0]
    return {
        "intervals": len(prompt_tp),
        "prompt_tp_mean_all": statistics.mean(prompt_tp),
        "prompt_tp_mean_active": statistics.mean(active) if active else 0.0,
        "prompt_tp_max": max(prompt_tp),
        "active_intervals": len(active),
        "external_hit_max": max(external) if external else None,
        "external_hit_mean": statistics.mean([e for e in external if e > 0])
                             if any(e > 0 for e in external) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", default=None, help="results dir to write the verdict into")
    args = ap.parse_args()

    prefill = parse(os.path.join(args.log_dir, "prefill.log"))
    decode = parse(os.path.join(args.log_dir, "decode.log"))
    if prefill is None or decode is None:
        print(f"ERROR: no engine logs with throughput lines under {args.log_dir}",
              file=sys.stderr)
        return 2

    # Both conditions, not either. A decode engine can show a high external hit rate
    # while still recomputing part of the prompt, and it can show near-zero prompt
    # throughput simply because no requests arrived.
    kv_crossed = (decode["external_hit_max"] or 0) >= 99.0
    decode_idle_on_prompts = decode["prompt_tp_mean_active"] < prefill["prompt_tp_mean_active"]
    prefill_worked = prefill["prompt_tp_max"] > 0

    verdict = {
        "prefill": prefill, "decode": decode,
        "kv_crossed": kv_crossed,
        "prefill_did_the_prefill": prefill_worked and decode_idle_on_prompts,
        "pass": bool(kv_crossed and prefill_worked and decode_idle_on_prompts),
    }

    print(f"prefill: prompt {prefill['prompt_tp_mean_active']:.1f} tok/s over "
          f"{prefill['active_intervals']} active intervals")
    print(f"decode : prompt {decode['prompt_tp_mean_active']:.1f} tok/s, "
          f"external prefix hit max {decode['external_hit_max']}%")
    print(f"VERDICT: {'PASS' if verdict['pass'] else 'FAIL'} "
          f"(kv_crossed={kv_crossed}, prefill_did_the_prefill={verdict['prefill_did_the_prefill']})")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "disagg_verification.json"), "w") as f:
            json.dump(verdict, f, indent=2)

    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
