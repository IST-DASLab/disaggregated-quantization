"""Single-user generation speed in vLLM: bf16 vs lloyd43, same weights.

    python vllm_serve.py --model Qwen/Qwen3-8B
    python vllm_serve.py --model Qwen/Qwen3-0.6B --quant lloyd43

PREFILL AND DECODE ARE MEASURED SEPARATELY, which matters because they are different
problems and only one of them is what this kernel is for. Decode at batch 1 is a GEMV and
is what the kernel accelerates; prefill is a GEMM over the whole prompt, takes the
dequantize-then-cuBLAS fallback, and would otherwise contaminate the headline number --
a slow prefill amortized over N output tokens looks exactly like slow decode.

Separation is by the two-point slope method: generate n1 tokens and n2 tokens from the
same prompt, then

    seconds per output token = (t2 - t1) / (n2 - n1)
    time to first token      = t1 - n1 * (that)

which needs no vLLM-internal metrics API and cancels prefill out of the decode figure
exactly. Prefix caching is disabled so the prompt is genuinely re-prefilled each time;
with it on, the second run would hit cache and TTFT would read as ~0.

Both arms load the SAME bf16 checkpoint; the lloyd43 arm packs it at load time (see
vllm_plugin), so the comparison isolates the kernel rather than comparing two different
checkpoints. Each arm runs in its own subprocess, because vLLM does not tear an engine
down cleanly enough to build a second one in the same interpreter.
"""

import argparse
import json
import os
from pathlib import Path
import os
import subprocess
import sys
import time

CHILD = "--_child"

# Arms. "none" and "lloyd43" run the base bf16 checkpoint -- lloyd43 packs it at load, so
# both see identical weights. The NVFP4 arms are vLLM-native compressed-tensors
# ("nvfp4-pack-quantized"): W4 group-16, with A4 for nvfp4 and 16-bit activations for
# nvfp4a16. vLLM reads the scheme from the checkpoint's config, so no --quantization flag.
#
# THOSE TWO ARE DIFFERENT WEIGHTS -- quantized by a third party from the same base model,
# not by us. For a LATENCY comparison that is fine (same architecture, same shapes, same
# kernels a served model would use) but it is not an accuracy comparison, and the arms are
# not bit-comparable the way "none" and "lloyd43" are.
# Converted locally by make_nvfp4.py; vLLM detects compressed-tensors from config.json.
NVFP4_DIR = os.environ.get("NVFP4_DIR", str(Path.home() / ".nvfp4_checkpoints"))
NVFP4_SCHEME = {"nvfp4": "NVFP4", "nvfp4a16": "NVFP4A16"}


# The *aq variants are the same weights and the same kernel as their plain counterpart,
# plus one discarded fp4 activation quantization per linear -- what a LUT format costs when
# it is NOT format-disaggregated. See lloyd43.cuda_gemv.act_quant_barrier.
LLOYD_ARMS = ("lloyd43", "lloyd21", "lloyd43aq", "lloyd21aq")


def _nvfp4_path(model: str, quant: str) -> str | None:
    """The converted checkpoint for this (model, scheme), or None if this is not an NVFP4 arm."""
    scheme = NVFP4_SCHEME.get(quant)
    if scheme is None:
        return None
    p = Path(NVFP4_DIR) / f"{model.split('/')[-1]}-{scheme}"
    if not p.exists():
        raise SystemExit(f"missing {p}; run: python make_nvfp4.py --model {model} "
                         f"--scheme {scheme}")
    return str(p)


def run_child(args):
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    if args.quant in LLOYD_ARMS:
        # One import registers all four; the flag below picks grid and whether the arm is
        # charged for activation quantization it cannot use (the *aq arms).
        import lloyd43.vllm_plugin  # noqa: F401


    # NVFP4 arms load a checkpoint we converted with make_nvfp4.py, through vLLM's OWN
    # compressed-tensors path -- no --quantization flag, the format is read from
    # config.json. Converting here (rather than pulling cortecs/*) means every arm runs the
    # same bf16 weights with the same ignore list, and Gemma is measurable at all.
    model_id = _nvfp4_path(args.model, args.quant) or args.model
    kw = dict(model=model_id, dtype="bfloat16", max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_util, enforce_eager=args.enforce_eager,
              disable_log_stats=True,
              # Off on purpose: with it on the repeat runs hit the prefix cache and the
              # prefill we are trying to measure never happens.
              enable_prefix_caching=False)
    if args.quant in LLOYD_ARMS:
        kw["quantization"] = args.quant
    if args.linear_backend != "auto":
        # vLLM picks NVFP4 GEMM kernels by a fixed priority list, not by measurement.
        # At DECODE its defaults are right -- batch 1 is a bandwidth-bound GEMV, so kernel
        # quality at large M does not transfer, and measured here neither alternative
        # helped: flashinfer_cudnn 2.84x -> 2.78x on NVFP4, humming 2.90x -> 2.76x on
        # NVFP4A16. The flag exists because that is worth being able to re-check on
        # another box rather than assumed, and because the same choice matters a great
        # deal at prefill, where the defaults are NOT right.
        kw["kernel_config"] = {"linear_backend": args.linear_backend}


    t_load = time.perf_counter()
    llm = LLM(**kw)
    load_s = time.perf_counter() - t_load

    prompt = "The history of computing hardware begins with " + ("the abacus. " * 40)

    def timed(n_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=n_tokens, ignore_eos=True)
        llm.generate([prompt], sp)                       # warm up this shape
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            out = llm.generate([prompt], sp)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return ts[len(ts) // 2], len(out[0].outputs[0].token_ids), len(out[0].prompt_token_ids)

    n1, n2 = args.short_len, args.out_len
    t1, got1, n_in = timed(n1)
    t2, got2, _ = timed(n2)

    per_tok = (t2 - t1) / max(got2 - got1, 1)
    ttft = t1 - got1 * per_tok

    print("RESULT " + json.dumps(dict(
        quant=args.quant, model=model_id, load_s=round(load_s, 1), in_tokens=n_in,
        short=dict(n=got1, s=round(t1, 4)), long=dict(n=got2, s=round(t2, 4)),
        decode_tok_s=round(1.0 / per_tok, 2), itl_ms=round(per_tok * 1e3, 3),
        ttft_ms=round(ttft * 1e3, 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out-len", type=int, default=128, help="long point of the slope")
    ap.add_argument("--short-len", type=int, default=8, help="short point of the slope")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip CUDA graph capture (shows what graphs are worth)")
    ap.add_argument("--quant", nargs="+", default=["none", "lloyd43"],
                    help="none | lloyd43 | lloyd21 | lloyd43aq | lloyd21aq | nvfp4 | "
                         "nvfp4a16 (the *aq arms are the LUT formats charged for the "
                         "activation quantization a non-disaggregated deployment pays)")
    ap.add_argument("--linear-backend", default="auto",
                    help="vLLM NVFP4 GEMM backend: auto | humming | flashinfer_cudnn | "
                         "marlin | cutlass | flashinfer_cutlass | flashinfer_b12x")
    ap.add_argument("--out", default="benchmarks/vllm_decode.csv",
                    help="CSV to merge results into, keyed on (model, quant)")
    ap.add_argument(CHILD, dest="_child", default=None)
    args = ap.parse_args()

    if args._child:
        args.quant = args._child
        return run_child(args)

    # bf16 FIRST, always: every other arm is reported as a ratio to it, so until it has
    # run the CSV holds numbers with no baseline and the plot can only draw a WIP card.
    # Reordered here rather than trusting the caller's argument order.
    arms = sorted(args.quant, key=lambda q: q != "none")
    if arms != list(args.quant):
        print(f"(running {' '.join(arms)} -- baseline first)")

    results = {}
    for q in arms:
        cmd = [sys.executable, __file__, "--model", args.model,
               "--out-len", str(args.out_len), "--short-len", str(args.short_len),
               "--max-len", str(args.max_len),
               "--reps", str(args.reps), "--gpu-util", str(args.gpu_util),
               "--linear-backend", args.linear_backend, CHILD, q]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        print(f"--- {args.model}  quant={q} ---", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            # One arm failing must not discard the arms that already worked. This used to
            # raise, which cost gemma-3-4b-it its none/lloyd43/lloyd21 results because the
            # nvfp4 arm died after them and the CSV is only written at the end.
            tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-25:])
            print(f"    !! child failed for quant={q} (rc={p.returncode}); continuing\n"
                  f"{tail}", flush=True)
            continue
        results[q] = json.loads(line[len("RESULT "):])
        print("   ", results[q], flush=True)
        # Flush after every arm, so a later crash cannot lose an earlier measurement.
        if args.out:
            _merge_csv(args.out, args.model, {q: results[q]})

    print(f"\n{'quant':<10}{'decode tok/s':>14}{'ITL ms':>9}{'TTFT ms':>10}{'load s':>9}"
          f"  {'vs bf16':>8}")
    if not results:
        raise SystemExit(f"every arm failed for {args.model}")
    base = results.get("none", {}).get("decode_tok_s")
    for q, r in results.items():
        rel = f"{r['decode_tok_s']/base:>7.2f}x" if base else f"{'-':>8}"
        print(f"{q:<10}{r['decode_tok_s']:>14.2f}{r['itl_ms']:>9.3f}"
              f"{r['ttft_ms']:>10.1f}{r['load_s']:>9.1f}  {rel}")
    if args.out and results:
        _merge_csv(args.out, args.model, results)

    if "none" in results and "lloyd43" in results:
        b, l = results["none"], results["lloyd43"]
        print(f"\nDECODE  {b['decode_tok_s']:.1f} -> {l['decode_tok_s']:.1f} tok/s"
              f"   {l['decode_tok_s']/b['decode_tok_s']:.2f}x   <-- what the kernel does")
        print(f"PREFILL {b['ttft_ms']:.0f} -> {l['ttft_ms']:.0f} ms TTFT"
              f"   {b['ttft_ms']/l['ttft_ms']:.2f}x   (dequant+cuBLAS fallback, not the GEMV)")


def _merge_csv(path: str, model: str, results: dict) -> None:
    """Merge these arms into the CSV, keyed on (model, quant).

    Merged rather than overwritten so a run covering one model, or one extra arm, updates
    exactly those rows -- the sweep is hours long and gets interrupted.
    """
    import csv as _csv
    import os as _os
    fields = ["model", "quant", "decode_tok_s", "itl_ms", "ttft_ms", "load_s"]
    rows = {}
    if _os.path.exists(path):
        with open(path, newline="") as f:
            for r in _csv.DictReader(f):
                rows[(r["model"], r["quant"])] = r
    for q, r in results.items():
        rows[(model, q)] = dict(model=model, quant=q,
                                decode_tok_s=round(r["decode_tok_s"], 3),
                                itl_ms=round(r["itl_ms"], 3),
                                ttft_ms=round(r["ttft_ms"], 1),
                                load_s=round(r["load_s"], 1))
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
