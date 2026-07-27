"""
Evaluate a QAD checkpoint (or the base model) with lm-eval using the vLLM backend.

Since QAD checkpoints are now saved as plain HF models (dequantized weights in
`weight`), vLLM can serve them directly. vLLM uses continuous batching — each
request finishes independently, so a few long generations no longer stall the
whole batch (the failure mode of the HF .generate() backend), and throughput is
much higher.

Usage:
    python eval_vllm.py --model Qwen/Qwen3-4B --quantizer ste3bit --iter 200 \
        --no-think --tasks gsm8k minerva_math500 aime25
    python eval_vllm.py --model Qwen/Qwen3-4B --unquantized --no-think --tasks gsm8k
"""

import os
# vLLM forks its engine-core subprocess; if CUDA is already initialized in the
# parent (importing qad / lm_eval touches torch CUDA), fork fails with "Cannot
# re-initialize CUDA in forked subprocess". Run the v1 engine in-process (no fork)
# and force spawn for any workers, before any CUDA-touching import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import sys
from pathlib import Path

_QAD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_QAD_DIR))

from quantizers import REGISTRY, build_quantizer_params


def resolve_checkpoint(ckpt_dir: Path, ckpt_tag: str, step: int) -> Path:
    hf_dir = ckpt_dir / ckpt_tag / "weights" / f"step_{step:07d}"
    if (hf_dir / "model.safetensors").exists() and (hf_dir / "config.json").exists():
        return hf_dir
    raise FileNotFoundError(f"No HF checkpoint at {hf_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="QAD vLLM evaluator")
    p.add_argument("--model", required=True, help="HF model id (also the tokenizer source)")
    p.add_argument("--quantizer", default="ste3bit", choices=list(REGISTRY))
    p.add_argument("--quantizer-params", default="")
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--unquantized", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ckpt-dir", default=str(_QAD_DIR / "checkpoints"))
    p.add_argument("--tasks", nargs="+", default=["gsm8k", "minerva_math500", "aime25"])
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-gen-toks", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--log-samples", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default=str(_QAD_DIR / "eval_results_vllm"))
    args = p.parse_args()

    if not args.unquantized and args.iter is None:
        p.error("--iter is required unless --unquantized is set")

    # Resolve model path + result tag. Tags match eval_transformers.py; the vLLM
    # results live under a separate output root (default eval_results_vllm/), so
    # the two backends never collide.
    if args.unquantized:
        model_path = args.model
        ckpt_tag = f"{args.model.replace('/', '-')}-unquantized"
        step_key = 0
    else:
        _, quant_hash = build_quantizer_params(args.quantizer, args.quantizer_params)
        run_name = args.run_name or f"qad-{args.model.replace('/', '-')}"
        ckpt_tag = f"{run_name}-{args.quantizer}-{quant_hash}"
        model_path = str(resolve_checkpoint(Path(args.ckpt_dir), ckpt_tag, args.iter))
        step_key = args.iter
    print(f"vLLM model: {model_path}  tasks={args.tasks}", flush=True)

    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM

    # NVFP4 checkpoints carry a compressed-tensors `quantization_config` in their
    # config.json (quant_method="compressed-tensors", nvfp4-pack-quantized), so vLLM
    # auto-detects the format and serves true W4A4 (CompressedTensorsW4A4Fp4: packed
    # FP4 weights + FP8 block scales + dynamic FP4 activations) with no extra flag.
    lm = VLLM(
        pretrained=model_path,
        tokenizer=args.model,              # tokenizer always from the base model id
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
    )

    if args.no_think:
        # Qwen3 suppresses thinking via the chat-template kwarg; patch the tokenizer
        # so lm-eval's apply_chat_template injects enable_thinking=False.
        tok = lm.tokenizer
        _orig = tok.apply_chat_template
        def _patched(conversation, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig(conversation, **kwargs)
        tok.apply_chat_template = _patched
        print("Thinking suppressed via tokenizer patch.", flush=True)

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        apply_chat_template=True,
        gen_kwargs=f"max_gen_toks={args.max_gen_toks}",
        log_samples=args.log_samples,
        limit=args.limit,
    )

    for task, metrics in results["results"].items():
        km = {k: v for k, v in metrics.items() if not k.endswith("_stderr")}
        print(f"  {task}: {km}", flush=True)

    out_dir = Path(args.output_dir) / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = results.pop("samples", None)
    if samples:
        for task, recs in samples.items():
            sp = out_dir / f"step_{step_key:07d}_samples_{task}.jsonl"
            with open(sp, "w") as f:
                for r in recs:
                    f.write(json.dumps({
                        "doc_id": r.get("doc_id"),
                        "target": r.get("target"),
                        "resps": r.get("resps"),
                        "filtered_resps": r.get("filtered_resps"),
                        "exact_match": r.get("exact_match"),
                        "arguments": r.get("arguments"),
                    }, default=str) + "\n")
            print(f"Samples → {sp}  ({len(recs)} docs)", flush=True)

    out_path = out_dir / f"step_{step_key:07d}.json"
    if out_path.exists():
        merged = json.loads(out_path.read_text())
        for key in ("results", "configs", "versions", "n-shot"):
            if key in results:
                if isinstance(merged.get(key), dict):
                    merged[key].update(results[key])
                else:
                    merged[key] = results[key]
        results = merged
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
