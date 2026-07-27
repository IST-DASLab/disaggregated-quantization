"""
Evaluate a QAD-trained quantized model on lm-eval benchmarks.

Loads the base model, applies the same quantizer used during training,
then loads master weights from the weights-only checkpoint for the given
step. Runs evaluation via lm_eval and saves results to JSON.

Usage:
    python eval_transformers.py \
        --model  Qwen/Qwen3-4B \
        --quantizer ste3bit \
        --iter   50 \
        --tasks  gsm8k math_500 aime_2025
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Locate the qad/ directory so we can reuse its quantization primitives
_QAD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_QAD_DIR))
sys.path.insert(0, str(_QAD_DIR.parent / "third_party" / "Liger-Kernel" / "src"))

from quantizers import REGISTRY, build_quantizer_params


def resolve_checkpoint(ckpt_dir: Path, ckpt_tag: str, step: int) -> Path:
    """Return the HF checkpoint directory for this step.

    New format is a self-contained HF model dir (config.json + model.safetensors)
    that loads directly with from_pretrained.
    """
    hf_dir = ckpt_dir / ckpt_tag / "weights" / f"step_{step:07d}"
    if (hf_dir / "model.safetensors").exists() and (hf_dir / "config.json").exists():
        return hf_dir
    raise FileNotFoundError(
        f"No HF checkpoint found for step {step} at {hf_dir}\n"
        f"  (expected {hf_dir}/model.safetensors + config.json)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QAD checkpoint evaluator")
    parser.add_argument("--model",            required=True, help="HuggingFace model id")
    parser.add_argument("--quantizer",        required=True, choices=list(REGISTRY))
    parser.add_argument("--quantizer-params", default="",   help="JSON overrides for quantizer")
    parser.add_argument("--iter",             type=int, default=None, help="checkpoint step (required unless --unquantized)")
    parser.add_argument("--run-name",         default=None,
                        help="run name prefix used during training (default: qad-<model_short>)")
    parser.add_argument("--ckpt-dir",         default=str(_QAD_DIR / "checkpoints"))
    parser.add_argument("--unquantized",      action="store_true",
                        help="evaluate the base BF16 model without any quantizer or checkpoint")
    parser.add_argument("--tasks",            nargs="+", default=["gsm8k", "math_500", "aime_2025"])
    parser.add_argument("--num-fewshot",      type=int, default=None)
    parser.add_argument("--no-think",         action="store_true",
                        help="disable Qwen3 thinking mode (sets generation_config.enable_thinking=False)")
    parser.add_argument("--batch-size",       type=int, default=16)
    parser.add_argument("--log-samples",      action="store_true",
                        help="dump per-sample generations to JSONL for inspection")
    parser.add_argument("--limit",            type=int, default=None,
                        help="limit eval to N docs per task (diagnostic only)")
    parser.add_argument("--output-dir",       default=str(_QAD_DIR / "eval_results"))
    args = parser.parse_args()

    if not args.unquantized and args.iter is None:
        parser.error("--iter is required unless --unquantized is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.unquantized:
        # Baseline: full-precision teacher model, no checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map=device, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model.eval()
        ckpt_tag = f"{args.model.replace('/', '-')}-unquantized"
        step_key = 0
        print(f"Model: {args.model}  [unquantized BF16 baseline]", flush=True)
    else:
        # Reconstruct ckpt_tag (matches qad.py) for path resolution / result naming.
        _, quant_hash = build_quantizer_params(args.quantizer, args.quantizer_params)
        model_short = args.model.replace("/", "-")
        run_name    = args.run_name or f"qad-{model_short}"
        ckpt_tag    = f"{run_name}-{args.quantizer}-{quant_hash}"

        ckpt_dir = resolve_checkpoint(Path(args.ckpt_dir), ckpt_tag, args.iter)
        print(f"Loading HF checkpoint: {ckpt_dir}", flush=True)

        # Checkpoints are saved as standard HF models (dequantized weights in `weight`),
        # so this is a single fast sharded load straight to GPU — no base-model read,
        # no quantizer wrapping, no manual load_state_dict.
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        ).to(device)
        model.eval()
        step_key = args.iter
        print(f"Model: {args.model}  quantizer={args.quantizer}  step={args.iter}", flush=True)
    if args.no_think:
        # Qwen3 suppresses thinking via the chat template kwarg, not generation_config.
        # The template checks: {%- if enable_thinking is defined and enable_thinking is false %}
        # Patch the tokenizer so lm_eval's apply_chat_template call injects it automatically.
        _orig_act = tokenizer.apply_chat_template
        def _no_think_act(conversation, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_act(conversation, **kwargs)
        tokenizer.apply_chat_template = _no_think_act
        print("Thinking suppressed via tokenizer.apply_chat_template patch.", flush=True)

    print(f"Tasks: {args.tasks}  batch_size={args.batch_size}", flush=True)

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        dtype="bfloat16",
        apply_chat_template=True,
        max_length=8192,  # total context (prompt + generation) cap
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        # Cap generation so tasks with large defaults (e.g. aime25=32768) don't
        # exceed max_length=8192.
        gen_kwargs="max_gen_toks=4096",
        log_samples=args.log_samples,
        limit=args.limit,
    )

    # Print summary
    for task, metrics in results["results"].items():
        key_metrics = {k: v for k, v in metrics.items() if not k.endswith("_stderr")}
        print(f"  {task}: {key_metrics}", flush=True)

    out_dir = Path(args.output_dir) / ckpt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump per-sample generations (prompt, resp, target, filtered) to JSONL so we can
    # inspect actual model outputs — is it truncating, looping, or just wrong?
    samples = results.pop("samples", None)
    if samples:
        for task, recs in samples.items():
            sp = out_dir / f"step_{step_key:07d}_samples_{task}.jsonl"
            with open(sp, "w") as f:
                for r in recs:
                    slim = {
                        "doc_id": r.get("doc_id"),
                        "target": r.get("target"),
                        "resps": r.get("resps"),
                        "filtered_resps": r.get("filtered_resps"),
                        "exact_match": r.get("exact_match"),
                        "arguments": r.get("arguments"),  # includes the prompt
                    }
                    f.write(json.dumps(slim, default=str) + "\n")
            print(f"Samples → {sp}  ({len(recs)} docs)", flush=True)

    # Save full results — merge with existing file so separate task runs accumulate.
    out_path = out_dir / f"step_{step_key:07d}.json"
    if out_path.exists():
        merged = json.loads(out_path.read_text())
        for key in ("results", "configs", "versions", "n-shot"):
            if key in results:
                if key in merged and isinstance(merged[key], dict):
                    merged[key].update(results[key])
                else:
                    merged[key] = results[key]
        results = merged
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
