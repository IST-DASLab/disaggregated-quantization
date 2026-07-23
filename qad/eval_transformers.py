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

from qad import _QUANTIZER_REGISTRY, _build_quantizer_params


def resolve_checkpoint(ckpt_dir: Path, ckpt_tag: str, step: int) -> Path:
    """Return the best available checkpoint path for this step (safetensors preferred)."""
    candidates = [
        ckpt_dir / ckpt_tag / "weights" / f"step_{step:07d}.safetensors",
        ckpt_dir / ckpt_tag / "weights" / f"step_{step:07d}.pt",
        ckpt_dir / ckpt_tag / f"step_{step:07d}" / "ckpt.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found for step {step} under {ckpt_dir / ckpt_tag}\n"
        + "\n".join(f"  tried: {p}" for p in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QAD checkpoint evaluator")
    parser.add_argument("--model",            required=True, help="HuggingFace model id")
    parser.add_argument("--quantizer",        required=True, choices=list(_QUANTIZER_REGISTRY))
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
        # Reconstruct ckpt_tag using the same logic as qad.py
        quant_params, quant_hash = _build_quantizer_params(args.quantizer, args.quantizer_params)
        model_short = args.model.replace("/", "-")
        run_name    = args.run_name or f"qad-{model_short}"
        run_tag     = f"{run_name}-{args.quantizer}"
        ckpt_tag    = f"{run_tag}-{quant_hash}"

        ckpt_path = resolve_checkpoint(Path(args.ckpt_dir), ckpt_tag, args.iter)
        print(f"Loading checkpoint: {ckpt_path}", flush=True)

        # Load base model in BF16 — base weights are overwritten by the checkpoint
        # anyway, so loading as FP32 just doubles IO (16 GB vs 8 GB) for no benefit.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )

        # Apply quantizer to restore the student architecture
        quant_entry = _QUANTIZER_REGISTRY[args.quantizer]
        quant_entry["apply"](model, **quant_params)

        # mmap=True: maps the file near-instantly; pages are read lazily during
        # load_state_dict rather than front-loading all 24 GB at once.
        if str(ckpt_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            # New format: _wq + non-quantized params only (~8 GB, BF16)
            state = load_file(str(ckpt_path), device="cpu")
        else:
            # Legacy format: full state dict (~24 GB, FP32 + BF16 mixed)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
            state = ckpt.get("model", ckpt)
        # strict=False: new checkpoints omit master weights and quant params intentionally
        model.load_state_dict(state, strict=False)
        del state

        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()
        step_key = args.iter
        print(f"Model: {args.model}  quantizer={args.quantizer}  step={args.iter}", flush=True)
    if args.no_think and hasattr(model, "generation_config"):
        model.generation_config.enable_thinking = False
        print("Thinking mode disabled.", flush=True)

    # Compile with dynamic shapes: handles variable sequence lengths during generation.
    model = torch.compile(model, dynamic=True)
    print("Model compiled (dynamic=True).", flush=True)

    print(f"Tasks: {args.tasks}  batch_size={args.batch_size}", flush=True)

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        dtype="bfloat16",   # autocast inside lm-eval for efficiency
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        gen_kwargs="max_gen_toks=4096",
        log_samples=False,
    )

    # Print summary
    for task, metrics in results["results"].items():
        key_metrics = {k: v for k, v in metrics.items() if not k.endswith("_stderr")}
        print(f"  {task}: {key_metrics}", flush=True)

    # Save full results
    out_path = Path(args.output_dir) / ckpt_tag / f"step_{step_key:07d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
