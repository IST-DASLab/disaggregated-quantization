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
import time
from pathlib import Path

_T0 = time.time()


def log(msg: str, since: float | None = None) -> None:
    """Wall-clock stage log. These jobs spend most of their life before generating a
    single token (container start, model load, checkpoint unpack, compile), and a
    silent 20-minute gap is indistinguishable from a hang."""
    el = f"  (+{time.time() - since:.1f}s)" if since is not None else ""
    print(f"[{time.time() - _T0:7.1f}s] {msg}{el}", flush=True)

import torch
from transformers import AutoTokenizer

# Locate the qad/ directory so we can reuse its quantization primitives
_QAD_DIR = Path(__file__).resolve().parent.parent   # eval/ -> qad
sys.path.insert(0, str(_QAD_DIR))
sys.path.insert(0, str(_QAD_DIR.parent / "third_party" / "Liger-Kernel" / "src"))

from export.save import _quant_layers, load_into, to_model_keys
from quantizers import (REGISTRY, QuantizedLinear, build_quantizer_params,
                        uses_compressed_tensors, is_dual, DEFAULT_DUAL_RUNTIME)
# Same loader as training: the class the checkpoint DECLARES, and the text stack
# resolved through a table rather than assumed to be `.model`. AutoModelForCausalLM
# silently mis-loads the multimodal Gemma-3 repos — see training/models.py.
from training.models import load_model, text_stack


def resolve_checkpoint(ckpt_dir: Path, ckpt_tag: str, step: int) -> Path:
    """Return the checkpoint directory for this step.

    Single-format quantizers write the step directory itself; the prefill/decode
    formats write <step>/prefill and <step>/decode beneath it.
    """
    hf_dir = ckpt_dir / ckpt_tag / "weights" / f"step_{step:07d}"
    if (hf_dir / "model.safetensors").exists() and (hf_dir / "config.json").exists():
        return hf_dir
    if (hf_dir / "prefill" / "model.safetensors").exists():
        return hf_dir
    raise FileNotFoundError(
        f"No checkpoint found for step {step} at {hf_dir}\n"
        f"  (expected {hf_dir}/model.safetensors or {hf_dir}/prefill/model.safetensors)"
    )


def build_quantized_model(base_model: str, runtime_quant: str, ckpt: Path, device):
    """Rebuild a quantized model from a checkpoint and load its weights.

    Handles both shapes with one path:

      homogeneous  one checkpoint, one format everywhere (nvfp4 = W4A4 throughout,
                   nvfp4a16 = W4A16 throughout). This is the honest baseline: the
                   model is evaluated in the format it was TRAINED for.
      dual         prefill/ + decode/. The phase follows sequence length, which is
                   exactly how HF generate() behaves — one multi-token pass over the
                   prompt, then one token at a time — so the decode phase attends to
                   a KV cache built by the prefill format, as it would across a
                   disaggregated pair of workers.

    Passing a single-format checkpoint with a dual `runtime_quant` gives the
    dual-inference-on-a-homogeneously-trained-model control, which separates the
    contribution of dual TRAINING from dual INFERENCE.
    """
    from safetensors.torch import load_file

    t = time.time()
    model = load_model(base_model, torch.bfloat16,
                       attn_implementation="flash_attention_2")
    log(f"base model loaded ({base_model})", t)
    t = time.time()
    params, _ = build_quantizer_params(runtime_quant, "")
    # Scoped to the text stack, exactly as training/qad.py does — otherwise the module
    # tree rebuilt here would not be the tree the checkpoint was written from (on a
    # multimodal wrapper this eval would additionally quantize the vision tower and then
    # find no weights for it in the checkpoint). Identical set of linears on a plain
    # CausalLM.
    REGISTRY[runtime_quant]["apply"](text_stack(model).base, **params)
    log(f"quantizer applied ({runtime_quant})", t)

    split_ckpt = (ckpt / "prefill" / "model.safetensors").exists()
    if split_ckpt:
        sources = {"prefill": ckpt / "prefill", "decode": ckpt / "decode"}
    elif is_dual(runtime_quant):
        sources = {"prefill": ckpt, "decode": ckpt}   # one checkpoint drives both
    else:
        sources = {None: ckpt}                        # homogeneous

    # Embeddings / norms / lm_head are trained too, so take them from the checkpoint
    # rather than the base model. Either variant carries the same copy.
    quant_paths = {n for n, m in model.named_modules() if isinstance(m, QuantizedLinear)}
    # A multimodal wrapper exports under text-only names (model.* rather than
    # model.language_model.*), so put them back on the live tree before matching them
    # against module paths. Identity for every plain CausalLM.
    first = to_model_keys(model,
                          load_file(str(next(iter(sources.values())) / "model.safetensors")))
    non_quant = {k: v for k, v in first.items() if k.rpartition(".")[0] not in quant_paths}
    _, unexpected = model.load_state_dict(non_quant, strict=False)
    if unexpected:
        raise RuntimeError(
            f"checkpoint has tensors the model does not expect: {sorted(unexpected)[:5]}")

    n_quant = len(_quant_layers(model))
    for variant, d in sources.items():
        t = time.time()
        n = load_into(model, load_file(str(d / "model.safetensors")), variant=variant)
        # ASSERT, do not merely log. `load_into` returning 0 means the checkpoint's
        # quantized weights were silently not applied and the eval would score the
        # PTQ-of-base-weights model while reporting success -- this project has already
        # shipped that exact failure twice (the dropped quantization_config, and the
        # --full-disag tag collision that re-scored and overwrote the plain baselines).
        # It is currently saved only by the unexpected-keys check above happening to share
        # `_text_prefixes` with the renamer; nothing enforces that coupling.
        if n != n_quant:
            raise RuntimeError(
                f"{d}: load_into applied {n} of {n_quant} quantized layers for variant "
                f"{variant or 'homogeneous'!r} — the checkpoint did not reach the model")
        log(f"  loaded {n} quantized layers for '{variant or 'homogeneous'}'", t)
    note = "" if split_ckpt else ("  (one checkpoint drives both phases)"
                                  if len(sources) > 1 else "  (single format throughout)")
    log(f"  non-quantized tensors restored: {len(non_quant)}{note}")
    t = time.time()
    model = model.to(device).eval()
    log("model moved to device", t)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="QAD checkpoint evaluator")
    parser.add_argument("--model",            required=True, help="HuggingFace model id")
    parser.add_argument("--quantizer",        default=None, choices=list(REGISTRY),
                        help="required unless --unquantized (the BF16 baseline has no "
                             "quantizer and no checkpoint)")
    parser.add_argument("--quantizer-params", default="",   help="JSON overrides for quantizer")
    parser.add_argument("--iter",             type=int, default=None, help="checkpoint step (required unless --unquantized)")
    parser.add_argument("--run-name",         default=None,
                        help="run name prefix used during training (default: qad-<model_short>)")
    parser.add_argument("--ckpt-dir",         default=str(_QAD_DIR / "checkpoints"))
    parser.add_argument("--unquantized",      action="store_true",
                        help="evaluate the base BF16 model without any quantizer or checkpoint")
    parser.add_argument("--dual",             action="store_true",
                        help="run a SINGLE-format checkpoint through dual prefill/decode "
                             "inference (prompt at W4A4, generated tokens at W4A16). Only "
                             "needed for that control — a dual-trained checkpoint is "
                             "detected automatically from its prefill/ + decode/ layout.")
    parser.add_argument("--runtime-quantizer", default=None,
                        help="layer class to run with (default: --quantizer; with --dual "
                             f"on a single-format checkpoint, {DEFAULT_DUAL_RUNTIME})")
    parser.add_argument("--tag-suffix",       default="",
                        help="appended to the results directory, to keep configurations "
                             "that share a checkpoint (e.g. homo-a4 vs sim-dual) apart")
    parser.add_argument("--tasks",            nargs="+", default=["gsm8k", "math_500", "aime_2025"])
    parser.add_argument("--num-fewshot",      type=int, default=None)
    # --think / --no-think, defaulting to thinking ON, matching run_eval_disagg.sh and
    # the results/*/think trees. An earlier version defaulted this OFF on the
    # premise that every eval here was run with thinking suppressed -- that premise was
    # wrong: eval_vllm.py's --no-think never took effect, so those results are all
    # thinking-ENABLED. A pair of explicit flags is what matters; the two paths must
    # agree on the default or the trees stop being comparable.
    parser.add_argument("--think",            action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Qwen3 thinking mode (default ON, matching the disagg path). Pass --no-think to suppress it.")
    parser.add_argument("--batch-size",       type=int, default=64,
                        help="lm-eval generation batch size. The dual format is "
                             "batch-size independent: the phase follows sequence "
                             "length, and batching changes only the batch dim.")
    parser.add_argument("--no-compile",       dest="compile", action="store_false",
                        default=True,
                        help="skip torch.compile (on by default, dynamic=True). Dynamic "
                             "shapes matter here: generation alternates a long prompt "
                             "pass with single-token steps, so static compilation would "
                             "recompile per sequence length. The dual format branches on "
                             "that same length, which compiles to two guarded graphs "
                             "(prefill and decode) rather than graph-breaking.")
    parser.add_argument("--max-gen-toks",     type=int, default=512,
                        help="generation cap per sample. NOT comparable across values: "
                             "a truncated answer scores 0, so runs with different caps "
                             "are different measurements. The vLLM sweep uses 4096 (for "
                             "AIME/MATH), so ITS numbers are not directly comparable to "
                             "these. 512 is ample for GSM8K with --no-think and avoids "
                             "one runaway sequence pinning a whole batch at the cap.")
    parser.add_argument("--log-samples",      action="store_true",
                        help="dump per-sample generations to JSONL for inspection")
    parser.add_argument("--limit",            type=int, default=None,
                        help="limit eval to N docs per task (diagnostic only)")
    parser.add_argument("--output-dir",       default=str(_QAD_DIR / "results" / "transformers"))
    args = parser.parse_args()

    if not args.unquantized:
        if args.quantizer is None:
            parser.error("--quantizer is required unless --unquantized is set")
        if args.iter is None:
            parser.error("--iter is required unless --unquantized is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.unquantized:
        # Baseline: full-precision teacher model, no checkpoint
        model = load_model(args.model, torch.bfloat16,
                           attn_implementation="flash_attention_2").to(device)
        model.eval()
        ckpt_tag = f"{args.model.replace('/', '-')}-unquantized"
        step_key = 0
        log(f"Model: {args.model}  [unquantized BF16 baseline]")
    else:
        # Reconstruct ckpt_tag (matches qad.py) for path resolution / result naming.
        _, quant_hash = build_quantizer_params(args.quantizer, args.quantizer_params)
        model_short = args.model.replace("/", "-")
        run_name    = args.run_name or f"qad-{model_short}"
        ckpt_tag    = f"{run_name}-{args.quantizer}-{quant_hash}"

        ckpt_dir = resolve_checkpoint(Path(args.ckpt_dir), ckpt_tag, args.iter)
        print(f"Loading checkpoint: {ckpt_dir}", flush=True)

        is_split = (ckpt_dir / "prefill" / "model.safetensors").exists()
        packed = uses_compressed_tensors(args.quantizer)
        if is_split or args.dual or packed:
            # Rebuild the quantized model and load the checkpoint into it. Required
            # for packed compressed-tensors checkpoints (transformers cannot read FP4
            # nibbles) and for the dual layouts. A homogeneous format is evaluated in
            # exactly the format it was trained for — no phase is imposed on it.
            runtime = args.runtime_quantizer or (
                args.quantizer if (is_dual(args.quantizer) or not args.dual)
                else DEFAULT_DUAL_RUNTIME)
            model = build_quantized_model(args.model, runtime, ckpt_dir, device)
            if is_dual(runtime):
                print("  phase follows sequence length: prompt=prefill, "
                      "generated tokens=decode", flush=True)
            else:
                print(f"  homogeneous {runtime}: one format at every position", flush=True)
        else:
            # Pseudo-quantized checkpoints are standard HF models (dequantized weights
            # in `weight`), so this is one fast sharded load straight to GPU.
            model = load_model(str(ckpt_dir), torch.bfloat16,
                               attn_implementation="flash_attention_2").to(device)
            model.eval()
        step_key = args.iter
        log(f"Model: {args.model}  quantizer={args.quantizer}  step={args.iter}")
    # Applies on EVERY path, including --unquantized: without it two baseline runs
    # differing only in a flag write to the same file and the second overwrites the
    # first, which is exactly how the cap=1024 diagnostic lost its result.
    ckpt_tag += args.tag_suffix

    if not args.think:
        # Qwen3 suppresses thinking via the chat template kwarg, not generation_config.
        # The template checks: {%- if enable_thinking is defined and enable_thinking is false %}
        # Patch the tokenizer so lm_eval's apply_chat_template call injects it automatically.
        _orig_act = tokenizer.apply_chat_template
        def _no_think_act(conversation, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_act(conversation, **kwargs)
        tokenizer.apply_chat_template = _no_think_act
        print("Thinking suppressed via tokenizer.apply_chat_template patch.", flush=True)

    if args.compile:
        # dynamic=True because generation runs two very different shapes: one
        # multi-token prompt pass then single-token steps. Static compilation would
        # recompile for every prompt length in the batch stream.
        t = time.time()
        model = torch.compile(model, dynamic=True)
        log("torch.compile(dynamic=True) requested", t)
        # compile() is lazy, so without a warmup a failure would only appear deep into
        # generation. Drive one prompt-shaped and one token-shaped forward to build
        # BOTH graphs now — which are also exactly the dual format's two phases.
        t = time.time()
        try:
            with torch.no_grad():
                ids = torch.ones(1, 8, dtype=torch.long, device=device)
                out = model(ids, use_cache=True)
                model(ids[:, :1], past_key_values=out.past_key_values, use_cache=True)
            log("compile warmup OK (prefill + decode graphs built)", t)
        except Exception as e:
            log(f"compile warmup FAILED: {type(e).__name__}: {str(e)[:200]}")
            raise

    log(f"Tasks: {args.tasks}  batch_size={args.batch_size}")

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
        # A batch runs until EVERY sequence stops or hits this cap, so one
        # non-terminating sequence drags all of them to the limit. GSM8K answers with
        # --no-think are a few hundred tokens, and 4096 (needed for AIME/MATH in the
        # vLLM sweep) made each batch of 64 take ~5 minutes.
        #
        # THE CAP CHANGES THE SCORE: a truncated answer scores 0, so runs at different
        # caps are different measurements. These results are internally consistent
        # (every config here uses the same cap) but are NOT comparable to the vLLM
        # sweep's numbers, which are produced at 4096.
        gen_kwargs=f"max_gen_toks={args.max_gen_toks}",
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
