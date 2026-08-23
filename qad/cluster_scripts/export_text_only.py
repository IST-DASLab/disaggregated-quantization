"""Export a BF16 text-only checkpoint from a stock repo, with no training involved.

    python cluster_scripts/export_text_only.py --model google/gemma-3-4b-it \
        --out checkpoints/bf16-text-only/google-gemma-3-4b-it

WHY THIS EXISTS
---------------
gemma-3-4b/12b ship ONLY as `Gemma3ForConditionalGeneration`, whose `model_type` is
"gemma3". vLLM's `ModelConfig.is_mm_prefix_lm` keys on that string ALONE -- not on
whether a vision tower is present, not on whether an image is ever passed -- and only
`flex_attention`/`triton_attn` implement `supports_mm_prefix()`. So the stock 4b/12b
repos cannot be served under FLASH_ATTN, which 270m/1b in turn REQUIRE (FlashInfer
asserts on head_dim=256 at block_size 16). No single backend serves the family as
shipped.

That makes this more than a convenience: **the BF16 baseline for 4b/12b cannot be the
stock repo.** A QAD checkpoint is text-only by construction, so comparing it against a
stock multimodal baseline would differ in architecture AND attention backend on top of
quantization -- not a control. This produces a baseline of exactly the shape the
quantized checkpoints have.

It is also the cheapest way to run the Phase 2.4 acceptance test (does vLLM actually
serve a text-only export?) without first spending a full 4b training run to get one.

Plain causal LMs (270m, 1b, every Qwen) are exported unchanged -- `save_checkpoint`
only rewrites the config when the model IS a wrapper -- so this is safe to point at
any model, and gives a like-for-like BF16 artifact for those too.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # qad root
from export.save import save_checkpoint, text_only_arch          # noqa: E402
from training.models import load_model, text_stack, verify_text_load   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF repo id")
    p.add_argument("--out", required=True, help="output checkpoint directory")
    p.add_argument("--attn", default="eager",
                   help="attn_implementation for the LOAD only; irrelevant to the "
                        "exported weights, and eager avoids needing FA2 just to save")
    args = p.parse_args()

    out = Path(args.out)
    print(f"loading {args.model} ...", flush=True)
    model = load_model(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    arch = text_only_arch(model)
    stack = text_stack(model)
    print(f"  class={type(model).__name__}  text stack at {stack.base_path!r}  "
          f"text_only_arch={arch}", flush=True)

    # The failure this guards is a load that reports success and returns re-initialized
    # tensors -- Gemma3ForCausalLM.from_pretrained(4b) does exactly that. Exporting such a
    # model would produce a confidently-wrong baseline.
    n = verify_text_load(model, args.model)
    print(f"  verified {n} text tensors against the checkpoint", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    count = save_checkpoint(model, out, step=0)
    print(f"wrote {count} tensors -> {out}", flush=True)

    # Acceptance checks, asserted here rather than discovered on a compute node 20 minutes
    # into a serving job.
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    bad = [k for k in keys if "vision" in k or "multi_modal" in k or "language_model" in k]
    assert not bad, f"text-only export still carries {len(bad)} wrapper tensors: {bad[:5]}"

    cfg = json.loads((out / "config.json").read_text())
    assert "vision_config" not in cfg, "vision_config leaked into the exported config"
    if arch is not None:
        assert cfg["architectures"] == [arch], cfg.get("architectures")
        assert cfg["model_type"] == "gemma3_text", cfg.get("model_type")
    if cfg.get("model_type") == "gemma3_text":
        # The silent defect: without this the sliding layers run at the full-attention
        # 1e6 and the model is quietly wrong rather than rejected.
        assert "rope_local_base_freq" in cfg, "missing rope_local_base_freq"
        assert "rope_parameters" not in cfg, "per-layer rope_parameters would be rejected"
    print(f"  config OK: model_type={cfg.get('model_type')} "
          f"architectures={cfg.get('architectures')} tensors={len(keys)}", flush=True)


if __name__ == "__main__":
    main()
