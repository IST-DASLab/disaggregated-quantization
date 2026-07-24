"""
Convert existing QAD weight checkpoints into standard HuggingFace model dirs.

Each quantized module's dequantized weight (_wq) is written to the standard
`<module>.weight` slot, producing a plain Qwen3ForCausalLM checkpoint that loads
with a single fast from_pretrained() — no quantizer, no base-model read.

Input per step (whichever exists, preferred first):
    weights/step_XXXXXXX.safetensors   (trimmed _wq-format from an earlier pass)
    weights/step_XXXXXXX.pt            (full training state dict)
Output:
    weights/step_XXXXXXX/config.json + model.safetensors [+ generation_config.json]

Usage:
    python convert_checkpoints.py --ckpt-dir <weights_dir> --base-model Qwen/Qwen3-4B
"""

import argparse
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, GenerationConfig


# quant-internal leaves that must never appear in a vanilla HF checkpoint
_QUANT_LEAVES = {
    "weight", "_mask", "scales", "quant_logits", "_temp", "_scale_val",
    "_values", "_idx",
}


def remap_to_hf(state: dict) -> dict:
    """Remap a quantized state dict to vanilla HF param names."""
    quant_prefixes = {k[: -len("._wq")] for k in state if k.endswith("._wq")}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        parent, _, leaf = key.rpartition(".")
        if parent in quant_prefixes:
            if leaf == "_wq":
                out[f"{parent}.weight"] = tensor.to(torch.bfloat16).contiguous()
            elif leaf == "bias":
                out[key] = tensor.to(torch.bfloat16).contiguous()
            # else: drop master weight / mask / scales / logits / schedule buffers
        elif leaf in _QUANT_LEAVES and leaf != "weight":
            # stray quant-internal outside a detected prefix — skip defensively
            continue
        else:
            out[key] = tensor.contiguous()
    return out


def load_state(pt_or_st: Path) -> dict:
    if pt_or_st.suffix == ".safetensors":
        return load_file(str(pt_or_st), device="cpu")
    ckpt = torch.load(pt_or_st, map_location="cpu", weights_only=False, mmap=True)
    return ckpt.get("model", ckpt)


def convert_step(weights_dir: Path, step: int, config, gen_config, dry_run: bool) -> None:
    out_dir = weights_dir / f"step_{step:07d}"
    if (out_dir / "model.safetensors").exists() and (out_dir / "config.json").exists():
        print(f"  step {step:7d}: already converted", flush=True)
        return

    src = None
    for cand in (weights_dir / f"step_{step:07d}.safetensors",
                 weights_dir / f"step_{step:07d}.pt"):
        if cand.exists():
            src = cand
            break
    if src is None:
        print(f"  step {step:7d}: NO SOURCE", flush=True)
        return

    print(f"  step {step:7d}: loading {src.name} …", flush=True)
    state = load_state(src)
    hf_state = remap_to_hf(state)
    n_w = sum(1 for k in hf_state if k.endswith(".weight"))
    size_gb = sum(t.numel() * t.element_size() for t in hf_state.values()) / 1e9
    print(f"             {len(state)} → {len(hf_state)} tensors "
          f"({n_w} .weight), ~{size_gb:.1f} GB", flush=True)

    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(hf_state, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "step": str(step)})
    config.save_pretrained(out_dir)
    if gen_config is not None:
        gen_config.save_pretrained(out_dir)
    print(f"             → {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="weights/ directory")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B", help="config source")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    weights_dir = Path(args.ckpt_dir)
    config = AutoConfig.from_pretrained(args.base_model)
    try:
        gen_config = GenerationConfig.from_pretrained(args.base_model)
    except Exception:
        gen_config = None

    steps = sorted({
        int(re.search(r"step_(\d+)", p.name).group(1))
        for p in weights_dir.glob("step_*")
        if re.search(r"step_(\d+)", p.name)
    })
    print(f"Found steps: {steps}", flush=True)
    for step in steps:
        convert_step(weights_dir, step, config, gen_config, args.dry_run)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
