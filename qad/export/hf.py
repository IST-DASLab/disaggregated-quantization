"""Pseudo-quantized checkpoint export: a plain HuggingFace bf16 model.

For quantizers with no matching inference kernel (STE, QuEST, GSQ, signed-Lloyd
3-bit), each QuantizedLinear's hard-quantized weight `_wq` is written to the
standard `<module>.weight` slot. The result is structurally identical to the
unquantized model, so evaluation is a single fast from_pretrained() and vLLM
serves it as an ordinary bf16 model — the quantization error is baked into the
weights ("pseudo-quantization").
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from quantizers import QuantizedLinear


def build_state_dict(student: nn.Module) -> dict[str, Tensor]:
    """Remap the quantized student into a vanilla HF state dict.

    Quant-internal tensors (FP32 master weight, _mask, scales, quant_logits,
    schedule buffers, act_amax) are dropped; embeddings, norms, lm_head and rotary
    buffers are kept at their native dtype.
    """
    quant_paths = {name for name, mod in student.named_modules()
                   if isinstance(mod, QuantizedLinear)}
    out: dict[str, Tensor] = {}
    for key, tensor in student.state_dict().items():
        parent, _, leaf = key.rpartition(".")
        if parent in quant_paths:
            if leaf == "_wq":
                out[f"{parent}.weight"] = tensor.detach().to(torch.bfloat16).cpu()
            elif leaf == "bias":
                out[key] = tensor.detach().to(torch.bfloat16).cpu()
            # drop all other quant-internal tensors
        else:
            out[key] = tensor.detach().cpu()
    return out


def save_checkpoint(student: nn.Module, out_dir: Path, step: int = 0) -> int:
    """Write an eval-ready HF model directory. Returns the tensor count."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_state_dict(student)
    save_file(state, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "step": str(step)})
    student.config.save_pretrained(out_dir)
    if getattr(student, "generation_config", None) is not None:
        student.generation_config.save_pretrained(out_dir)
    return len(state)
