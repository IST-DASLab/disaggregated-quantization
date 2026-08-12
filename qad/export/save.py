"""Model-level checkpoint assembly.

The FORMAT is owned by the layers (see QuantizedLinear.export_tensors /
export_config / load_tensors). This module only does the part that is inherently
model-level and cannot belong to any single layer:

  * walk the model and prefix each layer's tensors with its module path — the path
    is what vLLM looks up, so it has to come from named_modules() and not from
    anything the layer knows about itself;
  * carry over the tensors no quantizer owns (embeddings, norms, lm_head, rotary
    buffers);
  * write config.json, asking one quantized layer for the quantization_config.

A `variant` selects which checkpoint is being written for formats that emit more
than one per step (prefill vs decode). It is passed straight through to the layers;
None means "the only checkpoint", which is what every ordinary format returns from
export_variants().
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from quantizers.base import QuantizedLinear


def _quant_layers(model: nn.Module) -> dict[str, QuantizedLinear]:
    return {name: m for name, m in model.named_modules()
            if isinstance(m, QuantizedLinear)}


def build_state_dict(model: nn.Module, variant=None) -> dict[str, Tensor]:
    """Assemble the full tensor dict for one checkpoint variant."""
    quant = _quant_layers(model)
    out: dict[str, Tensor] = {}
    # Everything the quantizers do NOT own. Quant-internal state (the FP32 master,
    # _wq, act_amax, schedule buffers, logits) is skipped — the layer decides what
    # of itself is worth serializing, and it is never the training-time internals.
    for key, tensor in model.state_dict().items():
        parent, _, _ = key.rpartition(".")
        if parent in quant:
            continue
        out[key] = tensor.detach().cpu()
    for name, m in quant.items():
        for leaf, tensor in m.export_tensors(variant).items():
            out[f"{name}.{leaf}"] = tensor
    return out


def load_into(model: nn.Module, tensors: dict[str, Tensor], variant=None) -> int:
    """Restore quantized layers from a checkpoint dict. Returns layers restored.

    The inverse of build_state_dict for the quantized layers; non-quantized tensors
    are loaded by the caller through the usual load_state_dict path.
    """
    n = 0
    for name, m in _quant_layers(model).items():
        prefix = f"{name}."
        sub = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        if sub:
            m.load_tensors(sub, variant)
            n += 1
    return n



def export_variants(model: nn.Module) -> list:
    """Variants this model's quantizer emits ([None] for single-checkpoint formats).

    Asks the LAYER CLASS, so a format that emits prefill/ + decode/ is discovered from
    the model itself rather than from its name. quantizers.variants(name) is the
    equivalent lookup for callers that only have the CLI name and no built model.
    """
    quant = _quant_layers(model)
    if not quant:
        return [None]
    return type(next(iter(quant.values()))).export_variants()

from export.config_fix import fix_serving_fields


def save_checkpoint(model: nn.Module, out_dir: Path, variant=None, step: int = 0) -> int:
    """Write one checkpoint directory. Returns the tensor count."""
    from safetensors.torch import save_file
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_state_dict(model, variant)
    save_file(state, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "step": str(step)})
    model.config.save_pretrained(out_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out_dir)

    quant = _quant_layers(model)
    qcfg = next(iter(quant.values())).export_config(variant) if quant else None
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if qcfg is not None:
        cfg["quantization_config"] = qcfg
    fix_serving_fields(cfg)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return len(state)
