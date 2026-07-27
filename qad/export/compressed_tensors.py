"""Real NVFP4 checkpoint export in the compressed-tensors `nvfp4-pack-quantized`
format.

vLLM auto-detects this from config.json's quantization_config (quant_method
"compressed-tensors") and serves it with true 4-bit kernels — no --quantization
flag needed. Per linear it reads:

    weight_packed        uint8      [O, K//2]   packed FP4 (two E2M1 nibbles/byte)
    weight_scale         fp8_e4m3   [O, K//16]  per-block scale
    weight_global_scale  fp32 [1]   = 2688/amax      (RECIPROCAL of ModelOpt's)
    input_global_scale   fp32 [1]   = 2688/act_amax  (W4A4 only)

dequant = e2m1 * weight_scale / weight_global_scale.

Two hard requirements learned the hard way:
  * `quant_method` must be present, and NO `weight_shape` tensor may be emitted —
    vLLM does a bare params_dict[name] lookup and any extra tensor is a KeyError.
  * W4A16 must set input_activations=None AND omit input_global_scale: vLLM's
    _is_fp4a16_nvfp4() keys on `input_quant is None` to pick the weight-only
    scheme, which registers no such parameter.
"""

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from quantizers.blocked import GLOBAL_DEN
from quantizers.nvfp4 import NVFP4Linear, pack_nvfp4_weight

BLOCK = 16

_CT_GRP = {
    "actorder": None, "block_structure": None, "group_size": BLOCK,
    "num_bits": 4, "observer": "minmax", "observer_kwargs": {},
    "strategy": "tensor_group", "symmetric": True, "type": "float",
}


def ct_quant_config(quantize_act: bool = True) -> dict:
    """quantization_config for NVFP4. W4A4 uses input_activations.dynamic="local"
    (static per-tensor input_global_scale + dynamic per-block activation quant);
    W4A16 sets input_activations=None. Field set matches a real RedHatAI NVFP4
    checkpoint."""
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {**_CT_GRP, "dynamic": "local"} if quantize_act else None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {**_CT_GRP, "dynamic": False},
            }
        },
        "format": "nvfp4-pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def build_state_dict(student: nn.Module) -> dict[str, Tensor]:
    """Pack every NVFP4Linear; keep everything else (embeddings, norms, lm_head) as is.
    Quant-internal tensors (_wq, act_amax, the FP32 master weight) are dropped."""
    quant_mods = {name for name, m in student.named_modules()
                  if isinstance(m, NVFP4Linear)}
    out: dict[str, Tensor] = {}
    for name, m in student.named_modules():
        if not isinstance(m, NVFP4Linear):
            continue
        # shared fused-group global scale, so vLLM's weight_global_scale.max() is a no-op
        gscale = m.group_global_scale()
        packed, wscale, wscale2 = pack_nvfp4_weight(m.weight.data, m.block_size,
                                                    global_scale=gscale)
        out[f"{name}.weight_packed"] = packed.cpu()
        out[f"{name}.weight_scale"] = wscale.cpu()
        out[f"{name}.weight_global_scale"] = (1.0 / wscale2).reshape(1).cpu()
        if m.quantize_act:   # W4A4 only — the W4A16 scheme registers no such param
            act = m.act_amax.float().clamp(min=1e-8)
            out[f"{name}.input_global_scale"] = (GLOBAL_DEN / act).reshape(1).cpu()
        if m.bias is not None:
            out[f"{name}.bias"] = m.bias.data.to(torch.bfloat16).cpu()
    for key, tensor in student.state_dict().items():
        parent, _, _ = key.rpartition(".")
        if parent in quant_mods:
            continue     # master weight, _wq, act_amax, bias handled above
        out[key] = tensor.detach().cpu()
    return out


def save_checkpoint(student: nn.Module, out_dir: Path) -> int:
    """Write a compressed-tensors NVFP4 checkpoint dir (W4A4 or W4A16, matching how
    the model was quantized). Returns the tensor count."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_state_dict(student)
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    student.config.save_pretrained(out_dir)
    if getattr(student, "generation_config", None) is not None:
        student.generation_config.save_pretrained(out_dir)
    quantize_act = next(m.quantize_act for m in student.modules()
                        if isinstance(m, NVFP4Linear))
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = ct_quant_config(quantize_act)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return len(state)
