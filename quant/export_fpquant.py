"""Export a locally-quantized model into two on-disk checkpoints.

This repo's ``gptq_quantization`` / ``rtn_quantization`` replace each attention/MLP
``nn.Linear`` with a :class:`QuantizedLinear` that carries two fake-quantized weight
buffers (``weight_prefill`` = NVFP4, ``weight_decode``) plus an optional NVFP4 activation
quantizer.  This module writes those two weight sets out as two separate checkpoints:

  * **prefill** -> FP-Quant *pseudoquant* format: NVFP4 ``dqweight`` + identity rotation
    matrices + global scales, and a ``quantization_config`` with ``pseudoquantization``.
    Loads (with ``fp_quant`` installed) as W4A4 -- activations are NVFP4-quantized at run.
  * **decode**  -> a plain fake-quantized **bf16** HF model (standard ``nn.Linear``
    weights, no ``quantization_config``).  Runs W16A16: weights carry the quant error but
    nothing is quantized at runtime.

The pseudoquant config and the sharded-safetensors layout are copied from
``IST-DASLab/FP-Quant`` (``src/quantization/qconfig.py`` and
``model_quant.py::export_quantized_model``), specialised for the **identity** transform so
that we depend on neither the ``fp_quant`` package nor ``fast_hadamard_transform``.
"""

import copy
import json
import os
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from model_utils import QuantizedLinear

# Global scale FP-Quant's NVFP4 runtime uses by default (FPQuantLinear.pre_forward).
_NVFP4_GLOBAL_SCALE = 10.0
_DEFAULT_MAX_SHARD = 5 * 1024 ** 3


def make_pseudoquant_config(hadamard_group_size: int = 128) -> Dict:
    """FP-Quant NVFP4 pseudoquant ``quantization_config`` pinned to the identity transform.

    Mirrors ``prepare_quantization_config(format="nvfp", pseudoquantization=True)`` with one
    deliberate addition: ``transform_init="identity"``.  The fp_quant runtime rebuilds the
    online rotation from this field, so it MUST say identity -- otherwise activations get
    Hadamard-rotated against weights that were quantized without any rotation.
    """
    return {
        "forward_dtype": "nvfp4",
        "backward_dtype": "bf16",
        "forward_method": "abs_max",
        "hadamard_group_size": hadamard_group_size,
        "modules_to_not_convert": ["lm_head"],
        "quant_method": "fp_quant",
        "store_master_weights": False,
        "pseudoquantization": True,
        "transform_init": "identity",
    }


def _shard_and_save(state_dict: Dict[str, torch.Tensor], save_dir: str,
                    max_shard_size: int = _DEFAULT_MAX_SHARD) -> None:
    """Split ``state_dict`` into <= ``max_shard_size`` safetensors shards + an index json."""
    os.makedirs(save_dir, exist_ok=True)
    shards, cur, cur_sz = [], {}, 0
    for k, v in state_dict.items():
        sz = v.numel() * v.element_size()
        if cur and cur_sz + sz > max_shard_size:
            shards.append(cur)
            cur, cur_sz = {}, 0
        if sz > max_shard_size:            # a single tensor larger than a shard
            shards.append({k: v})
            continue
        cur[k] = v
        cur_sz += sz
    if cur:
        shards.append(cur)

    n = len(shards)
    width = len(str(max(n, 1)))
    weight_map = {}
    for i, shard in enumerate(shards):
        fname = f"model-{str(i + 1).zfill(width)}-of-{str(n).zfill(width)}.safetensors"
        save_file(shard, os.path.join(save_dir, fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
    with open(os.path.join(save_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def _quantized_linear_names(model) -> set:
    return {name for name, m in model.named_modules() if isinstance(m, QuantizedLinear)}


def _copy_shared_params(model, ql_names: set, dst: Dict[str, torch.Tensor],
                        dtype: torch.dtype) -> None:
    """Copy every parameter NOT owned by a QuantizedLinear (embeddings, norms, lm_head...)."""
    for k, v in model.state_dict().items():
        owner = k.rsplit(".", 1)[0]
        if owner in ql_names:              # weight_prefill / weight_decode / bias -> handled explicitly
            continue
        dst[k] = (v.to(dtype) if v.is_floating_point() else v).cpu()


def _write_config(model, save_dir: str, quantization_config: Optional[Dict]) -> None:
    cfg = copy.deepcopy(model.config)
    cfg.use_cache = True
    d = cfg.to_dict()
    d.pop("quantization_config", None)
    if quantization_config is not None:
        d["quantization_config"] = quantization_config
    for key in ("torch_dtype", "dtype"):   # ensure JSON-serialisable dtype
        if key in d and not isinstance(d[key], (str, type(None))):
            d[key] = str(d[key]).replace("torch.", "")
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(d, f, indent=2)
    try:
        model.generation_config.save_pretrained(save_dir)
    except Exception:
        pass


@torch.no_grad()
def export_prefill_pseudoquant(model, tokenizer, save_dir: str,
                               hadamard_group_size: int = 128,
                               dtype: torch.dtype = torch.bfloat16,
                               max_shard_size: int = _DEFAULT_MAX_SHARD) -> str:
    """Write the NVFP4 prefill weights in FP-Quant pseudoquant format (identity transform)."""
    ql_names = _quantized_linear_names(model)
    eye = torch.eye(hadamard_group_size, dtype=dtype)
    gscale = torch.tensor([_NVFP4_GLOBAL_SCALE], dtype=dtype)

    sd: Dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if not isinstance(m, QuantizedLinear):
            continue
        sd[f"{name}.dqweight"] = m.weight_prefill.to(dtype).cpu()
        sd[f"{name}.forward_hadamard_matrix"] = eye.clone()
        sd[f"{name}.backward_hadamard_matrix"] = eye.clone()
        sd[f"{name}.weight_global_scale"] = gscale.clone()
        sd[f"{name}.act_global_scale"] = gscale.clone()
        if m.bias is not None:
            sd[f"{name}.bias"] = m.bias.to(dtype).cpu()
    _copy_shared_params(model, ql_names, sd, dtype)

    _shard_and_save(sd, save_dir, max_shard_size)
    _write_config(model, save_dir, make_pseudoquant_config(hadamard_group_size))
    tokenizer.save_pretrained(save_dir)
    return save_dir


@torch.no_grad()
def export_decode_bf16(model, tokenizer, save_dir: str,
                       dtype: torch.dtype = torch.bfloat16,
                       max_shard_size: int = _DEFAULT_MAX_SHARD) -> str:
    """Write the decode weights as a plain fake-quantized bf16 HF model (no quant config)."""
    ql_names = _quantized_linear_names(model)

    sd: Dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if not isinstance(m, QuantizedLinear):
            continue
        sd[f"{name}.weight"] = m.weight_decode.to(dtype).cpu()
        if m.bias is not None:
            sd[f"{name}.bias"] = m.bias.to(dtype).cpu()
    _copy_shared_params(model, ql_names, sd, dtype)

    _shard_and_save(sd, save_dir, max_shard_size)
    _write_config(model, save_dir, quantization_config=None)
    tokenizer.save_pretrained(save_dir)
    return save_dir
