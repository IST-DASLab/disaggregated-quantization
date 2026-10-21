"""Structural checks for the real compressed-tensors NVFP4 serving artifact."""
import json
from pathlib import Path

import torch
from safetensors import safe_open


def validate_export(directory, expected_linears):
    directory = Path(directory)
    if not (directory / "tokenizer_config.json").is_file() or not any(
            (directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Serving export is missing its standalone tokenizer assets")
    config = json.loads((directory / "config.json").read_text())
    quant = config["quantization_config"]
    if quant.get("format") != "nvfp4-pack-quantized":
        raise ValueError("Export is not real packed NVFP4")
    if not config.get("text_config", config).get("use_cache", True):
        raise ValueError("Serving export must enable the inference cache")
    tensors = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as reader:
            for key in reader.keys():
                if key.endswith((".weight_packed", ".weight_scale")):
                    tensors[key] = (reader.get_slice(key).get_shape(), reader.get_slice(key).get_dtype())
                elif key.endswith((".weight_global_scale", ".input_global_scale")):
                    value = reader.get_tensor(key).float()
                    if not torch.isfinite(value).all() or not (value > 0).all():
                        raise ValueError(f"Invalid global scale: {key}")
                    tensors[key] = float(value.reshape(()))
    modules = [key.removesuffix(".weight_packed") for key in tensors if key.endswith(".weight_packed")]
    if len(modules) != expected_linears:
        raise ValueError(f"Expected {expected_linears} packed linears; found {len(modules)}")
    for name in modules:
        shape, dtype = tensors[name + ".weight_packed"]
        block_shape, block_dtype = tensors[name + ".weight_scale"]
        if dtype != "U8" or block_dtype != "F8_E4M3" or block_shape != [shape[0], shape[1] // 8]:
            raise ValueError(f"Invalid NVFP4 packed/block-scale layout: {name}")
        for scale in ("weight_global_scale", "input_global_scale"):
            if name + "." + scale not in tensors:
                raise ValueError(f"Missing {scale}: {name}")
        # These projections share the input and the weight-global quantization
        # group. GDN in_proj_qkv/in_proj_z intentionally remain independent.
        siblings = (("gate_proj", "up_proj") if name.endswith("gate_proj") else
                    ("q_proj", "k_proj", "v_proj") if name.endswith("q_proj") else
                    ("in_proj_qkv", "in_proj_z") if name.endswith("in_proj_qkv") and
                    config.get("text_config", config).get("prefiller_fuse_gdn_scales") else ())
        for leaf in siblings:
            sibling = name.rpartition(".")[0] + "." + leaf if "." in name else leaf
            for scale in ("weight_global_scale", "input_global_scale"):
                if tensors[sibling + "." + scale] != tensors[name + "." + scale]:
                    raise ValueError(f"Incompatible fused {scale}: {name} / {sibling}")
    return {"packed_nvfp4_linears": len(modules), "structural_validation": "passed",
            "native_kernel_parity": "not established by structural validation"}
