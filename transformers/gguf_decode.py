# SPDX-License-Identifier: Apache-2.0
# Layout restoration adapted from vllm-project/vllm-gguf-plugin:
# weights_adapter/qwen3_5.py and quantization/layout.py (contributors to vLLM).
# https://github.com/vllm-project/vllm-gguf-plugin
"""Stream a dense Qwen3.5/3.8 GGUF into HF text tensor order, without vLLM.

Only the packed GGUF is stored on disk. Each tensor is dequantized, transformed,
and copied straight into the student's frozen decoder parameters.
"""
import re

import gguf
import torch


LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a.weight": "linear_attn.A_log",
    "ssm_a": "linear_attn.A_log",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def restore_heads(weight, config, dim=0, head_dim=None):
    """GGML [repeat, key_heads, head_dim] -> HF [key_heads, repeat, head_dim]."""
    repeat = config.linear_num_value_heads // config.linear_num_key_heads
    if repeat == 1:
        return weight
    hd = config.linear_value_head_dim if head_dim is None else head_dim
    groups, remainder = divmod(weight.shape[dim], repeat * hd)
    if remainder:
        raise ValueError(f"Invalid head-tiled shape {weight.shape}")
    shape = weight.shape
    tiled = (*shape[:dim], repeat, groups, hd, *shape[dim + 1:])
    return weight.reshape(tiled).transpose(dim, dim + 1).reshape(shape).contiguous()


def restore_layout(name, weight, config):
    qk_rows = config.linear_key_head_dim * config.linear_num_key_heads * 2
    if name.endswith("linear_attn.out_proj.weight"):
        return restore_heads(weight, config, dim=1)
    if name.endswith("linear_attn.A_log"):
        return restore_heads(torch.log(-weight), config, head_dim=1)
    if name.endswith("linear_attn.dt_bias"):
        return restore_heads(weight, config, head_dim=1)
    if name.endswith("linear_attn.in_proj_z.weight"):
        return restore_heads(weight, config)
    if name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        return restore_heads(weight, config, head_dim=1)
    if name.endswith(("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")):
        weight = torch.cat((weight[:qk_rows], restore_heads(weight[qk_rows:], config)), dim=0)
        return weight.unsqueeze(1) if name.endswith("conv1d.weight") else weight
    # GGUF stores effective RMS scales; HF Qwen3.5 stores scale minus one.
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        return weight - 1
    return weight


def dense_gguf_weights(path, config):
    if config.model_type != "qwen3_5_text":
        raise ValueError("This GGUF converter supports dense Qwen3.5/3.8 text only")
    reader = gguf.GGUFReader(str(path), "r")
    architecture = reader.get_field("general.architecture").contents()
    if architecture != "qwen35":
        raise ValueError(f"Expected qwen35 GGUF, got {architecture}")
    global_names = {"token_embd.weight": "model.embed_tokens.weight",
                    "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
    for tensor in reader.tensors:
        name = global_names.get(tensor.name)
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
        if match:
            layer, suffix = int(match[1]), match[2]
            if layer >= config.num_hidden_layers:
                continue  # GGUF MTP weights are not used by the prefiller
            if suffix not in LAYER_NAMES:
                raise ValueError(f"Unmapped GGUF decoder tensor: {tensor.name}")
            name = f"model.layers.{layer}.{LAYER_NAMES[suffix]}"
        if name is None:
            raise ValueError(f"Unmapped GGUF decoder tensor: {tensor.name}")
        data = gguf.dequantize(tensor.data, tensor.tensor_type)
        weight = torch.from_numpy(data.copy()).float()
        yield name, restore_layout(name, weight, config).to(torch.bfloat16)
