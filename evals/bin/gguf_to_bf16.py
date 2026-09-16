"""Dequantize a GSQ-RCO GGUF (backbone + mmproj) into a BF16 HF checkpoint.

    python3 bin/gguf_to_bf16.py \
        --gguf   <dir>/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
        --mmproj <dir>/mmproj-Qwen3.8-27B-BF16.gguf \
        --base   <models>/Qwen3.8-27B \
        --out    <models>/Qwen3.8-27B-GSQ-RCO-IQ3_S-bf16

WHY NOT transformers' own GGUF loader: `from_pretrained(gguf_file=...)` only covers the
architectures in transformers' GGUF mapping table, and 5.17.0's table has qwen3/qwen3_moe
but NOT qwen35. This model is `qwen3_5`, so that path cannot read it at all.

WHY THIS IS FAITHFUL: dequantization reconstructs exactly the values the GGUF block
format encodes -- it does not recover the original unquantized weights. The quantization
error being measured is preserved bit for bit. GSQ-RCO is WEIGHT-ONLY, so activations
were always computed in higher precision anyway; serving the dequantized weights in BF16
through native vLLM measures the same thing the GGUF path does, without the GGUF runtime.

HOW THE LAYOUT IS GOT RIGHT: the GGUF tensor layout is NOT the HF layout for this
architecture -- the Gated-DeltaNet blocks need a value-head retiling, A_log needs
`log(-w)`, conv1d needs an unsqueeze, and in_proj_qkv needs its value rows retiled while
its qk rows are left alone. Rather than re-derive any of that, this reuses
vllm-gguf-plugin's own `Qwen35GGUFAdapter.transform_weights`, which is the code path that
already served this model correctly.

The one subtlety: that adapter takes a DENSE path and a PACKED path, and they differ --

    # Packed quantized tensors are only excluded from the dense out_proj
    # restore: reordering dim=1 would corrupt packed columns.
    if base.endswith("linear_attn.out_proj") and base not in quantized_bases:
        return layout.weight_to_vllm(weight, dim=1)

`quantized_bases` is populated from the `.weight_type` marker entries that the plugin's
own iterator emits for packed tensors. We dequantize BEFORE handing weights over and emit
NO `.weight_type` entries, so every tensor takes the dense branch -- which is exactly
right, and is the branch that applies the out_proj dim=1 restore the packed path defers
to runtime.
"""
import argparse
import gc
import json
import os
import shutil
import sys

import gguf
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.weights_adapter import get_weights_adapter


class _ModelConfigShim:
    """`build_name_map`/`transform_weights` read only `.hf_config` off the ModelConfig.

    Building a real vllm.config.ModelConfig would require a served model, a tokenizer and
    a device assignment for no benefit here.
    """

    def __init__(self, hf_config):
        self.hf_config = hf_config


def dense_weights(files, name_map):
    """Yield (hf_name, dense_tensor) for every mapped tensor, dequantizing as needed.

    Deliberately emits no `.weight_type` entries -- see the module docstring.
    """
    for path in files.all_files:
        reader = gguf.GGUFReader(path, "r")
        for tensor in reader.tensors:
            hf_name = name_map.get(tensor.name)
            if hf_name is None:
                continue
            qtype = tensor.tensor_type
            if qtype.name in ("F32", "F16", "BF16"):
                data = tensor.data
                if qtype.name == "BF16" and data.dtype.name == "uint8":
                    data = data.view("uint16")
                    out = torch.from_numpy(data.copy()).view(torch.bfloat16)
                else:
                    out = torch.from_numpy(data.copy())
            else:
                out = torch.from_numpy(gguf.dequantize(tensor.data, qtype).copy())
            yield hf_name, out.to(torch.float32)
        del reader
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="backbone .gguf")
    ap.add_argument("--mmproj", default=None, help="mmproj .gguf (required if multimodal)")
    ap.add_argument("--base", required=True,
                    help="dir with the base model's config.json + tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-bytes", type=int, default=4 * 1024**3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = GGUFModelFiles(backbone=(args.gguf,), mm_proj=args.mmproj)

    hf_config = AutoConfig.from_pretrained(args.base)
    adapter = get_weights_adapter(hf_config)
    print(f"adapter: {type(adapter).__name__}", flush=True)
    patched = adapter.patch_hf_config(files, hf_config)
    mc = _ModelConfigShim(patched)

    name_map = adapter.build_name_map(files, mc)
    print(f"name map: {len(name_map)} tensors", flush=True)

    shard_idx, shard, shard_bytes, total, index = 0, {}, 0, 0, {}

    def flush():
        nonlocal shard_idx, shard, shard_bytes
        if not shard:
            return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, os.path.join(args.out, fn), metadata={"format": "pt"})
        for k in shard:
            index[k] = fn
        print(f"  wrote {fn}  ({len(shard)} tensors, {shard_bytes/1e9:.2f} GB)", flush=True)
        shard, shard_bytes = {}, 0
        gc.collect()

    for name, weight in adapter.transform_weights(dense_weights(files, name_map), mc):
        if name.endswith(".weight_type"):
            continue                                  # defensive; we emit none
        w = weight.to(torch.bfloat16).contiguous()
        shard[name] = w
        shard_bytes += w.numel() * w.element_size()
        total += 1
        if shard_bytes >= args.shard_bytes:
            flush()
    flush()

    # Rename to the conventional -of- form now that the shard count is known.
    renamed = {}
    for i in range(1, shard_idx + 1):
        old, new = f"model-{i:05d}.safetensors", f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        os.rename(os.path.join(args.out, old), os.path.join(args.out, new))
        renamed[old] = new
    index = {k: renamed[v] for k, v in index.items()}

    nbytes = sum(os.path.getsize(os.path.join(args.out, f)) for f in set(index.values()))
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": nbytes}, "weight_map": index}, fh, indent=2)

    # Config/tokenizer come from the BASE repo, not from `patched`: the output is a plain
    # dense checkpoint served by NATIVE vLLM, which already supports qwen3_5, so it must
    # not carry a quantization_config or the plugin's GGUF-specific architecture edits.
    for fn in ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json"):
        src = os.path.join(args.base, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, fn))
    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    print(f"\n{total} tensors -> {shard_idx} shards, {nbytes/1e9:.1f} GB in {args.out}",
          flush=True)
    if total != len(name_map):
        print(f"NOTE: {len(name_map)} mapped vs {total} written "
              f"(transform_weights may merge/split tensors)", flush=True)


if __name__ == "__main__":
    main()
