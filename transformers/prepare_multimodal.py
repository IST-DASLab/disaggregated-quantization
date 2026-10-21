"""Attach identical frozen vision weights to the trained text prefiller and decoder.

Decoder conversion uses the training loader's exact GGUF layout restoration.
Prefiller conversion only renames already packed NVFP4 tensors; no requantization.
Run with the training environment, not the serving environment.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from gguf_decode import dense_gguf_weights


def outer_name(name):
    return "model.language_model." + name[len("model."):] if name.startswith("model.") else name


class Shards:
    def __init__(self, output):
        self.output = output
        self.tensors = {}
        self.bytes = self.total_bytes = 0
        self.index = {}
        self.number = 0

    def add(self, name, tensor):
        if name in self.index or name in self.tensors:
            raise ValueError(f"Duplicate tensor {name}")
        tensor = tensor.contiguous()
        self.tensors[name] = tensor
        size = tensor.numel() * tensor.element_size()
        self.bytes += size
        self.total_bytes += size
        if self.bytes >= 4 * 1024**3:
            self.flush()

    def flush(self):
        if not self.tensors:
            return
        self.number += 1
        filename = f"model-{self.number:05d}.safetensors"
        save_file(self.tensors, self.output / filename, metadata={"format": "pt"})
        self.index.update({name: filename for name in self.tensors})
        print(f"Wrote {filename}: {len(self.tensors)} tensors, {self.bytes / 1024**3:.2f} GiB", flush=True)
        self.tensors = {}
        self.bytes = 0

    def finish(self):
        self.flush()
        (self.output / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": self.total_bytes}, "weight_map": self.index}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gguf", type=Path)
    source.add_argument("--text-export", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.safetensors")):
        raise ValueError(f"Refusing to overwrite existing weights in {args.output}")
    cfg = json.loads((args.base / "config.json").read_text())
    cfg.pop("quantization_config", None)
    cfg["text_config"].pop("quantization_config", None)
    cfg["dtype"] = cfg["text_config"]["dtype"] = "bfloat16"
    cfg["text_config"]["use_cache"] = True
    writer = Shards(args.output)
    packed = 0
    if args.gguf:
        text_config = AutoConfig.from_pretrained(args.base).text_config
        for index, (name, tensor) in enumerate(dense_gguf_weights(args.gguf, text_config), 1):
            writer.add(outer_name(name), tensor)
            if index % 64 == 0:
                print(f"Decoder: {index} tensors converted", flush=True)
    else:
        text_cfg = json.loads((args.text_export / "config.json").read_text())
        cfg["text_config"].update({key: value for key, value in text_cfg.items()
                                 if key.startswith("prefiller_")})
        quant = text_cfg["quantization_config"]
        quant["ignore"] = [outer_name(name) for name in quant["ignore"]] + ["re:.*visual.*"]
        cfg["quantization_config"] = quant
        for filename in sorted(args.text_export.glob("*.safetensors")):
            with safe_open(filename, framework="pt") as tensors:
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    if name.endswith(".weight_packed"):
                        assert tensor.dtype in (torch.int32, torch.uint8), (name, tensor.dtype)
                        packed += 1
                    writer.add(outer_name(name), tensor)
        assert packed == 400, f"Expected 400 packed text linears, got {packed}"
    base_index = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    vision = [name for name in base_index if name.startswith("model.visual.")]
    assert vision, "No vision weights found"
    for filename in sorted({base_index[name] for name in vision}):
        with safe_open(args.base / filename, framework="pt") as tensors:
            for name in vision:
                if base_index[name] == filename:
                    tensor = tensors.get_tensor(name)
                    assert tensor.dtype == torch.bfloat16, (name, tensor.dtype)
                    writer.add(name, tensor)
    writer.finish()
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json"):
        if (args.base / name).is_file():
            shutil.copy2(args.base / name, args.output / name)
    (args.output / "evaluation_preparation.json").write_text(json.dumps({
        "base": str(args.base.resolve()),
        "source": str((args.gguf or args.text_export).resolve()),
        "packed_nvfp4_linears": packed,
        "shared_vision_tensors": len(vision),
        "decoder_layout": "transformers.gguf_decode.dense_gguf_weights",
    }, indent=2))
    print(f"Prepared {args.output}: {len(writer.index)} tensors, {packed} packed linears, {len(vision)} BF16 vision tensors", flush=True)


if __name__ == "__main__":
    main()
