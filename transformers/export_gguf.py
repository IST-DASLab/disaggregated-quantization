"""Export real packed NVFP4 as lossless, per-layer GGUF ODP blocks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import gguf
import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent / "odp"))
from pack_prefill import (P_BA, P_NAME, pack_layer_trained, write_block_gguf)

UPSTREAM = "https://github.com/IST-DASLab/disaggregated-llama.cpp/commit/967b479a5ab04c967a0591d02e130b4b35dbfe3d"


class ProjectionReader:
    def __init__(self, directory):
        self.directory = directory
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(directory / "model.safetensors", framework="pt") as reader:
                self.index = {key: "model.safetensors" for key in reader.keys()}
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.layers.")
                                                       for k in self.index) else "model."

    def get(self, name):
        name = name.replace("model.language_model.", self.prefix, 1)
        with safe_open(self.directory / self.index[name], framework="pt") as reader:
            return reader.get_tensor(name)


def unpack_ggml_bytes(data, n, k):
    """Inverse layout only: compare codes/scales without floating point maths."""
    blocks = np.asarray(data).view(np.uint8).reshape(n, k // 64, 36)
    scales = blocks[:, :, :4].reshape(n, k // 16)
    q = blocks[:, :, 4:].reshape(n, k // 16, 8)
    nibbles = np.concatenate((q & 15, q >> 4), axis=-1).reshape(n, k)
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4), scales


def verify_block(path, layer, linear, entries):
    reader = gguf.GGUFReader(path)
    if reader.get_field("odp.layer").contents() != layer:
        raise ValueError(f"Wrong layer metadata: {path}")
    if reader.get_field("odp.kind").contents() != (0 if linear else 1):
        raise ValueError(f"Wrong layer kind: {path}")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if set(tensors) != {P_NAME[e[0]] + ".weight" for e in entries}:
        raise ValueError(f"Wrong projection set: {path}")
    for pid, n, k, packed, scale, w_global, x_global in entries:
        name = P_NAME[pid]
        tensor = tensors[name + ".weight"]
        if list(tensor.shape) != [k, n]:
            raise ValueError(f"Wrong tensor shape: {path}/{name}: {tensor.shape}")
        if pid == P_BA:
            if tensor.tensor_type != gguf.GGMLQuantizationType.BF16 or not np.array_equal(
                    np.asarray(tensor.data).view(np.uint16).reshape(n, k), packed):
                raise ValueError(f"BF16 gating bytes changed: {path}/{name}")
        else:
            if tensor.tensor_type != gguf.GGMLQuantizationType.NVFP4:
                raise ValueError(f"Not real NVFP4: {path}/{name}")
            got, got_scale = unpack_ggml_bytes(tensor.data, n, k)
            if not np.array_equal(got, packed) or not np.array_equal(got_scale, scale):
                raise ValueError(f"FP4 codes or FP8 block scales changed: {path}/{name}")
            for field, expected in (("scale", 1 / w_global), ("act_scale", 1 / x_global)):
                if reader.get_field(f"odp.{field}.{name}").contents() != np.float32(expected):
                    raise ValueError(f"Global scale changed: {path}/{name}/{field}")


def export_gguf(source, output, *, tag, name="Qwen3.8-27B-ODP", allow_legacy=False,
                replace=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config", {}).get("format") != "nvfp4-pack-quantized":
        raise ValueError("Input must be a real packed NVFP4 export")
    text = config.get("text_config", config)
    if not text.get("prefiller_shared_decoder_norms", False) and not allow_legacy:
        raise ValueError("Legacy source specialized prefill norms. ODP shares decoder norms; "
                         "--allow-legacy explicitly exports projections with this limitation recorded.")
    manifest = {"upstream": UPSTREAM, "source": str(source), "tag": tag,
                "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
                "requantized": False, "global_scales_changed": False,
                "shared_decoder_norms": text.get("prefiller_shared_decoder_norms", False),
                "legacy_projection_export": allow_legacy,
                "handoff_boundary": text.get("prefiller_handoff_boundary", "prompt-end"),
                "runtime_parity": "not established by byte-level format validation",
                "limitations": [], "files": []}
    if not manifest["shared_decoder_norms"]:
        manifest["limitations"].append("Source trained specialized prefill norms, omitted by ODP. "
                                        "This projection export does not preserve the complete trained model.")
    if manifest["handoff_boundary"] == "last-prompt-decode":
        manifest["limitations"].append("Linked ODP must evaluate the final prompt token with decoder weights "
                                        "to match the training/NIXL handoff.")
    if output.exists():
        previous = output / "export_manifest.json"
        if not replace or not previous.is_file() or json.loads(previous.read_text()).get("source") != str(source):
            raise ValueError(f"Refusing to replace unrelated output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    heads = text["linear_num_key_heads"]
    gdn = {"n_k": heads, "v_per_k": text["linear_num_value_heads"] // heads,
           "head_k": text["linear_key_head_dim"], "head_v": text["linear_value_head_dim"]}
    count = text["num_hidden_layers"]
    sh = ProjectionReader(source)
    with tempfile.TemporaryDirectory(prefix=".odp-export-", dir=output.parent) as tmp:
        ready = Path(tmp) / "ready"
        ready.mkdir()
        for layer in range(count):
            linear = (text["layer_types"][layer] == "linear_attention" if "layer_types" in text else
                      (layer + 1) % text.get("full_attention_interval", 4) != 0)
            entries = pack_layer_trained(sh, layer, linear, gdn, validate_groups=not allow_legacy)
            filename = f"{name}-{tag}-{layer + 1:05d}-of-{count:05d}.gguf"
            path = ready / filename
            write_block_gguf(str(path), layer, linear, entries)
            verify_block(path, layer, linear, entries)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"].append({"name": filename, "bytes": path.stat().st_size, "sha256": digest,
                                      "verified_projections": len(entries)})
            print(f"GGUF {layer + 1}/{count}: {filename}; exact FP4/FP8/global-scale round trip passed", flush=True)
        manifest["byte_verification"] = "passed"
        (ready / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
        previous = Path(tmp) / "previous"
        if output.exists():
            output.rename(previous)
        try:
            ready.rename(output)
        except BaseException:
            if previous.exists():
                previous.rename(output)
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--name", default="Qwen3.8-27B-ODP")
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    manifest = export_gguf(args.source, args.output, tag=args.tag, name=args.name,
                           allow_legacy=args.allow_legacy, replace=args.replace_existing)
    print(f"Verified {len(manifest['files'])} GGUF blocks in {args.output}")


if __name__ == "__main__":
    main()
