"""Drop the `.inner.*` fp32 master copies from a QAD prefill checkpoint so vLLM can load it.

    python3 bin/strip_inner_weights.py --src <ckpt>/model.safetensors --out <serving dir>

WHY. The QAD trainer keeps fp32 masters (qad/training/qad.py loads the student in
float32) and its norm wrapper exposes the real parameter one level down, as
`...input_layernorm.inner.weight`. The exported checkpoint therefore carries BOTH:

    model.language_model.layers.0.input_layernorm.weight        BF16   <- what vLLM wants
    model.language_model.layers.0.input_layernorm.inner.weight  F32    <- training master

vLLM's Qwen3_5Model has no `.inner` submodule and refuses the whole load:

    ValueError: There is no module or parameter named 'layers.0.input_layernorm.inner'
                in Qwen3_5Model

DROPPING THEM IS LOSSLESS HERE, and that was checked rather than assumed: every one of
the 208 `.inner` tensors has a plain-named counterpart already present, and the pairs
agree to one BF16 ulp (max |diff| 0.000977 on a 5120-wide norm; several pairs are bit
identical). The plain tensor IS the BF16 cast of the master, and vLLM serves bfloat16,
so the fp32 copy could not have been used even if it loaded.

Only the 5 norm families are affected -- input_layernorm, post_attention_layernorm,
linear_attn.norm, self_attn.q_norm, self_attn.k_norm. Quantized Linear weights are
untouched, so this does not alter what is being measured.

Streams tensor by tensor into 4 GB shards, so peak memory is one shard rather than the
whole 25 GB checkpoint -- eight of these run side by side on one node.
"""
import argparse
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

SUFFIX = ".inner."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the checkpoint's model.safetensors")
    ap.add_argument("--out", required=True, help="serving directory to write into")
    ap.add_argument("--shard-bytes", type=int, default=4 * 1024**3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    shard, shard_bytes, shard_idx, index, dropped, kept = {}, 0, 0, {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, os.path.join(args.out, fn), metadata={"format": "pt"})
        for k in shard:
            index[k] = fn
        print(f"  wrote {fn} ({len(shard)} tensors, {shard_bytes/1e9:.2f} GB)", flush=True)
        shard, shard_bytes = {}, 0

    with safe_open(args.src, framework="pt") as f:
        for name in f.keys():
            if SUFFIX in name:
                # Verified precondition: the plain counterpart must exist, otherwise
                # dropping this tensor would silently remove a real weight.
                if name.replace(SUFFIX, ".") not in f.keys():
                    raise SystemExit(f"ERROR: {name} has no plain counterpart; refusing "
                                     f"to drop it")
                dropped += 1
                continue
            t = f.get_tensor(name)
            shard[name] = t
            shard_bytes += t.numel() * t.element_size()
            kept += 1
            if shard_bytes >= args.shard_bytes:
                flush()
    flush()

    renamed = {}
    for i in range(1, shard_idx + 1):
        old = f"model-{i:05d}.safetensors"
        new = f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        os.rename(os.path.join(args.out, old), os.path.join(args.out, new))
        renamed[old] = new
    index = {k: renamed[v] for k, v in index.items()}
    nbytes = sum(os.path.getsize(os.path.join(args.out, f)) for f in set(index.values()))
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": nbytes}, "weight_map": index}, fh, indent=2)

    print(f"kept {kept} tensors, dropped {dropped} '.inner' masters -> "
          f"{shard_idx} shards, {nbytes/1e9:.1f} GB in {args.out}", flush=True)


if __name__ == "__main__":
    main()
