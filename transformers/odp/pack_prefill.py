"""Pack an HF checkpoint into ODP prefill blocks: one file per transformer layer.

    python tools/odp/pack_prefill.py <hf-model-dir> <out-dir>

Only the transformer stack's LINEAR PROJECTIONS are written. Everything else -- the
embedding, the output head, every norm, conv1d, A_log, dt_bias -- is SHARED with decode
and comes from the GGUF at run time. That is the split qad/training/ trains under (see
docs/ODP.md); `--full-disag`, which would also dualize embeddings/norms/head, is a
different and deprecated mode we do not implement.

WHY NOTHING IS FUSED IN THE BLOCK FILE.

A fused projection has to be sliced by the runtime, and neither half of that is free in
ggml. Slicing the fused RESULT needs a ggml_cont, because a row slice of a [n, tokens]
tensor is strided and the consumers want contiguous -- that duplicates the fused output,
1088 MiB for gate_up at 8192 tokens, to produce tensors an unfused path writes directly.
Slicing the WEIGHT instead looks right and is not, so each projection is packed apart.
The 4-bit values are identical either way.

HISTORICAL NOTE ON FUSION, AND ITS LIMIT. llama.cpp keeps q/k/v and gate/up as separate tensors because
batch-one decode is bandwidth-bound and there is nothing to gain from one big GEMM.
Prefill is the opposite case, so those are packed FUSED.

What may be fused is decided by the CHECKPOINT, not by us. NVFP4 has one global scale per
tensor, so concatenating two projections is only exact if they already share it.
`qad/quantizers/blocked.py::link_fused_groups` deliberately gives q/k/v one shared scale
and gate/up another, precisely so an engine that fuses them can collapse the scales with
.max() without loss. It does NOT group in_proj_qkv with in_proj_z -- measured on the
trained checkpoint their global scales differ by 1.8x -- so fusing those two would
silently rescale one of them. They stay separate.

    full_attention     qkv   5120 -> 14336   (q,k,v share a global scale)
    linear_attention   qkv   5120 -> 10240   and z 5120 -> 6144, SEPARATE
    both               gate_up 5120 -> 34816 (gate,up share a global scale)
    both               down  17408 -> 5120,  out 6144 -> 5120

in_proj_a / in_proj_b (the per-head gating projections, 5120 x 48 each) are packed
UNQUANTIZED, as bf16. That matches nvfp4_linear.py's `skip=("lm_head", "in_proj_ba")`:
quantizing a 96-column matrix saves nothing measurable and costs accuracy.

ACTIVATION SCALE. Static, exactly as the benchmark: one calibrated amax per projection,
never a reduction over the live activation. `nvfp4_linear.py` uses a fixed ACT_AMAX=10.0
stand-in rather than a calibration pass, and this reproduces that -- so the LATENCY is
what a calibrated model would pay, and the ACCURACY is not something to draw conclusions
from. Real calibration would change only the numbers in the header, not the format.
"""

import argparse
import numpy as np
import json
import os
import sys
from pathlib import Path

from nvfp4_ggml import to_ggml_nvfp4

import torch

FP4_MAX = 6.0
E4M3_MAX = 448.0


# Wire format, mirrored by tools/odp/odp-blocks.h. The C side reads these by explicit byte
# offset rather than by casting a struct, so the two cannot drift over padding rules.

# Projection ids, mirrored by tools/odp/odp-blocks.h. Numbers, not names, so the runtime
# does not parse strings per layer.
# One id per tensor llama.cpp asks for. NOT fused: see the note on slicing below.
P_Q       = 0    # full attention: q (with its gate) / linear attention: in_proj_qkv
P_OUT     = 1
P_GATE    = 2
P_DOWN    = 3
P_BA      = 4    # unquantized, linear-attention layers only
P_Z       = 5    # in_proj_z, linear-attention layers only
P_K       = 6
P_V       = 7
P_UP      = 8

# Tensor names in the emitted GGUF. The point of writing real GGUF is that `gguf_dump` and
# the Hugging Face viewer show what a block contains, so these are words rather than ids.
P_NAME = {P_Q: "q", P_OUT: "out", P_GATE: "gate", P_DOWN: "down",
          P_BA: "ba", P_Z: "z", P_K: "k", P_V: "v", P_UP: "up"}

KIND_LINEAR = 0
KIND_FULL   = 1


class Shards:
    """Lazy per-shard safetensors reader.

    Loads one shard at a time and drops it: the 27B checkpoint is 47.6 GB of bf16 and
    slurping it whole to touch a handful of tensors per layer would not fit alongside the
    quantizer's CUDA context.
    """

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        index = model_dir / "model.safetensors.index.json"
        self.single = None
        if index.exists():
            self.map = json.load(open(index))["weight_map"]
        else:
            # A trained export is one file, not a shard set. safe_open reads a single
            # tensor without materializing the other 25 GB.
            from safetensors import safe_open
            self.single = safe_open(model_dir / "model.safetensors", framework="pt")
            self.map = {}
        self._cache_name = None
        self._cache = None

    def get(self, key: str) -> torch.Tensor:
        if self.single is not None:
            return self.single.get_tensor(key)
        shard = self.map[key]
        if shard != self._cache_name:
            from safetensors.torch import load_file
            self._cache = load_file(self.dir / shard)
            self._cache_name = shard
        return self._cache[key]


def reorder_v_heads(t: torch.Tensor, dim: int, n_k_heads: int, v_per_k: int,
                    head_dim: int) -> torch.Tensor:
    """HF's grouped V-head order -> ggml's tiled order.

    Copied from conversion/qwen.py's _LinearAttentionVReorderBase, because the GGUF
    converter applies it to every GDN projection and llama.cpp's graph then assumes it.
    HF stores V heads grouped by K head, [G0_v0..v{r-1}, G1_v0..., ...]; ggml's binary ops
    broadcast in tiled order, [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...], which lets a plain
    ggml_repeat replace an interleaved one.

    Skipping this is invisible: the shapes are unchanged, every assert passes, and the
    model just produces confident nonsense.
    """
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [n_k_heads, v_per_k, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


# ---------------------------------------------------------------- trained checkpoint path

def is_quantized_checkpoint(model_dir: Path) -> bool:
    """True if this export is compressed-tensors NVFP4 rather than plain bf16."""
    cfg = json.load(open(model_dir / "config.json"))
    return (cfg.get("quantization_config") or {}).get("format") == "nvfp4-pack-quantized"


def reorder_rows_packed(packed: torch.Tensor, scale: torch.Tensor, n_k: int, v_per_k: int,
                        head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """V-head reorder along the OUTPUT dimension, on already-packed weights.

    Rows are output features and the 4-bit packing runs along K, so a row permutation
    never touches a nibble: permute the rows of `packed` and the rows of `scale`
    identically and the result is exact.
    """
    n = packed.shape[0]
    idx = torch.arange(n).reshape(n_k, v_per_k, head).permute(1, 0, 2).reshape(n)
    return packed[idx].contiguous(), scale[idx].contiguous()


def reorder_cols_packed(packed: torch.Tensor, scale: torch.Tensor, n_k: int, v_per_k: int,
                        head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The same reorder along the INPUT dimension, for out_proj.

    This one is packed 2-per-byte and grouped 16-per-scale, so an arbitrary column
    permutation would need nibble surgery. This permutation is not arbitrary: it moves
    whole heads, and head_v_dim is 128 -- a multiple of both 2 and 16. So it is a
    permutation of 64-byte blocks and of 8-entry scale blocks, and stays exact.
    """
    assert head % 16 == 0, "head_v_dim must be a multiple of the 16-element scale group"
    kb = packed.shape[1] // (head // 2)          # blocks of `head` columns, in bytes
    sb = scale.shape[1] // (head // 16)          # the same blocks, in scale entries
    assert kb == n_k * v_per_k and sb == kb
    idx = torch.arange(kb).reshape(n_k, v_per_k).permute(1, 0).reshape(kb)
    pk = packed.reshape(packed.shape[0], kb, head // 2)[:, idx].reshape(packed.shape)
    sc = scale.reshape(scale.shape[0], sb, head // 16)[:, idx].reshape(scale.shape)
    return pk.contiguous(), sc.contiguous()


def pack_layer_trained(sh, il: int, is_linear: bool, gdn: dict, *, validate_groups: bool = True) -> list:
    """One layer from a TRAINED compressed-tensors NVFP4 export.

    Nothing is re-quantized here. The 4-bit values are the trained ones; all this does is
    concatenate the fusion groups (exact, because they share a global scale), apply the
    GDN V-head reorder the GGUF converter applies, and relay the values into ggml's
    block_nvfp4 (see nvfp4_ggml.py), which is lossless.
    """
    p = f"model.language_model.layers.{il}."
    entries = []

    def add(pid, packed, scale, w_gs, x_gs):
        n, k = packed.shape[0], packed.shape[1] * 2
        entries.append((pid, n, k,
                        packed.cpu().contiguous().numpy(),
                        scale.cpu().contiguous().view(torch.uint8).numpy(),
                        float(w_gs), float(x_gs)))

    def grp(names):
        """Concatenate a fusion group, refusing to do so if the scales disagree."""
        gs = [float(sh.get(p + n + ".weight_global_scale")) for n in names]
        if len(set(gs)) != 1:
            raise SystemExit(
                f"layer {il}: {names} do not share a weight_global_scale ({gs}); "
                "concatenating them would rescale one of the two")
        xs = [float(sh.get(p + n + ".input_global_scale")) for n in names]
        assert len(set(xs)) == 1, f"layer {il}: {names} disagree on input_global_scale"
        pk = torch.cat([sh.get(p + n + ".weight_packed") for n in names], dim=0)
        sc = torch.cat([sh.get(p + n + ".weight_scale")  for n in names], dim=0)
        return pk, sc, gs[0], xs[0]

    if is_linear:
        nk, vpk, hkd, hvd = gdn["n_k"], gdn["v_per_k"], gdn["head_k"], gdn["head_v"]

        pk, sc, w, x = grp(["linear_attn.in_proj_qkv"])
        qd = hkd * nk                                    # q and k are not reordered
        vp, vs = reorder_rows_packed(pk[2 * qd:], sc[2 * qd:], nk, vpk, hvd)
        add(P_Q, torch.cat([pk[:2 * qd], vp], 0), torch.cat([sc[:2 * qd], vs], 0), w, x)

        pk, sc, w, x = grp(["linear_attn.in_proj_z"])
        pk, sc = reorder_rows_packed(pk, sc, nk, vpk, hvd)
        add(P_Z, pk, sc, w, x)

        pk, sc, w, x = grp(["linear_attn.out_proj"])
        pk, sc = reorder_cols_packed(pk, sc, nk, vpk, hvd)
        add(P_OUT, pk, sc, w, x)
    else:
        # Still checked as a GROUP: q/k/v must share a global scale even though they are
        # now stored apart, because that is what the trained checkpoint guarantees and what
        # would make a future re-fusion exact.
        if validate_groups:
            grp(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])
        add(P_Q, *grp(["self_attn.q_proj"]))
        add(P_K, *grp(["self_attn.k_proj"]))
        add(P_V, *grp(["self_attn.v_proj"]))
        add(P_OUT, *grp(["self_attn.o_proj"]))

    if validate_groups:
        grp(["mlp.gate_proj", "mlp.up_proj"])
    add(P_GATE, *grp(["mlp.gate_proj"]))
    add(P_UP,   *grp(["mlp.up_proj"]))
    add(P_DOWN, *grp(["mlp.down_proj"]))

    if is_linear:
        nk, vpk = gdn["n_k"], gdn["v_per_k"]
        a = reorder_v_heads(sh.get(p + "linear_attn.in_proj_a.weight"), 0, nk, vpk, 1)
        b = reorder_v_heads(sh.get(p + "linear_attn.in_proj_b.weight"), 0, nk, vpk, 1)
        ba = torch.cat([b, a], dim=0).to(torch.bfloat16).contiguous()
        entries.append((P_BA, ba.shape[0], ba.shape[1],
                        ba.view(torch.uint16).numpy(), None, 0.0, 0.0))

    return entries


def write_block_gguf(path, il: int, is_linear: bool, entries) -> int:
    """One layer as a real GGUF of NVFP4 tensors, which ggml multiplies with its own kernels.

    The per-tensor global scale goes in the KV rather than the data: it is a scalar, matmul is
    linear, so the runtime applies it to the result. block_nvfp4 has nowhere to put it anyway.
    """
    import gguf

    w = gguf.GGUFWriter(path, "odp", endianess=gguf.GGUFEndian.LITTLE)
    w.add_uint32("odp.layer", il)
    w.add_uint32("odp.kind", KIND_LINEAR if is_linear else KIND_FULL)

    nbytes = 0
    for pid, n, k, packed, scale, w_gs, x_gs in entries:
        del n, k   # the shapes travel with the arrays
        name = P_NAME[pid]
        if pid == P_BA:
            # numpy has no bfloat16, so hand over the raw 16-bit pattern and name the type.
            w.add_tensor(f"{name}.weight", packed, raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            # gguf takes the byte shape and serializes it reversed, so (N, row_bytes) lands
            # as ne = [K, N], which is what ggml_mul_mat wants for a weight.
            blob = to_ggml_nvfp4(packed, scale)
            w.add_tensor(f"{name}.weight", blob, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
            # What the runtime multiplies the result by. Stored inverted so the hot path is
            # a multiply and cannot divide by a zero that a corrupt file could carry.
            w.add_float32(f"odp.scale.{name}", float(1.0 / w_gs))
            # The STATIC activation amax the checkpoint was calibrated with, in the form the
            # kernel wants: amax/(6*448), which is 1/input_global_scale. Without it ggml takes
            # each row's own amax, and this model was trained against the fixed one.
            w.add_float32(f"odp.act_scale.{name}", float(1.0 / x_gs))
            nbytes += blob.nbytes
        nbytes += 0 if pid != P_BA else packed.nbytes

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return nbytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=0, help="0 = all")
    ap.add_argument("--name", default="Qwen3.8-27B-ODP",
                    help="filename stem; the tag is appended to it")
    ap.add_argument("--tag", required=True,
                    help="decode quant these prefill weights were trained against, e.g. IQ2_S. "
                         "Becomes the variant tag llama.cpp selects with --odp-blocks repo:TAG. "
                         "Omit to write the legacy block_NNNN.bin layout.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not is_quantized_checkpoint(model_dir):
        raise SystemExit(
            f"{model_dir} is not a compressed-tensors NVFP4 export. The RTN path is gone: it "
            "existed to feed a hand-written kernel that ggml now replaces, and it was the only "
            "reason this tool needed vLLM. Quantize with the research pipeline instead.")
    print("source: trained NVFP4 export (nothing is re-quantized)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(model_dir / "config.json"))
    tcfg = cfg.get("text_config", cfg)
    n_layer = tcfg["num_hidden_layers"]
    interval = tcfg.get("full_attention_interval", 4)
    if args.layers:
        n_layer = min(n_layer, args.layers)

    n_k = tcfg["linear_num_key_heads"]
    n_v = tcfg["linear_num_value_heads"]
    gdn = {"n_k": n_k, "v_per_k": n_v // n_k,
           "head_k": tcfg["linear_key_head_dim"], "head_v": tcfg["linear_value_head_dim"]}
    print(f"GDN: {n_k} key heads, {n_v} value heads, head_v={gdn['head_v']} "
          f"-> V reorder {gdn['v_per_k']}x")

    sh = Shards(model_dir)
    total = 0
    for il in range(n_layer):
        is_linear = ((il + 1) % interval) != 0
        entries = pack_layer_trained(sh, il, is_linear, gdn)
        # llama.cpp's OWN split naming. The payload is ours, but the NAME is what its
        # Hugging Face machinery reads: the tag comes off the filename, one repo can then
        # hold every bitwidth, and `common_download_remove("repo:TAG")` evicts exactly one
        # of them. Nothing ODP-specific is needed on the cache side.
        path = out_dir / f"{args.name}-{args.tag}-{il + 1:05d}-of-{n_layer:05d}.gguf"
        n = write_block_gguf(str(path), il, is_linear, entries)
        total += n
        print(f"  block {il:3d}  {'linear' if is_linear else 'full  '}  "
              f"{n/2**20:8.1f} MiB", flush=True)

    print(f"\n{n_layer} blocks, {total/2**30:.2f} GiB in {out_dir}")
    if args.tag:
        print(f"\nuse locally:  --odp-blocks {out_dir}/{args.name}-{args.tag}-00001-of-{n_layer:05d}.gguf")
        print(f"publish:      tools/odp/upload_blocks.py --repo <user>/<repo> --blocks {out_dir}")
    else:
        print("\nno --tag given: wrote the legacy block_NNNN.bin layout, which cannot be "
              "published to a shared repo (nothing carries the variant name)")


if __name__ == "__main__":
    main()
