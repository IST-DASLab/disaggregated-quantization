"""The projection shapes every measurement is defined over.

Was also a benchmark harness; the sweep and the synthetic per-model decode estimate are
gone, superseded by the real vLLM numbers in `../../vllm_serve.py` -- that estimate timed
each projection in isolation and weighted it by how often a model contains it, which
excludes attention, norms, lm_head and launch overhead, and its weighting choice (unfused
q/k/v vs the merged QKV vLLM actually runs) moved the answer by up to 0.4x.

What is left is what `tune_cuda.py` needs: the model tables, the shape enumeration, and a
setup helper.
"""

import torch

from .format import LLOYD21, LLOYD43, pack_from_weight

BYTES_BF16 = 2.0
BYTES_LLOYD43 = 3 / 8 + 1 / 16
CEILING = BYTES_BF16 / BYTES_LLOYD43

# Straight out of the published configs (checked against the local HF cache), not
# approximated. An earlier hand-written shape list had (5120, 2560) for Qwen3-4B, which is
# not a projection that model has.
QWEN3_MODELS = {
    "0.6B": dict(hidden=1024, inter=3072,  layers=28, heads=16, kv=8, head_dim=128),
    "1.7B": dict(hidden=2048, inter=6144,  layers=28, heads=16, kv=8, head_dim=128),
    "4B":   dict(hidden=2560, inter=9728,  layers=36, heads=32, kv=8, head_dim=128),
    "8B":   dict(hidden=4096, inter=12288, layers=36, heads=32, kv=8, head_dim=128),
}

# Gemma 3, same treatment, from the published text configs (the 4b/12b numbers are the
# `text_config` of the multimodal checkpoint; the vision tower is not quantized here).
#
# Two things differ from Qwen3 and both show up in the shapes. head_dim is 256, twice
# Qwen3's, so q/k/v are wider than `hidden` would suggest -- 270m projects 640 -> 1024 for
# q, i.e. the attention block is bigger than the hidden size, which is unusual. And the
# small models are aggressively grouped: kv=1 means ONE key/value head, so the k and v
# projections are (256, 640) -- tiny, launch-bound shapes that flatter nothing.
GEMMA3_MODELS = {
    "270M": dict(hidden=640,  inter=2048,  layers=18, heads=4,  kv=1, head_dim=256),
    "1B":   dict(hidden=1152, inter=6912,  layers=26, heads=4,  kv=1, head_dim=256),
    "4B":   dict(hidden=2560, inter=10240, layers=34, heads=8,  kv=4, head_dim=256),
    "12B":  dict(hidden=3840, inter=15360, layers=48, heads=16, kv=8, head_dim=256),
}

FAMILIES = {"qwen3": QWEN3_MODELS, "gemma3": GEMMA3_MODELS}


def model_linears(cfg: dict) -> dict[tuple[int, int], int]:
    """{(N, K): count} over every nn.Linear in one model that QAD actually quantizes.

    A Qwen3 decoder layer has separate q/k/v/o and gate/up/down projections -- HF does not
    fuse them, and `quantizers.blocked.replace_linears` swaps each one individually.
    `lm_head` is in that function's `skip` list, so it is excluded here too; including it
    would flatter the estimate, since its (151936, hidden) weight is enormous and read
    once per token.
    """
    h, i, L = cfg["hidden"], cfg["inter"], cfg["layers"]
    q, kv = cfg["heads"] * cfg["head_dim"], cfg["kv"] * cfg["head_dim"]
    per_layer = [(q, h), (kv, h), (kv, h), (h, q), (i, h), (i, h), (h, i)]
    out: dict[tuple[int, int], int] = {}
    for shape in per_layer:
        out[shape] = out.get(shape, 0) + L
    return out


def model_linears_fused(cfg: dict) -> dict[tuple[int, int], int]:
    """{(N, K): count} with q/k/v fused into one projection and gate/up into another.

    This is what a serving stack actually runs: three separate GEMVs over the same x are
    one taller GEMV, which at batch 1 is strictly better -- same bytes, a third of the
    launches. These shapes are in the tuned dispatch table for that reason, even though
    nothing in this package fuses on its own.
    """
    h, i, L = cfg["hidden"], cfg["inter"], cfg["layers"]
    q, kv = cfg["heads"] * cfg["head_dim"], cfg["kv"] * cfg["head_dim"]
    per_layer = [(q + 2 * kv, h), (h, q), (2 * i, h), (h, i)]
    out: dict[tuple[int, int], int] = {}
    for shape in per_layer:
        out[shape] = out.get(shape, 0) + L
    return out


# (N, K) of every distinct projection across those models.
QWEN3_SHAPES = sorted({s for c in QWEN3_MODELS.values() for s in model_linears(c)})

# ... and with the fusions a serving stack would apply. Tuned as well, since the fused
# shapes are the ones the e2e benchmark actually calls.
QWEN3_SHAPES_FUSED = sorted({s for c in QWEN3_MODELS.values()
                             for s in model_linears_fused(c)})

# The dispatch table is tuned over EVERY family's shapes, not just Qwen3's. An untuned
# shape falls back to auto_config's heuristic, which is decent but not measured -- and
# Gemma 3's shapes are unlike Qwen3's (K=640, and 256-row kv projections), so leaving them
# to the heuristic would understate the format on exactly the models being added.
TUNE_SHAPES = sorted({s for models in FAMILIES.values() for c in models.values()
                      for s in (*model_linears(c), *model_linears_fused(c))})


def _setup(N, K, device="cuda", fmt=None):
    torch.manual_seed(0)
    w = torch.randn(N, K, device=device) * 0.02
    x = torch.randn(K, device=device, dtype=torch.bfloat16)
    packed, bs, gs = pack_from_weight(w, fmt=fmt)
    return x, w.to(torch.bfloat16), packed, bs, gs
