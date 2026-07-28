"""Quantizer registry.

Each entry maps a CLI name to:
  apply(model, **params)            – replace linears in-place
  param_groups(model, lr, **params) – DistAdamW param groups (None -> one group over all)
  post_update                       – callable(model, step, total_steps) or None
  defaults                          – hyperparameters merged with --quantizer-params
  export                            – "compressed_tensors" (real quantized checkpoint)
                                      or "dequantized" (pseudo-quant: plain bf16 HF model)
  stats                             – optional callable(model) -> dict of scalars to log

NOTE: `defaults` is hashed into the checkpoint tag (see build_quantizer_params), so
changing a defaults dict renames every checkpoint/eval directory for that method.
Current hashes: {} -> 99914b93, {"groupsize": 128} -> 1a17550c.
"""

import hashlib
import json

from .base import QuantizedLinear, post_update_all
from .fp8 import apply_fp8_linear
from .dual import (apply_nvfp4pdshared, apply_nvfp4pdsplit, prefill_mask_from_labels,
                   quant_phase)
from .gsq import apply_gsq2bit, apply_gsq3bit, gsq_param_groups
from .gsq_lloyd import apply_gsqlloyd3bit, assignment_stats
from .lloyd import apply_lloyd3bit
from .nvfp4 import apply_nvfp4, apply_nvfp4a16, calibrate_nvfp4
from .quest import apply_quest2bit, apply_quest3bit, apply_quest4bit
from .ste import apply_ste2bit, apply_ste3bit, apply_ste4bit

_GSQ_DEFAULTS = {
    "groupsize":   128,
    "std":         0.01,
    "strength":    6.0,
    "temp_start":  2.0,
    "temp_end":    0.05,
    "scale_start": 100.0,
    "scale_end":   500.0,
}

REGISTRY: dict = {
    "fp8": {
        "apply":        lambda model, **_: apply_fp8_linear(model),
        "param_groups": None,          # all params equally
        "post_update":  None,          # FP8 recomputes each forward via STE
        "defaults":     {},
        "export":       "dequantized",
    },
    "gsq2bit": {
        "apply":        apply_gsq2bit,
        "param_groups": gsq_param_groups,
        "post_update":  post_update_all,
        "defaults":     dict(_GSQ_DEFAULTS),
        "export":       "dequantized",
    },
    "gsq3bit": {
        "apply":        apply_gsq3bit,
        "param_groups": gsq_param_groups,
        "post_update":  post_update_all,
        "defaults":     dict(_GSQ_DEFAULTS),
        "export":       "dequantized",
    },
    "ste2bit": {
        "apply":        apply_ste2bit,
        "param_groups": None,          # single weight param, no split needed
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
        "export":       "dequantized",
    },
    "ste3bit": {
        "apply":        apply_ste3bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
        "export":       "dequantized",
    },
    "ste4bit": {
        "apply":        apply_ste4bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
        "export":       "dequantized",
    },
    "quest2bit": {
        "apply":        apply_quest2bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
        "export":       "dequantized",
    },
    "quest3bit": {
        "apply":        apply_quest3bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
        "export":       "dequantized",
    },
    "quest4bit": {
        "apply":        apply_quest4bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
        "export":       "dequantized",
    },
    "nvfp4": {
        # W4A4: NVFP4 fake-quant on both weights and activations (STE), block=16.
        "apply":        apply_nvfp4,
        "param_groups": None,          # single weight param per layer
        "post_update":  post_update_all,
        "defaults":     {},
        "export":       "compressed_tensors",
    },
    "nvfp4a16": {
        # W4A16: NVFP4 weights only — activations stay bf16 (no input_global_scale).
        "apply":        apply_nvfp4a16,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {},
        "export":       "compressed_tensors",
    },
    "nvfp4pdshared": {
        # W4A4 on prefill positions, W4A16 on decode positions, ONE master weight.
        # Both exported checkpoints hold identical weights and differ only in the
        # config / input_global_scale; what differs is that the weights were trained
        # to serve both regimes.
        "apply":        apply_nvfp4pdshared,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4pdsplit": {
        # As above but with two master weights, both initialised from the BF16 base
        # and trained separately, so the prefill and decode checkpoints diverge.
        # Costs 2x linear FLOPs and 2x master/optimizer memory.
        "apply":        apply_nvfp4pdsplit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "lloyd3bit": {
        # Weight-only signed-Lloyd 3-bit; pseudo-quantized (no 3-bit LUT kernel),
        # so the checkpoint holds dequantized bf16 weights and vLLM serves it as
        # an ordinary bf16 model.
        "apply":        apply_lloyd3bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16, "grid": "lloyd"},
        "export":       "dequantized",
    },
    "gsqlloyd3bit": {
        # Same deployable format as lloyd3bit, optimized with GSQ (learned per-element
        # level assignment) instead of STE. Block scales stay continuous: FP32 log2
        # master, straight-through E4M3 in the forward.
        "apply":        apply_gsqlloyd3bit,
        "param_groups": gsq_param_groups,
        "post_update":  post_update_all,
        "defaults":     {
            "block_size":  16,
            "grid":        "lloyd",
            "std":         0.01,
            "strength":    6.0,
            "noise":       0.0,   # 0 => init is exactly lloyd3bit's round-to-nearest
            # absolute LRs — logits and log2 scale deltas are not in weight units
            "logit_lr":    1e-4,
            "scale_lr":    3e-6,
            # Lion: sign-based, so it cannot stall on the vanishing gradients the
            # saturating Gumbel relaxation produces. AdamW froze the logits outright.
            "optim":       "lion",
            "betas":       [0.9, 0.99],
            # trainable master params are fp32, as everywhere else in this codebase
            "logits_dtype": "fp32",
            "adam_eps":    1e-16,   # only used when optim="adamw"
            "temp_start":  2.0,
            "temp_end":    0.05,
            "scale_start": 100.0,
            "scale_end":   500.0,
        },
        "export":       "dequantized",
        "stats":        assignment_stats,
    },
}


def build_quantizer_params(name: str, overrides_json: str) -> tuple[dict, str]:
    """Merge --quantizer-params over the registry defaults.
    Returns (params, 8-char hash) — the hash goes into the checkpoint tag."""
    params = dict(REGISTRY[name]["defaults"])
    if overrides_json:
        params.update(json.loads(overrides_json))
    h = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return params, h


def uses_compressed_tensors(name: str) -> bool:
    """True if this quantizer exports a real quantized (compressed-tensors) checkpoint."""
    return REGISTRY[name]["export"] == "compressed_tensors"


def variants(name: str) -> list:
    """Checkpoint variants this quantizer emits per step, WITHOUT building a model.

    The layer class is the source of truth (QuantizedLinear.export_variants); this
    is the lookup for callers that only have a quantizer name — eval scripts
    resolving a --variant path before anything is loaded.
    """
    return REGISTRY[name].get("variants") or [None]


__all__ = [
    "REGISTRY", "build_quantizer_params", "uses_compressed_tensors", "variants",
    "QuantizedLinear", "post_update_all", "calibrate_nvfp4",
    "quant_phase", "prefill_mask_from_labels",
]
