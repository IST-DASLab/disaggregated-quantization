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
from .dual import (apply_nvfp4prefill, apply_nvfp4decode,
                   apply_nvfp4lloyd43shared, apply_nvfp4lloyd43split,
                   apply_nvfp4lloyd43upcast, apply_nvfp4lloyd21upcast,
                   apply_nvfp4lloyd43upcastboth, apply_nvfp4lloyd21upcastboth,
                   apply_nvfp4lloyd21split,
                   apply_nvfp4pdshared, apply_nvfp4pdsplit,
                   prefill_mask_from_labels, quant_phase)
from .frozen_decode import (apply_nvfp4frozendec, load_frozen_decode,
                            NVFP4FrozenDecodeLinear, SKIP_LINEARS)
from .lloyd import apply_lloyd21, apply_lloyd3bit, apply_lloyd43
from .nvfp4 import apply_nvfp4, apply_nvfp4a16, calibrate_nvfp4
from .quest import apply_quest2bit, apply_quest3bit, apply_quest4bit
from .ste import apply_ste2bit, apply_ste3bit, apply_ste4bit


REGISTRY: dict = {
    "fp8": {
        "apply":        lambda model, **_: apply_fp8_linear(model),
        "param_groups": None,          # all params equally
        "post_update":  None,          # FP8 recomputes each forward via STE
        "defaults":     {},
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
    "nvfp4frozendec": {
        # NVFP4 W4A4 prefill trained against a FROZEN EXTERNAL decode checkpoint.
        #
        # Unlike nvfp4pdsplit, the decode half is not a second master: it is a black box
        # someone else quantized (the dequantized GSQ-RCO `*-bf16` models), loaded as-is,
        # never updated, never exported. So this costs 2x linear FLOPs like a split
        # format but only 1x master/optimizer memory, and the exported checkpoint is the
        # prefill half alone -- `variants` is ["prefill"], not the usual pair.
        #
        # `decode_model` is in defaults and therefore md5'd into the checkpoint tag, so
        # runs against different black boxes (IQ2_S vs IQ3_S ...) cannot collide. It has
        # no default value on purpose: there is no sensible one, and silently training
        # against the wrong decode model is the expensive mistake here.
        "apply":        apply_nvfp4frozendec,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16, "decode_model": "",
                         "skip_linears": SKIP_LINEARS},
        "export":       "compressed_tensors",
        "variants":     ["prefill"],
    },
    "nvfp4decode": {
        # PHASE-ISOLATION ABLATION. BF16 prefill, NVFP4 (W4A4) decode: one shared
        # master, only the decode phase quantized. Measures what quantizing DECODE
        # costs. Not deployable -- a BF16 phase needs BF16 weights resident, which is
        # the cost these formats exist to avoid.
        "apply":        apply_nvfp4decode,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4prefill": {
        # The mirror image: NVFP4 (W4A4) prefill, BF16 decode. Measures what
        # quantizing PREFILL costs. Both names state which phase runs NVFP4, so
        # nvfp4prefill quantizes PREFILL and nvfp4decode quantizes DECODE.
        "apply":        apply_nvfp4prefill,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4lloyd43shared": {
        # NVFP4 W4A4 prefill + Lloyd43 W3A16 decode, ONE master. Each phase gets the
        # format its bottleneck wants: prefill is compute-bound so quantizing
        # activations pays; decode is memory-bound on weights, so 3 bits of WEIGHT pays
        # and activations stay bf16. The halves differ in BOTH the grid (E2M1 vs the
        # 0/6-pinned Lloyd43) and the normalisation: Lloyd43 is signed (its grid is
        # asymmetric, so the block's extreme must land on the pinned +6.0), while the
        # NVFP4 half must stay unsigned because its block scale ships as UE4M3.
        # Mixed export: prefill is packed compressed-tensors, decode is pseudo-quantized
        # bf16 (no 3-bit kernel exists), exactly as lloyd3bit ships.
        "apply":        apply_nvfp4lloyd43shared,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4lloyd43split": {
        # As above but two masters, both initialised from the BF16 base and trained
        # separately, so the prefill and decode checkpoints genuinely diverge. Costs 2x
        # the linear FLOPs and 2x master/optimizer memory.
        "apply":        apply_nvfp4lloyd43split,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4lloyd43upcast": {
        # One master, but prefill is DERIVED from decode: the master is quantized to
        # Lloyd43, then that dequantized 3-bit weight is re-quantized to NVFP4.
        #   shared: W -> NVFP4        upcast: W -> Lloyd43 -> NVFP4
        # So only the 3-bit codebook has to be stored -- the FP4 prefill weight is a
        # pure function of it. Costs a second quantization error stacked on the 3-bit
        # one (Lloyd43's grid points are not E2M1-representable); measuring that
        # against nvfp4lloyd43shared is the point.
        "apply":        apply_nvfp4lloyd43upcast,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4lloyd43upcastboth": {
        # The upcast WITHOUT disaggregation: master -> Lloyd43 -> NVFP4, and BOTH phases
        # serve that NVFP4 weight. One checkpoint, one kernel, fast prefill and fast
        # decode -- the baseline the disaggregated upcast formats have to beat.
        # Against nvfp4lloyd43upcast it isolates what serving the 3-bit half on decode is
        # worth; against plain nvfp4, what the 3-bit bottleneck costs.
        "apply":        apply_nvfp4lloyd43upcastboth,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
    },
    "nvfp4lloyd21upcastboth": {
        # Same control at 2 bits.
        "apply":        apply_nvfp4lloyd21upcastboth,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
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
    "lloyd43": {
        # Same layer and pseudo-quantized export as lloyd3bit, on the 3-bit grid that is
        # MSE-optimal subject to 0.0 and +6.0 being grid points. The signed block scale
        # maps each block's max-abs element to exactly +6, so pinning 6.0 makes the
        # block outlier exact (lloyd3bit's top level is 5.788, a fixed 0.212 error on
        # every block), and pinning 0.0 gives a real flush-to-zero, which lloyd3bit
        # cannot express at all -- it straddles zero at -0.339 / +0.927.
        # Costs ~3% relative MSE on a Gaussian; see quantizers/grids.py for the numbers.
        "apply":        apply_lloyd43,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16, "grid": "lloyd43"},
        "export":       "dequantized",
    },
    "nvfp4lloyd21split": {
        # NVFP4 W4A4 prefill + Lloyd21 W2A16 decode, TWO masters that diverge -- the
        # disaggregated counterpart of nvfp4lloyd21upcast. The upcast gets 2-bit storage
        # by construction (prefill is a pure function of the decode weight); this instead
        # spends 2x master/optimizer memory and 2x linear FLOPs to let the phases
        # separate, and the comparison is what that separation is worth.
        # NOTE: does NOT fit at 8B (~215 GiB/GPU vs 179 available) -- a second master
        # drags its own gradient, _wq and optimizer shard. See qad-8b-memory-budget.
        "apply":        apply_nvfp4lloyd21split,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "nvfp4lloyd21upcast": {
        # The upcast format at 2 BITS: one master -> Lloyd21 (W2A16) decode -> that
        # dequantized weight re-quantized to NVFP4 for prefill. Same machinery as
        # nvfp4lloyd43upcast with a 4-level LUT instead of 8, so storage is 2 bits per
        # weight and the packed FP4 prefill checkpoint stays a pure function of it.
        # Expect a larger upcast penalty than the 3-bit version: the decode weight it
        # upcasts from already carries ~0.108 relative MSE (vs ~0.022 for lloyd43).
        "apply":        apply_nvfp4lloyd21upcast,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16},
        "export":       "compressed_tensors",
        "variants":     ["prefill", "decode"],
    },
    "lloyd21": {
        # The lloyd43 construction at 2 BITS: MSE-optimal subject to 0.0 and +6.0 being
        # grid points, four levels instead of eight. Same layer, same signed block scale,
        # same pseudo-quantized (bf16) export -- no 2-bit kernel exists either, so the
        # checkpoint holds dequantized weights and vLLM serves it as an ordinary model.
        # The grid is asymmetric (2 positive levels, 1 negative) for the same reason
        # lloyd43's is: the signed normalisation puts every block's extreme at exactly
        # +6, so the positive tail is what has to be represented well.
        "apply":        apply_lloyd21,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"block_size": 16, "grid": "lloyd21"},
        "export":       "dequantized",
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


# The runtime used to drive a HOMOGENEOUS checkpoint through dual inference (the
# --dual ablation in eval_transformers.py). It lives here because it is a property of
# the registry, not of any one caller: picking it by hardcoding a quantizer name in an
# eval script is how such a script ends up silently wrong when formats are renamed.
DEFAULT_DUAL_RUNTIME = "nvfp4pdshared"


def is_dual(name: str) -> bool:
    """True if this quantizer serves prefill and decode from separate checkpoints.

    Derived from the registry, never from the NAME. An earlier version of this test
    was `name.startswith("nvfp4pd")`, which silently excluded nvfp4prefill and
    nvfp4decode the moment they were added -- they would have been evaluated as
    homogeneous formats, quietly measuring the wrong thing.
    """
    return len(variants(name)) > 1


def variants(name: str) -> list:
    """Checkpoint variants this quantizer emits per step, WITHOUT building a model.

    The layer class is the source of truth (QuantizedLinear.export_variants); this
    is the lookup for callers that only have a quantizer name — eval scripts
    resolving a --variant path before anything is loaded.
    """
    return REGISTRY[name].get("variants") or [None]


__all__ = [
    "REGISTRY", "build_quantizer_params", "uses_compressed_tensors", "variants",
    "is_dual", "DEFAULT_DUAL_RUNTIME",
    "QuantizedLinear", "post_update_all", "calibrate_nvfp4",
    "quant_phase", "prefill_mask_from_labels",
]
