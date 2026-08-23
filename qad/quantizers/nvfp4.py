"""NVFP4 (W4A4 and W4A16) fake-quantization for QAD.

NVFP4 is NVIDIA's 4-bit block floating-point format, served by vLLM on Blackwell:
E2M1 4-bit elements, blocks of 16 along the contraction dim, FP8-E4M3 per-block
scales and one FP32 per-tensor global scale. The blocking / two-level-scaling /
fused-group-scale-sharing machinery lives in blocked.py — this module adds the
E2M1 grid, the activation path, and the FP4 bit-packing used by the real export.

  nvfp4     – W4A4:  weights AND activations quantized
  nvfp4a16  – W4A16: weights only, activations stay bf16

Activation quant (W4A4) uses a STATIC per-tensor global scale from a running-max
observer, matching what vLLM does at inference: it calls
`scaled_fp4_quant(x, input_global_scale)` with the calibrated scale from the
checkpoint and computes the per-block scales dynamically. Training therefore sees
exactly the inference-time scaling.

Export lives on the layer itself (NVFP4Linear.export_tensors / export_config /
load_tensors); export/save.py only assembles the files.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocked import (qlinear, BLOCK, GLOBAL_DEN, SCALE_REF, BlockScaledLinear,
                      blocked_quantize, replace_linears, ste, to_e4m3)
# torch.compile keys its cache on a guard signature, and e2m1_round is reached from
# several: fp32 weights under no_grad in post_update, activations in the forward, and
# the export packer. The default limit of 8 is below that count, and exceeding it makes
# dynamo fall back to EAGER PERMANENTLY -- paying compilation and losing the fusion.
# 32 is still bounded, which is what turns "compiles forever" into "falls back and keeps
# training" if the guards ever churn on something unexpected.
import torch._dynamo
for _limit in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _limit):
        setattr(torch._dynamo.config, _limit, 32)

from .grids import E2M1_BOUNDS, E2M1_DECODE, E2M1_LEVELS


@torch.compile(dynamic=True)
def e2m1_round(x: Tensor) -> Tensor:
    """Round to the nearest signed E2M1 magnitude (values >=5 clamp to 6).

    Kept as an explicit magnitude/midpoint table rather than a generic grid snap:
    this exact rounding is verified bit-identical to vLLM's kernels.

    COMPILED, and the boundary is chosen precisely. torch.bucketize returns int64, so
    eager this writes a temporary at TWICE the input's fp32 size, plus the gathered
    levels and the sign -- on gemma-3-12b's MLP down_proj activation (4 x 2048 x 15360 =
    125.8M elements) that is a 1007 MB int64 buffer and two 503 MB ones. Both 12b PP
    OOMs died right here. Fused, the indices never reach memory: measured peak on that
    tensor 3.81 -> 2.40 GiB.

    WHY THE DIVISION MUST STAY OUTSIDE. blocked_quantize calls rounder(xf / eff), so
    this function never sees a division -- and that is load-bearing, not incidental.
    Compiling an expression that contains `xf / eff` lets inductor rewrite it as
    `xf * eff.reciprocal()`, which shifts 12.3M elements by an ulp; 2 of them land on
    the far side of a bucketize boundary and quantize to a DIFFERENT E2M1 level.
    Measured, not feared: compiling nvfp4_quantize (which includes the division) changed
    exactly 2 elements per tensor by a full grid step, while compiling this
    division-free form is bit-identical at every shape tested. If a future change moves
    a division in here, the format changes silently.
    """
    levels = torch.tensor(E2M1_LEVELS, device=x.device, dtype=x.dtype)
    bounds = torch.tensor(E2M1_BOUNDS, device=x.device, dtype=x.dtype)
    return torch.sign(x) * levels[torch.bucketize(x.abs(), bounds)]


def e2m1_codes(x: Tensor) -> Tensor:
    """Map normalized values to 4-bit E2M1 codes (0..15):
    bit3 = sign, bits0-2 = magnitude index. code 8 (=-0) collapses to 0."""
    bounds = torch.tensor(E2M1_BOUNDS, device=x.device, dtype=torch.float32)
    mag = torch.bucketize(x.float().abs(), bounds).to(torch.uint8)  # 0..7
    code = mag | ((x < 0).to(torch.uint8) << 3)
    return torch.where(mag == 0, torch.zeros_like(code), code)


def nvfp4_quantize(x: Tensor, block: int = BLOCK, global_scale: Tensor | None = None,
                   signed: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Two-level NVFP4 fake-quant along the last dim.
    Returns (dequantized, block_scale_e4m3, global_scale).

    `signed` lets the per-block scale absorb the sign of the block's max-abs element.
    E2M1 is SYMMETRIC about zero, so this changes nothing about the representable set
    or the packed codes -- negating a block is the same as flipping every code's sign
    bit, and E4M3 block scales are signed, so the checkpoint holds it natively. It is
    off by default: flipping it would rename nothing but would change the bytes every
    existing nvfp4* checkpoint exports."""
    return blocked_quantize(x, e2m1_round, block, signed=signed, global_scale=global_scale)


def fake_quant_ste(x: Tensor, block: int, global_scale: Tensor | None) -> Tensor:
    """STE fake-quant for ACTIVATIONS, with the quantize itself under no_grad.

    ste(x, q) is `x + (q - x).detach()`: the forward value is q and the gradient goes
    straight to x, so q's autograd graph is DEAD -- nothing backward ever needs it.
    Building it anyway is what made this the peak-memory site of the whole forward.
    blocked_quantize materialises about six full-size temporaries (x.float(), xf/eff,
    the int64 bucketize output at TWICE fp32 size, the gathered levels, the sign, the
    result), and with requires_grad set every one of them is retained as a saved tensor
    until the graph is dropped. Under no_grad they are freed as the expression walks.

    Not an approximation: identical arithmetic, identical values, identical gradient
    (d/dx of the whole thing is 1 either way). Both 12b PP OOMs landed inside this call
    -- one on the 480 MiB fp32 copy, one on the 960 MiB int64 copy -- on the MLP
    down_proj input, 4 x 2048 x 15360 = 125.8M elements. Gradient checkpointing runs it
    twice per step, so the saving lands twice.
    """
    with torch.no_grad():
        q = nvfp4_quantize(x.detach(), block, global_scale)[0]
    return ste(x, q)


def pack_nvfp4_weight(w: Tensor, block: int = BLOCK, global_scale: Tensor | None = None,
                      signed: bool = False,
                      block_eff: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Encode a weight matrix into the real NVFP4 checkpoint tensors, using the
    SAME scales as nvfp4_quantize so the packed weight is bit-identical to the
    fake-quantized `_wq` the model trained with.  `global_scale` should be the
    shared fused-group scale (see blocked.link_fused_groups).

    Returns (packed_uint8 [O, K//2], weight_scale float8_e4m3fn [O, K//block],
             weight_scale_2 float32 [1]  = amax/2688).
    """
    O, K = w.shape
    assert K % block == 0, f"in_features {K} not divisible by block {block}"
    if block_eff is not None:
        # Caller supplies the EFFECTIVE per-block scale. Needed when the weight was
        # produced by something other than max-based block scaling: nvr2bit's vector
        # quantizer does not always put code 6.0 on a block's peak (2.66% of blocks peak
        # at 4), so recomputing block_amax/6 here re-rounds 1.08% of elements and turns
        # an exact re-encode into a lossy one.
        if global_scale is None:
            global_scale = (w.abs().amax() / GLOBAL_DEN).clamp(min=1e-8)
        block_scale = to_e4m3(block_eff.reshape(O, K // block).float() / global_scale)
    else:
        _, block_scale, global_scale = nvfp4_quantize(w, block, global_scale=global_scale,
                                                      signed=signed)
    # Guard the MAGNITUDE, not the value. `.clamp(min=1e-8)` floors negatives to +1e-8,
    # which is invisible while scales are all positive and catastrophic once `signed`
    # lets them go negative -- it discards exactly the sign the signed normalisation
    # just absorbed. Mirrors the same guard in blocked.blocked_quantize.
    eff = block_scale.unsqueeze(-1).float() * global_scale                   # [O, nB, 1]
    eff = torch.where(eff.abs() < 1e-8, torch.ones_like(eff), eff)
    wb = w.float().reshape(O, K // block, block)
    codes = e2m1_codes(wb / eff).reshape(O, K)                    # [O, K] uint8 0..15
    packed = (codes[:, 1::2] << 4) | codes[:, 0::2]               # low=even, high=odd
    return (packed.to(torch.uint8).contiguous(),
            block_scale.to(torch.float8_e4m3fn).contiguous(),
            global_scale.reshape(1).float())


def unpack_nvfp4_weight(packed: Tensor, weight_scale: Tensor,
                        weight_global_scale: Tensor, block: int = BLOCK) -> Tensor:
    """Inverse of pack_nvfp4_weight: rebuild the dequantized FP32 weight.

    `weight_global_scale` is stored RECIPROCAL in the checkpoint (2688/amax), which
    is what vLLM expects, so it divides rather than multiplies here:
        dequant = e2m1(code) * weight_scale / weight_global_scale

    Round-trips exactly with pack_nvfp4_weight — the packed codes are the same ones
    the model trained with, so this recovers `_wq` bit for bit.
    """
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    O, half = packed.shape
    codes = torch.empty(O, half * 2, dtype=torch.long, device=packed.device)
    codes[:, 0::2] = low            # low nibble holds the EVEN column
    codes[:, 1::2] = high
    vals = E2M1_DECODE.to(packed.device)[codes]                    # [O, K]
    bs = weight_scale.float().to(packed.device).reshape(O, -1, 1)
    wgs = weight_global_scale.float().to(packed.device).reshape(())
    K = codes.shape[1]
    return (vals.reshape(O, K // block, block) * bs / wgs).reshape(O, K)


class NVFP4Linear(BlockScaledLinear):
    """NVFP4 linear. W4A4 by default; `quantize_act=False` gives weight-only W4A16
    (activations stay bf16 and no input_global_scale is exported, which is what
    makes vLLM pick its weight-only CompressedTensorsW4A16Fp4 scheme)."""

    signed = False

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK,
                 quantize_act: bool = True):
        self.quantize_act = quantize_act
        self.calibrating = False
        self._observed = False   # has act_amax ever been set (>0)?
        super().__init__(weight, bias, block_size=block_size)
        # running-max |activation| = the static activation global-scale observer
        self.register_buffer("act_amax", torch.zeros((), device=weight.device))

    def rounder(self, x: Tensor) -> Tensor:
        return e2m1_round(x)

    # ------------------------------------------------------------------
    # compressed-tensors `nvfp4-pack-quantized` format
    # ------------------------------------------------------------------
    # Which weight/scale/activation setting a variant serves. Subclasses that hold
    # more than one master (the prefill/decode formats) override just these three.
    def _variant_weight(self, variant) -> Tensor:
        return self.weight

    def _variant_global_scale(self, variant) -> Tensor:
        return self.group_global_scale()

    def _variant_quantize_act(self, variant) -> bool:
        return self.quantize_act

    def export_config(self, variant=None) -> dict:
        """`quantization_config` for the compressed-tensors nvfp4-pack-quantized
        format. Two hard requirements, both learned the hard way:

          * `quant_method` must be present, and no `weight_shape` tensor may be
            emitted — vLLM does a bare params_dict[name] lookup and any extra tensor
            is a KeyError.
          * W4A16 must set input_activations=None AND omit input_global_scale:
            vLLM's _is_fp4a16_nvfp4() keys on `input_quant is None` to select the
            weight-only scheme, which registers no such parameter.
        """
        quantize_act = self._variant_quantize_act(variant)
        grp = {
            "actorder": None, "block_structure": None, "group_size": self.block_size,
            "num_bits": 4, "observer": "minmax", "observer_kwargs": {},
            "strategy": "tensor_group", "symmetric": True, "type": "float",
        }
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {**grp, "dynamic": "local"} if quantize_act else None,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {**grp, "dynamic": False},
                }
            },
            "format": "nvfp4-pack-quantized",
            "ignore": ["lm_head"],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }

    def export_tensors(self, variant=None) -> dict[str, Tensor]:
        # weight_global_scale is a FUSED-GROUP quantity, so this reads the layer's
        # siblings: vLLM collapses the per-shard scales with .max() when it loads
        # qkv_proj / gate_up_proj, and a per-layer scale would be silently rescaled.
        gscale = self._variant_global_scale(variant)
        packed, wscale, wscale2 = pack_nvfp4_weight(
            self._variant_weight(variant).data, self.block_size, global_scale=gscale,
            signed=self.signed)
        out = {
            "weight_packed": packed.cpu(),
            "weight_scale": wscale.cpu(),
            "weight_global_scale": (1.0 / wscale2).reshape(1).cpu(),
        }
        if self._variant_quantize_act(variant):
            # W4A4 only — the W4A16 scheme registers no such parameter, and emitting
            # one makes vLLM raise KeyError on a bare params_dict lookup.
            act = self.act_amax.float().clamp(min=1e-8)
            out["input_global_scale"] = (GLOBAL_DEN / act).reshape(1).cpu()
        if self.bias is not None:
            out["bias"] = self.bias.detach().to(torch.bfloat16).cpu()
        return out

    @torch.no_grad()
    def load_tensors(self, tensors: dict[str, Tensor], variant=None) -> None:
        """Dequantize a packed checkpoint back into `_wq` (and act_amax for W4A4).

        dequant = e2m1(code) * weight_scale / weight_global_scale — note the global
        scale is stored RECIPROCAL, matching what vLLM reads.
        """
        w = unpack_nvfp4_weight(tensors["weight_packed"], tensors["weight_scale"],
                                tensors["weight_global_scale"])
        self._wq.copy_(w.to(device=self._wq.device, dtype=self._wq.dtype))
        if "input_global_scale" in tensors:
            igs = tensors["input_global_scale"].float().reshape(()).to(self.act_amax.device)
            self.act_amax.copy_(GLOBAL_DEN / igs.clamp(min=1e-12))
            self._observed = True
        if self.bias is not None and "bias" in tensors:
            self.bias.data.copy_(tensors["bias"].to(self.bias.device, self.bias.dtype))

    def forward(self, x: Tensor) -> Tensor:
        w = self.wq if not self.training else self._differentiable_weight()
        if not self.quantize_act:
            return qlinear(x, w, self.bias)      # W4A16: activations stay bf16
        if self.training or self.calibrating:
            with torch.no_grad():
                self.act_amax = torch.maximum(
                    self.act_amax, x.detach().float().abs().amax())
            self._observed = True
        # Static observed activation scale (matches export/inference). The running
        # max is >= this batch's amax, so activations never saturate; falls back to
        # a dynamic per-forward scale only until the observer has seen data.
        gscale = (self.act_amax / GLOBAL_DEN) if self._observed else None
        return qlinear(fake_quant_ste(x, self.block_size, gscale), w, self.bias)


def apply_nvfp4(model: nn.Module, block_size: int = BLOCK, quantize_act: bool = True) -> None:
    """Replace every nn.Linear (except lm_head) with NVFP4Linear, in-place. W4A4."""
    replace_linears(
        model,
        lambda lin: NVFP4Linear.from_linear(lin, block_size=block_size,
                                            quantize_act=quantize_act),
    )


def apply_nvfp4a16(model: nn.Module, **kwargs) -> None:
    """Weight-only NVFP4 (W4A16): 4-bit NVFP4 weights, activations left in bf16."""
    apply_nvfp4(model, quantize_act=False, **kwargs)


# ---------------------------------------------------------------------------
# Activation calibration (static input_global_scale for the W4A4 export)
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_nvfp4(model: nn.Module, chunks, device, n_batches: int = 8,
                    batch_size: int = 4) -> None:
    """Record per-layer activation absmax over a few batches so the exporter can
    write a static input_global_scale.

    Runs on RANK 0 ALONE. Nothing here or in the forward is collective, so do not add a
    collective to this function without making every rank call it.
    """
    mods = [m for m in model.modules()
            if isinstance(m, NVFP4Linear) and m.quantize_act]
    if not mods:
        return
    was_training = model.training
    model.eval()
    # Do NOT zero act_amax: the running-max observer already holds the training-time
    # activation ranges; calibration only augments it with val-set ranges (max), so
    # the exported static scale never underestimates and clips at inference.
    for m in mods:
        m.calibrating = True
    n = min(n_batches * batch_size, len(chunks))
    for i in range(0, n, batch_size):
        batch = chunks[i:i + batch_size]
        if not batch:
            break
        ids = torch.stack([b[0] for b in batch]).to(device)
        # Calibrate UNDER THE PHASE MASK. Without it _phase_for() sees a multi-token
        # forward and calls every position prefill, so a format whose activation
        # quantization lives on DECODE would observe nothing from its own phase and
        # instead fold prefill activations into the running max -- inflating the
        # static scale baked into its checkpoint, on exactly the tensors that scale
        # governs. Chunks carry labels, and labels == -100 is the prefill mask.
        from .dual import prefill_mask_from_labels, quant_phase
        labels = (torch.stack([b[1] for b in batch]).to(device)
                  if len(batch[0]) > 1 else None)
        mask = prefill_mask_from_labels(labels) if labels is not None else None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), quant_phase(mask):
            model(input_ids=ids)
    for m in mods:
        m.calibrating = False
    if was_training:
        model.train()
