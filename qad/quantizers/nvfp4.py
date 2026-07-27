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

Export lives in export/compressed_tensors.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocked import (BLOCK, GLOBAL_DEN, BlockScaledLinear, blocked_quantize,
                      replace_linears, ste)
from .grids import E2M1_BOUNDS, E2M1_LEVELS


def e2m1_round(x: Tensor) -> Tensor:
    """Round to the nearest signed E2M1 magnitude (values >=5 clamp to 6).

    Kept as an explicit magnitude/midpoint table rather than a generic grid snap:
    this exact rounding is verified bit-identical to vLLM's kernels.
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


def nvfp4_quantize(x: Tensor, block: int = BLOCK, global_scale: Tensor | None = None
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Two-level NVFP4 fake-quant along the last dim.
    Returns (dequantized, block_scale_e4m3, global_scale)."""
    return blocked_quantize(x, e2m1_round, block, signed=False, global_scale=global_scale)


def fake_quant_ste(x: Tensor, block: int, global_scale: Tensor | None) -> Tensor:
    return ste(x, nvfp4_quantize(x, block, global_scale)[0])


def pack_nvfp4_weight(w: Tensor, block: int = BLOCK, global_scale: Tensor | None = None
                      ) -> tuple[Tensor, Tensor, Tensor]:
    """Encode a weight matrix into the real NVFP4 checkpoint tensors, using the
    SAME scales as nvfp4_quantize so the packed weight is bit-identical to the
    fake-quantized `_wq` the model trained with.  `global_scale` should be the
    shared fused-group scale (see blocked.link_fused_groups).

    Returns (packed_uint8 [O, K//2], weight_scale float8_e4m3fn [O, K//block],
             weight_scale_2 float32 [1]  = amax/2688).
    """
    O, K = w.shape
    assert K % block == 0, f"in_features {K} not divisible by block {block}"
    _, block_scale, global_scale = nvfp4_quantize(w, block, global_scale=global_scale)
    eff = (block_scale.unsqueeze(-1).float() * global_scale).clamp(min=1e-8)  # [O, nB, 1]
    wb = w.float().reshape(O, K // block, block)
    codes = e2m1_codes(wb / eff).reshape(O, K)                    # [O, K] uint8 0..15
    packed = (codes[:, 1::2] << 4) | codes[:, 0::2]               # low=even, high=odd
    return (packed.to(torch.uint8).contiguous(),
            block_scale.to(torch.float8_e4m3fn).contiguous(),
            global_scale.reshape(1).float())


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

    def forward(self, x: Tensor) -> Tensor:
        w = self._wq if not self.training else self._differentiable_weight()
        if not self.quantize_act:
            return F.linear(x, w, self.bias)      # W4A16: activations stay bf16
        if self.training or self.calibrating:
            with torch.no_grad():
                self.act_amax = torch.maximum(
                    self.act_amax, x.detach().float().abs().amax())
            self._observed = True
        # Static observed activation scale (matches export/inference). The running
        # max is >= this batch's amax, so activations never saturate; falls back to
        # a dynamic per-forward scale only until the observer has seen data.
        gscale = (self.act_amax / GLOBAL_DEN) if self._observed else None
        return F.linear(fake_quant_ste(x, self.block_size, gscale), w, self.bias)


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
    write a static input_global_scale. Safe to call on a single rank (no DDP sync)."""
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids)
    for m in mods:
        m.calibrating = False
    if was_training:
        model.train()
