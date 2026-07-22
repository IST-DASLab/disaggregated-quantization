"""Straight-Through Estimator (STE) integer quantization baselines for QAD.

ste2bit / ste3bit: per-group absmax scale, round-to-nearest, clamp,
straight-through gradient. No learnable logits — just the master bf16
weight updated by the distillation loss, with the hard-quantized value
cached in _wq and refreshed once per optimizer step via post_update().

STE forward:
    w_q  = round(w / scale).clamp(lo, hi) * scale
    fwd  = w_q   (quantized)
    grad = grad  (passes to w unchanged — straight-through)
    impl = w + (w_q - w).detach()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from quant import QuantizedLinear


def _int_quant(w: Tensor, bits: int, groupsize: int) -> Tensor:
    """Per-group absmax STE integer fake-quantization. Returns dequantized weight."""
    out, in_ = w.shape
    n_groups = (in_ + groupsize - 1) // groupsize
    pad = n_groups * groupsize - in_
    w_f = F.pad(w.float(), (0, pad)).reshape(out, n_groups, groupsize)

    lo = -(2 ** (bits - 1))
    hi =  2 ** (bits - 1) - 1
    scale = w_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / hi

    w_q = (w_f / scale).round().clamp(lo, hi) * scale  # dequantized
    return w_q.reshape(out, -1)[:, :in_].to(w.dtype)


class STEIntLinear(QuantizedLinear):
    """N-bit STE weight quantization: round-to-nearest + clamp, no learned params."""

    def __init__(
        self,
        weight: Tensor,
        bias,
        bits: int = 2,
        groupsize: int = 128,
    ):
        out, in_ = weight.shape
        super().__init__(out, in_, bias, dtype=weight.dtype, device=weight.device)
        self.weight = nn.Parameter(weight.clone())
        self.bits = bits
        self.groupsize = groupsize
        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        return _int_quant(self.weight, self.bits, self.groupsize)

    def _differentiable_weight(self) -> Tensor:
        # STE: cached _wq in forward, gradient flows through to self.weight
        return self.weight + (self._wq - self.weight).detach()

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "STEIntLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


def _apply_ste(model: nn.Module, bits: int, **kwargs) -> None:
    """Replace every nn.Linear (except lm_head) with STEIntLinear(bits=bits)."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, STEIntLinear.from_linear(module, bits=bits, **kwargs))


def apply_ste2bit(model: nn.Module, **kwargs) -> None:
    _apply_ste(model, bits=2, **kwargs)


def apply_ste3bit(model: nn.Module, **kwargs) -> None:
    _apply_ste(model, bits=3, **kwargs)


def apply_ste4bit(model: nn.Module, **kwargs) -> None:
    _apply_ste(model, bits=4, **kwargs)
