import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import QuantizedLinear


OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}

@torch.compile(dynamic=True)
def _quest_int_quant(w: Tensor, bits: int, groupsize: int) -> tuple[Tensor, Tensor]:
    """Per-group STD-based symmetric integer fake-quantization. Returns dequantized weight."""
    out, in_ = w.shape
    n_groups = (in_ + groupsize - 1) // groupsize
    pad = n_groups * groupsize - in_
    w_f = F.pad(w.float(), (0, pad)).reshape(out, n_groups, groupsize)

    gridmax = (2 ** bits - 1) / 2
    scale = w_f.std(dim=-1, keepdim=True).clamp(min=1e-8) * OPTIMAL_GAUSSIAN_SCALES[bits] / gridmax

    mask = (w_f / scale).abs() <= gridmax
    w_q = ((w_f / scale + 0.5).round().clamp(-gridmax + 0.5, gridmax + 0.5) - 0.5) * scale  # dequantized
    # Trim both outputs to [out, in_] to drop pad columns
    return w_q.reshape(out, -1)[:, :in_].to(w.dtype), mask.reshape(out, -1)[:, :in_]


class QuestIntLinear(QuantizedLinear):
    """N-bit Quest weight quantization: round-to-nearest + clamp, no learned params."""

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
        # Register _mask before _compute_wq() so the buffer exists when it tries to copy into it
        self.register_buffer("_mask", torch.empty(self.out_features, self.in_features, dtype=torch.bool,
                                                  device=self._wq.device))
        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        w_q, mask = _quest_int_quant(self.weight, self.bits, self.groupsize)
        self._mask.copy_(mask)
        return w_q

    def _differentiable_weight(self) -> Tensor:
        # Quest: cached _wq in forward, gradient flows through to self.weight
        return self.weight * self._mask.to(self.weight.dtype) + (self._wq - self.weight * self._mask.to(self.weight.dtype)).detach()

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "QuestIntLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


def _apply_quest(model: nn.Module, bits: int, **kwargs) -> None:
    """Replace every nn.Linear (except lm_head) with QuestIntLinear(bits=bits)."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, QuestIntLinear.from_linear(module, bits=bits, **kwargs))


def apply_quest2bit(model: nn.Module, **kwargs) -> None:
    _apply_quest(model, bits=2, **kwargs)


def apply_quest3bit(model: nn.Module, **kwargs) -> None:
    _apply_quest(model, bits=3, **kwargs)


def apply_quest4bit(model: nn.Module, **kwargs) -> None:
    _apply_quest(model, bits=4, **kwargs)
