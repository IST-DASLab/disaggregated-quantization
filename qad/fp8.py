import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_FP8_MAX = 448.0  # torch.float8_e4m3fn representable maximum


def fake_fp8(x: Tensor) -> Tensor:
    """Per-tensor absmax-scaled FP8 fake quant with straight-through estimator.

    scale = FP8_MAX / |x|_max clips x into the E4M3 representable range, casts
    to float8_e4m3fn and back for grid rounding, then rescales. STE: gradient
    passes through as identity.
    """
    scale = _FP8_MAX / x.detach().abs().max().clamp(min=1e-12)
    xq = (x * scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype) / scale
    return x + (xq - x).detach()


class FP8Linear(nn.Linear):
    """nn.Linear replacement: fake-quantizes both weight and input activation to FP8.

    Inherits from nn.Linear so isinstance(layer, nn.Linear) checks in HuggingFace
    (e.g. get_target_dtype in flash_attention_2) still find this layer.
    """

    def __init__(self, weight: nn.Parameter, bias: nn.Parameter | None):
        nn.Module.__init__(self)  # skip nn.Linear.__init__ to avoid duplicate weight
        self.in_features  = weight.shape[1]
        self.out_features = weight.shape[0]
        self.weight = weight  # nn.Module.__setattr__ registers nn.Parameter
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(fake_fp8(x), fake_fp8(self.weight), self.bias)


def apply_fp8_linear(model: nn.Module) -> None:
    """Replace every nn.Linear in model except lm_head with FP8Linear, in-place."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, FP8Linear(module.weight, module.bias))
