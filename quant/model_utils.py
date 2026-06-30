import gc
from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantizer import Quantizer

class ForwardInterrupt(Exception):
    pass

class InputCollector(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt
    

def clear_device_cache(garbage_collection=False):
    if garbage_collection: gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.xpu.is_available(): torch.xpu.empty_cache()


def to(obj, device):
    """Recursively move tensors in nested tuples/lists/dicts to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    return obj

def decompose_dim(size: int) -> Tuple[int, int]:
    a = int(size**0.5)
    if a**2 == size:
        return a, a
    for i in range(a, 0, -1):
        if size % i == 0:
            return i, size // i


def maybe_first_element(x):
    if isinstance(x, Sequence):
        x = x[0]
    return x

class QuantizedLinear(nn.Module):
    def __init__(
        self,
        weight_prefill: torch.Tensor,
        weight_decode: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        auto_mode: bool = False,
        act_quantizer: Optional[Quantizer] = None
    ):
        super().__init__()
        self.register_buffer("weight_prefill", weight_prefill)
        self.register_buffer("weight_decode", weight_decode)
        self.register_buffer("bias", bias)
        self.act_quantizer = act_quantizer
        self.auto_mode: bool = auto_mode
        self.mode = "prefill"
        # Whether activation quantization is applied (only possible when an
        # act_quantizer was attached at quantization time).
        self.act_quant = act_quantizer is not None

    def set_mode(self, mode: str = "prefill"):
        assert mode in ["prefill", "decode"]
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_mode:
            mode = "prefill" if x.shape[-2] > 1 else "decode"
        else:
            mode = self.mode

        if self.act_quantizer is not None and self.act_quant:
            scales, zeros = self.act_quantizer.get_quantization_params(x, dynamic=True)
            x = self.act_quantizer.quantize_dequantize(x, scales, zeros)

        w = self.weight_prefill if mode == "prefill" else self.weight_decode
        return F.linear(x, w, self.bias)


def _set_submodule(parent: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)