"""Gumbel-Softmax Quantization (GSQ) linear layers for QAD training.

Reference: "Gumbel-Softmax Quantization" (arXiv 2604.18556).

GSQLinear supports arbitrary bit widths via a symmetric uniform grid:
  bits=2: {-2,-1,0,1}  × scale   (4 levels)
  bits=3: {-4,...,3}   × scale   (8 levels)

Learnable parameters
--------------------
quant_logits : [out*n_levels, in]  log-probs over the levels, stored flat so
                                   DistAdamW reduce_scatter works (shape[0] divisible
                                   by world_size for standard hidden/intermediate sizes).
scales       : [out, n_groups]     per-group float32 scales; separate LR, no weight decay.

Schedule (linear over training steps)
--------------------------------------
temperature  : 2.0 → 0.05
scale_val    : 100 → 500
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import QuantizedLinear


def _make_grid(bits: int, dtype: torch.dtype, device) -> Tensor:
    """Symmetric uniform integer grid for `bits`-bit quantization."""
    lo = -(2 ** (bits - 1))
    hi =  2 ** (bits - 1)        # exclusive → gives 2**bits levels
    return torch.arange(lo, hi, dtype=dtype, device=device).float().to(dtype)


class _GumbelSoftmaxFn(torch.autograd.Function):
    """Gumbel-Softmax forward/backward with RNG replay. Level count is inferred
    from values.shape[0], so it works for any bit width."""

    @staticmethod
    def forward(ctx, logits_nl, scales, values, idx, temperature, scale_val, device):
        # logits_nl: [n_levels, out, in]
        n = values.shape[0]
        ctx.save_for_backward(logits_nl, scales)
        ctx.values = values
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale_val = scale_val
        ctx.device = device
        ctx.rng = torch.cuda.get_rng_state(device=device)
        ctx.n = n

        eps = 1e-8
        u = torch.rand_like(logits_nl)
        noise = -torch.log(-torch.log(u + eps) + eps).to(values.dtype)
        soft = F.softmax((logits_nl.to(values.dtype) * scale_val + noise) / temperature, dim=0)
        soft_w = (soft * values.view(n, 1, 1)).sum(0)
        return soft_w * scales[:, idx].to(values.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        logits_nl, scales = ctx.saved_tensors
        values, idx = ctx.values, ctx.idx
        T, sv, n = ctx.temperature, ctx.scale_val, ctx.n

        with torch.random.fork_rng(devices=[ctx.device]):
            torch.cuda.set_rng_state(ctx.rng, device=ctx.device)
            eps = 1e-8
            u = torch.rand_like(logits_nl)
            noise = -torch.log(-torch.log(u + eps) + eps).to(values.dtype)
            soft = F.softmax((logits_nl.to(values.dtype) * sv + noise) / T, dim=0)
            soft_w = (soft * values.view(n, 1, 1)).sum(0)

        scale_per_col = scales[:, idx].to(values.dtype)

        g_scale = grad_out * soft_w
        grad_scales = torch.zeros_like(scales)
        grad_scales.scatter_add_(1, idx.unsqueeze(0).expand_as(g_scale), g_scale.float())

        g_soft_w = grad_out * scale_per_col
        g_q = g_soft_w.unsqueeze(0) * values.view(n, 1, 1)
        dot = (g_q * soft).sum(0, keepdim=True)
        grad_logits = (soft * (g_q - dot) * sv / T).to(logits_nl.dtype)

        return grad_logits, grad_scales, None, None, None, None, None


class GSQLinear(QuantizedLinear):
    """N-bit GSQ weight quantization over a symmetric uniform grid × per-group scale.

    bits=2 → grid {-2,-1,0,1},  quant_logits shape [out*4, in]
    bits=3 → grid {-4,...,3},   quant_logits shape [out*8, in]
    """

    def __init__(
        self,
        weight: Tensor,
        bias,
        bits: int = 2,
        groupsize: int = 128,
        std: float = 0.01,
        strength: float = 6.0,
        logits_dtype: torch.dtype = torch.bfloat16,
        temp_start: float = 2.0,
        temp_end: float = 0.05,
        scale_start: float = 100.0,
        scale_end: float = 500.0,
    ):
        out, in_ = weight.shape
        dev = weight.device
        super().__init__(out, in_, bias, dtype=weight.dtype, device=dev)

        self.bits = bits
        self.n_levels = 2 ** bits
        self.groupsize = groupsize
        n_groups = (in_ + groupsize - 1) // groupsize

        grid = _make_grid(bits, dtype=weight.dtype, device=dev)  # [n_levels]
        self.register_buffer("_values", grid)
        self.register_buffer("_idx", torch.arange(in_, device=dev) // groupsize)

        # Per-group absmax scales: max abs / (2^(bits-1)) maps largest weight to ±max_grid
        w = weight.float()
        w_pad = F.pad(w, (0, n_groups * groupsize - in_))
        scales_init = (
            w_pad.reshape(out, n_groups, groupsize)
            .abs().amax(dim=-1).clamp(min=1e-8)
            / float(2 ** (bits - 1))
        )

        # Proximity logits: Gaussian kernel over grid points
        sc = scales_init[:, self._idx]
        w_norm = w / sc.clamp(min=1e-8)
        grid_f = grid.float().to(device=dev).view(self.n_levels, 1, 1)
        logits = -0.5 * (w_norm.unsqueeze(0) - grid_f) ** 2
        logits = logits - logits.mean(0, keepdim=True)
        init = std * (torch.randn_like(logits) + logits * strength)  # [n_levels, out, in]

        # Stored flat as [out*n_levels, in] for DistAdamW reduce_scatter
        self.quant_logits = nn.Parameter(init.to(logits_dtype).reshape(out * self.n_levels, in_))
        self.scales = nn.Parameter(scales_init.float())

        self.register_buffer("_temp",      torch.tensor(float(temp_start),  device=dev))
        self.register_buffer("_scale_val", torch.tensor(float(scale_start), device=dev))
        self._temp_start = temp_start
        self._temp_end = temp_end
        self._scale_start = scale_start
        self._scale_end = scale_end

        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    # ------------------------------------------------------------------
    # QuantizedLinear interface
    # ------------------------------------------------------------------

    @property
    def weight(self) -> Tensor:
        """Exposes _wq so HuggingFace's get_target_dtype (which calls
        isinstance(layer, nn.Linear) and then .weight.dtype) keeps working."""
        return self._wq

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        logits_nl = self.quant_logits.reshape(self.n_levels, self._out, self._in)
        hard_idx = logits_nl.argmax(0)
        return self._values[hard_idx] * self.scales[:, self._idx].to(self._values.dtype)

    def _differentiable_weight(self) -> Tensor:
        logits_nl = self.quant_logits.reshape(self.n_levels, self._out, self._in)
        return _GumbelSoftmaxFn.apply(
            logits_nl, self.scales, self._values, self._idx,
            self._temp.item(), self._scale_val.item(),
            self.quant_logits.device,
        )

    def _update_schedule(self, step: int, total_steps: int) -> None:
        t = step / max(total_steps - 1, 1)
        self._temp.fill_(self._temp_start + (self._temp_end - self._temp_start) * t)
        self._scale_val.fill_(self._scale_start + (self._scale_end - self._scale_start) * t)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "GSQLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


# Convenience aliases
class GSQ2BitLinear(GSQLinear):
    def __init__(self, weight, bias, **kwargs):
        kwargs.setdefault("bits", 2)
        super().__init__(weight, bias, **kwargs)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "GSQ2BitLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


class GSQ3BitLinear(GSQLinear):
    def __init__(self, weight, bias, **kwargs):
        kwargs.setdefault("bits", 3)
        super().__init__(weight, bias, **kwargs)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "GSQ3BitLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _apply_gsq(model: nn.Module, bits: int, **kwargs) -> None:
    """Replace every nn.Linear (except lm_head) with GSQLinear(bits=bits)."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, GSQLinear.from_linear(module, bits=bits, **kwargs))


def apply_gsq2bit(model: nn.Module, **kwargs) -> None:
    _apply_gsq(model, bits=2, **kwargs)


def apply_gsq3bit(model: nn.Module, **kwargs) -> None:
    _apply_gsq(model, bits=3, **kwargs)


def gsq_param_groups(model: nn.Module, lr: float, lr_scale_ratio: float = 0.5) -> list[dict]:
    """DistAdamW param groups: logits (lr), scales (lr*ratio, wd=0), others (lr)."""
    logit_params, scale_params, other_params = [], [], []
    owned: set[int] = set()
    for module in model.modules():
        if isinstance(module, GSQLinear):
            logit_params.append(module.quant_logits)
            scale_params.append(module.scales)
            owned.add(id(module.quant_logits))
            owned.add(id(module.scales))
    for p in model.parameters():
        if p.requires_grad and id(p) not in owned:
            other_params.append(p)
    return [
        {"params": logit_params},
        {"params": scale_params, "lr": lr * lr_scale_ratio, "weight_decay": 0.0},
        {"params": other_params},
    ]
