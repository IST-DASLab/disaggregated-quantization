"""Gumbel-Softmax Quantization (GSQ) linear layers for QAD training.

Reference: "Gumbel-Softmax Quantization" (arXiv 2604.18556).

Instead of rounding a master weight onto a grid (STE), GSQ gives every weight
element a learnable categorical distribution over the grid levels and trains it
with the Gumbel-Softmax relaxation, annealing temperature down and logit gain up
until the distribution is effectively a hard assignment.

This module holds the parts that are independent of *which* grid is used and of
*how* the scales are parameterised:

  _GumbelSoftmaxFn   the relaxed forward + analytic backward with RNG replay
  proximity_logits   grid-proximity initialisation (soft round-to-nearest)
  GSQLinearBase      logits, annealing schedule, hard/soft weight construction

`GSQLinear` below is the original variant: a symmetric uniform integer grid with
one raw FP32 scale per group of `groupsize` input channels.
  bits=2: {-2,-1,0,1}  × scale   (4 levels)
  bits=3: {-4,...,3}   × scale   (8 levels)

See gsq_lloyd.py for the two-level-block-scaled variant on the signed Lloyd grid.

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


def proximity_logits(w_norm: Tensor, grid: Tensor, spacing: float,
                     std: float, strength: float, noise: float = 1.0) -> Tensor:
    """Initial logits: a soft round-to-nearest over `grid`.

    `w_norm` is the weight divided by its effective scale, so it lives in the same
    units as `grid`. Distances are measured in units of `spacing` (the grid's mean
    level gap) so that `std`/`strength` mean the same thing on any grid.

    Only logit *differences* matter (softmax is shift-invariant), so the result is
    centred on the MAX rather than the mean: the winning level sits at 0 and its
    rivals just below it. That matters because the logits are stored in bfloat16,
    whose precision is relative — mean-centring leaves the far-away levels setting
    the magnitude (~0.6) while the decision is a ~0.02 gap between the top two, and
    bf16 rounding then reassigns a few percent of borderline weights away from
    round-to-nearest. Max-centring puts the full mantissa on the gap that decides.

    `noise` scales an additive Gaussian tiebreak. It is on by default for the
    original uniform-grid GSQ, but note that at std=0.01 / strength=6 it is
    comparable to the logit gap between the two nearest levels, so it flips a few
    percent of the borderline weights away from round-to-nearest. Set noise=0 to
    start from exactly the round-to-nearest assignment and leave exploration to the
    Gumbel noise, which is what the sampling is for.

    Returns [n_levels, out, in].
    """
    grid_f = grid.float().to(w_norm.device).view(-1, 1, 1)
    logits = -0.5 * ((w_norm.unsqueeze(0) - grid_f) / spacing) ** 2
    logits = logits - logits.amax(0, keepdim=True)
    if noise:
        logits = logits * strength + noise * torch.randn_like(logits)
    else:
        logits = logits * strength
    return std * logits


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
        grad_scales.scatter_add_(1, idx.unsqueeze(0).expand_as(g_scale), g_scale.to(scales.dtype))

        g_soft_w = grad_out * scale_per_col
        g_q = g_soft_w.unsqueeze(0) * values.view(n, 1, 1)
        dot = (g_q * soft).sum(0, keepdim=True)
        grad_logits = (soft * (g_q - dot) * sv / T).to(logits_nl.dtype)

        return grad_logits, grad_scales, None, None, None, None, None


class GSQLinearBase(QuantizedLinear):
    """Learned per-element level assignment over a fixed grid.

    Subclass contract:
      `_values`  buffer [n_levels]  – the grid, in the units of `_effective_scales()`
      `_idx`     buffer [in]        – input channel → scale-group index
      `_effective_scales()`         – [out, n_groups]; may be differentiable

    There is deliberately no master weight: the only trainable state is the
    assignment logits and whatever `_effective_scales()` is built from.

    `logit_optim` / `scale_optim` are extra DistAdamW group settings for the two
    quantization parameter classes (see gsq_param_groups). Empty by default, so the
    original uniform-grid GSQLinear keeps its historical optimizer settings.
    """

    logit_optim: dict = {}
    scale_optim: dict = {}

    def __init__(self, out: int, in_: int, bias, dtype, device,
                 grid: Tensor, idx: Tensor, logits_dtype: torch.dtype = torch.bfloat16,
                 temp_start: float = 2.0, temp_end: float = 0.05,
                 scale_start: float = 100.0, scale_end: float = 500.0):
        super().__init__(out, in_, bias, dtype=dtype, device=device)
        self.n_levels = grid.numel()
        self.logits_dtype = logits_dtype
        self.register_buffer("_values", grid.to(device=device, dtype=dtype))
        self.register_buffer("_idx", idx.to(device))

        self.register_buffer("_temp",      torch.tensor(float(temp_start),  device=device))
        self.register_buffer("_scale_val", torch.tensor(float(scale_start), device=device))
        self._temp_start, self._temp_end = temp_start, temp_end
        self._scale_start, self._scale_end = scale_start, scale_end

    # --- subclass contract -------------------------------------------------
    def _effective_scales(self) -> Tensor:
        raise NotImplementedError

    def scale_parameters(self) -> list[nn.Parameter]:
        """Trainable parameters behind `_effective_scales()` (own LR group, no wd)."""
        return []

    # --- QuantizedLinear interface ----------------------------------------
    @property
    def weight(self) -> Tensor:
        """Exposes _wq so HuggingFace's get_target_dtype (which calls
        isinstance(layer, nn.Linear) and then .weight.dtype) keeps working."""
        return self._wq

    def _logits_nl(self) -> Tensor:
        return self.quant_logits.reshape(self.n_levels, self._out, self._in)

    def level_indices(self) -> Tensor:
        """Hard (argmax) level assignment, [out, in]."""
        return self._logits_nl().argmax(0)

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        scales = self._effective_scales()
        return self._values[self.level_indices()] * scales[:, self._idx].to(self._values.dtype)

    def _differentiable_weight(self) -> Tensor:
        return _GumbelSoftmaxFn.apply(
            self._logits_nl(), self._effective_scales(), self._values, self._idx,
            self._temp.item(), self._scale_val.item(), self.quant_logits.device,
        )

    def _update_schedule(self, step: int, total_steps: int) -> None:
        t = step / max(total_steps - 1, 1)
        self._temp.fill_(self._temp_start + (self._temp_end - self._temp_start) * t)
        self._scale_val.fill_(self._scale_start + (self._scale_end - self._scale_start) * t)


class GSQLinear(GSQLinearBase):
    """N-bit GSQ over a symmetric uniform grid × a raw per-group FP32 scale.

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
        noise: float = 1.0,
        logits_dtype: torch.dtype = torch.bfloat16,
        temp_start: float = 2.0,
        temp_end: float = 0.05,
        scale_start: float = 100.0,
        scale_end: float = 500.0,
    ):
        out, in_ = weight.shape
        dev = weight.device
        grid = _make_grid(bits, dtype=weight.dtype, device=dev)  # [n_levels]
        idx = torch.arange(in_, device=dev) // groupsize
        super().__init__(out, in_, bias, dtype=weight.dtype, device=dev,
                         grid=grid, idx=idx, logits_dtype=logits_dtype,
                         temp_start=temp_start, temp_end=temp_end,
                         scale_start=scale_start, scale_end=scale_end)

        self.bits = bits
        self.groupsize = groupsize
        n_groups = (in_ + groupsize - 1) // groupsize

        # Per-group absmax scales: max abs / (2^(bits-1)) maps largest weight to ±max_grid
        w = weight.float()
        w_pad = F.pad(w, (0, n_groups * groupsize - in_))
        scales_init = (
            w_pad.reshape(out, n_groups, groupsize)
            .abs().amax(dim=-1).clamp(min=1e-8)
            / float(2 ** (bits - 1))
        )
        self.scales = nn.Parameter(scales_init.float())

        # The uniform grid has unit spacing, so no spacing normalisation is needed.
        init = proximity_logits(w / scales_init[:, self._idx].clamp(min=1e-8),
                                grid, spacing=1.0, std=std, strength=strength,
                                noise=noise)
        # Stored flat as [out*n_levels, in] for DistAdamW reduce_scatter
        self.quant_logits = nn.Parameter(init.to(logits_dtype).reshape(out * self.n_levels, in_))

        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    def _effective_scales(self) -> Tensor:
        return self.scales

    def scale_parameters(self) -> list[nn.Parameter]:
        return [self.scales]

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
    """DistAdamW param groups: logits, scales, everything else.

    Both quantization parameters live in units of their own, unrelated to the
    weight LR: the logits are consumed as `logits * scale_val` inside a softmax,
    and the scales of the blocked variant are log2 multipliers. Layers therefore
    pin their own group settings via `logit_optim` / `scale_optim`; when those are
    empty (the uniform-grid GSQLinear) the original `lr` / `lr * lr_scale_ratio`
    is used and nothing else changes.
    """
    logit_params, scale_params, other_params = [], [], []
    owned: set[int] = set()
    logit_optim, scale_optim = {}, {}
    for module in model.modules():
        if isinstance(module, GSQLinearBase):
            logit_params.append(module.quant_logits)
            owned.add(id(module.quant_logits))
            for p in module.scale_parameters():
                scale_params.append(p)
                owned.add(id(p))
            logit_optim, scale_optim = module.logit_optim, module.scale_optim
    for p in model.parameters():
        if p.requires_grad and id(p) not in owned:
            other_params.append(p)
    return [
        {"params": logit_params, "lr": lr, **logit_optim},
        {"params": scale_params, "lr": lr * lr_scale_ratio, "weight_decay": 0.0,
         **scale_optim},
        {"params": other_params},
    ]
