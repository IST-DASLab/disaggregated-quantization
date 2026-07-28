"""Shared core for block-scaled, grid-based quantizers (NVFP4, signed-Lloyd, ...).

All of these formats share the same four stages; only the *grid* and whether the
scale is signed differ:

  1. block      – split the contraction (last) dim into groups of `block` elements
  2. scale      – two-level: one FP32 per-tensor global + one FP8-E4M3 per-block
  3. round      – snap the normalized values to the nearest point of a fixed grid
  4. dequantize – value = grid_point * (block_scale * global_scale)

Two-level scaling (stage 2) in detail. With SCALE_REF = 6.0 (the reference grid
max) and E4M3_MAX = 448:

    global      = amax(|x|) / (SCALE_REF * E4M3_MAX)      # one FP32 per tensor
    block_scale = to_e4m3( (block_amax / SCALE_REF) / global )
    effective   = block_scale * global                    # per block

so a block's extreme element normalizes to ~±SCALE_REF and the block scales stay
inside E4M3's range. Fused-group scale sharing is part of this contract, not an
NVFP4 quirk: layers that an inference engine fuses (q/k/v -> qkv_proj, gate/up ->
gate_up_proj) must share ONE global scale, because the engine collapses the
per-shard global scales with .max() when it loads them. Deriving that shared
scale from the group-max amax makes the collapse a no-op instead of silently
rescaling the projections with a smaller amax. See link_fused_groups().

The `signed` variant lets the per-block scale absorb the SIGN of the block's
max-abs element, so after normalization the extreme element is always +SCALE_REF.
The value distribution is then asymmetric, which is why signed grids are
asymmetric too. The global scale is a magnitude and so is sign-independent —
fused-group sharing works identically for both variants.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import QuantizedLinear

E4M3_MAX = 448.0
SCALE_REF = 6.0                      # reference grid max (E2M1's max; also the
                                     # normalization used to fit the Lloyd grids)
GLOBAL_DEN = SCALE_REF * E4M3_MAX    # 2688
BLOCK = 16


def to_e4m3(x: Tensor) -> Tensor:
    """Emulate an FP8 E4M3 cast (grid rounding via a real f8 round-trip)."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def index_nearest(x: Tensor, grid: Tensor) -> Tensor:
    """Index of the nearest point of a sorted 1-D `grid` for every element of x.

    Ties go to the higher index (matches the reference implementation).
    """
    inds = torch.bucketize(x, grid)
    lo = torch.clamp(inds - 1, min=0, max=grid.shape[-1] - 1)
    hi = torch.clamp(inds, min=0, max=grid.shape[-1] - 1)
    return torch.where((grid[hi] - x) <= (x - grid[lo]), hi, lo)


def round_to_grid(x: Tensor, grid: Tensor) -> Tensor:
    """Round-to-nearest onto a sorted 1-D grid (values, not indices)."""
    return grid[index_nearest(x, grid.to(x.device))]


def grid_rounder(grid: Tensor):
    """A `rounder` callable for blocked_quantize() that snaps to `grid`."""
    return partial(round_to_grid, grid=grid)


def ste(x: Tensor, quantized: Tensor) -> Tensor:
    """Straight-through estimator: forward sees `quantized`, grad flows to `x`."""
    return x + (quantized - x).detach()


def blocked_quantize(x: Tensor, rounder, block: int = BLOCK, signed: bool = False,
                     global_scale: Tensor | None = None
                     ) -> tuple[Tensor, Tensor, Tensor]:
    """Block-scale `x`, round it with `rounder`, and dequantize.

    `rounder(x_normalized) -> x_on_grid` may be any grid snap (E2M1, Lloyd, ...).
    `global_scale` may be supplied (e.g. a shared fused-group or calibrated scale);
    otherwise it is derived from this tensor's own amax.

    Returns (dequantized, block_scale_e4m3, global_scale). block_scale_e4m3 is a
    float32 tensor holding E4M3-grid values, shape (..., n_blocks); for the signed
    variant it carries the sign of each block's max-abs element.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    n_blocks = (K + block - 1) // block
    pad = n_blocks * block - K
    xf = F.pad(x.float(), (0, pad)).reshape(*orig_shape[:-1], n_blocks, block)

    if global_scale is None:
        global_scale = xf.abs().amax().clamp(min=1e-8) / GLOBAL_DEN
    global_scale = global_scale.clamp(min=1e-8)

    if signed:
        # scale absorbs the sign of the block's max-abs element
        amax, idx = xf.abs().max(dim=-1, keepdim=True)
        block_amax = amax.clamp(min=1e-8) * xf.take_along_dim(idx, -1).sign()
    else:
        block_amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    block_scale = to_e4m3((block_amax / SCALE_REF) / global_scale)
    eff = block_scale * global_scale
    eff = torch.where(eff.abs() < 1e-12, torch.ones_like(eff), eff)

    q = rounder(xf / eff)
    deq = (q * eff).reshape(*orig_shape[:-1], n_blocks * block)[..., :K].to(x.dtype)
    return deq, block_scale.squeeze(-1), global_scale


# ---------------------------------------------------------------------------
# Fused-group scale sharing — a core part of two-level scaling (see module docstring)
# ---------------------------------------------------------------------------
FUSED_GROUPS = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]


class GroupScaled:
    """Mixin: owns the per-tensor global scale and its sharing across a fused group.

    Kept separate from BlockScaledLinear so that formats which do NOT keep a master
    weight (e.g. the GSQ-style learned-assignment layers) can take part in the same
    fused-group contract. A participant must provide `amax()`; everything else —
    who its siblings are, and what to do once they are known — is handled here.
    """

    _group: list | None = None      # fused-group siblings (set by link_fused_groups)

    def amax(self) -> Tensor:
        """Magnitude the global scale is derived from. Overridden by layers whose
        `weight` is a quantized view rather than the original master weight."""
        return self.weight.detach().float().abs().amax()

    def group_global_scale(self) -> Tensor:
        """Global scale shared across the fused group (group-max amax / 2688).
        Identical for every member, so an engine's global_scale.max() is a no-op.
        A magnitude, hence the same for signed and unsigned variants."""
        members = self._group if (self._group and len(self._group) > 1) else [self]
        amax = max(m.amax() for m in members)
        return (amax / GLOBAL_DEN).clamp(min=1e-8)

    @torch.no_grad()
    def on_group_linked(self) -> None:
        """Called once the fused group is known, i.e. once group_global_scale() is
        final. Default: refresh the cached hard-quantized weight."""
        self._wq.copy_(self._compute_wq())


def link_fused_groups(model: nn.Module) -> None:
    """Give every GroupScaled layer that an engine fuses a shared global scale.

    vLLM (and TensorRT) fuse q/k/v into qkv_proj and gate/up into gate_up_proj, then
    collapse the per-shard global scales with .max(). Sharing the group-max-derived
    scale makes that collapse exact; without it the projections with a smaller amax
    are silently rescaled and the model degrades badly (this cost ~25 points of
    MMLU-Pro before it was found). Applied to every blocked format for consistency.
    """
    for parent in model.modules():
        children = dict(parent.named_children())
        for group in FUSED_GROUPS:
            members = [children[g] for g in group
                       if isinstance(children.get(g), GroupScaled)]
            if len(members) >= 2:
                for m in members:
                    m._group = members
    for m in model.modules():
        if isinstance(m, GroupScaled):
            if m._group is None:
                m._group = [m]
            m.on_group_linked()


class BlockScaledLinear(GroupScaled, QuantizedLinear):
    """nn.Linear with block-scaled, grid-quantized weights (STE) — base for NVFP4,
    signed-Lloyd, and any other two-level-scaled format.

    Subclasses provide `rounder(x_normalized) -> x_on_grid` and set `signed`.
    Weights are fake-quantized with a straight-through estimator: the forward uses
    the cached hard-quantized `_wq` (refreshed once per optimizer step by
    post_update), while gradients flow to the FP32 master weight.
    """

    signed: bool = False

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        out, in_ = weight.shape
        super().__init__(out, in_, bias, dtype=weight.dtype, device=weight.device)
        self.weight = nn.Parameter(weight.clone())
        self.block_size = block_size
        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    # --- subclass contract -------------------------------------------------
    def rounder(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def quantize_weight(self) -> tuple[Tensor, Tensor, Tensor]:
        """(dequantized, block_scale, global_scale) for the current master weight."""
        return blocked_quantize(self.weight, self.rounder, self.block_size,
                                signed=self.signed,
                                global_scale=self.group_global_scale())

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        return self.quantize_weight()[0]

    def _differentiable_weight(self) -> Tensor:
        # STE: forward uses the cached hard-quant weight, grad flows to master.
        return ste(self.weight, self._wq)

    def forward(self, x: Tensor) -> Tensor:
        w = self._wq if not self.training else self._differentiable_weight()
        return F.linear(x, w, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "BlockScaledLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


def replace_linears(model: nn.Module, factory, skip=("lm_head",)) -> None:
    """Swap every nn.Linear (except `skip`) for factory(module), then link fused groups."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.rpartition(".")[2] in skip:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, factory(module))
    link_fused_groups(model)
