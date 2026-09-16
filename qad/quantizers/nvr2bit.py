import torch
import torch.nn as nn
from torch import Tensor

from .base import QuantizedLinear
from .blocked import GroupScaled, replace_linears, ste

try:
    from .luts_backend import NUM_PROBES, _recover_scales, nvr2bit_quantize  # noqa: F401
except ImportError:
    NUM_PROBES = 8

    def _unavailable(*_args, **_kwargs):
        raise ImportError("psx-luts is not built/on PYTHONPATH; nvr2bit/nvfp4nvr2bit* "
                           "formats are unavailable. See qad/bin/README.md #4.")
    _recover_scales = _unavailable
    nvr2bit_quantize = _unavailable

__all__ = ["NVR2BitLinear", "post_update_nvr2bit", "apply_nvr2bit",
           "nvr2bit_quantize", "_recover_scales", "NUM_PROBES"]


class NVR2BitLinear(GroupScaled, QuantizedLinear):
    NVR2BIT_HEAVY = True

    def __init__(self, weight: Tensor, bias, num_probes: int = NUM_PROBES):
        out, in_ = weight.shape
        super().__init__(out, in_, bias, dtype=weight.dtype, device=weight.device)
        self.weight = nn.Parameter(weight.clone())
        self.num_probes = int(num_probes)
        from .luts_backend import _load
        _load()
        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    def group_amax(self) -> Tensor:
        members = self._group if (self._group and len(self._group) > 1) else [self]
        return max(m.amax() for m in members)

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        return nvr2bit_quantize(self.weight.detach(), self.group_amax(),
                                self.num_probes)

    def _differentiable_weight(self) -> Tensor:
        return ste(self.weight, self.wq)

    @torch.no_grad()
    def refresh_buffers(self) -> None:
        self._wq.copy_(self._compute_wq())

    @torch.no_grad()
    def post_update(self, step: int, total_steps: int) -> None:
        self._update_schedule(step, total_steps)
        self.refresh_buffers()


@torch.no_grad()
def post_update_nvr2bit(model: nn.Module, step: int, total_steps: int) -> None:
    """Refresh every NVR2BIT_HEAVY layer, sharded across ranks."""
    import torch.distributed as dist

    layers = [m for m in model.modules() if getattr(m, "NVR2BIT_HEAVY", False)]
    for m in model.modules():
        if isinstance(m, QuantizedLinear) and not getattr(m, "NVR2BIT_HEAVY", False):
            m.post_update(step, total_steps)
    if not layers:
        return
    for l in layers:
        l._update_schedule(step, total_steps)
    live = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if live else 1
    rank = dist.get_rank() if live else 0
    for i, l in enumerate(layers):
        if i % world == rank:
            l.refresh_buffers()
    if world > 1:
        for i, l in enumerate(layers):
            dist.broadcast(l._wq, src=i % world)
            if hasattr(l, "_wq_dec"):
                dist.broadcast(l._wq_dec, src=i % world)


def apply_nvr2bit(model: nn.Module, num_probes: int = NUM_PROBES) -> None:
    replace_linears(model, lambda lin: NVR2BitLinear(
        lin.weight.data, lin.bias, num_probes=num_probes))
