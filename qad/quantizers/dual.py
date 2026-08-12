"""Prefill/decode dual-format quantization."""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocked import (BLOCK, GLOBAL_DEN, blocked_quantize, grid_rounder,
                      replace_linears, ste)
from .grids import LLOYD21_SIGNED_2BIT, LLOYD43_SIGNED_3BIT
from .nvfp4 import (NVFP4Linear, fake_quant_ste, pack_nvfp4_weight,
                    unpack_nvfp4_weight)
from .nvr2bit import NUM_PROBES as NVR2BIT_PROBES, nvr2bit_quantize

# [B, T] bool, True where the PREFILL format applies. None => infer from shape.
_PHASE_MASK: Tensor | None = None


@contextmanager
def quant_phase(prefill_mask: Tensor | None):
    """Route positions to prefill/decode formats for the enclosed block.

    Must span the backward pass too: gradient checkpointing recomputes the
    forward during backward, and a cleared mask would silently run every
    position through the decode format.
    """
    global _PHASE_MASK
    prev = _PHASE_MASK
    _PHASE_MASK = prefill_mask
    try:
        yield
    finally:
        _PHASE_MASK = prev


def prefill_mask_from_labels(labels: Tensor) -> Tensor:
    """[B, T] -> True on prefill (prompt/system) positions."""
    return labels == -100


def _phase_for(x: Tensor):
    """True (all prefill), False (all decode), or a [B, T, 1] bool mask."""
    m = _PHASE_MASK
    if m is None:
        return x.shape[-2] > 1
    if x.dim() != 3:
        raise RuntimeError(
            f"dual-format layer got a {x.dim()}D input under an explicit phase mask; "
            "the mask is per (batch, position) and cannot be aligned")
    return m.unsqueeze(-1)


class _DualActMixin:
    """Activation quantization restricted to prefill positions."""

    def _quant_act(self, x: Tensor, mask=None) -> Tensor:
        if self.training or self.calibrating:
            with torch.no_grad():
                a = x.detach().float().abs()
                if mask is not None:
                    a = torch.where(mask, a, torch.zeros((), device=a.device, dtype=a.dtype))
                self.act_amax = torch.maximum(self.act_amax, a.amax())
            self._observed = True
        gscale = (self.act_amax / GLOBAL_DEN) if self._observed else None
        return fake_quant_ste(x, self.block_size, gscale)


class DualSharedNVFP4Linear(_DualActMixin, NVFP4Linear):
    """One master weight; activations quantized on prefill only."""

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)

    def forward(self, x: Tensor) -> Tensor:
        w = self._wq if not self.training else self._differentiable_weight()
        phase = _phase_for(x)
        if not torch.is_tensor(phase):
            return F.linear(self._quant_act(x) if phase else x, w, self.bias)
        return F.linear(torch.where(phase, self._quant_act(x, phase), x), w, self.bias)

    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _variant_quantize_act(self, variant) -> bool:
        return variant != "decode"


class DualSplitNVFP4Linear(_DualActMixin, NVFP4Linear):
    """Two masters: `weight` for prefill (W4A4), `decode_weight` for decode (W4A16)."""

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)
        self.decode_weight = nn.Parameter(weight.clone())
        self.register_buffer("_wq_dec", torch.empty_like(self._wq))
        with torch.no_grad():
            self._wq_dec.copy_(self.quantize_decode_weight()[0])

    def decode_amax(self) -> Tensor:
        return self.decode_weight.detach().float().abs().amax()

    def decode_group_global_scale(self) -> Tensor:
        members = self._group if (self._group and len(self._group) > 1) else [self]
        amax = max(m.decode_amax() for m in members)
        return (amax / GLOBAL_DEN).clamp(min=1e-8)

    def quantize_decode_weight(self):
        return blocked_quantize(self.decode_weight, self.rounder, self.block_size,
                                signed=self.signed,
                                global_scale=self.decode_group_global_scale())

    @torch.no_grad()
    def on_group_linked(self) -> None:
        super().on_group_linked()
        self._wq_dec.copy_(self.quantize_decode_weight()[0])

    @torch.no_grad()
    def post_update(self, step: int, total_steps: int) -> None:
        super().post_update(step, total_steps)
        self._wq_dec.copy_(self.quantize_decode_weight()[0])

    def forward(self, x: Tensor) -> Tensor:
        train = self.training
        phase = _phase_for(x)
        if not torch.is_tensor(phase):
            if phase:
                w_p = ste(self.weight, self._wq) if train else self._wq
                return F.linear(self._quant_act(x), w_p, self.bias)
            w_d = ste(self.decode_weight, self._wq_dec) if train else self._wq_dec
            return F.linear(x, w_d, self.bias)
        w_p = ste(self.weight, self._wq) if train else self._wq
        w_d = ste(self.decode_weight, self._wq_dec) if train else self._wq_dec
        y_p = F.linear(self._quant_act(x, phase), w_p, self.bias)
        y_d = F.linear(x, w_d, self.bias)
        return torch.where(phase, y_p, y_d)

    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _variant_weight(self, variant) -> Tensor:
        return self.decode_weight if variant == "decode" else self.weight

    def _variant_global_scale(self, variant) -> Tensor:
        return (self.decode_group_global_scale() if variant == "decode"
                else self.group_global_scale())

    def _variant_quantize_act(self, variant) -> bool:
        return variant != "decode"

    @torch.no_grad()
    def load_tensors(self, tensors, variant=None) -> None:
        w = unpack_nvfp4_weight(tensors["weight_packed"], tensors["weight_scale"],
                                tensors["weight_global_scale"])
        if variant == "decode":
            self._wq_dec.copy_(w.to(self._wq_dec.device, self._wq_dec.dtype))
            self.decode_weight.data.copy_(self._wq_dec)
        else:
            self._wq.copy_(w.to(self._wq.device, self._wq.dtype))
            self.weight.data.copy_(self._wq)
            if "input_global_scale" in tensors:
                igs = tensors["input_global_scale"].float().reshape(())
                self.act_amax.copy_(GLOBAL_DEN / igs.clamp(min=1e-12).to(self.act_amax.device))
                self._observed = True
        if self.bias is not None and "bias" in tensors:
            self.bias.data.copy_(tensors["bias"].to(self.bias.device, self.bias.dtype))


def apply_nvfp4pdshared(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: DualSharedNVFP4Linear.from_linear(
        lin, block_size=block_size))


def apply_nvfp4pdsplit(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: DualSplitNVFP4Linear.from_linear(
        lin, block_size=block_size))


class _PhaseIsolatedNVFP4Linear(_DualActMixin, NVFP4Linear):
    """Shared master; NVFP4 on one phase, BF16 on the other. Diagnostic only."""

    NVFP4_PHASE: str = "prefill"

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)

    def _nvfp4_on_prefill(self) -> bool:
        return self.NVFP4_PHASE == "prefill"

    def _nvfp4_out(self, x: Tensor, mask=None) -> Tensor:
        w = self._differentiable_weight() if self.training else self._wq
        return F.linear(self._quant_act(x, mask), w, self.bias)

    def _bf16_out(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def forward(self, x: Tensor) -> Tensor:
        phase = _phase_for(x)
        if not torch.is_tensor(phase):
            is_nvfp4 = phase if self._nvfp4_on_prefill() else not phase
            return self._nvfp4_out(x) if is_nvfp4 else self._bf16_out(x)
        nv = phase if self._nvfp4_on_prefill() else ~phase
        return torch.where(nv, self._nvfp4_out(x, nv), self._bf16_out(x))

    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _is_bf16_variant(self, variant) -> bool:
        return variant != self.NVFP4_PHASE

    def _variant_quantize_act(self, variant) -> bool:
        return not self._is_bf16_variant(variant)

    def export_config(self, variant=None):
        return None if self._is_bf16_variant(variant) else super().export_config(variant)

    def export_tensors(self, variant=None) -> dict:
        if not self._is_bf16_variant(variant):
            return super().export_tensors(variant)
        out = {"weight": self.weight.detach().to(torch.bfloat16).cpu()}
        if self.bias is not None:
            out["bias"] = self.bias.detach().to(torch.bfloat16).cpu()
        return out

    @torch.no_grad()
    def load_tensors(self, tensors, variant=None) -> None:
        if not self._is_bf16_variant(variant):
            super().load_tensors(tensors, variant)
            return
        w = tensors["weight"]
        self.weight.data.copy_(w.to(self.weight.device, self.weight.dtype))
        self._wq.copy_(self.quantize_weight()[0])
        if self.bias is not None and "bias" in tensors:
            self.bias.data.copy_(tensors["bias"].to(self.bias.device, self.bias.dtype))


class BF16PrefillNVFP4DecodeLinear(_PhaseIsolatedNVFP4Linear):
    NVFP4_PHASE = "decode"


class NVFP4PrefillBF16DecodeLinear(_PhaseIsolatedNVFP4Linear):
    NVFP4_PHASE = "prefill"


def apply_nvfp4decode(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: BF16PrefillNVFP4DecodeLinear.from_linear(
        lin, block_size=block_size))


def apply_nvfp4prefill(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4PrefillBF16DecodeLinear.from_linear(
        lin, block_size=block_size))


class _NVFP4Lloyd43Linear(_DualActMixin, NVFP4Linear):
    """NVFP4 (W4A4) prefill, Lloyd43 (W3A16) decode."""

    # NVFP4 block scales are UE4M3 (unsigned), so the prefill half must be unsigned.
    # The decode half is pseudo-quantized to bf16 and is unaffected.
    signed = False
    DECODE_GRID = LLOYD43_SIGNED_3BIT

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)
        self._lloyd_grid = self.DECODE_GRID.to(weight.device)
        self._lloyd_round = grid_rounder(self._lloyd_grid)
        self.register_buffer("_wq_dec", torch.empty_like(self._wq))
        with torch.no_grad():
            self._wq_dec.copy_(self.quantize_decode_weight()[0])

    def _decode_master(self) -> Tensor:
        return self.weight

    def _round_decode(self, x: Tensor) -> Tensor:
        if self._lloyd_grid.device != x.device:
            self._lloyd_grid = self._lloyd_grid.to(x.device)
            self._lloyd_round = grid_rounder(self._lloyd_grid)
        return self._lloyd_round(x)

    def decode_amax(self) -> Tensor:
        return self._decode_master().detach().float().abs().amax()

    def decode_group_global_scale(self) -> Tensor:
        members = self._group if (self._group and len(self._group) > 1) else [self]
        amax = max(m.decode_amax() for m in members)
        return (amax / GLOBAL_DEN).clamp(min=1e-8)

    def quantize_decode_weight(self):
        return blocked_quantize(self._decode_master(), self._round_decode, self.block_size,
                                signed=True, global_scale=self.decode_group_global_scale())

    @torch.no_grad()
    def on_group_linked(self) -> None:
        super().on_group_linked()
        self._wq_dec.copy_(self.quantize_decode_weight()[0])

    @torch.no_grad()
    def post_update(self, step: int, total_steps: int) -> None:
        super().post_update(step, total_steps)
        self._wq_dec.copy_(self.quantize_decode_weight()[0])

    def forward(self, x: Tensor) -> Tensor:
        train = self.training
        phase = _phase_for(x)
        if not torch.is_tensor(phase):
            if phase:
                w_p = self._differentiable_weight() if train else self._wq
                return F.linear(self._quant_act(x), w_p, self.bias)
            w_d = (ste(self._decode_master(), self._wq_dec) if train else self._wq_dec)
            return F.linear(x, w_d, self.bias)
        w_p = self._differentiable_weight() if train else self._wq
        w_d = ste(self._decode_master(), self._wq_dec) if train else self._wq_dec
        y_p = F.linear(self._quant_act(x, phase), w_p, self.bias)
        y_d = F.linear(x, w_d, self.bias)
        return torch.where(phase, y_p, y_d)

    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _variant_quantize_act(self, variant) -> bool:
        return variant != "decode"

    def export_config(self, variant=None):
        return None if variant == "decode" else super().export_config(variant)

    def export_tensors(self, variant=None) -> dict:
        if variant != "decode":
            return super().export_tensors(variant)
        out = {"weight": self._wq_dec.detach().to(torch.bfloat16).cpu()}
        if self.bias is not None:
            out["bias"] = self.bias.detach().to(torch.bfloat16).cpu()
        return out

    @torch.no_grad()
    def load_tensors(self, tensors, variant=None) -> None:
        if variant != "decode":
            super().load_tensors(tensors, variant)
            return
        w = tensors["weight"].to(self._wq_dec.device, self._wq_dec.dtype)
        self._wq_dec.copy_(w)
        self._decode_master().data.copy_(w.to(self._decode_master().dtype))
        if self.bias is not None and "bias" in tensors:
            self.bias.data.copy_(tensors["bias"].to(self.bias.device, self.bias.dtype))


class NVFP4Lloyd43SharedLinear(_NVFP4Lloyd43Linear):
    """One shared master, both phases quantized from it."""


class NVFP4Lloyd43SplitLinear(_NVFP4Lloyd43Linear):
    """Two masters, trained separately."""

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size)
        self.decode_weight = nn.Parameter(weight.clone())
        with torch.no_grad():
            self._wq_dec.copy_(self.quantize_decode_weight()[0])

    def _decode_master(self) -> Tensor:
        return getattr(self, "decode_weight", self.weight)


class NVFP4Lloyd43UpcastLinear(_NVFP4Lloyd43Linear):
    """Prefill weight upcasted from the decode weight; only the decode master is stored."""

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size)
        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    def _upcast_source(self) -> Tensor:
        return getattr(self, "_wq_dec", self.weight)

    def amax(self) -> Tensor:
        return self._upcast_source().detach().float().abs().amax()

    def quantize_weight(self):
        return blocked_quantize(self._upcast_source(), self.rounder, self.block_size,
                                signed=self.signed,
                                global_scale=self.group_global_scale())

    def _variant_weight(self, variant) -> Tensor:
        return self._upcast_source()

    @torch.no_grad()
    def on_group_linked(self) -> None:
        self._wq_dec.copy_(self.quantize_decode_weight()[0])
        self._wq.copy_(self._compute_wq())

    @torch.no_grad()
    def post_update(self, step: int, total_steps: int) -> None:
        self._update_schedule(step, total_steps)
        self._wq_dec.copy_(self.quantize_decode_weight()[0])
        self._wq.copy_(self._compute_wq())


def apply_nvfp4lloyd43shared(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4Lloyd43SharedLinear.from_linear(
        lin, block_size=block_size))


def apply_nvfp4lloyd43upcast(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4Lloyd43UpcastLinear.from_linear(
        lin, block_size=block_size))


def apply_nvfp4lloyd43split(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4Lloyd43SplitLinear.from_linear(
        lin, block_size=block_size))


class NVFP4Lloyd21UpcastLinear(NVFP4Lloyd43UpcastLinear):
    DECODE_GRID = LLOYD21_SIGNED_2BIT


def apply_nvfp4lloyd21upcast(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4Lloyd21UpcastLinear.from_linear(
        lin, block_size=block_size))


class NVFP4Lloyd21SplitLinear(NVFP4Lloyd43SplitLinear):
    DECODE_GRID = LLOYD21_SIGNED_2BIT


def apply_nvfp4lloyd21split(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4Lloyd21SplitLinear.from_linear(
        lin, block_size=block_size))


class _NVFP4NVR2BitMixin:
    NVR2BIT_HEAVY = True
    num_probes = NVR2BIT_PROBES

    def _decode_group_amax(self) -> Tensor:
        members = self._group if (self._group and len(self._group) > 1) else [self]
        return max(m.decode_amax() for m in members)

    def quantize_decode_weight(self):
        return (nvr2bit_quantize(self._decode_master().detach(),
                                 self._decode_group_amax(), self.num_probes),)

    @torch.no_grad()
    def refresh_buffers(self) -> None:
        raise NotImplementedError


class NVFP4NVR2BitUpcastLinear(_DualActMixin, NVFP4Linear):
    NVR2BIT_HEAVY = True
    signed = False
    num_probes = NVR2BIT_PROBES

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)

    def group_amax(self) -> Tensor:
        members = self._group if (self._group and len(self._group) > 1) else [self]
        return max(m.amax() for m in members)

    def quantize_weight(self):
        return (nvr2bit_quantize(self.weight.detach(), self.group_amax(),
                                 self.num_probes),)

    @torch.no_grad()
    def refresh_buffers(self) -> None:
        self._wq.copy_(self._compute_wq())

    def forward(self, x: Tensor) -> Tensor:
        w = self._wq if not self.training else self._differentiable_weight()
        phase = _phase_for(x)
        if not torch.is_tensor(phase):
            return F.linear(self._quant_act(x) if phase else x, w, self.bias)
        return F.linear(torch.where(phase, self._quant_act(x, phase), x), w, self.bias)

    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _variant_weight(self, variant) -> Tensor:
        return self._wq

    def _variant_quantize_act(self, variant) -> bool:
        return variant != "decode"

    def export_tensors(self, variant=None) -> dict:
        # use codebook-recovered block scales, not block_amax/6
        from .luts_backend import _recover_scales
        gscale = self._variant_global_scale(variant)
        packed, wscale, wscale2 = pack_nvfp4_weight(
            self._wq.data, self.block_size, global_scale=gscale, signed=self.signed,
            block_eff=_recover_scales(self._wq.data, self.block_size))
        out = {"weight_packed": packed.cpu(), "weight_scale": wscale.cpu(),
               "weight_global_scale": (1.0 / wscale2).reshape(1).cpu()}
        if self._variant_quantize_act(variant):
            act = self.act_amax.float().clamp(min=1e-8)
            out["input_global_scale"] = (GLOBAL_DEN / act).reshape(1).cpu()
        if self.bias is not None:
            out["bias"] = self.bias.detach().to(torch.bfloat16).cpu()
        return out


class NVFP4NVR2BitSplitLinear(_NVFP4NVR2BitMixin, NVFP4Lloyd43SplitLinear):
    """Two masters: NVFP4 (W4A4) prefill, nvr2bit (W2A16) decode."""

    @torch.no_grad()
    def refresh_buffers(self) -> None:
        self._wq.copy_(self._compute_wq())
        self._wq_dec.copy_(self.quantize_decode_weight()[0])


def apply_nvfp4nvr2bitupcast(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4NVR2BitUpcastLinear.from_linear(
        lin, block_size=block_size))


def apply_nvfp4nvr2bitsplit(model: nn.Module, block_size: int = BLOCK) -> None:
    replace_linears(model, lambda lin: NVFP4NVR2BitSplitLinear.from_linear(
        lin, block_size=block_size))
