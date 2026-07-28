"""Prefill/decode dual-format quantization.

Inference splits into two regimes with opposite bottlenecks. Prefill is one large
compute-bound GEMM over the whole prompt, where quantizing activations to 4 bits
buys real speed. Decode is a memory-bound GEMV per token, where the cost is loading
weights and activation quantization buys little. So the natural pairing is
**NVFP4 (W4A4) for prefill, NVFP4A16 (W4A16) for decode** — and in a disaggregated
deployment those run on different workers, each loading its own checkpoint.

Two variants, differing only in whether the phases share a master weight:

  nvfp4pdshared  one master. Both checkpoints hold IDENTICAL weights (W4A4 and
                 W4A16 quantize weights the same way); they differ only in the
                 config and the presence of input_global_scale. The training
                 signal is what differs: the weights must serve both regimes.
  nvfp4pdsplit   two masters, both initialised from the BF16 base and trained
                 separately, so prefill and decode weights genuinely diverge.
                 Costs 2x the linear FLOPs (both paths are computed and combined
                 by mask) and 2x master/optimizer memory.

Which positions are which
-------------------------
`data.py` already gives it: labels[t] == input_ids[t] on assistant tokens and -100
elsewhere, so over INPUT positions `labels != -100` is exactly the decode phase.
That matches deployment: prompt tokens are processed in one prefill pass; each
generated assistant token is fed back one at a time. The causal shift in the loss
does not enter here — it selects prediction targets, not the format a position runs
under.

At inference there is no mask to thread: HF generate() runs one multi-token forward
(prefill) then single-token forwards (decode), so `x.shape[-2] > 1` recovers the
phase exactly. Decode attends to a KV cache built by the prefill format, which is
the deployment behaviour we want to measure.

The layers SUBCLASS NVFP4Linear rather than containing two of them, because the
exporter derives tensor keys from named_modules() — a contained submodule would emit
`...q_proj.prefill.weight_packed`, which vLLM cannot load. Everything the exporter
touches goes through the export_* accessors, so select_variant() can present either
weight under the correct key.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocked import BLOCK, GLOBAL_DEN, blocked_quantize, replace_linears, ste
from .nvfp4 import NVFP4Linear, fake_quant_ste, unpack_nvfp4_weight

# [B, T] bool, True where the PREFILL format applies. None => infer from shape.
_PHASE_MASK: Tensor | None = None


@contextmanager
def quant_phase(prefill_mask: Tensor | None):
    """Route positions to prefill/decode formats for the enclosed block.

    MUST span the backward pass as well as the forward: gradient checkpointing
    recomputes the forward during backward, and if the mask were cleared by then the
    recomputation would silently run every position through the decode format,
    producing gradients for a model that was never evaluated.
    """
    global _PHASE_MASK
    prev = _PHASE_MASK
    _PHASE_MASK = prefill_mask
    try:
        yield
    finally:
        _PHASE_MASK = prev


def prefill_mask_from_labels(labels: Tensor) -> Tensor:
    """[B, T] -> True on prefill (prompt/system) positions. Padding counts as
    prefill; it is excluded from the loss anyway."""
    return labels == -100


def _phase_for(x: Tensor):
    """True (all prefill), False (all decode), or a [B, T, 1] bool mask."""
    m = _PHASE_MASK
    if m is None:
        return x.shape[-2] > 1          # generate(): prompt pass vs per-token decode
    if x.dim() != 3:
        raise RuntimeError(
            f"dual-format layer got a {x.dim()}D input under an explicit phase mask; "
            "the mask is per (batch, position) and cannot be aligned")
    return m.unsqueeze(-1)


class _DualActMixin:
    """Activation quantization restricted to prefill positions.

    The observer must see prefill activations ONLY: input_global_scale is baked into
    the prefill checkpoint as a static scale, and decode-phase activations are never
    passed through it in deployment. Letting them into the running max would inflate
    the scale and cost precision on exactly the tensors it governs.
    """

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
    """One master weight, quantized once; activations quantized on prefill only."""

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)

    def forward(self, x: Tensor) -> Tensor:
        w = self._wq if not self.training else self._differentiable_weight()
        phase = _phase_for(x)
        # NOT `phase is True`: under torch.compile(dynamic=True) the shape comparison
        # yields a SymBool, which is neither singleton, so an identity test silently
        # falls through and hands a bool to torch.where(). Branch on whether this is a
        # per-position mask instead; truth-testing a SymBool guards the graph, which
        # is what produces the separate prefill and decode compilations.
        if not torch.is_tensor(phase):
            return F.linear(self._quant_act(x) if phase else x, w, self.bias)
        return F.linear(torch.where(phase, self._quant_act(x, phase), x), w, self.bias)

    # --- checkpoint format -------------------------------------------------
    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _variant_quantize_act(self, variant) -> bool:
        return variant != "decode"


class DualSplitNVFP4Linear(_DualActMixin, NVFP4Linear):
    """Two masters: `weight` serves prefill (W4A4), `decode_weight` serves decode
    (W4A16). Both start from the BF16 base, then diverge.

    Fused-group scale sharing needs no extra wiring: this layer IS a GroupScaled, so
    link_fused_groups() populates `_group` as usual, and the decode view simply takes
    its group max over the siblings' decode masters.
    """

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)
        self.decode_weight = nn.Parameter(weight.clone())
        self.register_buffer("_wq_dec", torch.empty_like(self._wq))
        with torch.no_grad():
            self._wq_dec.copy_(self.quantize_decode_weight()[0])

    # --- decode view of the two-level scaling ------------------------------
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
        # See DualSharedNVFP4Linear.forward: a SymBool under torch.compile is neither
        # `True` nor `False` by identity, so branch on tensor-ness.
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

    # --- checkpoint format -------------------------------------------------
    # Three tiny overrides are the whole difference: which master, which group
    # scale, and whether activations are quantized. Everything else — packing,
    # key names, config, the loader — is inherited from NVFP4Linear, so the two
    # exported directories are ordinary NVFP4 checkpoints that vLLM loads on their
    # own. That is what makes disaggregated prefill/decode serving possible.
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
        """Restore into the master this variant serves.

        Writes each variant's buffers DIRECTLY rather than going through the
        inherited path, so loading a prefill and a decode checkpoint into the same
        model is order-independent — routing decode through `_wq` would overwrite
        the prefill weight if prefill happened to be loaded first.
        """
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
