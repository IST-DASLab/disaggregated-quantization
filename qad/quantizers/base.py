"""Base class for quantized linear layers used in QAD training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# OPT-IN: recompute the hard-quantized weight in every forward instead of caching it.
#
# `_wq` (and `_wq_dec` for the split formats) is a full-size buffer per layer that FSDP
# and PP both leave unsharded, because the forward reads it directly. At gemma-3-12b
# split that is 19.8 GiB per pipeline stage -- the largest single item nothing else can
# shard, and the reason 12b split does not fit.
#
# Recomputing is EXACT, not an approximation: _wq is a pure function of the master
# weight, refreshed by post_update immediately after optimizer.step(), so computing it
# at forward time from that same unchanged master yields the same bits. What changes is
# only WHEN: once per step becomes once per forward, and gradient checkpointing reruns
# the forward, so roughly twice the quantize work per step.
#
# Off by default. It trades throughput for memory and is only worth it where the
# footprint is the binding constraint.
_RECOMPUTE_WQ = False


def set_recompute_wq(on: bool) -> None:
    """Set BEFORE the quantizers are constructed -- __init__ skips allocating the
    buffers, so flipping this afterwards would leave a model half in each mode."""
    global _RECOMPUTE_WQ
    _RECOMPUTE_WQ = bool(on)


def recompute_wq() -> bool:
    return _RECOMPUTE_WQ


def qlinear(x: Tensor, w: Tensor, bias: Tensor | None = None) -> Tensor:
    """F.linear with the ACTIVATION and BIAS cast to the weight's dtype.

    `_wq` is stored bf16 and every production forward runs under
    torch.amp.autocast(bfloat16), so x is already bf16 there and this is a no-op. It
    matters for direct calls that skip autocast -- tests, ad-hoc evaluation -- which would
    otherwise hit "expected mat1 and mat2 to have the same dtype".

    The ACTIVATION is cast, never the weight: casting the weight would allocate a full
    fp32 copy of every layer and undo the reason the buffer is bf16, and it would hand the
    GEMM an operand production never uses.

    The bias needs casting for the same reason and gets it for free: F.linear lowers to
    addmm(bias, x, w.T), where the bias is addmm's `self`, so an fp32 bias against a bf16
    weight fails with "self and mat2 must have the same dtype" -- an error that names
    neither the bias nor this function. It is one vector per layer, so unlike the weight
    there is no memory argument against casting it.
    """
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    if bias is not None and bias.dtype != w.dtype:
        bias = bias.to(w.dtype)
    return F.linear(x, w, bias)


class QuantizedLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with learnable quantization.

    Inherits from nn.Linear so that HuggingFace code which calls
    isinstance(layer, nn.Linear) — e.g. get_target_dtype in the flash-attention
    path — still finds our layers and can inspect .weight for the dtype.

    Maintains a hard-quantized weight buffer (_wq) that is refreshed once after
    each optimizer step via post_update().  forward() uses the subclass's
    differentiable quantization during training and the cached _wq during eval,
    so the quantization is only recomputed once per optimizer step — not once
    per microbatch.

    Subclass contract:
      from_linear(cls, linear, **kwargs)   – construct from nn.Linear
      _compute_wq()                        – hard-quantized weight (no grad)
      _differentiable_weight()             – soft/STE weight for training forward
      _update_schedule(step, total_steps)  – advance annealing scalars
    """

    def __init__(self, out_features: int, in_features: int, bias, dtype: torch.dtype,
                 device=None):
        # Skip nn.Linear.__init__ to avoid creating a duplicate weight parameter.
        # Call nn.Module.__init__ directly and set the attributes nn.Linear expects.
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self._out = out_features
        self._in = in_features
        self.bias = bias
        # Hard-quantized weight cache.  Refreshed by post_update() after each
        # optimizer step; used directly in eval forward and as an STE offset in
        # subclasses that want to avoid recomputing quantization every microbatch.
        #
        # BF16, NOT the master's fp32. Nothing here asks for the extra mantissa: this
        # buffer carries no gradient (refreshed under no_grad by post_update), and every
        # consumer discards the low bits anyway -- ste() is `x + (wq - x).detach()`, whose
        # forward value is exactly wq, and both the train and eval forwards run under
        # torch.amp.autocast(bfloat16), so the F.linear operand is rounded to BF16
        # regardless. Storing BF16 makes the GEMM input bit-identical to what fp32 storage
        # produced, while halving the buffer.
        #
        # Worth 20.0 GiB/GPU at gemma-3-12b for a single-buffer format (40.1 -> 20.0), and
        # 40.1 GiB for upcast/split, which hold two. That is what takes the 12b upcast
        # SETUP peak from 145.9 to 105.8 GiB, i.e. what lets 12b upcast run at all.
        #
        # The dtype is pinned here rather than inherited: `dtype` still describes the
        # master, and subclasses build _wq_dec with torch.empty_like(self._wq), so they
        # follow automatically.
        # Zero-element when recomputing: the attribute must still EXIST because
        # subclasses build _wq_dec with torch.empty_like(self._wq) and other code tests
        # for it, but it holds nothing and costs nothing. numel() == 0 is the mode flag
        # every reader below keys on, so a layer cannot be half-converted.
        _shape = (0,) if _RECOMPUTE_WQ else (out_features, in_features)
        self.register_buffer("_wq", torch.empty(*_shape, dtype=torch.bfloat16,
                                                device=device))

    @property
    def wq(self) -> Tensor:
        """The `_wq` buffer, unchanged. An alias, deliberately NOT a cast.

        No dtype conversion happens here: every real forward -- training and eval alike --
        runs under torch.amp.autocast(bfloat16), so bf16 is already the operand dtype
        F.linear wants, and upcasting would both undo the storage saving (a full fp32 copy
        of each layer's weight, per forward) and hand the GEMM an operand production never
        sees. Callers that build an F.linear by hand must supply BF16 activations, exactly
        as autocast does.

        Kept as a named accessor rather than reverting to `self._wq` everywhere because it
        marks the GEMM-operand uses, which is where a future dtype question belongs -- and
        because returning the buffer ITSELF preserves identity, which formats like
        upcastboth assert on (prefill and decode must ship one tensor).
        """
        if self._wq.numel() == 0:            # --recompute-wq: no cache to read
            # .to(bfloat16) is REQUIRED, not cosmetic. The cache is a bf16 buffer, so
            # the cached mode serves a bf16-rounded weight; _compute_wq returns fp32.
            # Without this the two modes differ -- recompute would quietly train against
            # a MORE precise weight than the cache ever provides, which is a silent
            # change to the format rather than a memory optimisation.
            return self._compute_wq().to(torch.bfloat16)
        return self._wq

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "QuantizedLinear":
        raise NotImplementedError

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        """Return the hard-quantized weight (no grad). Called by post_update."""
        raise NotImplementedError

    def _differentiable_weight(self) -> Tensor:
        """Return a differentiable pseudo-quantized weight for the training forward."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        w = self.wq if not self.training else self._differentiable_weight()
        return qlinear(x, w, self.bias)

    def _update_schedule(self, step: int, total_steps: int) -> None:
        """Override to advance temperature, scale, or other annealing scalars."""

    @torch.no_grad()
    def post_update(self, step: int, total_steps: int) -> None:
        """Called after optimizer.step(): advance schedule and refresh _wq buffer."""
        self._update_schedule(step, total_steps)
        if self._wq.numel():                 # nothing to refresh when recomputing
            self._wq.copy_(self._compute_wq())

    # ------------------------------------------------------------------
    # Checkpoint format — owned by the layer, not by an external exporter
    # ------------------------------------------------------------------
    # Each format decides what it serializes and how to read it back, so writer and
    # reader sit next to each other and cannot drift. export/save.py only assembles:
    # it walks the model, prefixes these keys with the module path, adds the
    # non-quantized tensors, and writes config.json.
    #
    # The default is pseudo-quantization: hand back the hard-quantized weight in the
    # standard `weight` slot, so the result is an ordinary HF model with the
    # quantization error baked in. STE / QuEST / Lloyd / GSQ need nothing more.

    @classmethod
    def export_variants(cls) -> list:
        """Checkpoints this format emits per step. [None] -> a single checkpoint
        written directly to the step directory. Formats that serve different phases
        (see quantizers/dual.py) return one name per phase."""
        return [None]

    def export_config(self, variant=None) -> dict | None:
        """`quantization_config` for config.json, or None to write a plain HF model."""
        return None

    def export_tensors(self, variant=None) -> dict[str, Tensor]:
        """Tensors this layer contributes, keyed RELATIVE to the layer."""
        out = {"weight": self.wq.detach().to(torch.bfloat16).cpu()}
        if self.bias is not None:
            out["bias"] = self.bias.detach().to(torch.bfloat16).cpu()
        return out

    @torch.no_grad()
    def load_tensors(self, tensors: dict[str, Tensor], variant=None) -> None:
        """Inverse of export_tensors: restore this layer from a checkpoint.

        Used to rebuild a quantized model for evaluation without re-running the
        quantizer, and to reconstruct the prefill/decode formats side by side.
        """
        w = tensors["weight"].to(device=self._wq.device, dtype=self._wq.dtype)
        self._wq.copy_(w)
        if self.bias is not None and "bias" in tensors:
            self.bias.data.copy_(tensors["bias"].to(self.bias.device, self.bias.dtype))


def post_update_all(model: nn.Module, step: int, total_steps: int) -> None:
    """Call post_update on every QuantizedLinear in model after optimizer.step()."""
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.post_update(step, total_steps)
