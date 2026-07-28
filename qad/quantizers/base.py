"""Base class for quantized linear layers used in QAD training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        self.register_buffer("_wq", torch.empty(out_features, in_features, dtype=dtype,
                                                 device=device))

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
        w = self._wq if not self.training else self._differentiable_weight()
        return F.linear(x, w, self.bias)

    def _update_schedule(self, step: int, total_steps: int) -> None:
        """Override to advance temperature, scale, or other annealing scalars."""

    @torch.no_grad()
    def post_update(self, step: int, total_steps: int) -> None:
        """Called after optimizer.step(): advance schedule and refresh _wq buffer."""
        self._update_schedule(step, total_steps)
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
        out = {"weight": self._wq.detach().to(torch.bfloat16).cpu()}
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
