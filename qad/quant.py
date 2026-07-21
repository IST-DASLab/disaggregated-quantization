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
