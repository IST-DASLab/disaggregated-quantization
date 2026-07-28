"""Signed Lloyd 3-bit weight-only quantization (pseudo-quantization).

Same two-level block scaling as NVFP4 (blocks of 16, FP32 per-tensor global +
FP8-E4M3 per-block, fused-group scale sharing) but with:

  * a SIGNED per-block scale — it absorbs the sign of the block's max-abs element,
    so after normalization the extreme element is always +6 and the value
    distribution is asymmetric;
  * an 8-level (3-bit) MSE-optimal Lloyd grid instead of the 16-level E2M1 grid.

On a Gaussian this is the best of the 3-bit options measured in ../../grids.ipynb
(relative MSE 0.0218, vs 0.0334 for the FP4-pair-downcast LUT and 0.0091 for 4-bit
NVFP4 itself).

There is no 3-bit LUT kernel to serve this, so it is **pseudo-quantized**: the
checkpoint stores the dequantized bf16 weights (the standard HF export path), and
vLLM runs it as an ordinary bf16 model. Weights only — activations stay bf16.
"""

import torch.nn as nn
from torch import Tensor

from .blocked import BlockScaledLinear, grid_rounder, replace_linears
from .grids import FP4_DOWNCAST_SIGNED_3BIT, LLOYD_SIGNED_3BIT

GRIDS = {
    "lloyd": LLOYD_SIGNED_3BIT,               # MSE-optimal (default)
    "downcast": FP4_DOWNCAST_SIGNED_3BIT,     # FP4-pair centers of mass
}


class SignedLloydLinear(BlockScaledLinear):
    """Weight-only 3-bit linear: signed block scale + MSE-optimal Lloyd grid."""

    signed = True

    def __init__(self, weight: Tensor, bias, block_size: int = 16, grid: str = "lloyd"):
        self.grid_name = grid
        self._grid = GRIDS[grid].to(weight.device)
        self._round = grid_rounder(self._grid)
        super().__init__(weight, bias, block_size=block_size)

    def rounder(self, x: Tensor) -> Tensor:
        if self._grid.device != x.device:       # first forward after .to(device)
            self._grid = self._grid.to(x.device)
            self._round = grid_rounder(self._grid)
        return self._round(x)


def apply_lloyd3bit(model: nn.Module, block_size: int = 16, grid: str = "lloyd") -> None:
    """Replace every nn.Linear (except lm_head) with SignedLloydLinear, in-place."""
    replace_linears(
        model,
        lambda lin: SignedLloydLinear.from_linear(lin, block_size=block_size, grid=grid),
    )
