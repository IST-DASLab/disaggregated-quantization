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
from .grids import (FP4_DOWNCAST_SIGNED_3BIT, LLOYD21_SIGNED_2BIT,
                    LLOYD43_SIGNED_3BIT, LLOYD_SIGNED_3BIT)

GRIDS = {
    "lloyd": LLOYD_SIGNED_3BIT,               # MSE-optimal (default)
    "lloyd43": LLOYD43_SIGNED_3BIT,           # MSE-optimal with 0.0 and 6.0 pinned
    "lloyd21": LLOYD21_SIGNED_2BIT,           # the lloyd43 construction at 2 bits
    "downcast": FP4_DOWNCAST_SIGNED_3BIT,     # FP4-pair centers of mass
}


class SignedLloydLinear(BlockScaledLinear):
    """Weight-only LUT linear: signed block scale + an MSE-optimal Lloyd grid.

    Bit width is whatever the grid says (8 levels = 3 bits, 4 = 2 bits); nothing here
    depends on it, because these formats ship pseudo-quantized to bf16 -- there is no
    packed layout whose width would have to match.
    """

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


def apply_lloyd43(model: nn.Module, block_size: int = 16, grid: str = "lloyd43") -> None:
    """Same layer as apply_lloyd3bit, on the 0/6-pinned grid.

    A separate entry point rather than a bare grid argument so the REGISTRY reads as one
    method per line, and so the two show up as distinct checkpoint tags -- they are
    different formats to compare, not two settings of one.
    """
    replace_linears(
        model,
        lambda lin: SignedLloydLinear.from_linear(lin, block_size=block_size, grid=grid),
    )


def apply_lloyd21(model: nn.Module, block_size: int = 16, grid: str = "lloyd21") -> None:
    """Same layer as apply_lloyd43, on the 2-bit 0/6-pinned grid.

    Its own entry point for the same reason apply_lloyd43 has one: the REGISTRY reads as
    one method per line, and the two get distinct checkpoint tags -- 2-bit vs 3-bit is a
    comparison, not a setting.
    """
    replace_linears(
        model,
        lambda lin: SignedLloydLinear.from_linear(lin, block_size=block_size, grid=grid),
    )
