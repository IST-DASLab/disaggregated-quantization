"""Crude PyTorch baseline: decode the packed 3-bit weight, then multiply in bf16.

This is the correctness anchor, not a fast path. It materializes the whole dense bf16
weight before doing any arithmetic, which is exactly the memory traffic the Triton kernel
exists to avoid -- so it is also the honest "what you would otherwise have to do" baseline
for the speedup numbers in bench.py.

Two implementations, both computing y = x @ W.T:

  gemv_reference       vectorized decode, one bf16 matmul. Slow because of the decode,
                       but usable at real model sizes.
  gemv_reference_loop  an explicit Python loop over blocks of 16, unpacking bit by bit.
                       Unusably slow and deliberately so -- it is written to be read and
                       believed, and the tests check the fast reference against it.
"""

import torch
from torch import Tensor

from .format import BITS, BLOCK, GROUP, LUT, dequantize, effective_scales

__all__ = ["gemv_reference", "gemv_reference_loop", "gemv_bf16"]


def gemv_bf16(x: Tensor, w: Tensor) -> Tensor:
    """The thing we are trying to beat: a dense bf16 vector-matrix product."""
    return torch.mv(w, x) if x.dim() == 1 else x @ w.T


def gemv_reference(x: Tensor, packed: Tensor, block_scale: Tensor,
                   global_scale: Tensor, K: int) -> Tensor:
    """y = x @ dequantize(packed).T, computed in bf16.

    x may be (K,) or (M, K). The decode runs in float32 and is cast to bf16 before the
    matmul, so the multiply itself is bf16 x bf16 -- the same arithmetic the Triton kernel
    accumulates, which keeps the comparison meaningful.
    """
    w = dequantize(packed, block_scale, global_scale, K, dtype=torch.bfloat16)
    return gemv_bf16(x.to(torch.bfloat16), w)


def gemv_reference_loop(x: Tensor, packed: Tensor, block_scale: Tensor,
                        global_scale: Tensor, K: int) -> Tensor:
    """The same product written out one block at a time. Ground truth, not a kernel.

    O(N * K/16) Python iterations -- keep N and K small (a few hundred) or it will take
    minutes. Nothing here is clever on purpose: if this disagrees with anything else in
    the directory, this one is right.
    """
    N = packed.shape[0]
    lut = LUT.to(packed.device, torch.float32)
    eff = effective_scales(block_scale, global_scale)        # (N, K // BLOCK)
    xf = x.to(torch.bfloat16).reshape(-1, K)
    M = xf.shape[0]
    out = torch.zeros(M, N, dtype=torch.float32, device=packed.device)

    words = packed.to(torch.int64)
    for n in range(N):
        for b in range(K // BLOCK):
            scale = eff[n, b].item()
            for t in range(BLOCK):
                k = b * BLOCK + t
                g, j = divmod(k, GROUP)
                idx = 0
                for p in range(BITS):
                    idx |= int((words[n, p, g].item() >> j) & 1) << p
                wk = lut[idx].item() * scale
                out[:, n] += xf[:, k].float() * torch.tensor(wk, dtype=torch.bfloat16).float()

    out = out.to(torch.bfloat16)
    return out.reshape(N) if x.dim() == 1 else out
