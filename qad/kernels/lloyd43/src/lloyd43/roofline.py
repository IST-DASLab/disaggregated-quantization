"""Measured memory roofline: what this device gives for a transfer of a given size.

Why this exists. The obvious way to state the lloyd43 speedup ceiling is the traffic
ratio, 2.0 / 0.4375 = 4.57x. That is wrong, and wrong in the optimistic direction:
achievable bandwidth is a function of HOW MUCH you read. A 28 MiB read cannot reach the
GB/s a 128 MiB read can (cold-L2 ramp, launch overhead amortized over less work, fewer
concurrent streams to hide latency with). So the quantized format is penalised by exactly
the property that makes it good -- it moves less data, and less data moves slower.

On GB10 the measured curve is steep: 107 GB/s at 7 MiB, 156 at 28 MiB, 223 at 128 MiB,
244 at 1 GiB. At N=4096 K=16384 that turns a nominal 4.57x into an actual 3.18x.

`speedup_ceiling` measures the honest number for a shape: time a flat contiguous read of
each format's footprint and take the ratio. It is an upper bound no kernel can beat,
because it does nothing but read.
"""

import torch
import triton
import triton.language as tl

__all__ = ["flat_read_ms", "flat_read_gbs", "speedup_ceiling"]


@triton.jit
def _sum_kernel(p, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in tl.range(pid * BLOCK, n, tl.num_programs(0) * BLOCK):
        o = off + tl.arange(0, BLOCK)
        acc += tl.load(p + o, mask=o < n, other=0.0)
    tl.store(out + pid, tl.sum(acc))


def flat_read_ms(nbytes: int, grid: int = 1024, block: int = 2048) -> float:
    """Milliseconds to read `nbytes` contiguously, L2 flushed as do_bench does.

    This is the floor for any kernel that must touch that many bytes: perfectly
    coalesced, no arithmetic, no structure.
    """
    n = max(int(nbytes) // 4, block)
    buf = torch.empty(n, device="cuda", dtype=torch.float32)
    out = torch.empty(grid, device="cuda", dtype=torch.float32)
    try:
        return triton.testing.do_bench(lambda: _sum_kernel[(grid,)](buf, out, n, BLOCK=block),
                                       warmup=25, rep=100)
    finally:
        del buf, out


def flat_read_gbs(nbytes: int) -> float:
    return nbytes * 1e-6 / flat_read_ms(nbytes)


def speedup_ceiling(N: int, K: int, bytes_base: float = 2.0,
                    bytes_quant: float = 0.4375) -> float:
    """Best speedup physically available at this shape: t_flat(base) / t_flat(quant).

    Not the traffic ratio. Always <= it, and on a small-cache / low-bandwidth part
    noticeably so.
    """
    return flat_read_ms(int(N * K * bytes_base)) / flat_read_ms(int(N * K * bytes_quant))
