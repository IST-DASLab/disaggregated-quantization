"""LUT rounding is chunked for peak memory; assert it changed NOTHING numerically.

round_to_grid materialises ~6 tensors the size of its input inside index_nearest, which
at gemma-3-12b means ~470 MB each for the int64 ones. Chunking bounds that. The whole
value of the change depends on it being EXACT -- every nvfp4lloyd*/lloyd* checkpoint on
disk was produced by the unchunked path, so a difference here would silently invalidate
comparisons against all of them, and the quantizer tag hashes would not notice (they hash
quantizer PARAMETERS, not this code).

Tested against an explicit unchunked reference rather than against itself.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from quantizers import blocked
from quantizers.blocked import BLOCK, blocked_quantize, grid_rounder, round_to_grid
from quantizers.grids import LLOYD21_SIGNED_2BIT, LLOYD43_SIGNED_3BIT

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def unchunked(x, grid):
    """The pre-change implementation, kept here as the reference."""
    g = grid.to(x.device)
    return g[blocked.index_nearest(x, g)]


def test_round_to_grid_is_bit_identical():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, grid in (("LUT3", LLOYD43_SIGNED_3BIT), ("LUT2", LLOYD21_SIGNED_2BIT)):
        g = grid.to(dev)
        for shape in ((2048, 512), (4096, 1024), (37, 13), (1, 4096)):
            torch.manual_seed(0)
            x = torch.randn(*shape, device=dev) * 3.0
            got, ref = round_to_grid(x, g), unchunked(x, g)
            check(f"{name} {shape}: chunked == unchunked",
                  torch.equal(got, ref),
                  f"max|Δ|={(got - ref).abs().max().item():.3e}")


def test_chunking_actually_engages():
    """A tensor over the budget must take the chunked path, or the test proves nothing."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    big = blocked._LUT_CHUNK_ELEMS * 2
    rows = 64
    x = torch.randn(rows, big // rows, device=dev)
    check("test tensor exceeds the chunk budget", x.numel() > blocked._LUT_CHUNK_ELEMS,
          f"{x.numel()} > {blocked._LUT_CHUNK_ELEMS}")
    got, ref = round_to_grid(x, LLOYD43_SIGNED_3BIT.to(dev)), unchunked(
        x, LLOYD43_SIGNED_3BIT.to(dev))
    check("chunked path is still bit-identical on a large tensor", torch.equal(got, ref))


def test_blocked_quantize_end_to_end_unchanged():
    """The caller that matters: blocked_quantize with a LUT rounder."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    w = torch.randn(2048, 1024, device=dev)
    g = LLOYD21_SIGNED_2BIT.to(dev)

    q_new, bs_new, gs_new = blocked_quantize(w, grid_rounder(g), BLOCK)
    saved = blocked.round_to_grid
    try:                                   # force the unchunked reference through
        blocked.round_to_grid = unchunked
        q_ref, bs_ref, gs_ref = blocked_quantize(w, grid_rounder(g), BLOCK)
    finally:
        blocked.round_to_grid = saved
    check("blocked_quantize weights identical", torch.equal(q_new, q_ref),
          f"max|Δ|={(q_new - q_ref).abs().max().item():.3e}")
    check("blocked_quantize block scales identical", torch.equal(bs_new, bs_ref))
    check("blocked_quantize global scale identical", torch.equal(gs_new, gs_ref))


def test_peak_memory_is_bounded():
    """The point of the change: temporaries must not scale with the whole tensor."""
    if not torch.cuda.is_available():
        print("  SKIP  peak memory needs cuda")
        return
    g = LLOYD43_SIGNED_3BIT.cuda()
    x = torch.randn(4096, 8192, device="cuda")          # 33.5M elements
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    round_to_grid(x, g)
    torch.cuda.synchronize()
    chunked_peak = torch.cuda.max_memory_allocated() - base

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    unchunked(x, g)
    torch.cuda.synchronize()
    full_peak = torch.cuda.max_memory_allocated() - base

    check("chunked peak is materially lower than unchunked",
          chunked_peak < full_peak * 0.6,
          f"{chunked_peak/2**20:.0f} MiB vs {full_peak/2**20:.0f} MiB")


def main():
    for fn in (test_round_to_grid_is_bit_identical,
               test_chunking_actually_engages,
               test_blocked_quantize_end_to_end_unchanged,
               test_peak_memory_is_bounded):
        fn()
    print("ALL PASS" if not FAILED else f"FAILED: {FAILED}")
    if FAILED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
