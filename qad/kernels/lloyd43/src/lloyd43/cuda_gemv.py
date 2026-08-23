"""CUDA GEMV for lloyd43 and lloyd21: the kernel everything else runs through.

The extension is JIT-compiled by `torch.utils.cpp_extension.load` on first use and cached
under ~/.cache/torch_extensions, so the package still installs with nothing but torch and
there is no build step to keep in sync with the working tree. `csrc/lloyd43_gemv.cu` is
the kernel; read its header for why it is shaped the way it is.

Set LLOYD43_CUDA_ARCH to override the target (e.g. "100a" for B200, "121a" for GB10).
By default it compiles for the visible device only, which keeps the first build to a few
seconds. The "a" suffix matters on Blackwell: sm_100/sm_103/sm_120/sm_121 need the
architecture-specific targets to get the right SASS.
"""

import os
import threading

import torch
from torch import Tensor

from .format import (Format, LLOYD43, BLOCK, GROUP, LUT, format_for_packed,
                     lut_for)

__all__ = ["gemv_lloyd43_cuda", "dequant_cuda", "gemv_op", "linear_op",
           "load_extension", "is_available", "CUDA_CONFIGS", "auto_config",
           "CUDA_SHAPE_TABLES"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "csrc", "lloyd43_gemv.cu"))

_lock = threading.Lock()
_ext = None

# (warps, rows_per_warp, gpl) the kernel is instantiated for.
#   warps * rows_per_warp = rows per CTA. More amortizes the staged x further; fewer keeps
#     more CTAs resident.
#   gpl = packed words each lane takes per step. A warp covers 1024*gpl weights, so gpl
#     must not exceed K/1024 or lanes idle -- that was worth up to 1.7x on the small
#     projections. Also selects the vector width of the packed load (int / int2 / int4).
#   stage_x = keep x in shared memory (True) or read it straight to registers (False).
#     Staging amortizes x across the CTA's rows, but at small K the buffer costs more
#     traffic than the weights it serves and adds two __syncthreads per chunk.
CUDA_CONFIGS = [(w, r, g, s)
                for w, r in ((16, 2), (16, 1), (8, 4), (8, 2), (8, 1),
                             (4, 4), (4, 2), (4, 1))
                for g in (1, 2, 4)
                for s in (True, False)]


def _arch() -> str:
    override = os.environ.get("LLOYD43_CUDA_ARCH")
    if override:
        return override.strip().lower().replace("sm_", "").replace(".", "")
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    # Blackwell wants the architecture-specific cubin; plain sm_100 silently loses
    # instructions the "a" target has.
    return arch + "a" if arch in {"100", "103", "120", "121"} else arch


def load_extension(verbose: bool = False):
    """Compile (once) and return the extension module."""
    global _ext
    if _ext is not None:
        return _ext
    with _lock:
        if _ext is None:
            from torch.utils.cpp_extension import load
            arch = _arch()
            _ext = load(
                name="lloyd43_cuda",
                sources=[_SRC],
                extra_cuda_cflags=[
                    "-O3", "--use_fast_math", "-std=c++17",
                    "-gencode", f"arch=compute_{arch},code=sm_{arch}",
                ],
                extra_cflags=["-O3", "-std=c++17"],
                verbose=verbose,
            )
    return _ext


def is_available() -> bool:
    try:
        load_extension()
        return True
    except Exception:
        return False


# Measured best (warps, rows_per_warp, gpl, stage_x) per (N, K), from tune_cuda.py on GB10.
# Keyed on the exact shape for every projection in the Qwen3 AND Gemma 3 models, fused and
# unfused; anything else falls back to `_heuristic`. Regenerate with
# `python tune_cuda.py --emit`.
#
# Gemma 3's shapes are unlike Qwen3's -- K=640, and kv projections only 256 rows tall -- and
# the heuristic does not serve them well: tuning is worth up to 1.21x on those, which is the
# difference between a fair per-model number and one that understates the format on exactly
# the models being compared.
_TABLE_LLOYD43: dict[tuple[int, int], tuple[int, int, int, bool]] = {
    (  256,   640): ( 4, 1, 1, True),
    (  256,  1152): ( 4, 1, 1, True),
    (  640,  1024): ( 8, 1, 1, False),
    (  640,  2048): ( 4, 4, 2, False),
    ( 1024,   640): ( 4, 2, 1, False),
    ( 1024,  1024): ( 4, 1, 1, True),
    ( 1024,  1152): ( 8, 1, 2, True),
    ( 1024,  2048): ( 8, 1, 1, False),
    ( 1024,  2560): ( 8, 1, 2, True),
    ( 1024,  3072): ( 8, 1, 1, False),
    ( 1024,  4096): ( 8, 1, 1, False),
    ( 1152,  1024): ( 8, 1, 2, True),
    ( 1152,  6912): ( 8, 1, 4, True),
    ( 1536,   640): ( 4, 2, 1, True),
    ( 1536,  1152): ( 8, 1, 2, True),
    ( 2048,   640): ( 4, 2, 1, True),
    ( 2048,  1024): ( 4, 2, 1, True),
    ( 2048,  2048): (16, 1, 1, False),
    ( 2048,  2560): ( 4, 1, 1, True),
    ( 2048,  3840): ( 8, 1, 2, True),
    ( 2048,  6144): ( 8, 1, 1, False),
    ( 2560,  2048): ( 4, 1, 1, False),
    ( 2560,  4096): ( 4, 1, 1, False),
    ( 2560,  9728): ( 8, 2, 2, True),
    ( 2560, 10240): ( 8, 1, 1, False),
    ( 3072,  1024): (16, 1, 1, True),
    ( 3840,  4096): ( 8, 1, 1, False),
    ( 3840, 15360): ( 4, 4, 1, False),
    ( 4096,   640): ( 4, 2, 1, True),
    ( 4096,  1024): (16, 1, 1, True),
    ( 4096,  2048): ( 8, 1, 2, True),
    ( 4096,  2560): (16, 1, 1, True),
    ( 4096,  3840): (16, 1, 4, True),
    ( 4096,  4096): (16, 1, 1, False),
    ( 4096, 12288): ( 8, 1, 2, True),
    ( 6144,  1024): ( 4, 1, 2, True),
    ( 6144,  2048): ( 4, 1, 2, True),
    ( 6144,  2560): (16, 1, 1, True),
    ( 6144,  4096): ( 8, 1, 2, True),
    ( 6912,  1152): ( 4, 1, 2, True),
    ( 8192,  3840): ( 4, 1, 4, True),
    ( 9728,  2560): ( 8, 1, 2, True),
    (10240,  2560): ( 8, 1, 4, False),
    (12288,  2048): ( 4, 1, 2, True),
    (12288,  4096): ( 4, 1, 4, True),
    (13824,  1152): ( 8, 1, 2, True),
    (15360,  3840): (16, 1, 4, True),
    (19456,  2560): (16, 1, 4, True),
    (20480,  2560): (16, 1, 4, True),
    (24576,  4096): ( 4, 1, 4, True),
    (30720,  3840): (16, 1, 4, True),
}

# lloyd21 gets its OWN table. The config space is identical -- W = K/32 whichever format,
# so the same (warps, rows, gpl, stage_x) tuples are legal -- but the optima are not: at 2
# bit-planes a lane loads 2/3 of the packed words per step, so the balance between staging
# x, amortizing rows and keeping CTAs resident lands elsewhere. Reusing lloyd43's table was
# measurably leaving performance behind on the shapes where the two disagree.
_TABLE_LLOYD21: dict[tuple[int, int], tuple[int, int, int, bool]] = {
    (  256,   640): (16, 1, 1, False),
    (  256,  1152): ( 8, 1, 1, False),
    (  640,  1024): (16, 1, 1, False),
    (  640,  2048): (16, 1, 1, False),
    ( 1024,   640): ( 4, 2, 1, False),
    ( 1024,  1024): ( 4, 1, 1, False),
    ( 1024,  1152): ( 4, 2, 1, False),
    ( 1024,  2048): ( 8, 1, 1, False),
    ( 1024,  2560): ( 4, 1, 1, False),
    ( 1024,  3072): (16, 1, 1, False),
    ( 1024,  4096): ( 8, 1, 1, False),
    ( 1152,  1024): ( 4, 1, 1, True),
    ( 1152,  6912): (16, 1, 4, True),
    ( 1536,   640): (16, 1, 1, True),
    ( 1536,  1152): ( 8, 1, 2, True),
    ( 2048,   640): ( 4, 2, 1, True),
    ( 2048,  1024): ( 4, 2, 1, True),
    ( 2048,  2048): ( 8, 1, 1, False),
    ( 2048,  2560): ( 4, 2, 1, True),
    ( 2048,  3840): ( 8, 1, 2, True),
    ( 2048,  6144): ( 8, 1, 1, False),
    ( 2560,  2048): (16, 1, 1, False),
    ( 2560,  4096): ( 8, 1, 1, False),
    ( 2560,  9728): ( 8, 2, 2, True),
    ( 2560, 10240): ( 8, 1, 1, False),
    ( 3072,  1024): ( 4, 2, 1, True),
    ( 3840,  4096): ( 8, 1, 1, False),
    ( 3840, 15360): ( 8, 4, 1, False),
    ( 4096,   640): ( 4, 1, 1, True),
    ( 4096,  1024): ( 8, 1, 1, False),
    ( 4096,  2048): (16, 1, 1, False),
    ( 4096,  2560): ( 4, 1, 1, False),
    ( 4096,  3840): ( 8, 1, 2, True),
    ( 4096,  4096): (16, 1, 1, False),
    ( 4096, 12288): ( 8, 1, 1, False),
    ( 6144,  1024): ( 4, 2, 1, True),
    ( 6144,  2048): ( 8, 1, 2, True),
    ( 6144,  2560): ( 8, 1, 1, False),
    ( 6144,  4096): ( 4, 1, 4, True),
    ( 6912,  1152): ( 4, 1, 2, True),
    ( 8192,  3840): ( 4, 1, 4, True),
    ( 9728,  2560): ( 8, 1, 1, False),
    (10240,  2560): ( 8, 1, 4, True),
    (12288,  2048): ( 4, 1, 2, True),
    (12288,  4096): ( 4, 1, 4, True),
    (13824,  1152): ( 4, 1, 2, True),
    (15360,  3840): ( 4, 1, 4, True),
    (19456,  2560): ( 8, 1, 1, False),
    (20480,  2560): ( 8, 1, 1, False),
    (24576,  4096): ( 8, 1, 4, True),
    (30720,  3840): ( 8, 2, 4, True),
}

CUDA_SHAPE_TABLES = {"lloyd43": _TABLE_LLOYD43, "lloyd21": _TABLE_LLOYD21}
# Back-compat alias: the 3-bit table is what existing callers meant.
CUDA_SHAPE_TABLE = _TABLE_LLOYD43


def _heuristic(N: int, K: int) -> tuple[int, int, int, bool]:
    """Fallback for shapes not in the measured table.

    gpl is the load-bearing one: a warp covers 1024*gpl weights per step, so a gpl larger
    than K/1024 leaves lanes with nothing to do. Rows per CTA then trades x-amortization
    against having enough CTAs to fill 48 SMs.
    """
    gpl = 4 if K >= 4096 else (2 if K >= 2048 else 1)
    warps, rows = ((16, 2) if N * K > (4 << 20) else (8, 1))
    return warps, rows, gpl, True


def auto_config(N: int, K: int, fmt: Format | None = None
                ) -> tuple[int, int, int, bool]:
    """(warps, rows_per_warp, gpl, stage_x) for a shape -- measured where we have it.

    Per format: the 2-bit kernel moves 2/3 the packed words per step, so it does not want
    the same tile as the 3-bit one. Falls back to lloyd43's table (then the heuristic) for
    a format with no measurement yet, which is better than the heuristic alone.
    """
    name = (fmt or LLOYD43).name
    table = CUDA_SHAPE_TABLES.get(name, _TABLE_LLOYD43)
    return table.get((N, K)) or _TABLE_LLOYD43.get((N, K)) or _heuristic(N, K)


def gemv_lloyd43_cuda(x: Tensor, packed: Tensor, block_scale: Tensor,
                      global_scale: Tensor, K: int, warps: int | None = None,
                      rows_per_warp: int | None = None,
                      gpl: int | None = None, stage_x: bool | None = None,
                      fmt: Format | None = None) -> Tensor:
    """y = x @ dequantize(packed).T for a single bf16 vector x of length K.

    Kept a plain positional signature so it is easy to call from tests and from
    bench.py and the tests. `global_scale` must be a device tensor and is read on the
    device: accepting a Python float would mean an .item() sync per call, which cost
    6.8 us of a 35 us kernel at (1024, 1024).
    """
    if x.dim() != 1:
        raise ValueError(f"gemv_lloyd43_cuda takes a vector; got {tuple(x.shape)}")
    if K % GROUP:
        raise ValueError(f"K={K} must be a multiple of {GROUP}")
    if not isinstance(global_scale, Tensor):
        raise TypeError("global_scale must be a device tensor, not "
                        f"{type(global_scale).__name__} -- a float would force a sync")

    if warps is None or rows_per_warp is None or gpl is None or stage_x is None:
        aw, ar, ag, as_ = auto_config(packed.shape[0], K, fmt or format_for_packed(packed))
        warps = aw if warps is None else warps
        rows_per_warp = ar if rows_per_warp is None else rows_per_warp
        gpl = ag if gpl is None else gpl
        stage_x = as_ if stage_x is None else stage_x

    ext = load_extension()
    x = x.contiguous().to(torch.bfloat16)
    packed = packed.contiguous()
    scale_u8 = block_scale.contiguous().view(torch.uint8)
    lut = lut_for(x.device, fmt or format_for_packed(packed))          # cached: see format.lut_for
    gs = global_scale.to(x.device, torch.float32).reshape(()).contiguous()

    return ext.gemv(x, packed, scale_u8, lut, gs, K, rows_per_warp, warps, gpl,
                    bool(stage_x))


# ------------------------------------------------------------------ torch.compile support
# Registered as a custom op so torch.compile treats the kernel as an opaque, side-effect
# free call it can put inside a graph. That is what lets vLLM (and mode="reduce-overhead")
# capture a whole decode step into a CUDA graph and pay the ~4.2 us launch overhead once
# for the step instead of once per projection -- which is most of what a batch-1 decode is.
#
# This only works because the kernel has no host sync in it: reading global_scale with
# .item() would break capture outright. See the note in gemv_lloyd43_cuda.
#
# THE LUT IS NOT AN ARGUMENT, DELIBERATELY. It is the same 8 values for every layer in the
# model, and inductor recognises that: passed as a tensor it gets folded into a host-side
# constant and copied in, which fails CUDA graph capture with "Cannot copy between CPU and
# CUDA tensors during CUDA graph capture" thrown from generated code that names nothing
# recognisable. Caching it per device here keeps it out of the graph entirely.
@torch.library.custom_op("lloyd43::gemv", mutates_args=())
def _gemv_op(x: Tensor, packed: Tensor, block_scale_u8: Tensor, gscale: Tensor,
             K: int, warps: int, rows_per_warp: int, gpl: int,
             stage_x: bool) -> Tensor:
    return load_extension().gemv(x, packed, block_scale_u8, lut_for(x.device, format_for_packed(packed)), gscale,
                                 K, rows_per_warp, warps, gpl, stage_x)


@_gemv_op.register_fake
def _(x, packed, block_scale_u8, gscale, K, warps, rows_per_warp, gpl, stage_x):
    return torch.empty(packed.shape[0], dtype=x.dtype, device=x.device)


def gemv_op(x: Tensor, packed: Tensor, block_scale_u8: Tensor, gscale: Tensor,
            K: int, cfg: tuple[int, int, int, bool]) -> Tensor:
    """Compile-safe entry point. Inputs must already be contiguous and correctly typed."""
    w, r, g, sx = cfg
    return torch.ops.lloyd43.gemv(x, packed, block_scale_u8, gscale, K, w, r, g, sx)


# A whole quantized linear as ONE opaque op, batch dispatch included. The obvious
# alternative -- branch in Python and let the M > 1 path fall back to `format.dequantize`
# -- looks fine and is not: dequantize is ordinary torch ops, so inductor traces and fuses
# the entire weight reconstruction into the graph. That cost us a CPU-constant LUT copy
# (illegal during CUDA graph capture) and then "cycle exists between partitions!" from
# vLLM's graph partitioner. Keeping both paths inside the op means the compiler sees one
# opaque call and nothing to fuse or reorder.
@torch.library.custom_op("lloyd43::linear", mutates_args=())
def _linear_op(x2d: Tensor, packed: Tensor, block_scale_u8: Tensor, gscale: Tensor,
               K: int, warps: int, rows_per_warp: int, gpl: int,
               stage_x: bool) -> Tensor:
    ext = load_extension()
    lut = lut_for(x2d.device, format_for_packed(packed))
    if x2d.shape[0] == 1:
        y = ext.gemv(x2d[0].contiguous().to(torch.bfloat16), packed, block_scale_u8,
                     lut, gscale, K, rows_per_warp, warps, gpl, stage_x)
        return y.unsqueeze(0)
    # Prefill / batched: reconstruct the weight and let cuBLAS do the GEMM. Uses the
    # CUDA dequant kernel, not format.dequantize -- the latter gathers through int64
    # indices (8 bytes moved per weight to produce 2) and cost more than an entire
    # generate() call when vLLM ran it once per layer at prefill.
    w = ext.dequant(packed, block_scale_u8, lut, gscale, K)
    return x2d.to(torch.bfloat16) @ w.T


@_linear_op.register_fake
def _(x2d, packed, block_scale_u8, gscale, K, warps, rows_per_warp, gpl, stage_x):
    return torch.empty(x2d.shape[0], packed.shape[0], dtype=torch.bfloat16,
                       device=x2d.device)


def linear_op(x2d: Tensor, packed: Tensor, block_scale_u8: Tensor, gscale: Tensor,
              K: int, cfg: tuple[int, int, int, bool]) -> Tensor:
    """(M, K) -> (M, N). M == 1 takes the GEMV kernel; anything else dequantizes."""
    w, r, g, sx = cfg
    return torch.ops.lloyd43.linear(x2d, packed, block_scale_u8, gscale, K, w, r, g, sx)


def dequant_cuda(packed: Tensor, block_scale: Tensor, global_scale: Tensor,
                 K: int) -> Tensor:
    """Packed 3-bit -> dense (N, K) bf16, on the GPU. Bitwise equal to format.dequantize.

    Exists because the PyTorch reference gathers through int64 indices -- 8 bytes moved to
    produce 2 -- which is fine for a correctness anchor and far too slow to sit in a
    serving prefill path.
    """
    ext = load_extension()
    return ext.dequant(packed.contiguous(), block_scale.contiguous().view(torch.uint8),
                       lut_for(packed.device, format_for_packed(packed)),
                       global_scale.to(packed.device, torch.float32).reshape(()),
                       K)
