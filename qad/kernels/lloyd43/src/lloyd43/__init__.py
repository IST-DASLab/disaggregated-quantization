"""Real inference for the lloyd43 3-bit weight format.

QAD trains lloyd43 pseudo-quantized -- a bf16 master, a dequantized checkpoint -- because
there was no kernel to serve 3 bits. This package supplies the missing pieces:

    format.py       the packed layout, and a packer that round-trips QAD's own
                    blocked_quantize bit for bit
    reference.py    crude PyTorch baseline: decode to dense bf16, then multiply
    bench.py        speedup vs dense bf16 GEMV (triton.testing.perf_report)

Depends on torch alone; triton is needed only for the kernel and is imported lazily, so
the packer and the reference stay usable on a machine without it.

    from lloyd43 import pack_from_weight, gemv_reference

    packed, block_scale, global_scale = pack_from_weight(w)      # w: (N, K)
    y = gemv_lloyd43(x, packed, block_scale, global_scale, K=w.shape[1])

"""

from .format import (BITS, BLOCK, E4M3_MAX, GLOBAL_DEN, GROUP, LUT, SCALE_REF,
                     dequantize, effective_scales, index_nearest, pack_from_weight,
                     pack_indices, quantize_lloyd43, to_e4m3, unpack_indices)
from .reference import gemv_bf16, gemv_reference, gemv_reference_loop

__version__ = "0.1.0"

__all__ = ["LUT", "BLOCK", "GROUP", "BITS", "E4M3_MAX", "SCALE_REF", "GLOBAL_DEN",
           "to_e4m3", "index_nearest", "quantize_lloyd43", "pack_indices",
           "unpack_indices", "dequantize", "pack_from_weight", "effective_scales",
           "gemv_reference", "gemv_reference_loop", "gemv_bf16", "gemv_lloyd43"]


def __getattr__(name):
    if name == "gemv_lloyd43":                  # triton only when the kernel is asked for
        return gemv_lloyd43
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
