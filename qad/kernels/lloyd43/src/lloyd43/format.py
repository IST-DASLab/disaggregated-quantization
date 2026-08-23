"""The packed lloyd43 storage format: 3-bit indices, signed E4M3 block scales, FP32 global.

QAD trains lloyd43 as a *pseudo*-quantized format -- `quantizers/lloyd.py` keeps a bf16
master and the checkpoint ships dequantized weights, because there was no 3-bit kernel to
serve it. This module defines the real packed layout that such a kernel needs, and the
packer is written so that unpacking reproduces QAD's own `blocked_quantize` output
*exactly*, not approximately. That equality is what tests/test_vs_qad.py asserts, and it
is the whole point: a kernel that is fast but disagrees with the trained weights is
worthless.

This package depends on torch alone -- the quantization math below is reimplemented here
rather than imported from QAD, so it can be installed and shipped on its own.

LAYOUT
------
For a weight W of shape (N, K), K a multiple of 32:

    packed        int32 (N, 3, K // 32)    3-bit indices, bit-planed (see below)
    block_scale   float8_e4m3fn (N, K // 16)   SIGNED -- see below
    global_scale  float32 scalar

and the dequantization is

    eff[n, b]  = float32(block_scale[n, b]) * global_scale
    W[n, k]    = LUT[idx[n, k]] * eff[n, k // 16]

with LUT = LLOYD43_SIGNED_3BIT, the 8-level grid.

WHY THE BLOCK SCALE IS SIGNED
-----------------------------
This is the one thing that will silently corrupt a checkpoint if you get it wrong. The
signed normalization makes each block's max-abs element land on exactly +6.0, so the
scale has to absorb that element's SIGN. A block whose extreme is negative therefore has
a negative scale. E4M3 (`float8_e4m3fn`) is a signed type and stores this fine -- but the
UE4M3 convention used for NVFP4 block scales does not, and storing these scales as
unsigned makes the model score 0.0 while looking structurally valid. Roughly half of all
blocks have a negative scale; `test_block_scales_are_signed` pins that.

WHY BIT-PLANES
--------------
3 bits does not divide a byte, and the usual fixes (10 indices per 32-bit word, or 8
indices per 3 bytes) either waste bits or need unaligned loads. Instead, 32 consecutive
indices are stored as 3 words, one per bit position:

    packed[n, p, g]  bit j  ==  bit p of idx[n, 32*g + j]

so idx[n, 32*g + j] = sum_p ((packed[n, p, g] >> j) & 1) << p. That is exactly 3 bits per
weight with nothing wasted, every load is a naturally aligned 32-bit word, and the unpack
in a kernel is three shifts and three masks with no cross-lane traffic. 32 is also two
whole blocks of 16, so a group never straddles a block-scale boundary.

The plane index comes BEFORE the group index so that, for one row and one bit-plane, the
words for consecutive k are contiguous. A kernel walking K then issues unit-stride loads;
with the planes innermost they would be strided by 3 and coalescing would be lost.

The words are stored as int32 rather than uint32: Triton's uint32 support is patchy, and
the extraction `(w >> j) & 1` is correct under an ARITHMETIC shift anyway -- sign
extension only ever fills bits above the one being masked off.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["LUT", "BLOCK", "GROUP", "BITS", "Format", "LLOYD43", "LLOYD21",
           "FORMATS", "E4M3_MAX", "SCALE_REF", "GLOBAL_DEN",
           "to_e4m3", "index_nearest", "quantize_lloyd43", "pack_indices",
           "unpack_indices", "dequantize", "pack_from_weight", "effective_scales",
           "lut_for", "format_for_bits", "format_for_packed"]

# These constants are duplicated from QAD rather than imported, so that this package
# installs and runs with nothing but torch. tests/test_vs_qad.py imports BOTH and asserts
# they agree -- which also makes the bitwise-equality test meaningful instead of circular.
LUT = torch.tensor(                 # LLOYD43_SIGNED_3BIT: MSE-optimal with 0.0 and 6.0 pinned
    [-4.7038, -2.8698, -1.3696, 0.0000, +1.2204, +2.5285, +4.0473, +6.0000]
)
BITS = 3
BLOCK = 16                          # weights per block scale
GROUP = 32                          # indices per bit-planed int32 triple
E4M3_MAX = 448.0
SCALE_REF = 6.0                     # the grid's top level; block extremes normalize here
GLOBAL_DEN = SCALE_REF * E4M3_MAX   # 2688


@dataclass(frozen=True)
class Format:
    """A grid plus its bit width. Everything else about the layout is shared.

    lloyd21 is lloyd43 with a different number of bit-planes and a different table, and
    nothing else: same signed E4M3 block scales, same fp32 global scale, same bit-planed
    int32 packing, same GROUP of 32 and BLOCK of 16. That is why this is one package with
    a parameter rather than two packages -- a second copy would be 90% duplicated and would
    drift, and the kernels take the bit count as a template/constexpr argument anyway.

    QAD's `quantizers/grids.py` is the source of both grids; `tests/test_vs_qad.py` asserts
    they agree, which is what keeps the bitwise-equality contract meaningful rather than
    circular.
    """

    name: str
    bits: int
    lut: Tensor

    @property
    def levels(self) -> int:
        return 1 << self.bits

    @property
    def bytes_per_weight(self) -> float:
        """Index bits plus one E4M3 scale per BLOCK weights."""
        return self.bits / 8 + 1 / BLOCK

    def __post_init__(self):
        if self.lut.numel() != self.levels:
            raise ValueError(f"{self.name}: {self.lut.numel()} levels for {self.bits} bits")


LLOYD43 = Format("lloyd43", 3, LUT)
# The same construction at 2 bits: MSE-optimal with 0.0 and +6.0 pinned. Two positive
# levels, one negative -- the asymmetry is deliberate and is the same one lloyd43 has,
# because the SIGNED block normalisation puts every block's max-abs element on exactly +6,
# so the positive tail is where the mass that must be exact lives. Rounding an
# UNSIGNED-normalised block onto this grid would clip every block with a negative extreme
# from 6.0 to 3.6517 -- a 39% error on the largest weight in the block.
LLOYD21 = Format("lloyd21", 2, torch.tensor([-3.6517, 0.0000, +2.5227, +6.0000]))

FORMATS = {f.name: f for f in (LLOYD43, LLOYD21)}

# 0.4375 vs 0.3125 bytes/weight, so lloyd21 moves 1.4x less than lloyd43 and 6.4x less than
# bf16. As always that is a TRAFFIC ratio and not an achievable speedup -- bandwidth is
# size-dependent, and reading fewer bytes means reading them at a lower GB/s.


_LUT_CACHE: dict[tuple, Tensor] = {}


def lut_for(device, fmt: "Format | None" = None) -> Tensor:
    """The grid on `device`, cached.

    LUT is a module-level CPU tensor, so `LUT.to(device)` allocates and copies 32 bytes
    host-to-device EVERY call. That is invisible on a large projection and ruinous on a
    small one: it was ~10 us of fixed cost per call, which at (1024, 1024) was most of the
    gap to NVFP4A16. Never call `.to()` on the constant in a hot path.
    """
    fmt = fmt or LLOYD43
    key = (str(device), fmt.name)
    t = _LUT_CACHE.get(key)
    if t is None:
        t = fmt.lut.to(device, torch.float32).contiguous()
        _LUT_CACHE[key] = t
    return t


def format_for_bits(bits: int) -> Format:
    """The Format with this many bit-planes."""
    for f in FORMATS.values():
        if f.bits == bits:
            return f
    raise ValueError(f"no format with {bits} bit-planes; have "
                     f"{ {f.name: f.bits for f in FORMATS.values()} }")


def format_for_packed(packed: Tensor) -> Format:
    """Infer the format from a packed tensor's plane count.

    The bit width is carried by the data -- `packed` is (N, bits, K//32) -- so a kernel
    entry point that only receives tensors can still pick the right grid. That is what lets
    the torch custom ops serve both formats without an extra argument, which they could not
    take anyway: custom-op schemas do not accept arbitrary Python objects.
    """
    return format_for_bits(int(packed.shape[1]))


def to_e4m3(x: Tensor) -> Tensor:
    """Round to the E4M3 grid via a real float8 round-trip."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def index_nearest(x: Tensor, grid: Tensor) -> Tensor:
    """Index of the nearest point of a sorted 1-D grid. Ties go to the HIGHER index."""
    inds = torch.bucketize(x, grid)
    lo = torch.clamp(inds - 1, min=0, max=grid.shape[-1] - 1)
    hi = torch.clamp(inds, min=0, max=grid.shape[-1] - 1)
    return torch.where((grid[hi] - x) <= (x - grid[lo]), hi, lo)


def effective_scales(block_scale: Tensor, global_scale: Tensor) -> Tensor:
    """eff[n, b] = float32(block_scale) * global_scale, the per-block dequant multiplier."""
    return block_scale.to(torch.float32) * global_scale.to(torch.float32)


def quantize_lloyd43(w: Tensor, global_scale: Tensor | None = None, block: int = BLOCK,
                     fmt: Format | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize a weight to (indices, block_scale_e4m3, global_scale).

    Deliberately mirrors `quantizers.blocked.blocked_quantize` line for line -- same amax,
    same signed block scale, same E4M3 cast, same tie rule via `index_nearest` -- but
    returns the grid INDICES rather than the dequantized values, because indices are what
    gets packed. `dequantize(...)` composed with this returns bit-identical results to
    blocked_quantize()'s first output.

    `global_scale` should be passed in when the layer belongs to a fused q/k/v or gate/up
    group, since those share one global scale; omitted, it is derived from this tensor.
    """
    N, K = w.shape
    if K % block:
        raise ValueError(f"K={K} must be a multiple of block={block}")
    wf = w.float().reshape(N, K // block, block)

    if global_scale is None:
        global_scale = wf.abs().amax().clamp(min=1e-8) / GLOBAL_DEN
    global_scale = global_scale.to(torch.float32).clamp(min=1e-8)

    # SIGNED: the scale carries the sign of the block's max-abs element, so that element
    # normalizes to exactly +6.0 -- the top grid level.
    amax, pos = wf.abs().max(dim=-1, keepdim=True)
    block_amax = amax.clamp(min=1e-8) * wf.take_along_dim(pos, -1).sign()

    block_scale = to_e4m3((block_amax / SCALE_REF) / global_scale)
    eff = block_scale * global_scale
    eff = torch.where(eff.abs() < 1e-12, torch.ones_like(eff), eff)

    fmt = fmt or LLOYD43
    idx = index_nearest(wf / eff, fmt.lut.to(w.device))
    return (idx.reshape(N, K).to(torch.uint8),
            block_scale.squeeze(-1).to(torch.float8_e4m3fn),
            global_scale.reshape(()))


def pack_indices(idx: Tensor, fmt: Format | None = None) -> Tensor:
    """(N, K) uint8 indices -> (N, bits, K//32) int32 bit-planes."""
    fmt = fmt or LLOYD43
    N, K = idx.shape
    if K % GROUP:
        raise ValueError(f"K={K} must be a multiple of {GROUP}")
    if int(idx.max()) >= fmt.levels:
        raise ValueError(f"index {int(idx.max())} out of range for a {fmt.levels}-level grid")
    i = idx.to(torch.int64).reshape(N, K // GROUP, GROUP)
    bit = torch.arange(GROUP, device=idx.device, dtype=torch.int64)
    planes = torch.stack([(((i >> p) & 1) << bit).sum(dim=-1) for p in range(fmt.bits)], dim=1)
    # Wrap into int32's range explicitly: bit 31 set means a negative int32, and relying
    # on an implicit narrowing cast to do that is exactly the kind of thing that differs
    # between torch versions.
    planes = planes - (planes >= 2 ** 31).to(torch.int64) * (2 ** 32)
    return planes.to(torch.int32).contiguous()


def unpack_indices(packed: Tensor, K: int, fmt: Format | None = None) -> Tensor:
    """(N, bits, K//32) int32 bit-planes -> (N, K) uint8 indices. Inverse of pack_indices."""
    fmt = fmt or LLOYD43
    N = packed.shape[0]
    w = packed.to(torch.int64) & 0xFFFFFFFF         # undo the int32 sign, work in 64 bits
    bit = torch.arange(GROUP, device=packed.device, dtype=torch.int64)
    idx = torch.zeros(N, K // GROUP, GROUP, dtype=torch.int64, device=packed.device)
    for p in range(fmt.bits):
        idx |= ((w[:, p, :].unsqueeze(-1) >> bit) & 1) << p
    return idx.reshape(N, K).to(torch.uint8)


def dequantize(packed: Tensor, block_scale: Tensor, global_scale: Tensor, K: int,
               dtype: torch.dtype = torch.bfloat16, block: int = BLOCK,
               lut: Tensor | None = None, fmt: Format | None = None) -> Tensor:
    """Packed 3-bit -> dense (N, K) weight. The definition the kernels must match.

    Pass `lut` to supply a grid tensor that already lives on the right device. The default
    is the module-level LUT, which is on the CPU: under torch.compile that becomes a
    host-side constant the graph has to copy in, and copying CPU->CUDA is illegal during
    CUDA graph capture. Callers inside a compiled region (the vLLM plugin's prefill path)
    must pass a device-resident LUT.
    """
    fmt = fmt or LLOYD43
    idx = unpack_indices(packed, K, fmt)
    eff = effective_scales(block_scale, global_scale)              # (N, K//block)
    grid = fmt.lut.to(packed.device) if lut is None else lut
    vals = grid[idx.to(torch.int64)]                               # (N, K) float32
    return (vals * eff.repeat_interleave(block, dim=1)).to(dtype)


def pack_from_weight(w: Tensor, global_scale: Tensor | None = None,
                     fmt: Format | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Convenience: dense weight -> (packed, block_scale, global_scale)."""
    fmt = fmt or LLOYD43
    idx, block_scale, gs = quantize_lloyd43(w, global_scale=global_scale, fmt=fmt)
    return pack_indices(idx, fmt), block_scale, gs
