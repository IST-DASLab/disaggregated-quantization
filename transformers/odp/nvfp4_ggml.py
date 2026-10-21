"""Convert compressed-tensors NVFP4 into ggml's block_nvfp4 layout.

LOSSLESS, and that is the point: the 4-bit values and the E4M3 block scales are the trained
ones, copied verbatim. Only their arrangement changes. Verified bit-identical over a whole
projection -- see `python nvfp4_ggml.py --selftest`.

Two offsets in ggml's format cancel rather than scale anything:

    kvalues_fp4  = {0,1,2,3,4,6,8,12}   standard E2M1 x 2
    scale decode = E4M3 x 0.5

so value*scale is exactly standard NVFP4 per block. What ggml has no room for is NVFP4's
per-tensor GLOBAL scale -- and it does not need one. A scalar commutes with the matmul, so
the runtime applies it once to the result instead of folding it into 8-bit scales, which
would have cost precision for nothing.
"""

import numpy as np

QK = 64          # values per block
QK_SUB = 16      # values per scale
N_SUB = QK // QK_SUB


def to_ggml_nvfp4(packed: np.ndarray, scale_e4m3: np.ndarray) -> np.ndarray:
    """(N, K/2) packed nibbles + (N, K/16) raw E4M3 bytes -> (N, K/64*36) block_nvfp4 bytes.

    Input nibble order is compressed-tensors': byte i holds element 2i in the low nibble and
    2i+1 in the high. ggml instead pairs element j with j+8 WITHIN each 16-element sub-block,
    so the nibbles are regrouped; none of them change value.
    """
    n, kh = packed.shape
    k = kh * 2
    if k % QK:
        raise ValueError(f"K={k} is not a multiple of {QK}")
    if scale_e4m3.shape != (n, k // QK_SUB):
        raise ValueError(f"scale shape {scale_e4m3.shape} != {(n, k // QK_SUB)}")

    nib = np.empty((n, k), np.uint8)
    nib[:, 0::2] = packed & 0xF
    nib[:, 1::2] = packed >> 4

    sub = nib.reshape(n, k // QK_SUB, QK_SUB)
    qs = (sub[:, :, :QK_SUB // 2] | (sub[:, :, QK_SUB // 2:] << 4)).astype(np.uint8)

    nblk = k // QK
    qs = qs.reshape(n, nblk, N_SUB * (QK_SUB // 2))      # 32 bytes per block
    d = scale_e4m3.reshape(n, nblk, N_SUB)               # 4 scale bytes per block

    # block_nvfp4 is { d[4]; qs[32] } -- scales first.
    return np.concatenate([d, qs], axis=2).reshape(n, nblk * (N_SUB + QK // 2))


def row_bytes(k: int) -> int:
    return k // QK * (N_SUB + QK // 2)


def _selftest() -> int:
    """Decode our output the way ggml does and compare against the reference dequant."""
    rng = np.random.default_rng(0)
    n, k = 8, 256
    packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    # E4M3 bytes, avoiding 0x7F (NaN) and the sign bit, which scales never use
    sc = rng.integers(1, 0x7F, (n, k // QK_SUB)).astype(np.uint8)

    E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    GG = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)

    def e4m3(b):
        b = b.astype(np.uint32); e = (b >> 3) & 0xF; m = b & 0x7
        v = np.where(e == 0, m * 2.0 ** -9,
                     (1.0 + m / 8.0) * np.exp2((e.astype(np.int32) - 7).astype(np.float32)))
        return np.where((b == 0) | (b == 0x7F), 0.0, v).astype(np.float32)

    nib = np.empty((n, k), np.uint8)
    nib[:, 0::2] = packed & 0xF
    nib[:, 1::2] = packed >> 4
    ref = E2M1[nib] * np.repeat(e4m3(sc), QK_SUB, axis=1)

    blob = to_ggml_nvfp4(packed, sc).reshape(n, k // QK, N_SUB + QK // 2)
    d, qs = blob[:, :, :N_SUB], blob[:, :, N_SUB:]
    qs = qs.reshape(n, k // QK_SUB, QK_SUB // 2)
    got_nib = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(n, k)
    got = GG[got_nib] * np.repeat(e4m3(d.reshape(n, -1)) * 0.5, QK_SUB, axis=1)

    ok = np.array_equal(got, ref)
    print(f"selftest: {'OK, bit-identical' if ok else 'FAIL'} over {ref.size} weights")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftest() if "--selftest" in sys.argv else 0)
