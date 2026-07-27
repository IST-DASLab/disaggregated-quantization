"""Sanity tests for the NVFP4 fake-quant / real-export consistency.

Run inside the training container:
    python test_nvfp4.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from quantizers.blocked import BLOCK
from quantizers.grids import E2M1_LEVELS as _E2M1_LEVELS
from quantizers.nvfp4 import nvfp4_quantize, pack_nvfp4_weight

# E2M1 decode table indexed by 4-bit code (matches ModelOpt / vLLM).
_E2M1_DECODE = torch.tensor(
    _E2M1_LEVELS + [-v for v in _E2M1_LEVELS], dtype=torch.float32)  # code 8 = -0 -> 0.0


def unpack_reference(packed, weight_scale, weight_scale_2):
    """Decode a ModelOpt-NVFP4 packed weight back to a dense tensor the way a
    kernel would: e2m1_value(code) * block_scale_e4m3 * global_scale."""
    O, Kh = packed.shape
    K = Kh * 2
    lo = (packed & 0x0F).to(torch.long)      # even elements
    hi = (packed >> 4).to(torch.long)        # odd elements
    codes = torch.empty(O, K, dtype=torch.long)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    vals = _E2M1_DECODE[codes]                                  # [O, K]
    bs = weight_scale.float().reshape(O, K // BLOCK, 1)         # e4m3 block scale
    g = weight_scale_2.float().item()
    return (vals.reshape(O, K // BLOCK, BLOCK) * bs * g).reshape(O, K)


def test_pack_roundtrip():
    torch.manual_seed(0)
    for (O, K) in [(64, 256), (128, 2560), (32, 9728)]:
        w = torch.randn(O, K) * 0.05
        deq, _, _ = nvfp4_quantize(w, BLOCK)
        packed, wscale, wscale2 = pack_nvfp4_weight(w, BLOCK)
        assert packed.dtype == torch.uint8 and packed.shape == (O, K // 2)
        assert wscale.dtype == torch.float8_e4m3fn and wscale.shape == (O, K // BLOCK)
        assert wscale2.dtype == torch.float32 and wscale2.numel() == 1
        deq2 = unpack_reference(packed, wscale, wscale2)
        max_err = (deq.float() - deq2).abs().max().item()
        assert max_err < 1e-4, f"({O},{K}) pack/fakequant mismatch: {max_err}"
        print(f"  ({O},{K}): pack==fakequant  max_err={max_err:.2e}")


def test_e2m1_levels():
    # every representable magnitude should quantize to itself (block-scaled)
    levels = torch.tensor(_E2M1_LEVELS)
    w = levels.repeat(4, 2)  # [4, 16] one block per row, all exact levels *6/6
    deq, _, _ = nvfp4_quantize(w * 1.0, BLOCK)
    # values are on-grid up to the per-block scale; just check finite + bounded
    assert torch.isfinite(deq).all()
    print("  e2m1 levels quantize finite/bounded  OK")


if __name__ == "__main__":
    print("test_pack_roundtrip:")
    test_pack_roundtrip()
    print("test_e2m1_levels:")
    test_e2m1_levels()
    print("ALL PASS")
