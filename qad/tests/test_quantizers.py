"""Regression + correctness tests for the blocked-quantizer core.

Covers the invariants that are expensive to discover the hard way:
  1. checkpoint-tag hashes are unchanged (a changed `defaults` renames every
     checkpoint and eval directory for that method);
  2. signed-Lloyd 3-bit reproduces the reference MSE from ../../grids.ipynb;
  3. fused-group scale sharing holds for EVERY blocked format (vLLM collapses the
     per-shard global scales with .max(), so they must already be identical);
  4. weight-only formats leave activations untouched.

Run inside the training container:
    python tests/test_quantizers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantizers import REGISTRY, build_quantizer_params
from quantizers.blocked import BlockScaledLinear, blocked_quantize, index_nearest
from quantizers.grids import FP4_GRID, LLOYD_SIGNED_3BIT, lloyd_grid
from quantizers.lloyd import SignedLloydLinear, apply_lloyd3bit
from quantizers.nvfp4 import NVFP4Linear, apply_nvfp4, apply_nvfp4a16, e2m1_round


def test_tag_hashes_unchanged():
    """Existing checkpoints live at .../qad-<model>-<method>-<hash>/ — these must not move.
    Verified against the pre-refactor registry; a changed `defaults` dict renames
    every checkpoint and eval directory for that method."""
    expected = {
        "fp8": "99914b93", "nvfp4": "99914b93", "nvfp4a16": "99914b93",
        "ste2bit": "1a17550c", "ste3bit": "1a17550c", "ste4bit": "1a17550c",
        "quest2bit": "1a17550c", "quest3bit": "1a17550c", "quest4bit": "1a17550c",
    }
    for method, want in expected.items():
        got = build_quantizer_params(method, "")[1]
        assert got == want, f"{method}: tag hash {got} != {want} (checkpoints would move)"
    print(f"  {len(expected)} tag hashes unchanged  OK")


def test_reference_mse():
    """Match ../../grids.ipynb on a Gaussian: NVFP4 0.0091, signed-Lloyd 0.0218."""
    torch.manual_seed(0)
    x = torch.randn(2 ** 22)
    ref = x.pow(2).mean()

    deq_fp4, _, _ = blocked_quantize(x, e2m1_round, 16, signed=False)
    err_fp4 = ((x - deq_fp4).pow(2).mean() / ref).item()

    grid = LLOYD_SIGNED_3BIT
    deq_ll, _, _ = blocked_quantize(x, lambda v: grid.to(v.device)[index_nearest(v, grid.to(v.device))],
                                    16, signed=True)
    err_ll = ((x - deq_ll).pow(2).mean() / ref).item()

    print(f"  NVFP4 (4-bit)          err={err_fp4:.4f}  (reference 0.0091)")
    print(f"  signed Lloyd (3-bit)   err={err_ll:.4f}  (reference 0.0218)")
    assert abs(err_fp4 - 0.0091) < 0.0015, err_fp4
    assert abs(err_ll - 0.0218) < 0.0025, err_ll
    assert err_ll < 0.030, "signed Lloyd should beat the 0.0334 FP4-downcast LUT"


def test_signed_normalization():
    """With a signed scale the block extreme normalizes to +6, never -6."""
    x = torch.randn(64, 16)
    x[0] = -x[0].abs()                     # a block whose max-abs element is negative
    deq, block_scale, gs = blocked_quantize(x, lambda v: v, 16, signed=True)
    xn = x.reshape(64, 1, 16) / (block_scale.reshape(64, 1, 1) * gs)
    ext = xn.reshape(64, 16).gather(1, xn.reshape(64, 16).abs().argmax(1, keepdim=True))
    assert (ext > 0).all(), "signed normalization must put the extreme at +6"
    print("  signed normalization -> extreme always positive  OK")


def _tiny_model():
    class Blk(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj = nn.Linear(64, 64, bias=False)
            s.k_proj = nn.Linear(64, 32, bias=False)
            s.v_proj = nn.Linear(64, 32, bias=False)
            s.gate_proj = nn.Linear(64, 128, bias=False)
            s.up_proj = nn.Linear(64, 128, bias=False)
            s.down_proj = nn.Linear(128, 64, bias=False)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.blk = Blk()
            s.lm_head = nn.Linear(64, 100, bias=False)
    m = M()
    # make the fused siblings have very different amax so sharing is observable
    with torch.no_grad():
        m.blk.q_proj.weight.mul_(10.0)
        m.blk.gate_proj.weight.mul_(10.0)
    return m


def test_fused_group_sharing():
    """Every blocked format must share one global scale across q/k/v and gate/up."""
    for name, apply in (("nvfp4", apply_nvfp4), ("nvfp4a16", apply_nvfp4a16),
                        ("lloyd3bit", apply_lloyd3bit)):
        m = _tiny_model()
        apply(m)
        qkv = [m.blk.q_proj, m.blk.k_proj, m.blk.v_proj]
        gu = [m.blk.gate_proj, m.blk.up_proj]
        for group, label in ((qkv, "q/k/v"), (gu, "gate/up")):
            scales = {round(float(l.group_global_scale()), 10) for l in group}
            assert len(scales) == 1, f"{name} {label} scales differ: {scales}"
        # a non-fused layer keeps its own scale, and lm_head is never quantized
        assert isinstance(m.blk.down_proj, BlockScaledLinear)
        assert not isinstance(m.lm_head, BlockScaledLinear), "lm_head must stay unquantized"
        print(f"  {name:9} fused-group scale shared (q/k/v, gate/up), lm_head skipped  OK")


def test_weight_only_leaves_activations():
    """lloyd3bit and nvfp4a16 must not touch activations; nvfp4 (W4A4) must."""
    x = torch.randn(2, 8, 64)
    for name, apply, expect_act_quant in (("nvfp4", apply_nvfp4, True),
                                          ("nvfp4a16", apply_nvfp4a16, False),
                                          ("lloyd3bit", apply_lloyd3bit, False)):
        m = _tiny_model()
        apply(m)
        lin = m.blk.q_proj
        lin.eval()
        # BF16 activations, because that is what production feeds these layers: every
        # forward runs under torch.amp.autocast(bfloat16), and _wq is stored bf16. An
        # fp32 x here would be testing a dtype combination the model never sees, and
        # would fail with "expected m1 and m2 to have the same dtype".
        xb = x.to(torch.bfloat16)
        plain = F.linear(xb, lin.wq, lin.bias)
        got = lin(xb)
        same = torch.equal(plain, got)
        assert same == (not expect_act_quant), f"{name}: activation quant mismatch"
        print(f"  {name:9} activations {'quantized' if expect_act_quant else 'untouched'}  OK")


def test_lloyd_ste_and_post_update():
    """STE passes gradient to the master weight; post_update refreshes _wq."""
    m = _tiny_model()
    apply_lloyd3bit(m)
    lin = m.blk.down_proj
    lin.train()
    out = lin(torch.randn(4, 128)).sum()
    out.backward()
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, "no STE gradient"
    with torch.no_grad():
        lin.weight.add_(torch.randn_like(lin.weight) * 0.1)
    before = lin._wq.clone()
    lin.post_update(0, 1)
    assert not torch.equal(before, lin._wq), "post_update did not refresh _wq"
    print("  lloyd3bit STE gradient + post_update refresh  OK")


def test_lloyd_grid_refit():
    """lloyd_grid() reproduces a grid close to the shipped constant on a Gaussian."""
    torch.manual_seed(0)
    x = torch.randn(2 ** 18, 16)
    scales = x.abs().amax(-1, keepdim=True)
    idx = x.abs().argmax(-1, keepdim=True)
    xn = x / (scales * x.gather(-1, idx).sign() / 6.0)
    fit = lloyd_grid(xn, 8, 50)
    ref = LLOYD_SIGNED_3BIT
    dev = (fit - ref).abs().max().item()
    print(f"  refit grid max|Δ| vs shipped constant = {dev:.3f}")
    assert dev < 0.6, f"refit grid diverged: {fit}"


def test_registry_consistency():
    exports = {n: e["export"] for n, e in REGISTRY.items()}
    assert exports["nvfp4"] == exports["nvfp4a16"] == "compressed_tensors"
    assert exports["lloyd3bit"] == "dequantized"
    assert "lloyd3bit" in REGISTRY and REGISTRY["lloyd3bit"]["post_update"] is not None
    print(f"  registry: {len(REGISTRY)} methods, export modes consistent  OK")


if __name__ == "__main__":
    for fn in (test_tag_hashes_unchanged, test_registry_consistency, test_reference_mse,
               test_signed_normalization, test_fused_group_sharing,
               test_weight_only_leaves_activations, test_lloyd_ste_and_post_update,
               test_lloyd_grid_refit):
        print(f"{fn.__name__}:")
        fn()
    print("\nALL PASS")
