"""lloyd21: the same layout at 2 bits, and proof the 3-bit path did not move.

The package is parameterized by `Format` rather than duplicated, so every one of these has
a second job: catching a change that fixes 2 bits by breaking 3. `test_lloyd43_unchanged`
is the explicit version of that, and the shared `fmt` parametrization is the implicit one.

Two bugs this file would have caught, both from the generalization and both silent:
  * the CUDA unpack built a 3-bit index unconditionally, so at 2 bits it read a third
    bit-plane out of the NEXT ROW and indexed a 4-entry table with 0..7;
  * the dequant kernel loaded `prow[2*W + g]` for the same reason.
Neither errors -- they return plausible-looking numbers, which is why the check is against
a reference rather than a smoke test.
"""

import pytest
import torch

from conftest import rel, requires_gpu
from lloyd43.format import (FORMATS, LLOYD21, LLOYD43, GROUP, dequantize,
                            pack_from_weight, pack_indices, unpack_indices)

FMTS = [pytest.param(f, id=f.name) for f in FORMATS.values()]


def test_grid_matches_the_spec():
    """The published lloyd21 grid, exactly. A drift here silently retrains nothing."""
    assert torch.equal(LLOYD21.lut,
                       torch.tensor([-3.6517, 0.0000, +2.5227, +6.0000]))
    assert LLOYD21.bits == 2 and LLOYD21.levels == 4


def test_zero_and_six_are_grid_points():
    """The construction pins 0.0 and +6.0; the signed block scale depends on both.

    +6.0 is where every block's max-abs element lands under the signed normalisation, and
    an exact 0.0 is what lets the bulk of a weight distribution be represented without
    bias. Lose either and the format is a different one.
    """
    for fmt in FORMATS.values():
        assert (fmt.lut == 0.0).any(), f"{fmt.name} has no exact zero"
        assert float(fmt.lut.max()) == 6.0, f"{fmt.name} top level is not 6.0"


@pytest.mark.parametrize("fmt", FMTS)
def test_pack_roundtrip_is_exact(device, fmt):
    torch.manual_seed(0)
    idx = torch.randint(0, fmt.levels, (32, 256), dtype=torch.uint8, device=device)
    packed = pack_indices(idx, fmt)
    assert packed.shape == (32, fmt.bits, 256 // GROUP)
    assert torch.equal(unpack_indices(packed, 256, fmt), idx)


@pytest.mark.parametrize("fmt", FMTS)
def test_spends_exactly_its_bit_budget(device, fmt):
    """bits/weight for the indices, plus one e4m3 scale per 16."""
    N, K = 64, 512
    w = torch.randn(N, K, device=device)
    packed, bs, _ = pack_from_weight(w, fmt=fmt)
    index_bits = packed.numel() * 32 / (N * K)
    assert index_bits == pytest.approx(fmt.bits)
    total = (packed.numel() * 4 + bs.numel()) / (N * K)
    assert total == pytest.approx(fmt.bytes_per_weight)


@pytest.mark.parametrize("fmt", FMTS)
def test_every_grid_level_is_reachable(device, fmt):
    """All 2**bits patterns survive a pack/unpack round trip."""
    idx = torch.arange(fmt.levels, dtype=torch.uint8, device=device)
    idx = idx.repeat(GROUP)[:GROUP].unsqueeze(0)
    got = unpack_indices(pack_indices(idx, fmt), GROUP, fmt)
    assert torch.equal(got, idx)
    assert set(got.unique().tolist()) == set(range(min(fmt.levels, GROUP)))


@pytest.mark.parametrize("fmt", FMTS)
def test_block_extreme_lands_on_the_top_level(device, fmt):
    """The signed normalisation puts every block's max-abs element on exactly +6.

    This is the property the asymmetric 2-bit grid is built around: one negative level is
    affordable only because the extreme is always mapped to the positive top.
    """
    torch.manual_seed(0)
    w = torch.randn(8, 64, device=device)
    packed, bs, gs = pack_from_weight(w, fmt=fmt)
    idx = unpack_indices(packed, 64, fmt).reshape(8, 4, 16)
    top = fmt.levels - 1
    # the max-abs element of each block must take the top index
    pos = w.reshape(8, 4, 16).abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(idx.gather(-1, pos), torch.full_like(pos, top, dtype=idx.dtype))


@requires_gpu
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("N,K", [(256, 512), (1024, 1024), (2048, 6144)])
def test_kernels_match_the_dequantized_reference(fmt, N, K):
    """The kernel against x @ dequantize(packed).T, which is the definition."""
    from lloyd43.cuda_gemv import dequant_cuda, gemv_lloyd43_cuda

    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda") * 0.02
    packed, bs, gs = pack_from_weight(w, fmt=fmt)
    x = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    dq = dequantize(packed, bs, gs, K, fmt=fmt).float()
    ref = x.float() @ dq.T

    assert rel(gemv_lloyd43_cuda(x, packed, bs, gs, K, fmt=fmt).float(), ref) < 1e-2
    # dequant is exact, not approximate -- it is the same arithmetic in a different order
    assert torch.equal(dequant_cuda(packed, bs, gs, K).float(), dq)


@requires_gpu
def test_lloyd21_moves_fewer_bytes_than_lloyd43():
    """0.3125 vs 0.4375 B/weight: 1.4x less traffic, 6.4x less than bf16."""
    assert LLOYD21.bytes_per_weight < LLOYD43.bytes_per_weight
    assert LLOYD21.bytes_per_weight == pytest.approx(0.3125)
    assert 2.0 / LLOYD21.bytes_per_weight == pytest.approx(6.4)


@requires_gpu
def test_lloyd43_unchanged(device):
    """The 3-bit path must be bit-for-bit what it was before lloyd21 existed.

    Guards the generalization: it would be easy to make 2 bits work by changing something
    the 3-bit path relies on, and the per-format tests above would all still pass.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 256, device=device) * 0.02
    packed, bs, gs = pack_from_weight(w)                      # default format
    assert packed.shape == (64, 3, 8)
    explicit, bs2, gs2 = pack_from_weight(w, fmt=LLOYD43)
    assert torch.equal(packed, explicit) and torch.equal(bs.view(torch.uint8),
                                                         bs2.view(torch.uint8))
    assert torch.equal(dequantize(packed, bs, gs, 256),
                       dequantize(explicit, bs2, gs2, 256, fmt=LLOYD43))
