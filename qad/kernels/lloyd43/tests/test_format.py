"""Self-contained format tests: packing, bit budget, grid usage. No QAD, no kernels."""

import pytest
import torch

from conftest import requires_gpu
from lloyd43 import (BITS, BLOCK, GROUP, LUT, dequantize, pack_from_weight,
                     pack_indices, quantize_lloyd43, unpack_indices)


def test_pack_roundtrip_is_exact(device):
    torch.manual_seed(0)
    idx = torch.randint(0, 8, (64, 512), device=device, dtype=torch.uint8)
    packed = pack_indices(idx)
    assert packed.dtype == torch.int32
    assert packed.shape == (64, BITS, 512 // GROUP)
    assert torch.equal(unpack_indices(packed, 512), idx)


def test_spends_exactly_three_bits_per_weight(device):
    idx = torch.zeros(8, 256, device=device, dtype=torch.uint8)
    packed = pack_indices(idx)
    assert packed.numel() * 32 / idx.numel() == pytest.approx(BITS)


def test_all_bit_patterns_survive(device):
    """Every index in every lane position -- catches an off-by-one in the bit-plane shift."""
    idx = (torch.arange(8 * GROUP, device=device, dtype=torch.uint8) % 8).reshape(1, -1)
    assert torch.equal(unpack_indices(pack_indices(idx), idx.shape[1]), idx)


def test_rejects_bad_shapes_and_indices(device):
    with pytest.raises(ValueError, match="multiple of"):
        pack_indices(torch.zeros(4, 48, device=device, dtype=torch.uint8))
    with pytest.raises(ValueError, match="out of range"):
        pack_indices(torch.full((4, 32), 8, device=device, dtype=torch.uint8))
    with pytest.raises(ValueError, match="multiple of block"):
        quantize_lloyd43(torch.randn(4, 24, device=device))


def test_high_bit_words_survive_the_int32_wrap(device):
    """Index 7 everywhere sets bit 31 in all three planes -> negative int32 words."""
    idx = torch.full((2, GROUP), 7, device=device, dtype=torch.uint8)
    packed = pack_indices(idx)
    assert (packed < 0).all(), "bit 31 set should make the word negative"
    assert torch.equal(unpack_indices(packed, GROUP), idx)


def test_every_grid_level_is_reachable(device):
    torch.manual_seed(0)
    w = torch.randn(256, 512, device=device)
    idx, _, _ = quantize_lloyd43(w)
    used = torch.bincount(idx.reshape(-1).to(torch.int64), minlength=8)
    assert int((used > 0).sum()) == 8, f"unused levels: {used.tolist()}"
    assert int(used.argmax()) == 3, "the pinned zero should be the modal level"


def test_block_scales_are_signed(device):
    """A negative block scale is correct here; storing it unsigned zeroes the model."""
    torch.manual_seed(0)
    w = torch.randn(256, 512, device=device)
    _, bs, _ = pack_from_weight(w)
    assert bs.dtype == torch.float8_e4m3fn
    frac_neg = (bs.to(torch.float32) < 0).float().mean().item()
    assert frac_neg > 0.4, f"only {frac_neg:.1%} negative -- sign is being dropped"


def test_block_extreme_lands_on_the_top_level(device):
    """The +6.0 pin: each block's max-abs weight quantizes to index 7, exactly."""
    torch.manual_seed(0)
    w = torch.randn(32, 512, device=device)
    idx, _, _ = quantize_lloyd43(w)
    wb = w.reshape(32, -1, BLOCK)
    pos = wb.abs().argmax(dim=-1, keepdim=True)
    top = idx.reshape(32, -1, BLOCK).to(torch.int64).take_along_dim(pos, -1)
    assert (top == 7).float().mean().item() > 0.99
    assert LUT[7].item() == 6.0


def test_dequantized_weight_tracks_the_original(device):
    torch.manual_seed(0)
    w = torch.randn(128, 1024, device=device)
    packed, bs, gs = pack_from_weight(w)
    deq = dequantize(packed, bs, gs, w.shape[1], dtype=torch.float32)
    rel_mse = ((deq - w) ** 2).mean().item() / (w ** 2).mean().item()
    # grids.py measures 0.0211 for this grid on a Gaussian in blocks of 16.
    assert rel_mse < 0.03, f"relative MSE {rel_mse:.4f} is worse than the grid's 0.0211"


@requires_gpu
def test_pack_is_device_agnostic():
    torch.manual_seed(0)
    w = torch.randn(32, 256)
    cpu = pack_from_weight(w)
    gpu = pack_from_weight(w.cuda())
    assert torch.equal(cpu[0], gpu[0].cpu())
    assert torch.equal(cpu[1].to(torch.float32), gpu[1].to(torch.float32).cpu())
