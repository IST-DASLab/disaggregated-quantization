"""External cross-check: the packed format must reproduce QAD's trained weights exactly.

This is the test that matters. `lloyd43` reimplements the quantization math so it can ship
standalone, which means the two implementations are genuinely independent -- a bitwise
match here is evidence, not a tautology. Skipped automatically when the QAD tree is not
importable (i.e. when the library is installed on its own).
"""

import pytest
import torch

from conftest import rel, requires_gpu, requires_qad
from lloyd43 import BLOCK, LUT, dequantize, pack_from_weight

pytestmark = [requires_qad, pytest.mark.qad]


def test_constants_match_qad():
    from quantizers.blocked import BLOCK as QAD_BLOCK
    from quantizers.blocked import E4M3_MAX, GLOBAL_DEN, SCALE_REF
    from quantizers.grids import LLOYD43_SIGNED_3BIT
    import lloyd43

    assert torch.equal(LUT, LLOYD43_SIGNED_3BIT), "the grid drifted from QAD's"
    assert (lloyd43.BLOCK, lloyd43.E4M3_MAX) == (QAD_BLOCK, E4M3_MAX)
    assert (lloyd43.SCALE_REF, lloyd43.GLOBAL_DEN) == (SCALE_REF, GLOBAL_DEN)


def test_helpers_match_qad(device):
    """Our reimplemented to_e4m3 / index_nearest must agree with QAD's, ties included."""
    from quantizers.blocked import index_nearest as qad_index_nearest
    from quantizers.blocked import to_e4m3 as qad_to_e4m3
    import lloyd43

    torch.manual_seed(0)
    x = torch.randn(4096, device=device) * 50
    assert torch.equal(lloyd43.to_e4m3(x), qad_to_e4m3(x))

    grid = LUT.to(device)
    # include exact midpoints, where the tie rule decides
    mids = ((grid[1:] + grid[:-1]) / 2).repeat(64)
    probe = torch.cat([torch.randn(4096, device=device) * 3, mids])
    assert torch.equal(lloyd43.index_nearest(probe, grid),
                       qad_index_nearest(probe, grid))


@pytest.mark.parametrize("shape", [(128, 512), (64, 1024), (256, 2048)])
def test_dequantized_weight_is_bitwise_equal_to_blocked_quantize(device, shape):
    from quantizers.blocked import blocked_quantize, grid_rounder
    from quantizers.grids import LLOYD43_SIGNED_3BIT

    torch.manual_seed(0)
    w = torch.randn(*shape, device=device)
    deq_qad, bs_qad, gs_qad = blocked_quantize(
        w, grid_rounder(LLOYD43_SIGNED_3BIT.to(device)), BLOCK, signed=True)

    packed, bs, gs = pack_from_weight(w)
    deq = dequantize(packed, bs, gs, w.shape[1], dtype=torch.float32)

    assert torch.equal(gs.reshape(()), gs_qad.reshape(()))
    assert torch.equal(bs.to(torch.float32), bs_qad.to(torch.float32))
    assert torch.equal(deq, deq_qad), \
        f"max|diff| {(deq - deq_qad).abs().max().item():.3e}"


@requires_gpu
def test_reproduces_a_real_layers_cached_weight():
    """End to end through the layer QAD actually trains, fused global scale and all."""
    from quantizers.lloyd import SignedLloydLinear

    torch.manual_seed(0)
    N, K = 512, 1024
    lin = torch.nn.Linear(K, N, bias=False).cuda()
    layer = SignedLloydLinear.from_linear(lin, block_size=BLOCK, grid="lloyd43")

    packed, bs, gs = pack_from_weight(layer.weight.data,
                                      global_scale=layer.group_global_scale())
    deq = dequantize(packed, bs, gs, K, dtype=torch.float32).to(layer._wq.dtype)
    assert torch.equal(deq, layer._wq), \
        f"max|diff| {(deq - layer._wq).abs().max().item():.3e}"


@requires_gpu
def test_gemv_matches_the_layer_forward():
    from lloyd43.cuda_gemv import gemv_lloyd43_cuda
    from quantizers.lloyd import SignedLloydLinear

    torch.manual_seed(0)
    N, K = 512, 1024
    lin = torch.nn.Linear(K, N, bias=False).cuda()
    layer = SignedLloydLinear.from_linear(lin, block_size=BLOCK, grid="lloyd43")
    packed, bs, gs = pack_from_weight(layer.weight.data,
                                      global_scale=layer.group_global_scale())

    x = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    want = layer(x.to(layer._wq.dtype)).float()
    got = gemv_lloyd43_cuda(x, packed, bs, gs, K).float()
    assert rel(got, want) < 1e-2, f"rel {rel(got, want):.2e}"
