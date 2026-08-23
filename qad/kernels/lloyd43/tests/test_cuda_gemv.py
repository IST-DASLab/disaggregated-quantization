"""The CUDA kernel is held to exactly the contract the Triton one is held to.

Nothing here is CUDA-specific except the entry point: the reference it is checked against
is the same `gemv_reference`, and the accuracy bar is the same. If these pass and
test_gemv.py passes, the two backends are interchangeable.
"""

import pytest
import torch

from conftest import rel, requires_gpu
from lloyd43.format import pack_from_weight
from lloyd43.reference import gemv_reference

pytestmark = requires_gpu


def _cuda():
    from lloyd43 import cuda_gemv
    if not cuda_gemv.is_available():
        pytest.skip("CUDA extension did not build (needs nvcc)")
    return cuda_gemv


def _case(N, K, device="cuda", seed=0):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=device) * 0.02
    x = torch.randn(K, device=device, dtype=torch.bfloat16)
    packed, bs, gs = pack_from_weight(w)
    return x, packed, bs, gs


@pytest.mark.parametrize("N,K", [(512, 1024), (256, 2560), (4096, 4096)])
def test_matches_the_reference(N, K):
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(N, K)
    want = gemv_reference(x, packed, bs, gs, K)
    got = cuda_gemv.gemv_lloyd43_cuda(x, packed, bs, gs, K)
    assert rel(got, want) < 1e-2, f"rel {rel(got, want):.2e}"



@pytest.mark.parametrize("N", [1, 37, 1000])
def test_handles_row_counts_that_do_not_fill_a_cta(N):
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(N, 1024)
    want = gemv_reference(x, packed, bs, gs, 1024)
    got = cuda_gemv.gemv_lloyd43_cuda(x, packed, bs, gs, 1024)
    assert rel(got, want) < 1e-2, f"rel {rel(got, want):.2e}"


@pytest.mark.parametrize("K", [32, 64, 96, 160])
def test_scalar_fallback_when_k_is_not_a_multiple_of_128(K):
    """K/32 not divisible by 4 means the int4 load is misaligned; the kernel must fall
    back to scalar loads rather than read garbage."""
    cuda_gemv = _cuda()
    assert (K // 32) % 4 != 0 or K == 160, "these K are meant to exercise the scalar path"
    x, packed, bs, gs = _case(128, K)
    want = gemv_reference(x, packed, bs, gs, K)
    got = cuda_gemv.gemv_lloyd43_cuda(x, packed, bs, gs, K)
    assert rel(got, want) < 1e-2, f"rel {rel(got, want):.2e}"


def test_every_launch_config_gives_the_same_answer():
    """Tile shape, gpl and x-staging are performance knobs; none may change the result.

    gpl in particular changes how many groups a lane owns and therefore the whole load
    pattern, and stage_x switches x between shared memory and registers -- both are easy
    to get subtly wrong at the tail of K.
    """
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(600, 2048)
    want = gemv_reference(x, packed, bs, gs, 2048)
    for warps, rows, gpl, stage_x in cuda_gemv.CUDA_CONFIGS:
        got = cuda_gemv.gemv_lloyd43_cuda(x, packed, bs, gs, 2048, warps=warps,
                                          rows_per_warp=rows, gpl=gpl, stage_x=stage_x)
        assert rel(got, want) < 1e-2, (
            f"w{warps} r{rows} gpl{gpl} stage_x={stage_x}: rel {rel(got, want):.2e}")


def test_tuned_dispatch_table_is_consistent_with_the_kernel():
    """Every table entry must be a config the kernel is actually instantiated for.

    A typo here would fall through to a TORCH_CHECK at launch, which is a runtime error on
    exactly the shapes we care most about.
    """
    cuda_gemv = _cuda()
    valid = set(cuda_gemv.CUDA_CONFIGS)
    for shape, cfg in cuda_gemv.CUDA_SHAPE_TABLE.items():
        assert cfg in valid, f"{shape} -> {cfg} is not an instantiated config"
    assert cuda_gemv._heuristic(4096, 4096) in valid


def test_works_under_torch_compile_fullgraph():
    """The custom-op registration must survive fullgraph capture.

    This is what lets a compiled decode step put the GEMV inside a CUDA graph, and it only
    holds while the kernel has no host sync in it -- reading global_scale with .item()
    would break capture.
    """
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(256, 1024)
    bs_u8 = bs.contiguous().view(torch.uint8)
    gsf = gs.reshape(()).float().contiguous()
    cfg = cuda_gemv.auto_config(256, 1024)

    def f(xx):
        return cuda_gemv.gemv_op(xx, packed, bs_u8, gsf, 1024, cfg)

    want = gemv_reference(x, packed, bs, gs, 1024)
    got = torch.compile(f, fullgraph=True)(x)
    assert rel(got, want) < 1e-2, f"rel {rel(got, want):.2e}"


def test_rejects_a_host_side_global_scale():
    """A float would force an .item() sync on every call; the API refuses it."""
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(128, 1024)
    with pytest.raises(TypeError, match="device tensor"):
        cuda_gemv.gemv_lloyd43_cuda(x, packed, bs, float(gs), 1024)


def test_negative_block_scales_survive_the_round_trip():
    """~51% of lloyd43 blocks have a negative scale. A ue4m3 (unsigned) read would look
    structurally fine and be silently wrong, so pin it here as well as in test_format."""
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(256, 1024, seed=3)
    assert (bs.float() < 0).any(), "test is vacuous without negative scales"
    want = gemv_reference(x, packed, bs, gs, 1024)
    got = cuda_gemv.gemv_lloyd43_cuda(x, packed, bs, gs, 1024)
    assert rel(got, want) < 1e-2, f"rel {rel(got, want):.2e}"


def test_dequant_kernel_is_bitwise_equal_to_the_reference():
    """The prefill fallback reconstructs the weight with a CUDA kernel instead of
    format.dequantize. Bitwise, not close: it is the same definition, so any difference
    is a bug in the kernel rather than rounding."""
    from lloyd43.format import dequantize
    cuda_gemv = _cuda()
    for N, K in ((512, 1024), (1024, 3072), (128, 96), (37, 64)):
        _, packed, bs, gs = _case(N, K)
        want = dequantize(packed, bs, gs, K, dtype=torch.bfloat16)
        got = cuda_gemv.dequant_cuda(packed, bs, gs, K)
        assert torch.equal(got, want), f"N={N} K={K}: max|diff| " \
            f"{(got.float() - want.float()).abs().max().item():.3e}"


def test_linear_op_matches_reference_for_batched_input():
    """M > 1 takes the dequantize+GEMM path; it must still agree with the reference."""
    cuda_gemv = _cuda()
    x, packed, bs, gs = _case(256, 1024)
    bs_u8 = bs.contiguous().view(torch.uint8)
    gsf = gs.reshape(()).float().contiguous()
    cfg = cuda_gemv.auto_config(256, 1024)
    for M in (1, 2, 8):
        xm = x.unsqueeze(0).repeat(M, 1).contiguous()
        want = gemv_reference(xm, packed, bs, gs, 1024)
        got = cuda_gemv.linear_op(xm, packed, bs_u8, gsf, 1024, cfg)
        assert rel(got, want) < 1e-2, f"M={M}: rel {rel(got, want):.2e}"
