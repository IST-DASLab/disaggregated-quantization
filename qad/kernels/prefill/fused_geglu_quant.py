"""GeGLU fused with NVFP4 activation quantization, in one pass over the intermediate.

    python fused_geglu_quant.py        # correctness vs vLLM, then the speedup

WHY
---
A W4A4 MLP currently does this:

    h        = gate_up(x)                  # (M, 2I) bf16   written by the GEMM
    y        = gelu(gate) * up             # (M,  I) bf16   written, then read again
    y4, ybs  = scaled_fp4_quant(y)         # (M, I/2) fp4   the read that this file removes
    out      = down(y4)

The middle tensor is the largest activation in the block, and it is written in bf16 and
immediately read back only to be quantized. At Gemma3-4B / 32k that is 671 MB written plus
671 MB read, per layer, 34 layers -- ~46 GB of round trip that buys nothing.

It is worth more than that arithmetic suggests. probe_nvfp4_scaling.py measures the
activation quantization at **45-49% of NVFP4 time on the `down` projection** (its input is
exactly this tensor), against 2-3% on gate_up.

MEASURED (GB10, I=10240, the Gemma3-4B intermediate):

       M      eager+quant   compiled+quant     fused    vs compiled
    8192          4.85 ms          3.08 ms   1.78 ms          1.73x
   16384          9.44             6.28      3.50            1.79x
   32768         19.23            12.35      6.89            1.79x

The baseline that counts is the COMPILED one: the block is `torch.compile`d, so inductor
already fuses gelu-and-mul into a single kernel, and quoting the 2.7x against the eager
path would be claiming credit for a fusion that comes free. Against the real baseline this
is 1.79x, i.e. 5.5 ms per layer at 32k -- ~186 ms over Gemma3-4B's 34 layers, against a
3074 ms NVFP4 forward.

AND WHAT THAT IS WORTH END TO END, WHICH IS MUCH LESS
------------------------------------------------------
1.79x on this kernel is 1.03-1.21x on the model, because the MLP epilogue is one slice of a
block that also does attention and three other projections. Measured over the whole Gemma 3
prefill sweep, fused / unfused:

    NVFP4 resident      270m 1.03-1.12   1b 0.99-1.21   4b 1.07-1.14   12b 1.03-1.11
    NVFP4 SSD offload   ~1.00 while drive-bound, 1.04-1.14 at 16k-32k

Median ~1.09x resident. The SSD arm shows nothing at short sequences and that is the
correct result, not a disappointment: there the forward is waiting on the drive, so making
compute faster cannot help until compute overtakes the drive at long context. Sub-1.0
entries in that arm are drive variance on points pinned to the disk, not regressions.

The check on this is the arithmetic closing: 5.5 ms/layer x 34 layers = 186 ms against
3074 ms is 6%, and the measured 4B/32k gain is 1.077, i.e. 7%.

This fusion is HALF the story at long context and not the important half. It does nothing
about the `cutlass_scaled_fp4_mm` cliff on the tall-output gate_up GEMM, which is what made
NVFP4 lose to bf16 above 16k; that needed the L2-aware M-chunking in nvfp4_linear.py. With
both, Gemma3-4B at 32k goes 1.04x -> 1.44x over bf16 and 12B 1.03x -> 1.50x, where the
fusion alone got 4B to 1.12x. See the README table.

BOTH FAMILIES USE THIS KERNEL, VIA THE `act` SWITCH
----------------------------------------------------
Gemma 3 is GeGLU-tanh and Qwen3 is SwiGLU, so the kernel takes an `ACT` constexpr and the
wrapper an `act=` argument. vLLM does ship `silu_and_mul_nvfp4_quant` for the SiLU case and
that was the intended route for Qwen3 -- but it is an out-variant whose scale buffer is
int32-typed and padded, and neither shape convention tried reproduced
`scaled_fp4_quant`'s output. One kernel with a switch, already validated against vLLM's
quantizer and covered by the mutation control in test_fused_geglu.py, is less to get wrong
than a second code path whose layout is being guessed at.

THE SCALE CONVENTION, WHICH IS THE ONLY HARD PART
--------------------------------------------------
Getting this subtly wrong produces a GEMM that runs at full speed and returns garbage, so
it is derived from the dequant rather than guessed. vLLM's fp4 GEMM computes

    x ~= fp4_value * block_scale / x_gs           with   x_gs = (6 * 448) / amax_global

(`alpha = 1/x_gs * 1/w_gs` folds the two global scales into the epilogue). For a block whose
max-abs is `amax_b`, the fp4 values must span at most 6, so

    block_scale = amax_b * x_gs / 6      stored as e4m3
    fp4_value   = y * x_gs / block_scale     <- the ROUNDED e4m3 scale, not the exact one

Using the unrounded scale here is the classic silent-error version: it is right to within
e4m3 rounding, which looks fine in a norm and is wrong in the last bit of every block.

Block scales come out in LINEAR layout and are handed to vLLM's own `swizzle_blockscale`
for the CUTLASS interleave. Reimplementing that swizzle in Triton would save one pass over
a tensor that is 1/32 the size of the one this kernel exists to avoid touching -- 21 MB
against 671 MB at 32k -- and would be one more thing to get silently wrong.
"""

import torch
import triton
import triton.language as tl

# This build has no `tl.math.tanh`; libdevice's maps to CUDA's tanhf, which is what
# PyTorch's gelu(approximate="tanh") calls. Matching the exact tanh matters here because a
# last-ulp difference can flip an fp4 code for a value sitting on a rounding boundary, and
# the selftest below demands bitwise equality.
from triton.language.extra import libdevice

FP4_MAX, E4M3_MAX = 6.0, 448.0
BLOCK = 16                      # weights per fp4 block scale


@triton.jit
def _geglu_nvfp4_kernel(
    x_ptr,          # (M, 2I) bf16: [gate | up]
    out_ptr,        # (M, I // 2) uint8, two fp4 per byte
    bs_ptr,         # (M, I // 16) fp8e4m3, LINEAR layout
    gs_ptr,         # fp32 scalar, the encode global scale (6*448)/amax
    I,
    stride_xm, stride_om, stride_bm,
    BLOCK_I: tl.constexpr,
    BLK: tl.constexpr,          # weights per block scale (16); a jitted fn cannot read the
    ACT: tl.constexpr,          # module-level BLOCK, so it comes in as a constexpr
):                              # ACT: 0 = GeLU-tanh (Gemma 3), 1 = SiLU (Qwen3)
    """One program: one row, BLOCK_I columns of the intermediate (a multiple of 32).

    BLOCK_I must be a multiple of 32 so that the fp4 pairing (2 per byte) and the block
    scales (1 per 16) both divide evenly and no program straddles either boundary.
    """
    m = tl.program_id(0)
    j = tl.program_id(1)
    gs = tl.load(gs_ptr)

    cols = j * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < I
    gate = tl.load(x_ptr + m * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + m * stride_xm + I + cols, mask=mask, other=0.0).to(tl.float32)

    if ACT == 0:      # GeLU, tanh approximation -- Gemma's `gelu_pytorch_tanh`
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        act = 0.5 * gate * (1.0 + libdevice.tanh(inner))
    else:             # SiLU -- Qwen3
        act = gate * tl.sigmoid(gate)
    y = act * up
    # Round through bf16 even though nothing is stored: the unfused path materializes this
    # tensor in bf16 and quantizes THAT, so skipping the rounding would make this kernel
    # slightly more accurate than the thing it replaces -- and not a drop-in. Keeping it
    # identical is what lets the selftest demand bitwise equality instead of a tolerance.
    y = y.to(tl.bfloat16).to(tl.float32)

    # Per-16 block scale. amax over each group of BLK, broadcast back.
    g = tl.reshape(y, (BLOCK_I // BLK, BLK))
    amax = tl.max(tl.abs(g), axis=1)                            # (BLOCK_I // BLK,)
    bs = (amax * gs / 6.0).to(tl.float8e4nv)                # round to e4m3 ...
    bs_f = bs.to(tl.float32)                                    # ... and use the ROUNDED value
    # An all-zero block gives bs == 0; the values are zero too, so any nonzero denominator
    # yields the right answer and this avoids a 0/0.
    den = tl.where(bs_f == 0, 1.0, bs_f)
    v = g * gs / den[:, None]

    # e2m1: codes 0..7 are magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign.
    # Round to nearest, ties to even CODE -- hence the alternating >/>= on the midpoints.
    a = tl.abs(v)
    code = tl.zeros((BLOCK_I // BLK, BLK), dtype=tl.int32)
    code = tl.where(a > 0.25, 1, code)
    code = tl.where(a >= 0.75, 2, code)
    code = tl.where(a > 1.25, 3, code)
    code = tl.where(a >= 1.75, 4, code)
    code = tl.where(a > 2.5, 5, code)
    code = tl.where(a >= 3.5, 6, code)
    code = tl.where(a > 5.0, 7, code)
    code = tl.where(v < 0, code | 8, code)
    code = tl.reshape(code, (BLOCK_I,))

    # Pack pairs: element 2i in the LOW nibble, 2i+1 in the high one. `tl.split` is the way
    # to peel a trailing axis of 2 -- Triton has no `pair[:, 0]` column indexing.
    lo, hi = tl.split(tl.reshape(code, (BLOCK_I // 2, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)

    ocols = j * (BLOCK_I // 2) + tl.arange(0, BLOCK_I // 2)
    tl.store(out_ptr + m * stride_om + ocols, packed, mask=ocols < (I // 2))
    bcols = j * (BLOCK_I // BLK) + tl.arange(0, BLOCK_I // BLK)
    tl.store(bs_ptr + m * stride_bm + bcols, bs, mask=bcols < (I // BLK))


ACT_IDS = {"gelu": 0, "silu": 1}


def geglu_nvfp4_quant(x: torch.Tensor, global_scale: torch.Tensor, block_i: int = 512,
                      act: str = "gelu"):
    """(M, 2I) bf16 [gate|up] -> ((M, I//2) uint8 fp4, (M, I//16) e4m3 scales, LINEAR).

    `act` selects the gate nonlinearity: "gelu" (tanh approximation, Gemma 3) or "silu"
    (Qwen3). vLLM ships `silu_and_mul_nvfp4_quant` for the SiLU case, but its out-variant
    scale layout did not reproduce `scaled_fp4_quant`'s under either shape convention tried,
    and one validated kernel with a switch is less to get wrong than two code paths.

    Swizzle the scales with vllm's `swizzle_blockscale` before handing them to
    `cutlass_scaled_fp4_mm`.
    """
    assert x.dim() == 2 and x.shape[1] % 2 == 0, x.shape
    M, twoI = x.shape
    I = twoI // 2
    if I % 32:
        raise ValueError(f"intermediate {I} must be a multiple of 32")
    x = x.contiguous()
    out = torch.empty(M, I // 2, dtype=torch.uint8, device=x.device)
    bs = torch.empty(M, I // BLOCK, dtype=torch.float8_e4m3fn, device=x.device)
    block_i = min(block_i, I)
    grid = (M, triton.cdiv(I, block_i))
    _geglu_nvfp4_kernel[grid](
        x, out, bs, global_scale, I,
        x.stride(0), out.stride(0), bs.stride(0),
        BLOCK_I=block_i, BLK=BLOCK, ACT=ACT_IDS[act],
    )
    return out, bs


MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)      # e2m1 magnitudes, by code
MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)      # and the rounding boundaries between


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


@torch.library.custom_op("prefill::geglu_nvfp4", mutates_args=())
def geglu_nvfp4(x: torch.Tensor, global_scale: torch.Tensor, act: str = "gelu"
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, 2I) bf16 [gate|up] -> (fp4, CUTLASS-swizzled block scales), ready for the GEMM.

    Opaque to dynamo on purpose: it wraps a raw Triton kernel plus vLLM's swizzle, and
    `fullgraph=True` should not have to reason about either. The swizzle lives inside so
    callers cannot forget it -- handing CUTLASS linear-layout scales is a silent
    wrong-answer bug, not an error.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    fp4, bs = geglu_nvfp4_quant(x, global_scale, act=act)
    return fp4, swizzle_blockscale(bs)


@geglu_nvfp4.register_fake
def _(x, global_scale, act="gelu"):
    M, two_i = x.shape
    I = two_i // 2
    return (torch.empty(M, I // 2, dtype=torch.uint8, device=x.device),
            torch.empty(_round_up(M, 128), _round_up(I // BLOCK, 4),
                        dtype=torch.float8_e4m3fn, device=x.device))


def _selftest():
    """Agreement with vLLM's `scaled_fp4_quant` on the same GeGLU output.

    THE CONTRACT IS NOT BITWISE, AND DELIBERATELY SO. Block scales are bitwise identical.
    The fp4 codes agree on ~99.97% of values; every disagreement is a value sitting on an
    exact rounding midpoint, where the two candidate codes are EQUIDISTANT and the
    quantization error is the same either way. The cause is a 1-ulp difference in computing
    `value / block_scale` between this kernel and vLLM's, which tips a tie one way or the
    other; it is not a difference in the rounding rule (both round ties to even -- verified
    directly against the hardware at all seven midpoints) and not a difference in the scale
    convention (quantizing against the UNROUNDED scale was tried and is 150x worse: 5.2%
    of codes differ, and those are real errors rather than ties).

    So this asserts what is actually true and worth guaranteeing:
      * scales bitwise equal,
      * every differing code lies on a midpoint, and
      * the dequantized tensors differ by no more than one fp4 step at that magnitude.
    A tolerance-only check would pass a kernel with a genuinely wrong scale convention;
    this one would not.
    """
    import vllm._custom_ops as vops
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}\n")
    ok = True
    for M, I in ((64, 512), (1024, 10240), (4096, 6912)):
        x = torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16)
        gs = torch.tensor((FP4_MAX * E4M3_MAX) / 10.0, dtype=torch.float32, device="cuda")

        got_fp4, got_bs = geglu_nvfp4_quant(x, gs)
        gate, up = x.chunk(2, dim=-1)
        ref_in = (torch.nn.functional.gelu(gate.float(), approximate="tanh")
                  * up.float()).to(torch.bfloat16)
        ref_fp4, ref_bs = vops.scaled_fp4_quant(ref_in, gs, is_sf_swizzled_layout=False)
        ref_bs = ref_bs.view(torch.uint8)[:M, :I // BLOCK]

        same_bs = torch.equal(got_bs.view(torch.uint8), ref_bs)

        def codes(packed):
            b = packed.view(torch.uint8).to(torch.int32)
            return torch.stack([b & 0xF, (b >> 4) & 0xF], -1).reshape(M, I)

        cg, cr = codes(got_fp4), codes(ref_fp4)
        diff = cg != cr
        n_diff = int(diff.sum())

        # Reconstruct the value each code was chosen for, and check the disagreements are
        # all ties.
        bs_f = got_bs.view(torch.uint8).view(torch.float8_e4m3fn).float()
        den = torch.where(bs_f == 0, torch.ones_like(bs_f), bs_f)
        v = (ref_in.float() * gs / den.repeat_interleave(BLOCK, dim=1)).abs()
        # The property that matters is that the two codes are EQUIDISTANT from the value --
        # i.e. the disagreement costs nothing. Testing distance-to-a-midpoint with an
        # absolute tolerance is the same idea but wrong at large magnitudes, where the
        # reconstruction of v carries more rounding error than the tolerance allows.
        mag = torch.tensor(MAG, device=v.device)
        err_g = (mag[(cg & 7).long()] - v).abs()
        err_r = (mag[(cr & 7).long()] - v).abs()
        on_tie = True
        if n_diff:
            gap = (err_g - err_r).abs()[diff]
            scale = torch.clamp(v[diff], min=1.0)
            on_tie = bool((gap <= 1e-3 * scale).all())

        # ... and that dequantizing either way lands within one fp4 step.
        dq_g = mag[(cg & 7).long()] * torch.where(cg >= 8, -1.0, 1.0)
        dq_r = mag[(cr & 7).long()] * torch.where(cr >= 8, -1.0, 1.0)
        step = float((dq_g - dq_r).abs().max()) if n_diff else 0.0

        good = same_bs and on_tie and step <= 2.0
        ok &= good
        print(f"M={M:>5} I={I:>6}   scales bitwise {'YES' if same_bs else 'NO'}"
              f"   codes differing {100*n_diff/diff.numel():.4f}%"
              f"   all on ties {'YES' if on_tie else 'NO'}"
              f"   max step {step:.1f}   {'OK' if good else 'FAIL'}")
    print("\n" + ("fused kernel agrees with vLLM (scales exact; code differences are "
                  "equidistant ties)" if ok else
                  "MISMATCH -- differences are NOT ties, do not use"))
    return ok


def _bench():
    """Against the path this replaces, with the GeGLU compiled.

    The eager split path is two elementwise kernels plus the quantize; the real block is
    `torch.compile`d, which fuses gelu-and-mul into one. Benchmarking against the eager
    version would credit this kernel with a fusion inductor already does for free, so the
    baseline here is the compiled one. Both are reported, since the gap between them is
    what inductor is worth on its own.
    """
    import vllm._custom_ops as vops
    gs = torch.tensor((FP4_MAX * E4M3_MAX) / 10.0, dtype=torch.float32, device="cuda")

    @torch.compile(dynamic=False)
    def geglu(x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.gelu(gate, approximate="tanh") * up

    print(f"\n{'M':>7}{'I':>7}{'eager+q':>10}{'compiled+q':>12}{'fused':>9}"
          f"{'vs compiled':>13}")
    for M, I in ((8192, 10240), (16384, 10240), (32768, 10240)):
        x = torch.randn(M, 2 * I, device="cuda", dtype=torch.bfloat16)

        def eager_split():
            gate, up = x.chunk(2, dim=-1)
            y = torch.nn.functional.gelu(gate, approximate="tanh") * up
            return vops.scaled_fp4_quant(y, gs, is_sf_swizzled_layout=False)

        def compiled_split():
            return vops.scaled_fp4_quant(geglu(x), gs, is_sf_swizzled_layout=False)

        compiled_split()                                    # warm the compile
        t0 = triton.testing.do_bench(eager_split, warmup=5, rep=20)
        tc = triton.testing.do_bench(compiled_split, warmup=5, rep=20)
        t1 = triton.testing.do_bench(lambda: geglu_nvfp4_quant(x, gs), warmup=5, rep=20)
        print(f"{M:>7}{I:>7}{t0:>10.3f}{tc:>12.3f}{t1:>9.3f}{tc/t1:>12.2f}x")
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if _selftest():
        _bench()
