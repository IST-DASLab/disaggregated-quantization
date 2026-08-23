"""NVFP4 (w4a4) linear on top of vLLM's CUTLASS kernels, for prefill benchmarking.

Prefill is a GEMM, not a GEMV, so it is compute-bound rather than bandwidth-bound and the
3-bit decode kernel in ../lloyd43 is the wrong tool entirely. NVFP4 is the right
comparison for prefill: 4-bit weights AND 4-bit activations feeding Blackwell's fp4 tensor
cores, so it cuts both the bytes streamed and the math.

Both matter here. For an offloaded model the weights have to cross a link every forward,
and NVFP4 makes that 4x smaller than bf16 -- which moves the point where streaming stops
being the bottleneck (see offload_forward.py).

    vLLM ops: scaled_fp4_quant -> packed uint8 (2 values/byte) + fp8_e4m3 block scales
              cutlass_scaled_fp4_mm(a, b, sa, sb, alpha, out_dtype)

SCALE CONVENTIONS, which are easy to get backwards
--------------------------------------------------
`scaled_fp4_quant` takes the ENCODE global scale (what the input is multiplied by):

    gs_enc = (FP4_MAX * E4M3_MAX) / amax        # 6 * 448 / amax

and the kernel's `alpha` undoes both sides, so it is the product of the two DECODE scales:

    alpha = (1 / gs_enc_x) * (1 / gs_enc_w)

Getting this inverted does not fail loudly -- it produces output off by a large constant
factor, which looks like a broken model rather than a broken scale. `_selftest` below
checks against a bf16 matmul.

Verified on GB10 (sm_121); `ops.cutlass_scaled_mm_supports_fp4(121)` is True.
"""

import torch
import torch.nn as nn

FP4_MAX = 6.0
E4M3_MAX = 448.0


def _ops():
    import vllm._custom_ops as ops
    return ops


def global_encode_scale(t: torch.Tensor) -> torch.Tensor:
    """(6 * 448) / amax -- the factor `scaled_fp4_quant` expects."""
    amax = t.abs().amax().float().clamp(min=1e-8)
    return ((FP4_MAX * E4M3_MAX) / amax).to(torch.float32)


# Stand-in for a calibrated activation amax. A served NVFP4 checkpoint carries a per-layer
# `input_global_scale` produced by a calibration pass, and vLLM's w4a4 scheme reads it
# straight from the checkpoint; it does NOT reduce over the activation at runtime. Doing
# that reduction per call was costing more than the fp4 GEMM saved at long context (see
# README), so this benchmark uses one fixed value rather than rebuilding calibration.
#
# 10.0 is a plausible magnitude, not a measured one. It is the right shape of thing for a
# LATENCY benchmark -- the kernels, the traffic and the launch count are all exactly what a
# calibrated model would do -- and the wrong thing to draw an accuracy conclusion from.
ACT_AMAX = 10.0



class NVFP4Linear(nn.Module):
    """Drop-in for nn.Linear. Weights quantized once, activation scale fixed at build time.

    Both `x_gs` and `alpha` are fp32 TENSORS, not Python floats, for two reasons: a Python
    float would be baked into the graph as a constant computed on the host, and a fresh
    scalar tensor per call would allocate inside a CUDA graph capture. Registered buffers
    live at stable addresses, so capture works and the values could be updated in place by
    a real calibration pass without recapturing.

    `alpha` folds both decode scales -- 1/x_gs and 1/w_gs -- into one constant, so the
    forward is exactly two kernels: quantize the activation, then the fp4 GEMM.
    """

    def __init__(self, lin: nn.Linear, act_amax: float = ACT_AMAX):
        super().__init__()
        ops = _ops()
        w = lin.weight.data.to(torch.bfloat16).cuda()
        self.out_features, self.in_features = w.shape

        w_gs = global_encode_scale(w)
        w_fp4, w_bs = ops.scaled_fp4_quant(w, w_gs, is_sf_swizzled_layout=True,
                                           backend="cutlass")
        self.register_buffer("w_fp4", w_fp4)
        self.register_buffer("w_bs", w_bs)

        # Static activation scale: (6 * 448) / amax, the same formula the dynamic path used,
        # evaluated once against the calibrated amax instead of the live tensor.
        x_gs = torch.tensor((FP4_MAX * E4M3_MAX) / max(act_amax, 1e-8),
                            dtype=torch.float32, device=w.device)
        self.register_buffer("x_gs", x_gs)
        self.register_buffer("alpha", ((1.0 / x_gs) * (1.0 / w_gs)).to(torch.float32))
        self.bias = lin.bias

    @property
    def nbytes(self) -> int:
        """Bytes that would have to be streamed to use this layer."""
        return self.w_fp4.numel() * self.w_fp4.element_size() + \
            self.w_bs.numel() * self.w_bs.element_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = _ops()
        shape = x.shape
        x2d = x.reshape(-1, self.in_features).to(torch.bfloat16)
        rows = x2d.shape[0]
        chunk = self._chunk_rows(rows)
        if chunk >= rows:
            out = self._mm(ops, x2d)
        else:
            # See _chunk_rows: the fp4 GEMM collapses on tall outputs, and feeding it the
            # same work in slices it handles recovers most of the loss. Bitwise identical
            # to the unchunked call -- each slice is an independent row block.
            out = torch.cat([self._mm(ops, x2d[i:i + chunk])
                             for i in range(0, rows, chunk)], 0)
        out = out.reshape(*shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    def _mm(self, ops, x2d: torch.Tensor) -> torch.Tensor:
        x_fp4, x_bs = ops.scaled_fp4_quant(x2d, self.x_gs, is_sf_swizzled_layout=True,
                                           backend="cutlass")
        return ops.cutlass_scaled_fp4_mm(x_fp4, self.w_fp4, x_bs, self.w_bs, self.alpha,
                                         torch.bfloat16)

    # Output elements above which `cutlass_scaled_fp4_mm` falls off a cliff on this box.
    # Measured in probe_nvfp4_scaling.py: on (N=20480, K=2560) it holds 2.4x over bf16 up
    # to M=10240 (210M output elements) and collapses to 1.51x at M=12288 (251M) and 0.75x
    # at M=32768. Shapes with a short output -- `down` at N=2560, `qkv` at N=4096 -- never
    # reach the threshold and are never chunked, which matches the measurement that they
    # hold their speedup all the way to 32k.
    #
    # The likely mechanism is L2: the fp4 gate_up weight is ~29 MB against 24 MiB of L2,
    # and a chunk's output write evicts it. Chunking does not recover the full 2.4x for
    # that reason (four chunks of 8192 cost 26.5 ms where a lone M=8192 call is 3.52 ms),
    # but it turns 0.75x into 1.26x on the shape that dominates the MLP.
    MAX_OUT_ELEMS = 200 * 1024 * 1024
    # GB10's L2. The weight has to MISS it for the cliff to exist -- which is why the
    # criterion is not output size alone. Measured the hard way: keying only on output
    # elements chunked Gemma3-1B's gate_up (an 8 MB fp4 weight that sits happily in L2, so
    # it never had a cliff) and cost it 1.44x -> 1.13x at 32k, while the same rule was
    # worth 1.04x -> 1.44x on 4B, whose 26 MB weight does miss.
    L2_BYTES = 24 * 1024 * 1024

    def _weight_bytes(self) -> int:
        """fp4 payload plus one e4m3 scale per 16 weights."""
        return int(self.out_features * self.in_features * (0.5 + 1 / 16))

    # Measured best chunk per (out_features, in_features), from sweeping every candidate
    # on this box. A static table rather than a runtime sweep: the block is compiled with
    # fullgraph=True, so `_chunk_rows` is traced -- benchmarking inside it is impossible,
    # and `torch.cuda.is_current_stream_capturing()` is not traceable either. A tuner was
    # tried and broke compilation outright.
    #
    # It has to be a table because no rule fits: 12B's gate_up wants 6784 rows and its down
    # wants 2048, which are per-chunk outputs of 417 MB and 15.7 MB. An output-size
    # threshold that is right for one is silently wrong for the other -- it never fired for
    # down at all, leaving 1.28x on that projection.
    #
    # Worth 1.28x on the projection in isolation but only ~1.03x on a whole block: the
    # torch.cat that reassembles the chunks hands most of it back. Measure the block, not
    # the GEMM, before extending this.
    CHUNK_TABLE = {
        (30720, 3840): 6784,     # Gemma3-12B gate_up
        (3840, 15360): 2048,     # Gemma3-12B down
        (20480, 2560): 10240,    # Gemma3-4B  gate_up
        (24576, 4096): 8192,     # Qwen3-8B   gate_up
    }

    def _chunk_rows(self, rows: int) -> int:
        """Rows per GEMM call. Pure function of shape, so it survives tracing.

        The weight missing L2 is necessary: shapes that fit (Gemma3-1B's 8 MB gate_up,
        Gemma3-4B's 14 MB down) have no cliff, and chunking them only adds passes -- keying
        on output size alone cost 1B 1.44x -> 1.13x at 32k.
        """
        if self._weight_bytes() <= self.L2_BYTES:
            return rows
        tuned = self.CHUNK_TABLE.get((self.out_features, self.in_features))
        if tuned is not None:
            return min(tuned, rows)
        return self._chunk_heuristic(rows)

    def _chunk_heuristic(self, rows: int) -> int:
        """The old output-size rule. Only a fallback for when tuning cannot run."""
        if rows * self.out_features <= self.MAX_OUT_ELEMS:
            return rows
        # A multiple of 128: the swizzled block-scale layout is built in 128-row tiles, so
        # an unaligned split would pad every chunk and change the scale layout.
        return max(128, (self.MAX_OUT_ELEMS // self.out_features) // 128 * 128)

    def forward_prequantized(self, x_fp4: torch.Tensor, x_bs: torch.Tensor,
                             lead_shape) -> torch.Tensor:
        """The GEMM alone, for an input that is already fp4 with swizzled scales.

        Exists so a producer can emit fp4 directly and skip the separate quantization
        pass -- see fused_geglu_quant.py, which folds it into the GeGLU that writes this
        layer's input. That input is the largest activation in the block, and quantizing it
        as its own pass costs 45-49% of this layer's time.
        """
        ops = _ops()
        rows = x_fp4.shape[0]
        chunk = self._chunk_rows(rows)
        if chunk >= rows:
            out = ops.cutlass_scaled_fp4_mm(x_fp4, self.w_fp4, x_bs, self.w_bs, self.alpha,
                                            torch.bfloat16)
        else:
            # The swizzled block-scale layout groups 128 rows contiguously, so slicing it
            # at a multiple of 128 is exactly the slice for those rows -- verified bitwise
            # down to chunk=128. Without that, this path could not be chunked at all, and
            # `down` (the one the GeGLU fusion routes here) would keep missing its 1.28x.
            out = torch.cat([ops.cutlass_scaled_fp4_mm(x_fp4[i:i + chunk], self.w_fp4,
                                                       x_bs[i:i + chunk], self.w_bs,
                                                       self.alpha, torch.bfloat16)
                             for i in range(0, rows, chunk)], 0)
        out = out.reshape(*lead_shape, self.out_features)
        return out if self.bias is None else out + self.bias


def convert(model: nn.Module, skip=("lm_head",), act_amax: float = ACT_AMAX) -> int:
    """Swap every nn.Linear for an NVFP4 one, in place. Returns the count."""
    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(module, name, NVFP4Linear(child, act_amax=act_amax))
                n += 1
    return n


def _selftest():
    """Two separate things, deliberately not conflated.

    `matched` uses the activation's true amax, so it tests the SCALE CONVENTIONS -- get
    encode/decode or alpha backwards and this blows up. `static` uses the benchmark's fixed
    ACT_AMAX against the same data, so it shows what the stand-in costs in accuracy. Only
    the first is a correctness assertion; the second is reported, because a fixed amax is a
    latency choice and its error depends entirely on how close the guess is.
    """
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}  fp4 supported: "
          f"{_ops().cutlass_scaled_mm_supports_fp4(121)}   ACT_AMAX={ACT_AMAX}\n")
    for N, K, M in ((2048, 2048, 512), (6144, 4096, 1024), (4096, 12288, 256)):
        lin = nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        want = (x.float() @ lin.weight.data.float().T)

        true_amax = x.abs().amax().item()
        matched = NVFP4Linear(lin, act_amax=true_amax)
        static = NVFP4Linear(lin, act_amax=ACT_AMAX)
        rel_m = ((matched(x).float() - want).norm() / want.norm()).item()
        rel_s = ((static(x).float() - want).norm() / want.norm()).item()

        dense = lin.weight.numel() * 2
        print(f"  N={N:<6} K={K:<6} M={M:<5} amax={true_amax:5.2f}  "
              f"rel matched={rel_m:.4f}  static={rel_s:.4f}   "
              f"weight {dense/2**20:.0f} -> {matched.nbytes/2**20:.0f} MiB "
              f"({dense/matched.nbytes:.2f}x)")
        assert rel_m < 0.15, \
            f"NVFP4 error {rel_m:.3f} with a matched amax -- check scale conventions"

    # The scale must be a stable fp32 tensor, or CUDA graph capture cannot see it.
    q = NVFP4Linear(nn.Linear(512, 512, bias=False).cuda().to(torch.bfloat16))
    assert q.x_gs.dtype == torch.float32 and q.x_gs.device.type == "cuda"
    assert q.alpha.dtype == torch.float32 and q.alpha.numel() == 1
    assert "x_gs" in dict(q.named_buffers()) and "alpha" in dict(q.named_buffers())
    print("\nscales are fp32 CUDA buffers (graph-capturable, and streamed with the block)")
    print("self-test OK (w4a4 error is percent-level by nature, not 1e-3)")


if __name__ == "__main__":
    _selftest()
