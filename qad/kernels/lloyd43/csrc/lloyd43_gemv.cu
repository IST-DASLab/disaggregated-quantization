// Blackwell CUDA GEMV for the lloyd43 packed 3-bit format.
//
//   y[n] = sum_k LUT[idx[n,k]] * bscale[n, k/16] * gscale * x[k]
//
// with idx stored bit-planed as int32 (N, 3, K/32), bscale as SIGNED float8_e4m3fn
// (N, K/16) and gscale an fp32 scalar. See ../src/lloyd43/format.py -- that file is the
// spec and this kernel must agree with it bit for bit, which tests/test_cuda_gemv.py
// checks against the same reference the Triton kernel is held to.
//
// WHY IT IS SHAPED THIS WAY
// -------------------------
// The Triton work (HANDOFF.md 5.1-5.2) established that this problem is bound by the
// memory ACCESS PATTERN, not by arithmetic: ablating the unpack and the LUT lookup out of
// the Triton kernel entirely left it no faster. So the design goal is long contiguous
// reads, enough of them in flight, and no cross-lane traffic in the k loop.
//
//   * Each lane pulls GPL consecutive packed words as ONE vector load (int/int2/int4), so
//     a warp issues a single 128/256/512-byte contiguous request per bit-plane.
//   * Each lane keeps a private scalar accumulator per row for the whole of K. The warp
//     reduces exactly once, at the end.
//   * The block scale is applied once per 16 weights, after the inner product, and the
//     global scale is folded into the LUT registers, so neither costs anything per weight.
//   * The 8-entry LUT lookup is a single `__shfl_sync`: lanes 0..7 hold the (already
//     global-scaled) table and the 3-bit index is the source lane, so every lane gets its
//     own entry in one instruction -- no shared memory, no bank conflicts, no divergence.
//     This is the trick worth stealing from this file.
//   * x is staged in shared memory once per CTA and reused by every row the CTA owns. The
//     buffer is padded to SX_STRIDE = GPL*32 + 1 floats per lane-chunk. The stride is ODD,
//     so lane t's run starts on bank (t * SX_STRIDE) % 32 and the 32 lanes hit 32
//     different banks. Without the pad every lane starts on bank 0 and the read serializes
//     32 ways.
//
// GPL IS THE PARAMETER THAT MATTERS MOST, AND IT IS SHAPE-DEPENDENT
// ----------------------------------------------------------------
// A warp covers 32 * GPL groups = 1024 * GPL weights per step. If K is smaller than that,
// the surplus lanes have nothing to do: at GPL=4 and K=1024 (32 groups) only 8 of 32 lanes
// were live, which is why the first version of this kernel managed just 1.14x on the small
// Qwen3 projections while the large ones sat at 98% of roofline. GPL is now a template
// parameter picked per shape -- see `auto_config` in cuda_gemv.py, whose table is measured
// (tune_cuda.py), not guessed.
//
// The row loop is INNERMOST, inside the j loop over weights. That is deliberate: x is the
// same for every row, so one shared-memory read feeds ROWS fused multiply-adds. With the
// rows outermost (the obvious way, and how this kernel was first written) x is re-read
// once per row and the LDS traffic is ROWS times higher.
//
// TRIED AND REJECTED, so nobody repeats them:
//   * Marlin-style bit_op->fp16 dequant (csrc/quantization/marlin/dequant.h in vLLM):
//     constructs values by OR-ing bits into a fixed exponent, which is a LINEAR map of the
//     index. lloyd43's grid is an arbitrary non-uniform Lloyd grid, so it cannot be built
//     that way. The shuffle is already one instruction; there is nothing to win.
//   * GPTQ-style packed 3-bit (qdq_3.cuh) needs cross-word shuffles to reassemble indices
//     that straddle 32-bit boundaries. The bit-planed layout here has no straddling at all
//     and is strictly cheaper to unpack.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <type_traits>

namespace {

constexpr int GROUP = 32;   // weights per bit-planed int32 word
constexpr int QBLOCK = 16;  // weights per fp8 block scale
// NBITS is a TEMPLATE parameter on the kernels, not a constant: lloyd21 is this exact
// layout at 2 bit-planes with a 4-level grid. Only the plane count and the LUT width
// differ, so one kernel serves both and the dispatch picks the instantiation.

__device__ __forceinline__ float fp8e4m3_to_float(unsigned char raw) {
    // Signed: ~51% of lloyd43 block scales are negative, and e4m3 (not ue4m3) is what
    // format.py stores. Reading these unsigned scores 0.0 while looking structurally fine.
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(raw, __NV_E4M3)));
}

// GPL consecutive packed words for one row and one bit-plane, as a single vector load
// when the row stride allows it (W % GPL == 0). The scalar path is both the unaligned
// fallback and the ragged last chunk; out-of-range words decode to index 0, harmless
// because the matching x has been zero-filled.
template <int GPL, bool VEC>
__device__ __forceinline__ void load_words(const int* __restrict__ plane, int g, int gmax,
                                           unsigned int out[GPL]) {
    if (VEC && g + GPL <= gmax) {
        if (GPL == 4) {
            const int4 v = *reinterpret_cast<const int4*>(plane + g);
            out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
        } else if (GPL == 2) {
            const int2 v = *reinterpret_cast<const int2*>(plane + g);
            out[0] = v.x; out[1] = v.y;
        } else {
            out[0] = plane[g];
        }
    } else {
#pragma unroll
        for (int i = 0; i < GPL; ++i) out[i] = (g + i < gmax) ? plane[g + i] : 0u;
    }
}

// One group's worth of x (32 values) straight into registers, no shared memory. Used by
// the STAGE_X=false path: at small K the staging buffer costs MORE traffic than the
// weights it serves -- at (1024,1024) a CTA writes 4 KB of x to smem to read 1.5 KB of
// packed weight -- and it also forces two __syncthreads per chunk.
__device__ __forceinline__ void load_x_group(const __nv_bfloat16* __restrict__ x,
                                             int k, int K, float out[GROUP]) {
    if (k + GROUP <= K) {
#pragma unroll
        for (int i = 0; i < GROUP; i += 8) {
            union { int4 v; __nv_bfloat162 h[4]; } u;
            u.v = *reinterpret_cast<const int4*>(x + k + i);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(u.h[q]);
                out[i + 2 * q] = f.x;
                out[i + 2 * q + 1] = f.y;
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < GROUP; ++i)
            out[i] = (k + i < K) ? __bfloat162float(x[k + i]) : 0.f;
    }
}

template <int NBITS, int ROWS, int WARPS, int GPL, bool VEC, bool STAGE_X>
__global__ __launch_bounds__(WARPS * 32)
void lloyd43_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                         const int* __restrict__ packed,
                         const unsigned char* __restrict__ bscale,
                         const float* __restrict__ lut,
                         const float* __restrict__ gscale,
                         __nv_bfloat16* __restrict__ y,
                         int N, int K, int W, int NBLK) {
    constexpr int WPL = GPL * GROUP;        // weights per lane per step
    constexpr int CHUNK = 32 * WPL;         // weights per CTA step
    constexpr int SX_STRIDE = WPL + 1;      // odd on purpose, see header

    extern __shared__ float sx[];           // [32][SX_STRIDE]

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nthreads = WARPS * 32;
    const int row0 = (blockIdx.x * WARPS + warp) * ROWS;

    // Lanes 0..(2^NBITS - 1) carry the LUT, pre-multiplied by the global scale so it never
    // appears in the inner loop. gscale is read on the DEVICE: pulling it to the host cost
    // a 6.8 us sync per call, which at small shapes was most of the kernel. The mask makes
    // every lane hold a valid entry, so the __shfl_sync below is unconditionally legal for
    // any index the unpack can produce -- 8 entries at 3 bits, 4 at 2.
    constexpr int NLEV = 1 << NBITS;
    const float lut_reg = lut[lane & (NLEV - 1)] * (*gscale);

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;

    for (int k0 = 0; k0 < K; k0 += CHUNK) {
        if (STAGE_X) {
            __syncthreads();
            for (int i = threadIdx.x; i < CHUNK; i += nthreads) {
                const int k = k0 + i;
                sx[(i / WPL) * SX_STRIDE + (i % WPL)] =
                    (k < K) ? __bfloat162float(x[k]) : 0.f;
            }
            __syncthreads();
        }

        const int gmax = W - k0 / GROUP;              // groups left
        const int wbase = k0 / GROUP + lane * GPL;    // this lane's first group
        const int bbase = k0 / QBLOCK + lane * (WPL / QBLOCK);
        const float* __restrict__ xp = &sx[lane * SX_STRIDE];

        // All ROWS rows' words live at once so the row loop can sit inside the j loop and
        // amortize each shared-memory read of x over ROWS FMAs.
        unsigned int wr[ROWS][NBITS][GPL];
        float sc[ROWS][2 * GPL];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            // Clamp rather than branch: rows are warp-uniform, and reading row N-1 twice
            // is harmless because the store is guarded. Keeps the warp fully converged so
            // every __shfl_sync below is unconditionally legal.
            const int n = min(row0 + r, N - 1);
            const int* __restrict__ prow = packed + (size_t)n * NBITS * W;
#pragma unroll
            for (int p = 0; p < NBITS; ++p)
                load_words<GPL, VEC>(prow + p * W, wbase, W, wr[r][p]);

            const unsigned char* __restrict__ srow = bscale + (size_t)n * NBLK;
#pragma unroll
            for (int b = 0; b < 2 * GPL; ++b) {
                const int bi = bbase + b;
                sc[r][b] = (bi < NBLK) ? fp8e4m3_to_float(srow[bi]) : 0.f;
            }
        }

#pragma unroll
        for (int g = 0; g < GPL; ++g) {
            const float* __restrict__ xg = xp + g * GROUP;
            float xreg[STAGE_X ? 1 : GROUP];
            if (!STAGE_X)
                load_x_group(x, k0 + lane * WPL + g * GROUP, K, xreg);
            // Two blocks of 16 share one word, each with its own scale.
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                float loc[ROWS];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) loc[r] = 0.f;
#pragma unroll
                for (int jj = 0; jj < QBLOCK; ++jj) {
                    const int j = half * QBLOCK + jj;
                    // one LDS (or one register), reused by every row
                    const float xv = STAGE_X ? xg[j] : xreg[STAGE_X ? 0 : j];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        // Over NBITS planes, not a hardcoded three: at 2 bits a third
                        // term would set a bit the 4-entry LUT does not have.
                        int idx = 0;
#pragma unroll
                        for (int p = 0; p < NBITS; ++p)
                            idx |= ((wr[r][p][g] >> j) & 1u) << p;
                        loc[r] = fmaf(__shfl_sync(0xffffffffu, lut_reg, idx), xv, loc[r]);
                    }
                }
#pragma unroll
                for (int r = 0; r < ROWS; ++r)
                    acc[r] = fmaf(loc[r], sc[r][2 * g + half], acc[r]);
            }
        }
    }

    // The only cross-lane reduction in the kernel: once per row, after all of K.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        const int n = row0 + r;
        if (lane == 0 && n < N) y[n] = __float2bfloat16(v);
    }
}

// ---------------------------------------------------------------- dequantize to bf16
// For prefill / batched decode, where a GEMV is the wrong shape entirely. Reconstructing
// the dense weight and handing it to cuBLAS is not clever, but it IS the right fallback,
// and it has to be fast: the obvious PyTorch version in format.py gathers through int64
// indices, which moves 8 bytes per weight to produce 2, and dominated a whole vLLM
// generate() call when it ran once per layer per prefill.
//
// One thread per packed word: NBITS words in (12 B at 3 bits, 8 at 2) plus 2 fp8
// scales, 32 bf16 out (64 B).
template <int NBITS>
__global__ void lloyd43_dequant_kernel(const int* __restrict__ packed,
                                       const unsigned char* __restrict__ bscale,
                                       const float* __restrict__ lut,
                                       const float* __restrict__ gscale,
                                       __nv_bfloat16* __restrict__ out,
                                       int N, int K, int W, int NBLK) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)N * W) return;
    const int n = (int)(tid / W), g = (int)(tid % W);

    const int* __restrict__ prow = packed + (size_t)n * NBITS * W;
    unsigned int wp[NBITS];
#pragma unroll
    for (int p = 0; p < NBITS; ++p) wp[p] = prow[p * W + g];

    const float gs = *gscale;
    const unsigned char* __restrict__ srow = bscale + (size_t)n * NBLK;
    const float s0 = fp8e4m3_to_float(srow[2 * g]) * gs;
    const float s1 = fp8e4m3_to_float(srow[2 * g + 1]) * gs;

    __nv_bfloat16* __restrict__ o = out + (size_t)n * K + (size_t)g * GROUP;
#pragma unroll
    for (int j = 0; j < GROUP; ++j) {
        int idx = 0;
#pragma unroll
        for (int p = 0; p < NBITS; ++p) idx |= ((wp[p] >> j) & 1u) << p;
        o[j] = __float2bfloat16(lut[idx] * (j < QBLOCK ? s0 : s1));
    }
}

#define LAUNCH_ONE(ROWS, WARPS, GPL, VEC, STAGE)                                     \
    do {                                                                               \
        constexpr size_t smem = STAGE                                                  \
            ? (size_t)32 * (GPL * GROUP + 1) * sizeof(float) : 0;                      \
        lloyd43_gemv_kernel<NBITS_V, ROWS, WARPS, GPL, VEC, STAGE>                              \
            <<<grid, WARPS * 32, smem, stream>>>(                                      \
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),         \
            packed.data_ptr<int>(),                                                     \
            reinterpret_cast<const unsigned char*>(bscale.data_ptr<uint8_t>()),         \
            lut.data_ptr<float>(), gscale.data_ptr<float>(),                            \
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),               \
            N, K, W, NBLK);                                                             \
    } while (0)

#define LAUNCH_STAGE(ROWS, WARPS, GPL, VEC)                                            \
    do {                                                                               \
        if (stage_x) LAUNCH_ONE(ROWS, WARPS, GPL, VEC, true);                          \
        else LAUNCH_ONE(ROWS, WARPS, GPL, VEC, false);                                 \
    } while (0)

#define LAUNCH_VEC(ROWS, WARPS, GPL)                                                   \
    do {                                                                               \
        if (vec) LAUNCH_STAGE(ROWS, WARPS, GPL, true);                                 \
        else LAUNCH_STAGE(ROWS, WARPS, GPL, false);                                    \
    } while (0)

#define LAUNCH_GPL(ROWS, WARPS)                                                        \
    do {                                                                               \
        if (gpl == 4) LAUNCH_VEC(ROWS, WARPS, 4);                                      \
        else if (gpl == 2) LAUNCH_VEC(ROWS, WARPS, 2);                                 \
        else LAUNCH_VEC(ROWS, WARPS, 1);                                               \
    } while (0)

}  // namespace

at::Tensor lloyd43_gemv_cuda(at::Tensor x, at::Tensor packed, at::Tensor bscale,
                             at::Tensor lut, at::Tensor gscale, int64_t K,
                             int64_t rows_per_warp, int64_t warps, int64_t gpl_in,
                             bool stage_x) {
    TORCH_CHECK(x.is_cuda() && packed.is_cuda(), "all tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be a vector; got ", x.sizes());
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(packed.scalar_type() == at::kInt, "packed must be int32");
    TORCH_CHECK(bscale.scalar_type() == at::kByte, "bscale must be viewed as uint8");
    TORCH_CHECK(K % GROUP == 0, "K must be a multiple of ", GROUP, "; got ", K);
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && bscale.is_contiguous(),
                "x, packed and bscale must be contiguous");
    TORCH_CHECK(x.numel() == K, "x has ", x.numel(), " elements, expected K=", K);

    const int N = packed.size(0);
    const int W = K / GROUP;
    const int NBLK = K / QBLOCK;
    // Bit width comes from the tensor, not a constant: 3 planes is lloyd43, 2 is lloyd21.
    const int bits = (int)packed.size(1);
    TORCH_CHECK(bits == 2 || bits == 3, "packed must have 2 or 3 bit-planes; got ", bits);
    TORCH_CHECK(lut.numel() == (1 << bits), "lut has ", lut.numel(),
                " levels, expected ", (1 << bits), " for ", bits, " bits");
    TORCH_CHECK(packed.size(2) == W, "packed must be (N, bits, K/32)");
    TORCH_CHECK(bscale.numel() == (int64_t)N * NBLK, "bscale must be (N, K/16)");

    auto y = at::empty({N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int gpl = (int)gpl_in;
    TORCH_CHECK(gpl == 1 || gpl == 2 || gpl == 4, "gpl must be 1, 2 or 4; got ", gpl);
    const bool vec = (W % gpl == 0);

    const int per_cta = (int)(warps * rows_per_warp);
    const dim3 grid((N + per_cta - 1) / per_cta);

    // The (warps, rows) chain, instantiated once per bit width. Textual duplication rather
    // than a template function because the macros below close over a dozen locals; the
    // compiler sees two constant NBITS_V values and specializes each fully.
#define DISPATCH_WARPS()                                                               \
    if (warps == 16 && rows_per_warp == 2) LAUNCH_GPL(2, 16);                          \
    else if (warps == 16 && rows_per_warp == 1) LAUNCH_GPL(1, 16);                     \
    else if (warps == 8 && rows_per_warp == 4) LAUNCH_GPL(4, 8);                       \
    else if (warps == 8 && rows_per_warp == 2) LAUNCH_GPL(2, 8);                       \
    else if (warps == 8 && rows_per_warp == 1) LAUNCH_GPL(1, 8);                       \
    else if (warps == 4 && rows_per_warp == 4) LAUNCH_GPL(4, 4);                        \
    else if (warps == 4 && rows_per_warp == 2) LAUNCH_GPL(2, 4);                        \
    else if (warps == 4 && rows_per_warp == 1) LAUNCH_GPL(1, 4);                        \
    else TORCH_CHECK(false, "unsupported (warps, rows_per_warp) = (", warps, ", ",      \
                     rows_per_warp, ")");

    if (bits == 3) { constexpr int NBITS_V = 3; DISPATCH_WARPS(); }
    else           { constexpr int NBITS_V = 2; DISPATCH_WARPS(); }
#undef DISPATCH_WARPS

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor lloyd43_dequant_cuda(at::Tensor packed, at::Tensor bscale, at::Tensor lut,
                                at::Tensor gscale, int64_t K) {
    TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kInt, "packed int32 cuda");
    TORCH_CHECK(bscale.scalar_type() == at::kByte, "bscale must be viewed as uint8");
    TORCH_CHECK(K % GROUP == 0, "K must be a multiple of ", GROUP);
    const int N = packed.size(0), W = K / GROUP, NBLK = K / QBLOCK;
    const int bits = (int)packed.size(1);
    TORCH_CHECK(bits == 2 || bits == 3, "packed must have 2 or 3 bit-planes; got ", bits);
    auto out = at::empty({N, (int64_t)K},
                         packed.options().dtype(at::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long total = (long long)N * W;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    auto launch = [&](auto nb) {
        lloyd43_dequant_kernel<decltype(nb)::value><<<blocks, threads, 0, stream>>>(
        packed.data_ptr<int>(),
        reinterpret_cast<const unsigned char*>(bscale.data_ptr<uint8_t>()),
        lut.data_ptr<float>(), gscale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            N, (int)K, W, NBLK);
    };
    if (bits == 3) launch(std::integral_constant<int, 3>{});
    else           launch(std::integral_constant<int, 2>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv", &lloyd43_gemv_cuda, "lloyd43 3-bit GEMV (CUDA)",
          py::arg("x"), py::arg("packed"), py::arg("bscale"), py::arg("lut"),
          py::arg("gscale"), py::arg("K"), py::arg("rows_per_warp") = 2,
          py::arg("warps") = 8, py::arg("gpl") = 4, py::arg("stage_x") = true);
    m.def("dequant", &lloyd43_dequant_cuda, "lloyd43 3-bit -> dense bf16 (CUDA)",
          py::arg("packed"), py::arg("bscale"), py::arg("lut"), py::arg("gscale"),
          py::arg("K"));
}
