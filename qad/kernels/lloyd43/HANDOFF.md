# lloyd43 inference kernels — handoff

**Status: correctness done, and it now runs in vLLM. Single-user decode on GB10 is 2.1x
(Qwen3-0.6B) to 3.4x (Qwen3-8B) faster than dense bf16, end to end, measured in vLLM.**
Standalone, the CUDA kernel is 3.13x over bf16 on the 8B projections; the Triton kernel
2.47x. Both were 1.33x when this was handed over. On B200 the Triton kernel was 0.28x;
neither backend has been re-measured there since, so those numbers are stale.

**Prefill is a regression, badly: TTFT is 6-20x worse.** See §9. Decode is what this
kernel is for; prefill takes a dequantize-then-cuBLAS fallback and needs a real M>1
kernel before anyone serves this to users.

Written for an agent picking this up on a **baremetal GPU box**, away from the SLURM
cluster it was developed on. Section 5 now carries measurements from two very different
machines — read the one you are on, and re-measure before trusting either.

**Read §5.1 before quoting the 4.57x ceiling at anyone.** It is a traffic ratio, not an
achievable speedup, and the difference is large enough to change what "done" means.

---

> **Note (cleanup).** The Triton GEMV, the NVFP4 python baselines and the synthetic
> per-model decode estimate have been removed: the CUDA kernel superseded the first, real
> vLLM checkpoints the second, and real vLLM end-to-end the third. Sections below that
> describe them are kept as the record of how the numbers were arrived at.

## 1. What this is, and why it exists

QAD (`../../`) trains weight-quantized LLMs. One of its formats, `lloyd43`, is a 3-bit
weight-only format: an 8-level lookup table, blocks of 16, two-level scaling. It has been
trained and evaluated, but it has never actually run at 3 bits — QAD ships it
*pseudo-quantized*, meaning the checkpoint stores dequantized bf16 weights and vLLM runs
it as an ordinary bf16 model. From `quantizers/lloyd.py`:

> There is no 3-bit LUT kernel to serve this, so it is **pseudo-quantized**.

So the format's accuracy numbers are real, but its *speed* claim is entirely unproven.
This package is the missing half: a real packed layout, a reference decoder, and a kernel
that dequantizes on the fly so the weight is never materialized.

The prize is memory traffic. A GEMV at batch size 1 — the decode step of inference — is
purely bandwidth-bound; runtime is (bytes of weight read) / (achievable bandwidth). So:

| | bytes per weight |
|---|---|
| bf16 | 2.0 |
| lloyd43 (3-bit index + fp8 scale per 16) | 3/8 + 1/16 = **0.4375** |
| **traffic ratio** | **4.57x** |

4.57x is how much less we read. It is *not* the speedup available, because bandwidth is
size-dependent: reading 4.57x fewer bytes means reading them at a lower GB/s. See §5.1.

---

## 2. The format (the part you must not get wrong)

For a weight `W` of shape `(N, K)`, K a multiple of 32:

```
packed        int32          (N, 3, K // 32)    3-bit indices, bit-planed
block_scale   float8_e4m3fn  (N, K // 16)       SIGNED
global_scale  float32        scalar
```

Dequantization is exactly:

```
eff[n, b] = float32(block_scale[n, b]) * global_scale
W[n, k]   = LUT[idx[n, k]] * eff[n, k // 16]
```

with the 8-level grid

```python
LUT = [-4.7038, -2.8698, -1.3696, 0.0000, +1.2204, +2.5285, +4.0473, +6.0000]
```

### Two things that will silently corrupt results

**The block scale is signed.** lloyd43 uses a *signed* normalization: the scale absorbs
the sign of each block's max-abs element, so that element always maps to exactly `+6.0`,
the top grid level. About **51% of blocks therefore have a negative scale**.
`float8_e4m3fn` is a signed type and stores this fine — but the UE4M3 convention used for
NVFP4 block scales elsewhere in this repo does *not*. Storing these unsigned produces a
checkpoint that looks structurally valid and scores 0.0. This exact bug has already cost
this project once. `test_block_scales_are_signed` pins it.

**The grid is asymmetric on purpose.** Four positive levels, three negative, plus a pinned
zero. That is not a bug and not a symmetric-grid opportunity — the `+6.0` and `0.0` pins
are what the format buys over plain Lloyd (see `../../quantizers/grids.py` for the
measured trade: ~3% worse relative MSE for an exact block outlier and a real
flush-to-zero).

### Why bit-planes

3 bits does not divide a byte. Rather than waste bits (10 indices per 32-bit word) or do
unaligned loads (8 indices per 3 bytes), 32 consecutive indices are stored as 3 words, one
per bit position:

```
packed[n, p, g]  bit j  ==  bit p of idx[n, 32*g + j]
idx[n, 32*g + j] = sum_p ((packed[n, p, g] >> j) & 1) << p
```

Exactly 3 bits per weight, nothing wasted, every load naturally aligned, and the unpack is
three shifts and three masks with no cross-lane traffic. 32 is two whole blocks of 16, so
a group never straddles a block-scale boundary.

The plane index precedes the group index so that, for one row and one bit-plane, words for
consecutive k are contiguous — a kernel walking K gets unit-stride loads. Words are
`int32`, not `uint32`: Triton's uint32 support is patchy, and `(w >> j) & 1` is correct
under an arithmetic shift anyway, since sign extension only fills bits above the masked
one.

**If you change the layout, change it in `format.py` and let the tests tell you what
broke.** They will: the packing, the reference, the loop reference and the kernel all
independently encode it.

---

## 3. Layout of the package

```
lloyd43/
├── pyproject.toml           setuptools, src layout, installable standalone
├── HANDOFF.md               this file
├── csrc/
│   └── lloyd43_gemv.cu      the CUDA kernel (JIT-built; see §6.1)
├── benchmarks/              per-model CSV + PNG, regenerated by bench.py
├── exp.py probe_bw.py probe_size.py   diagnostics, not shipped code
├── tune_cuda.py             sweeps launch configs, emits CUDA_SHAPE_TABLE
├── vllm_serve.py            single-user decode/prefill benchmark in vLLM (see §9)
├── src/lloyd43/
│   ├── format.py            the layout + packer. Depends on torch ONLY.
│   ├── reference.py         crude baseline: decode to dense bf16, then multiply
│   ├── cuda_gemv.py         JIT loader + wrapper for the CUDA kernel
│   ├── roofline.py          measured per-size bandwidth ceiling (see §5.1)
│   ├── vllm_plugin.py       the vLLM quantization method (see §9)
│   └── bench.py             triton.testing.perf_report harness + per-model report
└── tests/
    ├── conftest.py          skip logic for GPU / QAD availability
    ├── test_format.py       self-contained: packing, bit budget, grid usage
    ├── test_gemv.py         loop -> reference -> triton, each against the last
    ├── test_cuda_gemv.py    the CUDA kernel against the same reference
    └── test_vs_qad.py       external cross-check against the training code
```

`format.py` **reimplements** QAD's quantization math (`to_e4m3`, `index_nearest`, the
constants) rather than importing it. That is deliberate and load-bearing in two ways: the
package installs with nothing but torch, and the bitwise-equality test against QAD becomes
real evidence instead of a tautology. Do not "simplify" this by importing from
`quantizers.blocked`.

---

## 4. Running it on baremetal

Requirements: Python ≥3.10, torch ≥2.4, triton ≥3.0, pytest. Developed against torch
2.10 / triton 3.6 on a **B200 (sm_100)**; re-measured against torch 2.13 / triton 3.7 on a
**GB10 (sm_121)**.

```bash
cd qad/kernels/lloyd43
pip install -e . --no-deps --no-build-isolation      # --no-deps: don't touch the torch you have
python -m pytest                                      # 44 tests (37 + 7 qad, see below)
python -m lloyd43.bench --sweep k                     # K sweep at N=4096
python -m lloyd43.bench --providers torch triton cuda # both backends side by side
python -m lloyd43.bench --shapes qwen3                # real projection shapes
python vllm_serve.py --model Qwen/Qwen3-8B \
    --quant none lloyd43 lloyd21 nvfp4 nvfp4a16       # real vLLM decode -> benchmarks/
python -m lloyd43.bench --providers torch triton reference --save-path results/
python probe_bw.py ; python probe_size.py             # device roofline (diagnostics)
python exp.py ablate | variants | memsweep | configs  # kernel ablations & tile sweeps
```

On a uv-managed venv, `VIRTUAL_ENV=/path/to/.venv uv pip install -e . --no-deps
--no-build-isolation` does the same thing.

An editable install is the point: the benchmark then always runs the working tree, with no
stale copy to chase. If you would rather not install at all, `PYTHONPATH=$PWD/src` works
identically. `conftest.py` prints the resolved `lloyd43.__file__` in the pytest header —
check it if you ever suspect you are benchmarking something other than what you edited.

### The QAD cross-check skips here — and not for the documented reason

`test_vs_qad.py` imports `quantizers.*` from the QAD tree three levels up, so the original
advice was "bring the whole repo". That is necessary but no longer sufficient: even with
the full tree present, `quantizers/__init__.py` eagerly imports `nvr2bit`, which imports
`quantizers.luts_backend` — **deliberately withheld from this tree**. So the import fails,
the header says `qad cross-check: SKIPPED`, and 7 tests skip.

Those 7 are the strongest correctness signal (bitwise equality with `blocked_quantize`).
Without them the remaining 19 prove internal consistency only. **Do not paper over this by
stubbing `luts_backend` or bypassing the package `__init__`** — a fake module that lets an
equality test pass is worse than a skipped test. Run the cross-check on a machine where
that module exists.

### Verification status

`python -m pytest` → **37 passed, 7 skipped** on GB10. The skips are the QAD cross-check above. Every kernel variant in §5 was also
checked against `gemv_reference` inside `exp.py` before being timed, so nothing was
benchmarked without first being shown correct.

---

## 5. Measurements so far

B200, N=4096, bf16 x, M=1, GB/s of *weight traffic* (each provider over its own byte
count, so compare via the derived speedup, not the raw column):

| K | torch bf16 (cuBLAS) | triton v1 | triton v2 | triton v3 |
|---|---|---|---|---|
| 1024 | 740 | 25.2 | 29.3 | 32.0 |
| 2048 | 1243 | 37.7 | 46.9 | 54.8 |
| 4096 | 1949 | 59.2 | 83.3 | 73.9 |
| 8192 | 2843 | 77.8 | 134.0 | 115.6 |
| 16384 | 3383 | 92.4 | **210.8** | 146.1 |

- **v1** — flat-k indexing. Every lane re-loaded the packed word it shared with 31 others,
  and the fp8 scale it shared with 15 others. ~8x slower than cuBLAS.
- **v2** — tile shaped `(BLOCK_N, NB, 16)`: NB quantization blocks of 16. Each
  scale is loaded once per block, each word twice (two blocks share a 32-group). 2.3x
  better than v1. *Was the shipped kernel until the GB10 work below; superseded by v7.*
- **v3** — v2 plus an LUT select tree (7 `tl.where` on the three bits) instead of
  `tl.load(lut_ptr + idx)`. **Slower at K ≥ 4096**, better below it. Reverted. The LUT sits
  in cache; the extra ALU and register pressure cost more than the gather did. Recorded so
  you do not repeat it.

The versions after that are all GB10 work; the Triton backend has since been removed:

- **v4** — v2 with the reduction deferred (accumulator stays `(BLOCK_N, NB)` across the K
  loop) and the block scale applied per block rather than per weight. With a wide
  `BLOCK_K` this is where nearly all of the gain came from.
- **v5** — split-K. Never written; see §6 for why it got deprioritized.
- **v6** — `tl.dot` / tensor cores with M padded to 16. **0.59x.** Rejected.
- **v7** — v4 with the packed words loaded at group rather than block granularity, so each
  distinct word is requested once at unit stride. Worth ~2%; kept because it is also the
  clearer way to write it. **This is the current kernel.**

**Where that leaves us on B200.** At K=16384, v2 achieved 210.8 GB/s where cuBLAS
demonstrates ~3383 GB/s — ~6% of achievable, a 0.28x speedup. That machine has not been
revisited since the v7 rewrite below.

### 5.1 GB10 (DGX Spark, sm_121) — and why the 4.57x ceiling is wrong

Second machine, wildly different balance: 48 SMs, 24 MiB L2, unified LPDDR5X. Measured
peak streaming read **238 GB/s** (vs B200's ~3400). Run `python probe_bw.py`.

The thing that changes the whole framing: **achievable bandwidth depends strongly on how
much you read** (`python probe_size.py`, L2 flushed exactly as `do_bench` does):

| footprint | flat-read GB/s | floor |
|---|---|---|
| 7 MiB | 107 | |
| 28 MiB — *lloyd43 packed at N=4096,K=16384* | **156** | 189 µs |
| 128 MiB — *bf16 dense, same shape* | **223** | 601 µs |
| 1 GiB | 244 | |

So at that shape the honest ceiling is 601/189 = **3.18x, not 4.57x**. The quantized
format is penalised by its own virtue: a smaller transfer gets less bandwidth. cuBLAS bf16
measures 596 µs against a 601 µs flat-read floor — it is *exactly* at the roof, so the
comparison is fair and the baseline is not beatable by tuning.

Do not "fix" this by benchmarking at small K to make the ratio look better. And note that
L2 capacity is *not* a confound here: a GEMV reads each weight exactly once, so there is no
reuse for L2 to serve regardless of footprint.

N=4096, GB/s of each provider's own weight traffic:

| K | torch bf16 | v2 (old) | **v7 (now)** | speedup vs bf16 |
|---|---|---|---|---|
| 1024 | 104 | 29.2 | **37.9** | 1.66x |
| 2048 | 130 | 39.8 | **56.0** | 1.97x |
| 4096 | 148 | 51.9 | **74.6** | 2.30x |
| 8192 | 211 | 61.2 | **90.0** | 1.95x |
| 16384 | 225 | 68.2 | **102.0** | 2.07x |

Real qwen3 projection shapes: **min 1.65x, median 2.10x, max 2.72x**.

At K=16384 the kernel is at 291 µs against the 189 µs floor for 28 MiB — **65% of
achievable**, and 2.07x of the 3.18x that is actually on the table.

### 5.2 What the ablation says the cost centres are (`python exp.py ablate`)

Measured on GB10 at N=4096, K=16384. Only mode 0 is numerically correct; the others exist
to remove one cost at a time while keeping every load live so nothing is dead-coded away.

| | µs |
|---|---|
| full | 542 |
| no LUT gather | 448 |
| cheap unpack | 467 |
| **no unpack AND no gather** | **475** |

Removing *both* is no faster than removing either — the ordering is inside the noise.
**Neither the unpack nor the LUT gather is the binding constraint**, which retires a lot of
theorizing (including the select-tree question §5's v3 chased). Register spilling was the
other suspicion and is also not happening: the winning config reports 127 registers,
0 spills. The bottleneck was tile shape and the reduction structure — see the
the CUDA kernel's header for what actually won.

---

### 5.3 Per-model decode (real vLLM, `vllm_serve.py`)

Writes `benchmarks/qwen3_model_speedup.{csv,png}` and `qwen3_shape_detail.csv`. Every
projection is timed once at M=1 and weighted by how many times the model contains it, so
the number is `sum(t_bf16) / sum(t_lloyd43)` over one full pass through the linears QAD
quantizes. Layer inventories come from the real published configs; `lm_head` is excluded
because `replace_linears` skips it.

| model | hidden | quantized linears | triton | **cuda** | roofline ceiling |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 196 | 1.35x | **1.58x** | 3.14x |
| Qwen3-1.7B | 2048 | 196 | 2.05x | **2.39x** | 3.30x |
| Qwen3-4B | 2560 | 252 | 2.08x | **2.50x** | 3.02x |
| Qwen3-8B | 4096 | 252 | 2.50x | **3.04x** | 3.09x |

**This is not an end-to-end token/s number** and should not be quoted as one. Attention,
norms, embeddings, the lm_head and all launch overhead are excluded, and each layer is
measured cold-cache in isolation. It is the ceiling the linear layers place on decode
speedup.

At 8B the CUDA kernel is at **98% of the roofline ceiling**, so there is essentially
nothing left at that shape — further work has to come from the format or from batching,
not from the kernel.

**Small shapes are the remaining weakness, but less than it first appeared.** An earlier
version of this section reported `(1024, 1024)` at 0.77x, i.e. slower than bf16. Most of
that was an artefact: `gemv_lloyd43` took `global_scale` as a Python float, so every call
did an `.item()` host sync — 6.8 µs out of a 35 µs kernel. Both backends now read the
global scale on the device and **reject a float outright**, and the same shape is 1.14x
(CUDA). What remains is genuine: at 1 MiB of packed weight there is not enough work to
fill the machine, so the format's premise weakens. That is why 0.6B lags at 1.58x.

**A measured speedup can legitimately exceed the "ceiling" column.** `(12288, 4096)`
reaches 3.63x against a 3.06x roof, because the roof is *ideal bf16 vs ideal lloyd43*
while the speedup is *actual vs actual* — and cuBLAS runs that tall-skinny shape at only
~77% of its own flat-read floor. The ceiling bounds the kernel, not the baseline.

Note also that the previous shape list was partly wrong — it carried `(5120, 2560)` for
Qwen3-4B, a projection that model does not have; shapes are now derived from the configs.

### 5.4 Against NVFP4, the format that actually competes

bf16 is the baseline, but it is not the alternative anyone would deploy. The alternative is
NVFP4 -- what Blackwell has hardware for and what vLLM ships -- in two variants:

    NVFP4     W4A4   4-bit weights AND activations, CUTLASS fp4 tensor cores
    NVFP4A16  W4A16  4-bit weights, bf16 activations, Marlin weight-only GEMM

`make_nvfp4.py` builds both from the SAME bf16 weights lloyd43 packs, so the
comparison is about format and kernel rather than about whose checkpoint was calibrated
better. Bytes per weight is very nearly the whole story at batch 1:

| | bytes/weight |
|---|---|
| bf16 | 2.0 |
| NVFP4 (either variant) | 4/8 + 1/16 = 0.5625 |
| lloyd43 | 3/8 + 1/16 = **0.4375** |

so lloyd43 is 1.29x ahead on traffic alone. Standalone, per-layer, weighted by layer count
(`vllm_serve.py`, plotted by `plot_decode.py`):

| model | lloyd43 | NVFP4 | NVFP4A16 | lloyd43 / A16 |
|---|---|---|---|---|
| 0.6B | **3.05x** | 1.84x | 2.52x | 1.21x |
| 1.7B | **3.35x** | 2.12x | 2.67x | 1.25x |
| 4B | **2.89x** | 2.16x | 2.53x | 1.14x |
| 8B | **3.37x** | 2.50x | 2.92x | 1.15x |
| 14B | **3.07x** | 2.35x | 2.64x | 1.16x |

End to end in vLLM at 8B (`vllm_serve.py --quant none lloyd43 nvfp4 nvfp4a16`, the NVFP4
arms served from `cortecs/Qwen3-8B-NVFP4{,A16}`): bf16 13.8, lloyd43 **47.3**, NVFP4 39.4,
NVFP4A16 40.1 tok/s -- so 3.42x over bf16 and 1.18x over the better NVFP4 variant.

Three things to take from this:

1. **W4A16 beats W4A4 at decode, by 15-32%.** Decode has no arithmetic intensity, so the
   fp4 tensor cores have nothing to do while W4A4 still quantizes the activation on every
   call. W4A4 is a prefill format. Do not benchmark it at batch 1 and conclude NVFP4 is
   slow.
2. **lloyd43's margin is close to the 1.29x its bytes buy** -- 1.14-1.25x. What is left is
   kernel efficiency against Marlin, which is well tuned; there is not a lot of headroom
   here any more.
3. **lloyd43 leads on every model**, by 1.14-1.25x against the 1.29x its bytes buy, so it
   now captures 88-97% of its traffic advantage. It used to LOSE on 0.6B (2.00x against
   2.39x) -- see §5.5, which was a per-call bug rather than anything about small shapes.

The two harnesses disagree (1.09x standalone vs 1.18x in vLLM at 8B) because `bench.py`
times UNFUSED projections while vLLM runs fused qkv/gate_up. lloyd43's dispatch table covers
the fused shapes explicitly, so it gains more from fusion than the Marlin path does. The
vLLM number is what a user gets; the standalone one is the cleaner format-vs-format test.

**And the caveat that matters most: TTFT.** In the same vLLM run, prefill was 17 ms for
NVFP4 and 41 ms for NVFP4A16 against **450 ms for lloyd43** -- both NVFP4 variants have real
prefill kernels and lloyd43 falls back to dequantize-then-cuBLAS. On any prompt of length,
that erases much of the decode win. §6 item 3 is not optional if this is ever served.

### 5.5 The bug that made small models look bad, and how it hid

lloyd43 measured 2.00x on Qwen3-0.6B against NVFP4A16's 2.39x, and the loss was
concentrated in one shape -- (1024, 1024), the model's most common projection, at 23.2 us
against Marlin's 14.5. That is 19 GB/s for 448 KB of packed weight where Marlin managed 40.

**It was not the format, the tile shape, or small-shape physics. Both eager wrappers ran
`LUT.to(x.device)` on every call**, allocating and copying the 32-byte grid host-to-device
each time. Invisible at (12288, 4096); most of the runtime at (1024, 1024). Fixed by
`format.lut_for(device)`, a per-device cache both backends share. Result:

| shape | before | after |
|---|---|---|
| (1024, 1024) | 23.2 us | **11.1** |
| (1024, 2048) | 22.8 | 15.7 |
| (3072, 1024) | 28.0 | 22.5 |
| (12288, 4096) | 168 | 167 |

Large shapes unchanged, which is the signature of a fixed per-call cost. 0.6B went
2.00x -> 3.05x and every other model gained 4-8%.

**Two ways this hid, both worth remembering:**

*The config sweep could not see it.* Every one of the 48 configs paid the same overhead, so
they clustered at 18.8-19.4 us and the tuner reported a confident optimum. It was ranking
noise. What exposed it was extrapolating latency against N -- 18.8, 23.7, 28.2 us for
N = 1024, 2048, 3072 -- to a ~14 us intercept, against a 4.2 us launch floor. **If a sweep
comes back flat, stop tuning and go looking for a constant.**

*It had already been fixed once, elsewhere.* The custom-op path added `lut_for` when CUDA
graph capture forced the issue (§9.2), but the eager wrappers kept the old line. That is
exactly why vLLM measured 1.18x over NVFP4A16 while the standalone harness said 1.09x, and
the discrepancy was attributed to projection fusion at the time. Fusion was part of it; this
was the larger part. **A harness disagreement is evidence, not noise.**

The tuned table was retuned from scratch afterwards, not just extended for 14B: the old
entries were partly fitted to the artifact, and the optima did move for most small shapes
(1024x1024 went `w16 r1 g1 smem` -> `w8 r1 g1 reg`). Any config chosen while a dominant
constant sat in the measurement is suspect.

The same class of bug was in the NVFP4 comparison -- `apply_fp4_marlin_linear` imported
inside `forward` -- and is fixed there too. A baseline handicapped by its wrapper is worse
than no baseline.

## 6. What to try next

Several items that used to be here are now answered; keeping the corpses visible so nobody
digs them up again.

**Settled — do not redo:**

- ~~The 3-D tile / double reduction~~ — was real, and fixing it was most of the win.
  The accumulator now stays `(BLOCK_N, NB)` across the K loop and collapses once at the
  end. Combined with wide `BLOCK_K` this took 1.33x → 2.07x.
- ~~The autotune space~~ — was the quiet villain. It capped `BLOCK_K` at 256 and floored
  `BLOCK_N` at 8, so the winning tile (**4×2048**) was *outside the search*; autotuning
  could never have found it. Now centred on wide-K/few-rows. If you add shapes, sanity
  check that the optimum is not sitting on a boundary of the space again.
- ~~LUT gather vs select tree~~ — settled by ablation, §5.2. Not the constraint. Leave it.
- ~~Register spilling~~ — measured, 0 spills.
- ~~x is re-read by every program~~ — a red herring at least on GB10. x is 32 KB at
  K=16384 and trivially L2-resident; the traffic is ~16 MB against a 411 µs kernel,
  i.e. noise. Do not spend shared memory on it without measuring first.
- ~~`tl.dot` / tensor cores at M=1~~ — tried, **0.59x**. Padding M to 16 is not the
  problem; the transposes go through shared memory and it blows the 99 KB smem limit
  above `BLOCK_N=16`. For M=1 this is a dead end. It is still the right structure for
  M>1 prefill, where the padding waste disappears.

### 6.1 The CUDA kernel (`csrc/lloyd43_gemv.cu`)

Written against the same spec and the same tests; `src/lloyd43/cuda_gemv.py` JIT-builds it
with `torch.utils.cpp_extension.load` on first use, so the package still installs with
nothing but torch. It is **1.2–1.4x faster than the Triton kernel** and is the one to
extend. Read the file header for the reasoning; the short version:

- one `int4` of packed words per lane (16 B = 4 groups = 128 weights), so a warp issues
  one **512-byte contiguous** request per bit-plane — the "wide contiguous runs" lesson
  from §5.1 taken as far as it goes;
- a private scalar accumulator per lane per row for the whole of K, with **exactly one**
  warp reduction at the very end;
- the block scale applied once per 16 weights and the **global scale folded into the LUT
  registers**, so neither is a per-weight cost;
- **the 8-entry LUT lookup is a single `__shfl_sync`.** Lanes 0..7 hold the table and the
  3-bit index is used as the source lane, so every lane gets its own entry in one
  instruction — no shared memory, no bank conflicts, no divergence. This is the trick
  worth reusing elsewhere;
- x is staged in shared memory per CTA, **padded to `WPL + 1` floats per lane-chunk**.
  Without the pad all 32 lanes walk their own 128-element run and land on the same bank,
  which serializes the read 32 ways.

Tuning is `(warps, rows_per_warp)`; `auto_config` picks 16x2 for large shapes and 8x1 for
small, from measurement. `tests/test_cuda_gemv.py` pins correctness across every launch
config, the scalar fallback for `K % 128 != 0`, `N` values that do not fill a CTA, and the
signed-scale case.

**Not yet done in CUDA:** the plane interleaving in item 1 below (the reason the port was
wanted), split-K, and M>1.

**Still open, in the order I would try them:**

1. **Interleave the bit-planes: `(N, K/32, 3)` instead of `(N, 3, K/32)`.** This is the
   main remaining structural lever and the one to decide *before* writing CUDA, because
   changing it later means rewriting the checkpoint. Today a row is three separate 2 KB
   streams; interleaved it is one contiguous 6 KB run, cutting concurrent DRAM streams
   per program by 3x. The measured gap is suggestive: a flat contiguous read of 28 MiB
   takes 189 µs, the same bytes through our row/plane structure take ~280 µs.
   **Triton cannot express this** — it needs power-of-2 tile dims and the fast axis would
   be 3 — which is a concrete reason to want the CUDA port, where it is just index
   arithmetic. Note the original layout choice (§2) was made for unit-stride loads along
   K, which interleaving preserves; it is the *stream count* that improves.

2. **Split-K.** Was the previous author's main suspicion and is still untested. Less
   compelling than it looked: at `BLOCK_N=4` there are already 1024 programs for 48 SMs,
   so this is not a parallelism shortage, and the deferred reduction removed the
   loop-carried cross-lane dependency that made the serial K walk expensive. Worth trying
   for small K (the 1.66x at K=1024 is the weakest point on the curve) more than for large.

3. **M>1 prefill.** The reference and the format are already M-agnostic; `gemv_lloyd43` is
   the only piece hard-coding a vector. This is where `tl.dot` earns its keep.

4. ~~**CUDA/CUTLASS.**~~ Done — see §6.1. It is a `PROVIDERS` entry (`--providers torch
   triton cuda`) and a third series in the per-model plot.

5. **Re-measure on B200.** Everything above is GB10, whose memory system is ~15x slower
   and whose size-dependent bandwidth curve is what makes the ceiling 3.18x rather than
   4.57x. B200 has far more bandwidth and far more SMs; the tile shapes that won here
   (few rows, very wide K) are tuned to 48 SMs and may well be wrong there. Re-run
   `probe_size.py` first — the ceiling itself will be different.

### Profiling note

`ncu` is installed (`/usr/local/cuda/bin/ncu`) but **counter collection needs root on this
box** — `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1`, so it fails with
`ERR_NVGPUCTRPERM` for a normal user. Either get that flag flipped
(`nvidia-modprobe -u -c=0` / the `NVreg_RestrictProfilingToAdminUsers=0` module option) or
profile as root. In the meantime, Triton's own `kernel.n_regs` / `kernel.n_spills` answer
the register question without any privilege, and `exp.py`'s ablation harness answers "what
is this costing" more directly than a counter dump would. `nsys` traces without
privileges if you only need kernel timings.

### Adding an implementation to the benchmark

One entry in `PROVIDERS` in `bench.py` and one name on the command line:

```python
PROVIDERS = {
    "torch":  ("torch bf16 (cuBLAS)", lambda x, wbf, p, bs, gs, K: lambda: gemv_bf16(x, wbf), 2.0),
    "triton": ("triton lloyd43",      lambda x, wbf, p, bs, gs, K: lambda: gemv_lloyd43(x, p, bs, gs, K), 0.4375),
    "cuda":   ("cuda lloyd43",        lambda x, wbf, p, bs, gs, K: lambda: my_cuda_gemv(x, p, bs, gs, K), 0.4375),
}
```

It is built on `triton.testing.perf_report` / `Benchmark` / `do_bench`, so the sweep, the
CSV, the PNG and the multi-line comparison come for free and stay consistent as
implementations accumulate. The third tuple element is bytes/weight and is what makes the
GB/s columns meaningful across providers with different footprints.

---

## 7. The correctness contract

Whatever you write, it must satisfy these. They are already encoded in `tests/`:

1. `unpack(pack(idx)) == idx` for all indices, including index 7 everywhere (which sets
   bit 31 in all three planes and makes the words negative int32).
2. Storage is **exactly** 3 bits per weight.
3. `dequantize(pack_from_weight(w))` is **bitwise equal** to QAD's
   `blocked_quantize(w, grid_rounder(LLOYD43_SIGNED_3BIT), 16, signed=True)[0]` — not
   close, equal. This is the test that matters.
4. Our `to_e4m3` and `index_nearest` agree with QAD's elementwise, *including at exact
   midpoints* where the tie rule (ties go to the higher index) decides.
5. A packed `SignedLloydLinear` reproduces that layer's own cached `_wq` exactly.
6. The kernel matches the reference to <1e-2 relative, and is no worse than the reference
   against an fp32 ground truth computed on the same dequantized weight. (That last
   comparison is what separates kernel error from the quantization error the kernel is
   faithfully reproducing — do not compare against the *original* weight and conclude the
   kernel is inaccurate.)
7. N need not divide the tile: `test_triton_handles_a_non_multiple_of_block_n` uses
   N=1000.

The three-layer structure of `test_gemv.py` is deliberate — crude Python loop → vectorized
reference → kernel, each checked against the one below. When something breaks, the layer
that fails tells you where the bug is.

---

## 8. Loose ends

- ~~`bench.py`'s speedup summary printed nothing~~ — **fixed.** `perf_report` appends the
  ylabel to every column, so `"triton lloyd43"` never matched `"triton lloyd43 (GB/s
  (weight traffic))"`. It matches on the prefix now.
- ~~The autotune space has never been examined~~ — **rewritten**, see §6. It was excluding
  the winning tile outright.
- `gemv_lloyd43` requires `K % 32 == 0` and rejects batched `x`. Both are checked and
  raise; neither is fundamental.
- **The QAD cross-check cannot run here.** `quantizers/__init__.py` eagerly imports
  `nvr2bit` → `quantizers.luts_backend`, which is deliberately withheld from this tree, so
  `import quantizers.blocked` fails and all 7 of `test_vs_qad.py` skip. The other 19 tests
  pass and prove internal consistency, but nothing currently verifies agreement with the
  trained weights on this box. Do not "fix" this by stubbing `luts_backend`. If you need
  the cross-check, run it where that module exists.
- Nothing here is wired into QAD's export path yet. Producing a real packed checkpoint
  from a trained run means teaching `export/save.py` this layout; today `pack_from_weight`
  is the only producer and it works off a live weight tensor.
- `exp.py`, `probe_bw.py` and `probe_size.py` at the package root are diagnostics, not
  shipped code — they are what produced §5.1 and §5.2 and are worth keeping until the CUDA
  port has its own. `src/lloyd43/roofline.py` *is* package code: `bench.py` uses it for the
  per-shape ceiling, so the honest ceiling and the measured speedup cannot drift apart.

---

## 9. Serving it in vLLM

`src/lloyd43/vllm_plugin.py` registers a `lloyd43` quantization method via
`register_quantization_config`, and `pyproject.toml` exposes it as a `vllm.general_plugins`
entry point. Then:

```bash
python vllm_serve.py --model Qwen/Qwen3-8B          # bf16 vs lloyd43, decode and prefill
python vllm_serve.py --model Qwen/Qwen3-0.6B --quant lloyd43
```

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B", quantization="lloyd43")   # any bf16 checkpoint
```

**The entry point is not optional.** vLLM builds the model in a separate EngineCore
process; importing the plugin in the caller registers it only there, and the engine dies
with an unhelpful "Engine core initialization failed". The entry point is loaded in every
process vLLM spawns.

**It quantizes on the fly.** There is still no packed-lloyd43 checkpoint format (§8), so
the method loads an ordinary bf16 checkpoint and packs each linear in
`process_weights_after_loading`, then drops the dense copy. Load takes ~1.5-3x longer.
The upside is that both arms of the benchmark run *the same weights*, so the comparison
isolates the kernel.

vLLM already fuses q/k/v and gate/up, so a decoder layer issues 4 weight GEMVs, not 7.
Those fused shapes are in `CUDA_SHAPE_TABLE` — that is why `TUNE_SHAPES` includes them.

### 9.1 Measured, GB10, batch 1, 168-token prompt

Decode and prefill are separated by the two-point slope method (generate 8 tokens and 128
tokens, difference over difference), so prefill cannot contaminate the decode figure.
Prefix caching is off, or the repeat run would hit cache and TTFT would read as zero.

| model | bf16 decode | **lloyd43 decode** | speedup | bf16 TTFT | lloyd43 TTFT |
|---|---|---|---|---|---|
| Qwen3-0.6B | 127.8 tok/s | **267.2 tok/s** | **2.09x** | 4.9 ms | 29.5 ms |
| Qwen3-1.7B | 48.6 tok/s | **138.8 tok/s** | **2.86x** | 3.0 ms | 89.5 ms |
| Qwen3-4B | 22.6 tok/s | **71.7 tok/s** | **3.18x** | 5.4 ms | 227 ms |
| Qwen3-8B | 14.1 tok/s | **47.1 tok/s** | **3.36x** | 22.4 ms | 434 ms |

The decode numbers agree closely with the standalone per-layer estimate in §5.3 (2.09 vs
2.10 predicted at 0.6B), which is the cross-check worth having: two independent harnesses,
same answer. 8B comes out *above* its standalone figure because vLLM's fused projections
are taller, and taller is where the kernel does best.

**TTFT is 6-20x worse and that is the honest headline caveat.** Prefill has M > 1, so it
takes the fallback in `cuda_gemv._linear_op`: reconstruct the dense weight with the CUDA
dequant kernel, then cuBLAS. For 8B that is ~16 GB of writes per prefill. It is correct
and it is not acceptable for serving. **Fixing it means the M>1 kernel (§6 item 3)** --
that is now the highest-value piece of work left, ahead of anything in the GEMV.

### 9.2 Three traps, all of which cost hours

If you extend this, these will bite again:

1. **A CPU constant inside a traced region kills CUDA graph capture.** `format.dequantize`
   uses the module-level `LUT`, which lives on the CPU. Traced by inductor it becomes a
   host-side constant that must be copied in, and the failure is
   `Cannot copy between CPU and CUDA tensors during CUDA graph capture` thrown from
   generated code naming nothing recognisable. `dequantize` now takes an optional
   device-resident `lut=`; the op caches one per device via `cuda_gemv.lut_for`.
2. **Do not let the fallback be traced at all.** Branching in Python and calling
   `dequantize` for M>1 puts the entire weight reconstruction in the graph, which then
   failed vLLM's graph partitioner with `cycle exists between partitions!`. The whole
   linear is now one opaque custom op (`lloyd43::linear`) that branches internally, so the
   compiler sees one call with nothing to fuse or reorder.
3. **No host syncs, ever.** `.item()` on the global scale would break capture outright.
   This is the same fix as §8's `.item()` note, and it is what makes the op graph-safe.

### 9.3 Environment notes for this box

Reproducing the above needed three things that are nothing to do with lloyd43:

- vLLM was a source checkout, not installed. `VLLM_USE_PRECOMPILED=1 uv pip install -e .
  --no-deps --no-build-isolation` (plus `setuptools_rust`, `setuptools_scm`) fetches the
  prebuilt binaries and takes seconds; a full source build would be hours. `--no-deps`
  matters: the runtime deps are safe (`requirements/common.txt` does not pin torch) but
  `requirements/cuda.txt` does, and letting it reinstall torch would break the CUDA build
  everything else here depends on.
- `torchvision` is needed at runtime (an unrelated minimax_m3 import during kernel warmup)
  and must be installed `--no-deps` for the same reason.
- **flashinfer 0.6.16 is broken on Python 3.11** (`array.array[int]` needs 3.12) and vLLM's
  guard in `compilation/passes/fusion/allreduce_rms_fusion.py` only catches `ImportError`,
  so a `TypeError` took down the engine. Patched locally to `except Exception`. That is an
  upstream bug worth reporting; only `flashinfer.comm` is affected, so uninstalling
  flashinfer would be over-broad.
