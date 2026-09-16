# Prefill: offloading and NVFP4

Prefill is a GEMM over the whole prompt — compute-bound, and a completely different problem
from the batch-1 decode GEMV in `../lloyd43`. Two questions here:

1. If a model does not fit in GPU memory, how much does streaming it in block by block
   actually cost?
2. What does NVFP4 (4-bit weights *and* activations, Blackwell fp4 tensor cores) do to that
   picture?

Everything below is GB10, batch 1, four Qwen3 and four Gemma 3 sizes, 128 → 32k tokens.

## The answer

`qwen3_prefill_offload.png` — one panel per model, shared x, independent y. Offloaded
prefill is **flat** while the drive dominates, then converges onto the resident curve once
compute overtakes it. Every forward re-reads the whole model off the SSD with the page
cache dropped, so nothing is served from RAM.

| model | quant | block | SSD floor | SSD within 10% of resident from | SSD @128 | SSD @32k | resident @32k |
|---|---|---|---|---|---|---|---|
| 0.6B | bf16 | 30 MiB | 0.15 s | seq ≥ 8k | 238 ms | 1987 | 1969 |
| 0.6B | NVFP4 | 8 MiB | 0.04 s | seq ≥ 8k | 111 | 1894 | 1867 |
| 1.7B | bf16 | 96 MiB | 0.49 s | seq ≥ 16k | 558 | 2833 | 2824 |
| 1.7B | NVFP4 | 27 MiB | 0.14 s | **seq ≥ 8k** | 227 | 2437 | 2386 |
| 4B | bf16 | 192 MiB | 1.27 s | seq ≥ 16k | 1377 | 6959 | 6843 |
| 4B | NVFP4 | 54 MiB | 0.36 s | **seq ≥ 8k** | 453 | 7116 | 7044 |
| 8B | bf16 | 368 MiB | 2.44 s | seq ≥ 16k | 2554 | 9743 | 9826 |
| 8B | NVFP4 | 104 MiB | 0.69 s | seq ≥ 16k | 801 | 9868 | 9424 |

**Offloading to the SSD is ruinous at short context and free at long.** At 8B it costs 31x
at 128 tokens and nothing at 32k. NVFP4 reaches parity roughly one power of two earlier,
because 3.5x fewer bytes per block lowers the floor 3.5x.

**NVFP4's advantage decays with context — but attention was NOT the main reason, and the
original version of this section got that wrong.** It read: *"the share NVFP4 can touch
shrinks as context grows and the ratio walks to 1.0 … no kernel change fixes it."* Two
kernel changes fixed most of it. The story was superficially plausible — attention is O(S²),
unquantized, and does grow — which is exactly why it survived unchallenged until the decay
was measured instead of reasoned about.

What the measurement showed: bf16 latency doubles cleanly at
every doubling of S, while NVFP4 takes a one-time 3.7x step between 8k and 16k. Dilution is
smooth; a step is a kernel falling over. It was `cutlass_scaled_fp4_mm` collapsing on
tall-output shapes once the weight misses L2. Fixing that plus fusing the activation
quantization took Gemma3-12B at 32k from 1.03x to **1.50x** and Qwen3-4B from 0.97x
(slower than bf16) to 1.09x — see the table further down.

Attention is real but secondary, and its weight depends on the architecture: at Gemma3-12B /
16k it is 7% of a sliding layer and 34% of a global one. The per-component breakdown below quantifies the
whole decomposition; the short version is that **the dense fp4 : bf16 ratio on this box is
2.38x, not 8x**, and that ceiling — not attention — is the dominant term. The 8x figure is
sparsity-inclusive; even the "dense fp4 is 4x" claim that used to sit here is above what
this box delivers through CUTLASS against cuBLAS.

The SSD floor is **measured per (model, quant)** by `load_floor.py`. It runs the real
pipeline -- SSD → pinned host buffer → GPU slot, double buffered, cold page cache -- with the
compute removed, so it is a lower bound including the H2D leg and per-block overheads.

There are two floors because there are two offload modes, and the plots must read the one
matching the curve they draw. `load_floor_ssd.csv` is P only. `load_floor_zero_ssd.csv` is
P + C -- the prefill checkpoint plus the decode carve-out zero-ssd restores -- and is what
`plot_for_paper.py` draws, since `zero-ssd` is the default offload mode. Quoting the `ssd`
floor under a zero-ssd curve understates the drive-bound region by exactly the carve-out.

The zero-ssd floor is not a *strict* lower bound: it reads P then C back to back, while the
benchmark overlaps the C restore with the last layer's compute. Measured / floor at 128
tokens runs 0.96-1.13 across the nine models, and the ones under 1.0 are that overlap plus
a few percent of drive variation.

It used to be `n_blocks × block_bytes / 5.7 GB/s`, and that was wrong in a direction worth
knowing about: 5.7 GB/s was measured on Qwen3-8B's 386 MB blocks, and the
drive does not deliver that on a small read. Measured throughput ranges from **2.63 GB/s on
0.6B's 8 MiB NVFP4 blocks to 5.92 GB/s on 8B's 368 MiB bf16 ones** — a 2.25x spread. The
single-rate estimate was ~2.3x too optimistic at the small end and about right at the large
end, which flattered exactly the models that offload worst.

## Files

| | |
|---|---|
| `offload_forward.py` | the real interleaved forward: `--modes resident ssd zero-ssd ram`. SSD path is read → pinned staging buffer → GPU slot → compute, double buffered, page cache dropped every pass |
| `qwen3_block.py` | fused-QKV/gate_up Qwen3 block; `python qwen3_block.py` checks it against transformers |
| `gemma3_block.py` | the same for Gemma 3, plus its 5:1 sliding/global attention split; `python gemma3_block.py` checks **both** layer types against transformers |
| `nvfp4_linear.py` | NVFP4 linear over vLLM's `scaled_fp4_quant` + `cutlass_scaled_fp4_mm`; `python nvfp4_linear.py` self-tests it |
| `test_fused_geglu.py` | wiring + compile + capture checks, and `--mutate` to prove they can fail |

```bash
python offload_forward.py --model Qwen/Qwen3-8B --quant bf16 nvfp4 --modes resident ssd
python offload_forward.py --model google/gemma-3-4b-it --quant bf16 nvfp4 --modes resident ssd
python load_floor.py          # per-(model, quant) SSD floors, for every model in the CSV
python plot_for_paper.py      # one 2x2 figure per family
# figures are written to ../../notebooks/figures
```

## Gemma 3: the sliding window is the measurement

Gemma 3 alternates **five sliding-window layers with one global one** (`config.layer_types`;
window 512 on 270m/1b, 1024 on 4b/12b). At 32k a local layer attends to 1024 keys rather
than 32768, so running the stack full-causal throughout — the default if you hand SDPA
`is_causal=True` and move on — inflates Gemma 3 prefill by **~3.2x at 32k** while producing
numbers that look entirely reasonable.

No single attention backend wins both jobs on this box, so `gemma3_block.py` uses two.
Measured at the 4b geometry (H=8, KV=4, D=256, window 1024):

| backend | S=8192 | S=32768 | |
|---|---|---|---|
| SDPA `is_causal` | 3.14 ms | **50.6 ms** | used for **global** layers |
| FA2 varlen, full-causal | 3.75 | 53.0 | |
| FA4 (`flash_attn.cute`), full-causal | 3.06 | 49.8 | |
| flex_attention, full-causal mask | — | 123.2 | |
| flex_attention, sliding `BlockMask` | 2.03 | 8.7 | |
| FA4, native `window_size` | 3.02 | 49.0 | window buys ~nothing |
| **FA2 varlen, native `window_size`** | **1.47** | **6.0** | used for **local** layers |

Notes, since "use flash attention" is the obvious instinct and only half right here:

* **FA4 is installed** (`flash_attn` is a namespace package holding only `cute`) and runs on
  sm121 only with a hand-forced small tile — its default hd=256 config asks for 128 KiB of
  shared memory and this arch allows 101376 bytes, so the launch is rejected outright. With
  `tile_mn=(64, 64)` it is correct, but its `window_size` gives 49.0 ms against 49.8 ms
  unwindowed: the mask is applied without out-of-window key blocks being skipped.
* **FA2 comes from vLLM's bundled build**; the standalone `flash_attn` wheel here has no
  `flash_attn_func` at all, and FA3 refuses to load (needs 9.x).
* FA2 needs a `custom_op` + `register_fake` wrapper or `fullgraph=True` fails with
  "Operator does not support running with fake tensors".
* `window_size=(W-1, 0)` is exactly Gemma's `q - k < W`. Pinned by a test against a dense
  masked reference: neighbouring widths are off by ~0.3 rel, so it discriminates sharply.

The harness carries this as an **attention-variant axis**: one compiled callable, one kwarg
set per layer type, and one captured graph per (weight buffer, variant). Qwen3 has a single
variant and is unaffected. Offload output is checked **bitwise** against resident on both
families, eager and captured — the check that has caught every real bug in this file.

## Three benchmarking choices, and why

**1. A custom decoder block, not transformers'.** `qwen3_block.py` fuses q/k/v into one
projection and gate/up into another — 4 GEMM launches per layer instead of 7 — uses SDPA
with `is_causal` and `enable_gqa` so no S×S mask is ever materialized, and takes fixed
positional arguments with one tensor in and one out. transformers' layer threads a Cache
object whose `layer_idx` dynamo guards on (that alone forces one compilation per layer),
plus mask construction, `**kwargs` and output tuples, all of which add guards or break
graphs.

It is checked against transformers' own layer: **rel 3.1e-3** — bf16 rounding from the
fusion — with an exact parameter-count match. Worth it: NVFP4 resident at 0.6B/128 went
**10.4 → 6.5 ms**. The Qwen3 detail that is easy to get wrong, and which that check pins, is
`q_norm`/`k_norm`: RMSNorms over `head_dim`, applied per head after the projection and
*before* RoPE. Omitting them produces plausible garbage.

**2. The timed region is the decoder blocks only.** Embeddings, the RoPE tables and the
final norm are evaluated once beforehand and the stack is fed a fixed hidden state. None of
them has anything to do with where the weights live, all three run outside the compiled
block, and including them adds uncompiled eager noise to both arms while diluting exactly
the quantity being compared. It also makes the RoPE tables fixed tensors, which CUDA graph
capture requires.

**3. CUDA graphs on the two slot pointers.** A captured graph refers to its weights by
*address*. The slot buffers never move — only their contents do, rewritten by the DMA — so
two graphs captured against the two slots stay valid for the whole model, and block *i*
replays `graph[i % 2]` once its weights have landed. At replay there is no Python at all: no
`bind()`, no dynamo guards, no per-kernel launches.

That fixed per-block CPU cost is exactly what caps the drive-bound regime, where the drive
is only kept busy if the consumer gets out of its way. The cost is one activation copy per
block: each graph writes its output at a fixed pool address, which has to be copied into the
shared input before the next replay. An exact two-buffer ping-pong would need each graph
captured with the *other's* output as its input, which is circular — so one copy it is. It
is ~1% at long context and negligible at short, i.e. cheapest exactly where the launch
saving is worth most. Resident captures one graph per block and offload one per slot, so
both arms pay the same copy and stay comparable. `--no-graphs` falls back, and a capture
failure falls back loudly rather than silently reporting a different execution path as
though it were the same one.

Measured effect of 2 and 3 together on 0.6B: **−37%** for NVFP4 resident at 128 tokens,
−10% for bf16, and **+1–4% at 32k** — the activation copy, which grows with sequence length
while the launch saving is constant.

## Things that will mislead you

**NVFP4 resident prefill loses to bf16 above ~12k tokens, and it is a kernel cliff, not
Amdahl.** Every model's resident NVFP4 speedup decays to ~1.0 at 32k (Gemma3-4B 2.56x at 128
tokens → 1.04x at 32k; Qwen3-4B 2.82x → 0.97x). The tempting explanation — attention is not
quantized, so its growing share dilutes the linear-layer speedup — is *wrong here*, and the
increments say so: bf16 doubles cleanly at every doubling of S while NVFP4 takes a one-time
3.7x step between 8k and 16k. Dilution is smooth; this is a step.

Measured by isolating the projections from attention entirely (script since removed; the numbers below are the record). The collapse is in
`cutlass_scaled_fp4_mm` and only on the **tall-output** GEMM:

| bf16/nvfp4 | M=8192 | M=12288 | M=16384 | M=32768 |
|---|---|---|---|---|
| `gate_up` (N=20480, K=2560) | 2.39 | 1.51 | 0.91 | **0.75** |
| `down` (N=2560, K=10240) | 2.01 | 1.94 | 1.90 | 1.90 |
| `qkv` (N=4096, K=2560) | 1.93 | 1.88 | 1.99 | 2.00 |

It starts between M=10240 and M=12288 — where the bf16 output crosses ~500 MB. Chunking the
M axis recovers part of it (32768: 44.6 → 26.5 ms, 0.75x → **1.26x**, bitwise identical) but
not all: four chunks of 8192 cost 26.5 ms while a standalone M=8192 call is 3.52 ms, so each
chunk runs ~1.9x slower in situ. That points at L2 — the fp4 `gate_up` weight is ~29 MB
against this box's 24 MiB L2, and each chunk's 336 MB output write evicts it. bf16's 105 MB
weight never fit, which is why bf16 scales linearly and never shows the cliff.

`NVFP4Linear` now chunks automatically, and **both** conditions are load-bearing: the weight
must miss L2 *and* the output must be large. Keying on output size alone chunked
Gemma3-1B's `gate_up` — an 8 MB weight that sits happily in L2 and never had a cliff — and
cost it **1.44x → 1.13x** at 32k, while the identical rule was worth **1.04x → 1.44x** on
4B, whose 26 MB weight does miss. A one-sided heuristic here is not a smaller win; it is a
regression on half the models.

Combined with the fused activation-quantization, NVFP4 vs bf16 on **resident** prefill,
before → after:

| | 2048 | 8192 | 16384 | 32768 |
|---|---|---|---|---|
| Gemma3-270M | 1.20 → 1.30 | 1.13 → 1.24 | 1.12 → 1.23 | 1.07 → 1.16 |
| Gemma3-1B | 1.40 → 1.62 | 1.32 → 1.57 | 1.22 → 1.41 | 1.26 → 1.44 |
| Gemma3-4B | 1.88 → 2.10 | 1.64 → 1.87 | 1.12 → **1.44** | 1.04 → **1.44** |
| Gemma3-12B | 2.45 → 2.67 | 1.52 → 1.56 | 1.04 → **1.59** | 1.03 → **1.50** |
| Qwen3-0.6B | 1.30 → 1.37 | 1.21 → 1.27 | 1.10 → 1.14 | 1.05 → 1.08 |
| Qwen3-1.7B | 1.66 → 1.85 | 1.44 → 1.57 | 1.12 → 1.16 | 1.18 → 1.23 |
| Qwen3-4B | 1.86 → 2.03 | 1.47 → 1.57 | 1.13 → 1.30 | 0.97 → **1.09** |
| Qwen3-8B | 2.50 → 2.69 | 1.56 → 1.64 | 1.20 → **1.51** | 1.04 → **1.25** |

NVFP4 no longer loses to bf16 anywhere (Qwen3-4B at 32k was 0.97). What remains is a step
between 8k and 16k rather than a collapse.

**Gemma gains more than Qwen, and that is structural, not a kernel difference.** Qwen3 is
full-causal at every layer, so attention grows quadratically and is bf16 in both arms; at
Qwen3-1.7B / 32k the attention FLOPs (~4.4 T/layer) exceed the linear ones (~3.15 T), which
caps *any* weight-format speedup near 1.7x however good the GEMMs are. Gemma 3's 5:1 sliding
window keeps attention cheap, so its linear layers stay the dominant term and fixing them
shows up. Two of the Qwen models also never had the cliff at all: Qwen3-0.6B and 1.7B have
`gate_up` weights of 3.5 and 14 MB, which fit in L2, so they get only the fusion.

### Why NVFP4 prefill is 1.5x and not 8x (Gemma3-12B, S=16384)

A per-component breakdown separates the three candidate explanations — Amdahl, slow fp4 kernels,
and an already-good bf16 baseline — by timing every component of a block in both formats
against an empirically measured peak. It reproduces the end-to-end number (predicted 1.54x
vs measured 1.50x), so the decomposition can be trusted:

```
8x    (marketing)      sparsity-inclusive fp4 vs bf16
2.38x (measured peak)  dense fp4 266.4 vs cuBLAS bf16 112.0 TFLOP/s on this box
2.06x (Amdahl)         27% of the fp4 forward is attention + elementwise + act-quant
1.54x (achieved)       projections run at 66% of the measured fp4 peak
```

| component | bf16 ms | fp4 ms | speedup | fp4 TFLOP/s | % of peak |
|---|---|---|---|---|---|
| qkv | 10.29 | 5.26 | 1.96 | 195.9 | 74 |
| o | 5.55 | 2.34 | 2.37 | 220.5 | 83 |
| gate_up | 37.01 | 22.84 | 1.62 | 169.2 | 64 |
| down | 21.36 | 11.36 | 1.88 | 170.1 | 64 |
| geglu(+quant) | 6.62 | 5.59 | 1.18 | — | — |
| act quant x3 | 0.00 | 2.40 | — | — | — |
| attn global | 25.35 | 25.35 | 1.00 | 86.8 | 77 (of bf16 peak) |
| attn sliding | 3.59 | 3.59 | 1.00 | 76.6 | 68 (of bf16 peak) |

Reading it:

* **The 8x premise is the biggest single error.** Dense fp4 through CUTLASS against cuBLAS
  bf16 measures 2.38x on this box. Everything after that is a fraction of 2.38, not of 8.
* **Attention is healthy and is not the bottleneck.** 68–77% of bf16 peak. Because Gemma is
  40 sliding + 8 global layers it is only 7% of a sliding layer (34% of a global one), so
  the model is projection-dominated — which is why the GEMM fixes moved it so much, and why
  Qwen3, full-causal at every layer, gains less.
* **`gate_up` and `down` at 64% are the remaining lever**, and `gate_up` is already chunked.
  It is the largest single cost (22.8 ms of a 53 ms sliding layer); taking it to `o`'s 83%
  would be worth roughly another 1.1x overall.
* **W4A4 pays 2.4 ms/layer of activation quantization** that bf16 never pays — ~4.5% of the
  fp4 layer, and the reason W4A16 wins at decode.

Measuring the peak is itself a trap: the first version used one 8192^3 GEMM, which lands in
the fp4 cliff documented above, reported a peak of 105 TFLOP/s, and had four of the model's
own projections scoring 160–214% of it. It now sweeps 8 shapes and takes the best, so the
failure mode is a component visibly exceeding 100% rather than a plausible wrong answer.

### Offload overhead

With the pipeline working, `SSD ≈ max(drive floor, resident compute)` per block predicts the
measured offload cost to within ±10% everywhere except the 8k crossover, where fetch and
compute are comparable and neither fully hides the other (it under-predicts by ~20% there).
The overhead ratio zero-ssd/resident for NVFP4, all arms measured in one session:

| | 128 | 2048 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| Gemma3-12B | 29.7 | 4.96 | 1.11 | 1.06 | 1.03 |
| Qwen3-8B | 33.8 | 4.92 | 1.11 | 1.01 | 1.00 |
| Qwen3.8-27B | 29.3 | 3.85 | 1.10 | 1.03 | 1.02 |

i.e. **29-34x at 128 tokens and within +-6% beyond 16k**. Below 2k the absolute overhead is
constant per model because it *is* the drive read. Perfect overlap would drive the
long-context overhead to ~0; the residual is the CUDA-graph activation copy (~1%) and --
untested, so a hypothesis -- the H2D leg spending the same LPDDR5X bandwidth the GEMMs need,
which would not apply on a discrete GPU.

Those ratios charge the first block's fetch, because `zero-ssd` does: nothing preceded this
request, so there is no previous forward to have filled the slot. `ssd` takes the other view
-- ring buffers are circular, and under sustained serving the slot is filled while the
PREVIOUS forward computes -- and pre-loads block 0 outside the timed region. The gap between
the two is the cold-start question, not a measurement dispute, and it scales inversely with
model size: at 128 tokens it is ~1% on 12B and ~33% on gemma-3-270m, which is the one read
in a pass with nothing at all to hide behind.

**Activation quantization is nearly half the cost of the `down` projection.** W4A4 pays an
`scaled_fp4_quant` pass per linear that bf16 does not, and its cost tracks the *input* size —
so it is 2-3% on `gate_up` and **45-49% on `down`**, whose input is the big M x intermediate
tensor. `fused_geglu_quant.py` folds it into the GeGLU that produces that tensor (vLLM's own
`silu_and_mul_nvfp4_quant` does the same for Qwen3's SiLU, so only Gemma's GeGLU-tanh needed
a kernel). Quoting a W4A4 speedup from the GEMM alone overstates it by roughly 2x on that
projection.

**A kernel speedup is not a model speedup — quote both.** That fusion is **1.79x** on the
MLP epilogue and **1.03–1.21x (median ~1.09x)** on NVFP4 resident prefill end to end, because
the epilogue is one slice of a block that also does attention and three other projections.
On the SSD arm it is ~1.00 while drive-bound and 1.04–1.14x at 16k–32k, which is the correct
shape: a faster MLP cannot help a forward that is waiting on the disk.

Two further traps worth naming, since the first one produced a *passing* test for an
obviously broken kernel:

* **Do not verify a W4A4 path on small activations.** The first wiring test scaled every
  parameter by 0.05, which put activations far below the static `ACT_AMAX=10`; every block
  scale underflowed e4m3 to zero, the MLP quantized to nothing, and the test compared zero
  against zero — reporting "bitwise identical" for a kernel computing SiLU instead of GeLU.
  `test_fused_geglu.py` now refuses to run if the MLP branch output is ~zero.
* **A test that cannot fail is worse than no test.** `python test_fused_geglu.py --mutate`
  breaks the kernel five ways and asserts the checks notice. Four are caught; the fifth
  (flipping the tie rule at 0.75) is tolerated *by design*, since it only changes which of
  two equidistant fp4 codes wins.

**`--modes ram` is not an offload on this box.** Host and device memory on GB10 are the
same physical LPDDR5X over NVLink-C2C, so streaming from pinned host RAM is a memcpy inside
one pool — it measures the interconnect, not an offload. It is kept because it is the
ceiling SSD streaming would approach with an infinitely fast drive, and because on a
**discrete GPU (a 5090) host RAM really is across PCIe**, which makes that mode the
realistic one there. It is no longer measured by default; `--modes ssd` is the real offload
here.

**The 55 GiB/s pinned H2D that `ram` mode reaches is real but architectural.** Checked for
caching directly: 4 GiB of never-reused pinned buffers gives 54.4 GiB/s against 55.1 for one
buffer copied eight times. It is that fast because no bus is involved — an "H2D copy" is a
memcpy inside one pool (~110 GB/s of traffic against the ~238 GB/s this box sustains; a D2D
copy of the same 4 GiB runs at 101.6 GiB/s).

**Pinned memory is load-bearing, not a micro-optimization.** Pageable host memory cannot be
DMA'd asynchronously; the driver stages it through a bounce buffer, which makes the copy
effectively synchronous and silently destroys the overlap. No error, just worse numbers.

**Three hazards guard the pipeline, and one of them is easy to miss.** `h2d_done` (the host
buffer is free to overwrite), `ready` (the weights have landed) and `compute_done` (the GPU
slot is no longer being read). The last was missing at first: without it the H2D for block
i+2 lands on top of block i's weights while block i is still computing. It never fires at
short sequences — a 68 ms read dwarfs a 0.2 ms compute — so a correctness check at seq=128
passes happily. Verify at 16k or 32k, where compute is ~260 ms per block against that same
68 ms read.

**Residency:** in `ssd` mode all blocks live on the drive and at most **2** are on the GPU at
any instant, plus 2 pinned host staging buffers. That 2-block budget is the constraint being
simulated. In `ram` mode the whole model sits in pinned host RAM instead.

**`zero-ssd` is the honest version of `ssd`, and it is the default.** `ssd` grants the
prefill two streaming slots as if they were free memory, and preloads block 0 outside the
timer. Neither is free on a box that is already full: the slots have to come from somewhere,
and the first block's read is part of the request's latency. So `zero-ssd` (protocol
`odp_carveout_cold_v1`) carves the slots out of decode-only weights, times block 0 inside
the clock, and restores the carve-out from the drive before generation may start, overlapped
with the last layer's compute. It therefore moves P + C bytes per request against `ssd`'s P,
and it is what the figures report. Both modes stay in the CSV; do not relabel `ssd` rows as
ODP.

**The page cache must be dropped every pass**, and `SSDOffloadRunner.pre_rep` does it with
`posix_fadvise(DONTNEED)`. This box has 128 GB of RAM, so after one pass the whole model is
cached and every later read is served from memory at ~20x the speed — measuring memcpy, not
the drive. It is the easiest way to get a spectacular and completely fake result.

**`offload_forward.py` runs without a KV cache** so that all blocks are structurally
identical and one `torch.compile` covers them all — a cache would make `layer_idx` differ,
which dynamo guards on, costing one compilation per layer. Compare within this harness, not
against a whole-model baseline compiled with `reduce-overhead` and using a
static cache.

**`dynamic=False` means one compile per sequence length**, which is intended, and is why
`recompile_limit` is raised to 64: dynamo's default of 8 fails a 9-length sweep on the last
point and takes the whole run with it.

**The NVFP4 activation scale is STATIC and is not calibrated.** vLLM's w4a4 scheme reads a
per-layer `input_global_scale` from the checkpoint and does no runtime reduction; this
benchmark does the same thing with one fixed `ACT_AMAX = 10.0` rather than rebuilding
calibration. Every kernel, every byte of traffic and every launch is what a calibrated model
would do, so the latency is representative — the accuracy is not, and no accuracy claim
should be read off it.

An earlier version reduced over the activation per call to get the amax. That cost more than
the fp4 GEMM saved at long context and was the reason NVFP4 lost to bf16 on 4B at 32k.
Removing it was worth **2.4x** at 0.6B/128 resident (6.49 → 2.68 ms) and closed the
regression to parity. Interestingly the accuracy barely moved (rel 0.1391 static vs 0.1393
matched): NVFP4 carries per-16 fp8 block scales, so the global scale only has to keep those
inside e4m3's range, and being 2x off hardly matters.

**NVFP4 is w4a4**, so its ~14% relative error against bf16 is expected and mostly comes from
quantizing the *activations*. These are latency measurements; nothing here says the accuracy
is acceptable.
