# Disaggregated Quantization (DQ)

Code for the paper *Disaggregated Quantization of Large Language Models* (`manuscript.tex`).

LLM inference has two regimes: compute-bound **prefill** and memory-bound **decode**.
This repo trains and evaluates quantization formats that treat them as separate targets
rather than one, via **disaggregation-aware distillation (DQAD)** — a single
forward/backward in which a per-position mask, taken from the SFT label mask, selects
which pseudo-quantization scheme applies to each token.

- `qad/` — training, quantizers, export, evaluation harness (the paper's experiments)
- `qad/kernels/` — NVFP4 / LUT3 kernels, offloaded-prefill benchmarks, and their figures
- `notebooks/` — plotting (`plots.ipynb`), schematics, grid derivation (`grids.ipynb`)

---

## The formats

The paper's names map onto `--quantizer` values as follows. "Prefill" and "decode" refer
to the phase each representation serves.

| Paper name | `--quantizer` | Prefill | Decode | Masters | Disaggregation |
|---|---|---|---|---|---|
| BF16 | *(baseline, no training)* | BF16 | BF16 | – | – |
| NVFP4 | `nvfp4` | W4A4 | W4A4 | 1 | none |
| NVFP4A16 | `nvfp4a16` | W4A16 | W4A16 | 1 | none |
| NVFP4PD | `nvfp4pdshared` | W4A4 | W4A16 | 1 | format |
| NVFP4PD (full) | `nvfp4pdsplit` | W4A4 | W4A16 | 2 | full |
| LUT3-NVFP4 | `nvfp4lloyd43upcastboth` | NVFP4 | NVFP4 | 1 | none |
| LUT3-NVFP4 (format) | `nvfp4lloyd43upcast` | NVFP4 | LUT3 | 1 | format |
| LUT3-NVFP4 (full) | `nvfp4lloyd43split` | NVFP4 | LUT3 | 2 | full |
| LUT2-NVFP4 | `nvfp4lloyd21upcastboth` | NVFP4 | NVFP4 | 1 | none |
| LUT2-NVFP4 (format) | `nvfp4lloyd21upcast` | NVFP4 | LUT2 | 1 | format |
| LUT2-NVFP4 (full) | `nvfp4lloyd21split` | NVFP4 | LUT2 | 2 | full |
| LUT3-WO / LUT2-WO | `lloyd43` / `lloyd21` | LUT | LUT | 1 | none |

**Phase-isolation ablation** (Sec. 2.2 of the paper): `nvfp4prefill` quantizes *only*
prefill (BF16 decode) and `nvfp4decode` quantizes *only* decode (BF16 prefill). Neither
is deployable — a BF16 phase needs BF16 weights resident — they exist to separate the
two phases' contributions to the total quantization error.

"upcast" means the prefill weight is *derived* from the decode weight
(`W -> LUT -> NVFP4`); "upcastboth" is the non-disaggregated control where that
NVFP4 weight serves both phases; "split" gives each phase its own master weight.

---

## Reproducing the experiments

### 1. Training (DQAD)

All runs use the same recipe: KL against the frozen BF16 teacher on 100M tokens of
Tülu 3, 2048-token sequences, global batch 64, constant LR 3e-6 after 100 warmup steps.
Full hyperparameters are in Appendix B of the manuscript.

```bash
cd qad

# one format, one model (8 GPUs, 1 node)
MODEL=Qwen/Qwen3-4B ./bin/run_qad.sh --quantizer nvfp4pdshared

# the paper's Qwen3 sweep
for q in nvfp4 nvfp4a16 nvfp4pdshared nvfp4pdsplit \
         nvfp4lloyd43upcastboth nvfp4lloyd43upcast nvfp4lloyd43split \
         nvfp4lloyd21upcastboth nvfp4lloyd21upcast nvfp4lloyd21split \
         nvfp4prefill nvfp4decode; do
  MODEL=Qwen/Qwen3-4B ./bin/run_qad.sh --quantizer $q
done

# Gemma-3 uses RUN_PREFIX=gemma3 to namespace the checkpoints
MODEL=google/gemma-3-4b-it RUN_PREFIX=gemma3 ./bin/run_qad.sh --quantizer nvfp4
```

`RUN_PREFIX` (default `qad3x`) namespaces checkpoints as
`<prefix>-<model>-<quantizer>-<hash>`; the hash covers the quantizer's parameters, so
changing them never overwrites an existing run.

**Scaling knobs.** Larger models need more than one node, and the split formats hold two
master weights, which roughly doubles parameter memory:

| Config | Flags |
|---|---|
| ≤4B, any format | *(defaults: 1 node, mbs 8)* → world 8, DP 8 |
| 8B, single-master | *(defaults)* → world 8, DP 8 |
| 8B, split | `--pp 2 --nodes 2 --micro-batch-size 8` → world 16, PP 2, DP 8 |
| 12B, single-master | `--nodes 2 --micro-batch-size 4` → world 16, DP 16 |
| 12B, split | `--pp 2 --nodes 4 --micro-batch-size 4 --recompute-wq` → world 32, PP 2, DP 16 |

Every configuration keeps `grad_accum = 1` (`gbs = mbs × accum × DP`). Gradient clipping
uses the RMS of per-rank norms, so its effective strength depends on DP width — logged
`grad_norm` is comparable between runs at the same DP, not across DP 8 and DP 16.

`--pp 2` is pipeline parallelism: the decoder layers of student *and* teacher are split
in half across adjacent (same-node) rank pairs, stage 0 holding the embeddings and
stage 1 the loss. `--recompute-wq` drops the persistent quantized-weight caches and
recomputes them each forward — exact, but roughly doubles the quantize work.

The 4h wall-clock cap needs `--chain N` for anything past ~4B: N jobs share one job name
under `--dependency=singleton` and resume from `state/` (written every 100 steps).
A chained job whose flags differ from the original will refuse to resume rather than
silently reinterpret the state.

**Fully-disaggregated serving ablation** (Sec. 2.4): add `--full-disag`, which also gives
the run its own checkpoint hash so it cannot collide with the plain run.

### 2. Evaluation

Evaluations run under **real disaggregated serving** — a prefill engine and a decode
engine on separate GPUs exchanging a KV cache through vLLM — because a single-engine
eval cannot show a format that behaves differently per phase.

```bash
cd qad

# every format x both thinking modes x all exported steps, for one model
./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-4B
./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-4B --dry-run   # inspect first

# a single checkpoint
./bin/run_eval_disagg.sh --model Qwen/Qwen3-4B --quantizer nvfp4pdshared \
    --run-name qad3x-Qwen-Qwen3-4B --iter 2250
```

Benchmarks are GSM8K (5-shot), MATH-500 (4-shot) and MMLU-Pro (5-shot), each run with
reasoning **enabled and disabled** — the two modes are separate benchmarks, not variants,
since decode length differs sharply between them. Results land in
`qad/results/disagg/{think,nothink}/<tag>/step_*.json`.

The paper reports accuracy averaged over the three benchmarks, both modes, and the last
five checkpoints of each run.

### 3. Figures

The paper's figures come from three places, not one:

| Figures | Source |
|---|---|
| `bars_*`, `lines_*`, `pareto` | `notebooks/plots.ipynb` (reads `qad/results/` directly) |
| `quantized_linear`, `dq_linear`, `full_disag_linear` | `notebooks/schematics/fig_dq_linear.py` |
| `odp_timeline` | `notebooks/schematics/fig_odp_timeline.py` |
| `qwen3_prefill_offload_paper` | `qad/kernels/prefill/plot_for_paper.py` |
| `offload_cost` | `qad/kernels/prefill/plot_offload_cost.py` |
| decode-speed comparison | `qad/kernels/lloyd43/plot_decode.py` |

The accuracy figures are regenerated from the results tree; the kernel and offloading
figures depend on latency measurements taken on the target hardware (DGX Spark for the
ODP numbers) and are not reproducible from this repo alone.

---

## Kernels

The accuracy results above are measured under real disaggregated serving, but the *cost*
claims in the paper come from two separate kernel efforts under `qad/kernels/`. Both hold
real measurements, not projections, and both carry caveats worth reading before quoting
them.

### `kernels/prefill` — offloaded disaggregated prefill (ODP)

Real latency measurements for streaming the prefill model off SSD block by block while
computing, which is what makes fully-disaggregated quantization viable on a single
device (Sec. 2.5). Measured on GB10 / DGX Spark, batch 1, four Qwen3 and four Gemma-3
sizes, 128 → 32k tokens, with the page cache dropped before every forward so nothing is
served from RAM.

The headline is asymptotic: loading cost is constant, prefill compute scales with context,
so offloaded prefill is flat while the drive dominates and then converges onto the
resident curve. NVFP4 reaches within 10% of resident at **seq ≥ 8k** for 1.7B and 4B;
in BF16 the same crossover is at 16k. Raw numbers are in `prefill/README.md` and the
CSVs beside it (`qwen3_prefill_latency_new.csv`, `offload_prefill.csv`, `load_floor.csv`).

```bash
python qad/kernels/prefill/offload_forward.py     # the interleaved load/compute forward
python qad/kernels/prefill/plot_for_paper.py      # -> qwen3_prefill_offload_paper.pdf
```

### `kernels/lloyd43` — LUT3 decode kernels

CUDA and Triton kernels for the 3-bit LUT format, plus the vLLM integration used for the
decode-speed claim (Sec. 3.1). Single-user decode on GB10 is **2.1x (Qwen3-0.6B) to 3.4x
(Qwen3-8B)** faster than dense BF16, measured end to end in vLLM rather than in a
microbenchmark; standalone the CUDA kernel is 3.13x over BF16 on the 8B projections.

Two caveats the kernel's own handoff records: **prefill is a large regression** (TTFT 6–20x
worse, since prefill falls back to dequantize-then-cuBLAS and has no real M>1 kernel) —
which is precisely why the paper pairs LUT decode with an NVFP4 prefill path rather than
serving LUT for both phases. And the B200 numbers are stale; only GB10 has been
re-measured. See `lloyd43/HANDOFF.md`.

```bash
python qad/kernels/lloyd43/vllm_serve.py    # end-to-end vLLM decode, bf16 vs lloyd43
python qad/kernels/lloyd43/plot_decode.py   # -> the decode-speed figure
```

---

## Coverage

What has actually been trained and evaluated:

| Model | Formats |
|---|---|
| Qwen3 0.6B / 1.7B / 4B | 19 |
| Qwen3 8B | 16 |
| Gemma-3 270m / 1b / 4b / 12b | 12 |

The Qwen3 ≤4B sizes additionally carry the STE/QuEST/nvr2bit formats and the
`--full-disag` ablation (0.6B only); the Gemma-3 sizes carry the 12 formats in the table
above. Coverage per model is worth checking before quoting a cross-model trend.

---

## Testing

```bash
cd qad && ./bin/run_tests.sh              # everything
./bin/run_tests.sh nvfp4 dual pipeline    # selected files
```

`test_pipeline` and `test_checkpoint` are distributed and run under `torchrun` on 8 GPUs.
Note that pipeline bugs are frequently invisible at world=2 — a single stage pair cannot
race, `dp_size` is 1, and the intra-node layout assertion is vacuous — so the pipeline
tests always request the full node.

---
