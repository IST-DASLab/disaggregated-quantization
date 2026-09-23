# Disaggregated Quantization: Specializing LLM Prefill and Decode

[![arXiv: 2609.26333](https://img.shields.io/badge/arXiv-2609.26333-b31b1b.svg)](https://arxiv.org/abs/2609.26333)
[![Hugging Face: Qwen3.8 prefillers](https://img.shields.io/badge/Hugging_Face-Released_prefillers-FFD21E.svg)](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller)

[Training](qad/training/) · [Evaluation](evals/) · [Plots](notebooks/plots.ipynb)

Prefill and decode reward different quantization choices: hardware-native low-precision
arithmetic accelerates prompt processing, while compact weight encodings reduce memory
traffic during generation. **Disaggregated quantization (DQ)** specializes computation
formats, weights and storage placement to each phase, training the pathways toward a
common response objective.

## Improve an existing quantized decoder

A released weight-only checkpoint need not be retrained to benefit. We train an NVFP4
**prefiller** around its frozen decoder, retaining the existing decode weights
and kernels. **Offloaded disaggregated prefill (ODP)** streams the additional prefill
checkpoint from SSD without increasing device weight residency.

Download the [released Qwen3.8-27B NVFP4 prefillers on Hugging Face](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller).

[![Qwen3.8-27B accuracy versus device weight size on MMLU-Pro and MMMU-Pro, alongside llama.cpp time to first token.](docs/figures/pareto_gsq_rco_both.png)](notebooks/figures/pareto_gsq_rco_both.pdf)

On Qwen3.8-27B with released Unsloth GGUF decoders:

- **IQ1_S accuracy more than doubles:** 29.04% → 61.54% on MMLU-Pro and
  24.39% → 59.65% on MMMU-Pro, with no change to the decoder.
- **Text-only training transfers to visual reasoning.** The strongest gains occur at
  low bitwidths; some higher-precision formats lose a little MMMU-Pro accuracy.
- **ODP reduces time to first token from 12.27 s to 6.90 s at 8K context** in our
  custom llama.cpp integration. Against native llama.cpp inference with Unsloth IQ1_S,
  speedups range from 1.38× to 1.78× over the measured 4K–32K contexts. Shorter prompts
  remain slower because of SSD loading.

Accuracy uses disaggregated vLLM serving with GGUF weights dequantized to BF16;
the size axis reports encoded GGUF backbone weights, not evaluation-backend allocations.
TTFT is measured separately in llama.cpp on DGX Spark. See the
[exact accuracies](notebooks/tables/qadd27b_accuracy.tex) and
[TTFT measurements](notebooks/data/qwen3_8_27b_llama_cpp_ttft.csv).

## How it works

| Scheme | What changes | Benefit and cost |
|---|---|---|
| **Format disaggregation** | Shared weights; native NVFP4 prefill and weight-only decode | Removes decode activation quantization and, for LUT formats, weight re-quantization, without extra weight storage. |
| **Full disaggregation** | Separate compute-native prefill and compact decode weights | Improves low-bit accuracy beyond weight-only inference while enabling fast prefill; stores an additional checkpoint. |
| **Offloaded disaggregated prefill** | Streams prefill weights through reusable block buffers | Retains full-disaggregation accuracy without additional device weight residency; pays SSD storage and loading latency. |

**Quantization-aware distillation with disaggregation (QADD)** uses the SFT token mask
both to select the phase-specific linear pathway and to identify response targets.
Distillation gradients reach prefill through the representations consumed by decode,
training both pathways in one forward-backward pass. The same construction supports
shared weights, separate weights, or a frozen external decoder.

[![QADD training schematic: the SFT mask routes prompt and response positions through different quantized linear pathways.](docs/figures/qadd_training.png)](notebooks/figures/qadd_training.pdf)

### Decode-heavy and prefill-heavy workloads

The core experiments cover Qwen 3 (0.6B, 1.7B, 4B, 8B) and Gemma 3
(270M, 1B, 4B, 12B), with NVFP4, LUT3 and LUT2 formats.
Decode-heavy evaluation uses GSM8K, MATH-500 and MMLU-Pro; prefill-heavy evaluation
uses RULER's 13 tasks at 4K, 8K, 16K and 32K context lengths.

[![Family-mean accuracy for uniform, format-disaggregated and fully-disaggregated quantization, on decode-heavy and prefill-heavy workloads.](docs/figures/bars_disag_both.png)](notebooks/figures/bars_disag_both.pdf)

Format disaggregation primarily improves decode-heavy accuracy. Full disaggregation
improves low-bit accuracy on both workload types, with larger gains on prefill-heavy
tasks. At 2-bit decode, it exceeds trained weight-only baselines by **7.1 and 4.5 points**
on decode-heavy tasks and **12.6 and 8.9 points** on prefill-heavy tasks, for Qwen 3
and Gemma 3 respectively. At 3-bit decode, it matches or exceeds weight-only accuracy
while enabling native NVFP4 prefill.

Family means give equal weight to model sizes and exclude Gemma-270M in both workloads.
Scores average five late QADD checkpoints; error bars describe checkpoint variation,
not independent training seeds. The [full cost/accuracy table](notebooks/tables/minimal_comparison.tex)
also reports decode speedups and device weight allocations.

As a separate large-scale validation, shared-weight format disaggregation without
retraining improves point estimates in 11 of 13 model–benchmark comparisons on models
up to 2.8T parameters, with six significant gains and no significant degradations
under per-comparison paired tests. See the [PTQ results](notebooks/tables/bigmodels.tex).

### Serving the extra prefill checkpoint locally

During prefill, ODP borrows buffer space from temporarily unused decode weights,
streams prefill blocks from SSD, and restores the decode carve-out before generation.
Decode weights are resident during generation; prefill weights need not be.

[![ODP timeline: a cold first-block load, overlapping subsequent loads and computation, and restoration of the decode carve-out.](docs/figures/odp_timeline.png)](notebooks/figures/odp_timeline.pdf)

The schematic illustrates the core Qwen 3 pipeline, scaled to aggregate measurements;
it is not a per-block profiler trace. At context lengths above 16K, ODP adds under 5%
to resident NVFP4 prefill latency on Qwen 3 and under 8% on Gemma 3 in the custom
transformer-stack benchmark. These are separate measurements from the 27B llama.cpp
TTFT results above. ODP avoids additional **weight** memory, not KV-cache or other
runtime allocations, and still stores the extra checkpoint on SSD.

## Repository guide

| Path | Contents |
|---|---|
| [latex/main.tex](latex/main.tex) | Manuscript, protocols, limitations and figure captions |
| [qad/training/](qad/training/), [qad/quantizers/](qad/quantizers/), [qad/export/](qad/export/) | QADD, phase-specific formats and checkpoint export |
| [qad/serving/](qad/serving/), [qad/eval/](qad/eval/) | Disaggregated vLLM serving and core benchmark drivers |
| [evals/](evals/) | Large-model PTQ and frozen-decoder evaluations, per-item scores and comparison drivers |
| [qad/kernels/prefill/](qad/kernels/prefill/) | NVFP4 prefill and SSD-offloading benchmarks |
| [qad/kernels/lloyd43/](qad/kernels/lloyd43/) | LUT2/LUT3 CUDA and Triton kernels, vLLM integration and decode measurements |
| [notebooks/](notebooks/) | Plotting notebook, schematics, table generators and recorded llama.cpp measurements |

## Reproducing the experiments

The launch scripts target a SLURM/container environment, not a standalone laptop.
Before submitting jobs, adapt the account, partition, container and filesystem paths
in the scripts to your cluster. The current training launcher uses four GPUs per node
and defaults to two nodes. See the
[entrypoint guide](qad/bin/README.md); the scripts are authoritative for resource defaults.

### Training and core evaluation

Core QADD runs use 100M Tülu 3 tokens, 2048-token sequences, global batch 64,
and a constant learning rate of 3e-6 after 100 warmup steps.
The Qwen3.8-27B prefillers instead use text-only reasoning traces, an 8192-token limit,
global batch 32 and the final checkpoint at step 980; see the manuscript for the full setup.

```bash
cd qad

# Train shared-weight format-disaggregated NVFP4.
MODEL=Qwen/Qwen3-4B ./bin/run_qad.sh --quantizer nvfp4pdshared

# Inspect the core reasoning-evaluation sweep before submitting it.
./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-4B --dry-run

# Evaluate an exported checkpoint on the paper's RULER context lengths.
./bin/run_eval_ruler.sh --model Qwen/Qwen3-4B --quantizer nvfp4pdshared \
    --run-name qad3x-Qwen-Qwen3-4B --iter 2250 \
    --seqlens 4096,8192,16384,32768
```

Qwen 3 reasoning evaluations use both thinking modes; Gemma 3 has no mode switch.
Core results live in [qad/results/disagg/](qad/results/disagg/) and
[qad/results/ruler/](qad/results/ruler/). Larger-model evaluations use
[evals/bin/run_eval.sh](evals/bin/run_eval.sh), with results and score exports under
[evals/results/](evals/results/) and [evals/scores/](evals/scores/).

<details>
<summary>Paper formats and quantizer identifiers</summary>

| Format or scheme | `--quantizer` | Prefill / decode computation |
|---|---|---|
| Uniform NVFP4 | `nvfp4` | NVFP4 / NVFP4 |
| Weight-only NVFP4 | `nvfp4a16` | NVFP4A16 / NVFP4A16 |
| Format-disaggregated NVFP4 | `nvfp4pdshared` | NVFP4 / NVFP4A16, shared weights |
| Fully-disaggregated NVFP4 | `nvfp4pdsplit` | NVFP4 / NVFP4A16, separate weights |
| Uniform LUT3 autocast | `nvfp4lloyd43upcastboth` | LUT3 → NVFP4 in both phases |
| Format-disaggregated LUT3 | `nvfp4lloyd43upcast` | LUT3 → NVFP4 / weight-only LUT3 |
| Fully-disaggregated LUT3 | `nvfp4lloyd43split` | Separate NVFP4 / weight-only LUT3 |
| Uniform LUT2 autocast | `nvfp4lloyd21upcastboth` | LUT2 → NVFP4 in both phases |
| Format-disaggregated LUT2 | `nvfp4lloyd21upcast` | LUT2 → NVFP4 / weight-only LUT2 |
| Fully-disaggregated LUT2 | `nvfp4lloyd21split` | Separate NVFP4 / weight-only LUT2 |
| Weight-only LUT3 / LUT2 | `lloyd43` / `lloyd21` | Weight-only in both phases |
| Frozen-decoder prefiller | `nvfp4frozendec` | Trainable NVFP4 / fixed external decode weights |

`nvfp4prefill` and `nvfp4decode` are phase-isolation controls: one phase uses NVFP4
and the other BF16. `--full-disag` is a separate ablation that also duplicates normally
shared unquantized parameters; it is not required for the fully-disaggregated formats.

For frozen-decoder training, supply `decode_model` through `--quantizer-params`,
pointing to the dequantized external checkpoint. See
[frozen_decode.py](qad/quantizers/frozen_decode.py) and
[reasoning.py](qad/training/reasoning.py).

</details>

### Figures, tables and checks

[notebooks/plots.ipynb](notebooks/plots.ipynb) reads the recorded core and large-model
results. Schematics live in [notebooks/schematics/](notebooks/schematics/);
prefill and decode benchmark figures have their own plotting scripts under `qad/kernels/`.
Rendering plots from recorded results does not rerun the GPU experiments.

From the repository root, with the analysis dependencies installed:

```bash
python notebooks/table_generators/generate.py all --check
python notebooks/table_generators/generate.py all --write
python -m unittest discover -s notebooks/table_generators
```

Tables are generated into [notebooks/tables/](notebooks/tables/). For a manuscript build,
copy that directory and `notebooks/figures/` beside the manuscript as `tables/` and
`figures/`. See the [table-generator guide](notebooks/table_generators/README.md).
The README's four PNGs are rasterizations of the paper PDFs, not independently plotted
results. On macOS, refresh them with `swift notebooks/export_readme_figures.swift`.

GPU and distributed tests use the configured cluster environment:

```bash
cd qad
./bin/run_tests.sh
./bin/run_tests.sh nvfp4 dual pipeline
```

The manuscript distinguishes measured latency from proxy timings, and documents the
accuracy protocols and their limitations. Highly batched serving, multi-turn cache
rebuilds and agentic behavior are not evaluated.

## Cite this work

```bibtex
@misc{panferov2026disaggregatedquantizationspecializingllm,
      title={Disaggregated Quantization: Specializing LLM Prefill and Decode},
      author={Andrei Panferov and Maximilian Kleinegger and Sweta Priyadarshi and Tijmen Blankevoort and Dan Alistarh},
      year={2026},
      eprint={2609.26333},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.26333},
}
```
