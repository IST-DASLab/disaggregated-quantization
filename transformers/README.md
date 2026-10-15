# Qwen3.8 prefiller training with Transformers

This is a Hugging Face `Trainer` port of `qad/training/qad.py`'s
`nvfp4frozendec` path. It uses the existing root `.venv` through UV, and runs
directly on the local NVIDIA GPUs without Slurm.

See [HANDOFF.txt](HANDOFF.txt) for validation evidence, local checkpoint locations,
known limitations, and the starting plan for a B200 x 8 machine.

The defaults are `Qwen/Qwen3.8-27B-FP8` as the **real FP8 teacher**, FP32 student
master weights initialized by HF's `FineGrainedFP8Config(dequantize=True)`, and
`ISTA-DASLab/GSQ-RCO-small/Qwen3.8-27B-GSQ-RCO-IQ1_S.gguf` as the frozen decoder.
The decoder is dequantized to BF16 one tensor at a time directly into the model.
There is no intermediate dense checkpoint on disk and no vLLM dependency.

## Sharing and gradients

| Component | Prefill | Decode | Initialization |
| --- | --- | --- | --- |
| Text linear projections except `in_proj_a`, `in_proj_b` | Trainable FP32 master, NVFP4 weight/activation STE | Separate frozen BF16 weight | Teacher for prefill, IQ1_S for decode |
| Text RMS norms, including gated and Q/K norms | Trainable specialized FP32 scale | Separate frozen scale | Teacher for prefill, IQ1_S for decode |
| Token embedding and LM head | Same frozen parameter used by both phases | Same parameter | IQ1_S |
| `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, convolution and remaining text parameters | Shared, frozen | Shared, frozen | IQ1_S |

The trainable/shared partition reuses the reference quantizer unchanged.
Shared frozen values come from the decoder, as required for inference reuse;
the original script instead retains their base-model initialization.
Norms are specialized in the reference; they are **not** shared trainable weights.
The frozen teacher is independent of the student's frozen decode pathway.
Vision and MTP modules are unused for text training and omitted.

KL is `KL(teacher || student)` at shifted assistant target positions, ignoring
padding. `--include-prefill-loss` additionally scores prompt positions without
changing phase routing. Logit projections are checkpointed in chunks (default 1024 scored tokens),
avoiding a sequence-by-vocabulary allocation. The phase mask remains active during backward.

## Setup and run

From the repository root:

```bash
UV_CACHE_DIR="$PWD/transformers/.cache/uv" uv pip install \
  --python .venv/bin/python -r transformers/requirements.txt

# Build optimized causal convolution against the installed Torch/CUDA stack.
UV_CACHE_DIR="$PWD/transformers/.cache/uv" uv pip install \
  --python .venv/bin/python --no-build-isolation -r transformers/requirements-kernels.txt

# Tiny hybrid architecture with real HF FP8 linear kernels; saves/resumes via FSDP.
NPROC_PER_NODE=2 bash transformers/run.sh --smoke-test \
  --max-steps 2 --warmup-steps 0 --save-steps 1 --eval-steps 1 \
  --gradient-accumulation-steps 1 --output-dir transformers/outputs/smoke

# Three-step actual-model validation on all eight B200 GPUs, FP32 masters on GPU.
bash transformers/run.sh --max-steps 3 --warmup-steps 0 \
  --gradient-accumulation-steps 1 --train-tokens 100000 --val-tokens 10000 \
  --max-seq-len 8192 --logit-chunk-size 1024 --no-save-checkpoints \
  --output-dir transformers/outputs/b200-validation

# Eight GPUs give the paper's global batch 32 (1 example x 4 accumulation x 8).
NPROC_PER_NODE=8 bash transformers/run.sh --logit-chunk-size 1024

# Continue the same configuration from a saved checkpoint.
bash transformers/run.sh --logit-chunk-size 1024 \
  --resume-from-checkpoint transformers/outputs/qwen38-iq1s/checkpoint-50
```

The default eight-GPU global batch is 32. Sequence limit is 8192; the reasoning
corpus is the reference's `faunix/Qwen3.8-27B-Distillation-40K`, with tool-use rows
filtered, reasoning split into the template's reasoning field, and overlong
examples dropped. Trainer shards the dataloader; chunks are not pre-sharded.
Use `--decode-model /path/to/file.gguf` or a dense safetensors directory to change
the decoder. Dense wrapper checkpoints use `--decode-prefix model.language_model.`;
text-only checkpoints use `--decode-prefix model.`. Model and dataset downloads
use the normal Hugging Face cache and login, honoring
`HF_HOME`, `HF_HUB_CACHE` and `HF_DATASETS_CACHE`. Prepared token chunks and the UV
cache live under `transformers/.cache/`; checkpoints live under `transformers/outputs/`.

## FSDP and memory

FSDP2 shards both text stacks and the student's decoder parameters. Frozen decode
weights are Parameters with `requires_grad=False` so they can be sharded;
quantized prefill weights are cached as lazy BF16 buffers. Each cached layer quantizes
once per optimizer update and reuses that buffer across accumulation microbatches
and backward checkpoint recomputation. The Trainer invalidates caches after
optimizer.step, before evaluation or saving. Caches are nonpersistent and rebuild
after resume; FP32 masters and the checkpoint tensor schema are unchanged.
FSDP replicates these buffers, adding about 45 GiB per GPU for the 27B model.
Use `--no-cache-quantized-weights` to recompute instead. Activation quantization
remains input-dependent and runs for each microbatch.
Use `--weight-cache-max-gib 4` to cap replicated caches at 4 GiB per rank,
prioritizing the largest matrices; zero (the default) allows the full cache.
The forward explicitly enables BF16 autocast for compute without
casting stored FP8 teacher weights or FP32 student masters; Trainer
`bf16=True` alone did not provide compute autocast in this FSDP2 preparation path. All parameter shards,
gradients, optimizer state and saved activations stay on GPU; CPU and selective
offload options have been removed. Activation checkpointing remains enabled.
TP/CP are not implemented: the custom quantizer and hybrid DeltaNet recurrence
need additional work for those modes.

FSDP tuning controls keep model storage and quantizer arithmetic unchanged:

| Argument | Effect |
| --- | --- |
| `--batch-size 2 --gradient-accumulation-steps 2` | Larger microbatches with the same global batch 32 on eight ranks |
| `--sync-each-batch` (default) | Reduce-scatter gradients every microbatch so accumulation retains sharded gradients; `--no-sync-each-batch` trades more memory for fewer synchronizations |
| `--optimizer adamw_torch_fused` | Fused AdamW with FP32 masters and optimizer state |
| `--weight-cache-max-gib N` | Limit replicated BF16 weight buffers per GPU; 0 allows all weights |
| `--activation-checkpoint-every N` | Checkpoint every Nth student decoder block; 1 checkpoints all, 2 checkpoints half, 0 disables block checkpointing |
| `--fsdp-prefetch-depth N` | Explicit FSDP2 block prefetch; 0 uses implicit scheduling, 1/2 issue future gathers earlier |
| `--fsdp-retain-student-layers N` | Keep the first N student blocks gathered between forward and backward to avoid their second gather |
| `--compile-blocks` | Compile repeated student/teacher blocks in place with static shapes, preserving FSDP hooks |
| `--compile-mode max-autotune-no-cudagraphs` | Optional additional Inductor GEMM tuning; default is `default` |
| `--profile-step N` | Record rank-zero CPU/CUDA activity for an absolute optimizer step; writes `profile/` traces and operator summaries |

FSDP2's explicit prefetch APIs are used directly: Accelerate 1.15 rejects its
FSDP1 `forward_prefetch` configuration for FSDP2. Checkpointing applies only to
student decoder blocks; the frozen teacher runs under no_grad. More prefetching,
retained blocks, fewer checkpoints, larger microbatches and BF16 caches all consume
GPU headroom. Tune one variable at a time and compare padded and real throughput
at the same global batch. Steps marked `profiled_step=true` include profiler
overhead and should be excluded from throughput estimates.

Compiled blocks call an opaque NVFP4 quantization operation that retains eager
division/scaling and the existing compiled E2M1 rounder. Compiling division and
rounding together previously changed quantization codes near thresholds. LM-head
and KL loops remain eager because the number of scored positions varies; block
compilation targets the repeated fixed-shape computation.
Quantization parity tests cover the opaque operation. Entire compiled blocks can
still differ numerically through other floating-point operations; compilation is
optional and has not been validated for convergence.

Practical references: [PyTorch FSDP2 prefetching](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html),
[Trainer FSDP2 configuration](https://huggingface.co/docs/transformers/v5.19.0/fsdp),
[selective block checkpointing](https://pytorch.org/blog/maximizing-training/),
and [gradient accumulation memory](https://huggingface.co/docs/accelerate/main/concept_guides/gradient_synchronization).

Only rank zero reads model weights. Other ranks construct metadata and uninitialized
storage, then receive FSDP shards. The launcher monitors job RSS (default limit
768 GiB, configurable through `HOST_MEMORY_LIMIT_GIB`) and reserves host RAM.
`fsdp_compat.py` works around Accelerate 1.15's persistent-buffer initialization bug
locally; installed packages are not patched. The single-host launcher uses loopback
for rendezvous and prioritizes CUDA libraries from the active `.venv`.

Checkpoints are sharded, resumable Trainer states with model, optimizer, scheduler
and RNG state. They include frozen weights, so expect large checkpoints. At most
two are retained. Resume constructs metadata and restores checkpoint shards directly;
it does not reload/reconvert the original teacher and GGUF decoder. Token caches
are memory mapped so ranks share their filesystem cache pages.
A checkpoint is not itself a deployable prefiller.

Every optimizer step logs wall time, real-token throughput, peak allocated GPU
memory, and two conventional student `6N` MFU proxies (real and padded tokens).
`hfu_qad_execution_estimate` additionally counts both dense phase branches, input
gradients through frozen weights, teacher projections, and checkpoint recomputation.
It approximates attention FLOPs and excludes quantizer/vector/DeltaNet overhead.
The default dense per-GPU peaks are 2250 BF16 and 4500 FP8 TFLOPS for B200.
[NVIDIA's HGX specifications](https://www.nvidia.com/en-au/data-center/hgx/)
list sparse eight-GPU peaks; dense peaks are half those values, divided by eight.
Adjust `--bf16-peak-tflops` and `--fp8-peak-tflops` for other hardware. The first
step of each process, including after resume, includes compilation/warmup; use
later steps for throughput comparisons.

Post-fix four-B200 measurements (GPUs 0,5,6,7), sequence 8192, batch/rank 1,
accumulation 1, KL chunks 1024, fused AdamW, cache disabled and all student blocks
checkpointed:

| Mode | Warm step 2 / 4 | Peak allocated per GPU |
| --- | --- | --- |
| Eager blocks, explicit BF16 autocast | 8.534 / 8.854 s | 135.90 GiB |
| Regional compiled blocks, explicit BF16 autocast | 8.397 / 8.341 s | 135.89 GiB |

Cold initialization and profiled step 3 are excluded. Mean warm latency improved
about 4% with compilation. Student padded-token MFU is about 7%; it excludes the
extra frozen decode branch, teacher and checkpoint recomputation. The comparable
pre-fix eager warm step was 52.975 seconds. BF16 reduced profiled `aten::mm` time
from 44.17 to 1.59 seconds and enabled cuDNN BF16 attention. Artifacts are under
`outputs/b200-4gpu-{bf16,compiled}-baseline/`. These short runs validate execution
and throughput, not convergence: eager/compiled final eval KL was 5.525/5.970.
Only four GPUs are authorized for current tuning; wait for explicit user approval
before expanding to eight.

The next four-GPU candidate uses batch/rank 2, a 4 GiB weight-cache cap and
explicit prefetch depth 1, with other settings unchanged. Warm steps 2/4 took
14.404/15.044 seconds for twice the padded tokens, at 146.44 GiB peak allocated.
The run completed train/eval, but CUDA allocation retries indicate fragmentation.
An otherwise matching repeat with `PYTORCH_ALLOC_CONF=expandable_segments:True`
and profiling disabled completed without allocation-retry warnings. Warm steps
2/3/4 took 15.117/14.146/14.426 seconds (mean 14.563), at 146.33 GiB peak
allocated. That is about 4500 padded tokens/s across four GPUs, approximately
15% above compiled batch 1; cache, batch and prefetch were changed together, so
this does not isolate their individual gains. Losses/grad norms and eval KL
(5.062707) matched the preceding batch-2 run exactly. Cold-step improvements
are not attributed to the allocator: the repeat reused the compiler disk cache.
These are short validation runs. Keep checkpointing enabled for the current
four-GPU configuration.

To reproduce the bounded four-GPU candidate:

```bash
CUDA_VISIBLE_DEVICES=0,5,6,7 NPROC_PER_NODE=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True bash transformers/run.sh \
  --max-steps 4 --warmup-steps 0 --batch-size 2 \
  --gradient-accumulation-steps 1 --optimizer adamw_torch_fused \
  --compile-blocks --weight-cache-max-gib 4 --fsdp-prefetch-depth 1 \
  --train-tokens 100000 --val-tokens 10000 --max-seq-len 8192 \
  --logit-chunk-size 1024 --no-save-checkpoints \
  --output-dir transformers/outputs/b200-4gpu-bf16-batch2-expandable
```

The historical measurements below predate the explicit compute-autocast fix.
A later four-GPU profile found FP32 SIMT GEMMs dominating runtime (44.17 seconds
of a 53.49-second step). They demonstrate correctness and fit, but are not BF16
Tensor Core performance results.

The original eight-B200 validation, before BF16 caching and the larger KL default,
completed three fresh optimizer steps and evaluation
with GPU-resident state, sequence length 8192, batch/rank 1, accumulation 1,
logit chunks 128, optimized causal convolution and 100K/10K train/validation token
budgets. The two warm steps took 53.564 and 53.711 seconds, processing 514 and
256 real tokens/s respectively (different padding). Mean student MFU was 1.095%
with padded tokens and 0.345% with real tokens; estimated QAD utilization was
2.597%. Maximum allocated GPU memory across ranks was 82.47 GiB; aggregate host RSS
peaked at 199.5 GiB. Final eval KL was 6.48697. The cold first step took 248.13
seconds and is excluded from these means. These measurements validate fit and
execution; they do not establish convergence. No full-model checkpoint was saved.

The initial 10-L40S validation used global batch 10, sequence length 8192,
logit chunks 128, and full CPU offload. Two warm steps took 137.6 and 136.1
seconds: 234–248 real tokens/s, 2.67% mean padded-token student MFU, about
1.07% real-token student MFU, and 6.34% estimated QAD hardware utilization.
Peak allocated GPU memory was 33.9 GiB. These are short validation measurements,
not convergence results; sequence padding substantially affects the MFU proxy.
Use `--no-save-checkpoints` for disposable benchmarks to avoid writing another
large model/optimizer checkpoint; metrics and final evaluation are still saved.

```bash
# Export only the trained prefill variant to a serving-compatible NVFP4 checkpoint.
UV_CACHE_DIR="$PWD/transformers/.cache/uv" uv run --no-project \
  --python .venv/bin/python transformers/export_prefiller.py \
  --checkpoint transformers/outputs/qwen38-iq1s/checkpoint-980 \
  --output-dir transformers/outputs/qwen38-iq1s/prefill

# CPU checks: reference trainable partition, decoder-source shared values,
# bit-identical frozen tensors and decode outputs after updates, KL gradients.
UV_CACHE_DIR="$PWD/transformers/.cache/uv" uv run --no-project \
  --python .venv/bin/python -m pytest transformers/test_prefiller.py -q
```

The GGUF head-layout transforms are adapted from the converter already used in
`evals/bin/gguf_to_bf16.py`, specifically vllm-gguf-plugin revision
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. Attribution/license are retained in
`gguf_decode.py` and `GGUF_LAYOUT_LICENSE`.
