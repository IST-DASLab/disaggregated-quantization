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
changing phase routing. Logit projections are checkpointed in chunks, avoiding a
sequence-by-vocabulary allocation. The phase mask remains active during backward.

## Setup and run

From the repository root:

```bash
UV_CACHE_DIR="$PWD/transformers/.cache/uv" uv pip install \
  --python .venv/bin/python -r transformers/requirements.txt

# Tiny hybrid architecture with real HF FP8 linear kernels; saves/resumes via FSDP.
NPROC_PER_NODE=2 bash transformers/run.sh --smoke-test \
  --max-steps 2 --warmup-steps 0 --save-steps 1 --eval-steps 1 \
  --gradient-accumulation-steps 1 --output-dir transformers/outputs/smoke

# Actual model, all 10 local L40S GPUs, FP32 masters and CPU offload.
bash transformers/run.sh --logit-chunk-size 128

# Eight GPUs give the paper's global batch 32 (1 example x 4 accumulation x 8).
NPROC_PER_NODE=8 bash transformers/run.sh --logit-chunk-size 128

# Continue the same configuration from a saved checkpoint.
bash transformers/run.sh --logit-chunk-size 128 \
  --resume-from-checkpoint transformers/outputs/qwen38-iq1s/checkpoint-50
```

The default ten-GPU global batch is 40. Sequence limit is 8192; the reasoning
corpus is the reference's `faunix/Qwen3.8-27B-Distillation-40K`, with tool-use rows
filtered, reasoning split into the template's reasoning field, and overlong
examples dropped. Trainer shards the dataloader; chunks are not pre-sharded.
Use `--decode-model /path/to/file.gguf` or a dense safetensors directory to change
the decoder. Dense wrapper checkpoints use `--decode-prefix model.language_model.`;
text-only checkpoints use `--decode-prefix model.`. Downloads and token caches live
under `transformers/.cache/`; checkpoints live under `transformers/outputs/`.

## FSDP and memory

FSDP2 shards both text stacks and the student's decoder parameters. Frozen decode
weights are Parameters with `requires_grad=False` so they can be sharded; cached
full-size quantized weights are disabled. Autocast handles BF16 compute without
casting FP8 teacher storage or FP32 student masters. CPU offload is enabled for
masters, gradients and optimizer state. `--no-cpu-offload` is available for larger
GPUs. TP/CP are not implemented: the custom quantizer and hybrid DeltaNet recurrence
need additional work for those modes.

Selective offload is controlled per FSDP unit:

| Argument | Effect |
| --- | --- |
| `--teacher-resident` | Keep frozen teacher block and embedding shards on GPU |
| `--student-resident-layers N` | Keep the first N student blocks' parameter, gradient and Adam shards on GPU |
| `--activation-offload` | Store activation-checkpoint inputs on CPU to free GPU headroom |
| `--no-cpu-offload` | Keep all shards and Adam state on GPU |

Selective residency requires FSDP and CPU offload. The remaining units retain CPU
offload, including root-owned heads. Parameter sharding and checkpoint format stay
the same when residency changes. The launcher prints local shard storage by device;
measure training peaks when increasing residency. These policies are applied in a
local Accelerate preparation wrapper, without editing installed packages.

```bash
# Use GPU headroom for teacher shards and 12 student blocks.
bash transformers/run.sh --logit-chunk-size 128 --teacher-resident \
  --student-resident-layers 12 --activation-offload
```

Only rank zero reads model weights. Other ranks construct metadata and uninitialized
storage, then receive FSDP shards. The launcher monitors job RSS (default limit
768 GiB, configurable through `HOST_MEMORY_LIMIT_GIB`) and reserves host RAM.
`fsdp_compat.py` works around Accelerate 1.15's persistent-buffer initialization bug
locally; installed packages are not patched.

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
The default dense peaks are 362.05 BF16 and 733 FP8 TFLOPS per L40S, from
[NVIDIA's specifications](https://www.nvidia.com/en-gb/data-center/l40s/).
Adjust `--bf16-peak-tflops` and `--fp8-peak-tflops` for other hardware. The first
step of each process, including after resume, includes compilation/warmup; use
later steps for throughput comparisons.

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
