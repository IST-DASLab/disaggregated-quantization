# Standalone NVFP4 prefiller training

This directory can be copied to another repository or renamed. It contains all
project code needed for training, resuming, real NVFP4 export, ODP GGUF export,
and stock-vLLM serving checks. It does not import code from a parent repository.
Run the commands below from this directory.

## Environments

Training uses Python 3.11. Serving checks use a separate Python 3.12 environment
because vLLM requires a different Torch version. CUDA training needs compatible
NVIDIA drivers, a CUDA toolkit/compiler, and the causal-convolution extension.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python ninja==1.13.2
uv pip install --python .venv/bin/python --no-build-isolation -r requirements-kernels.txt

uv venv --python 3.12 .venv-serving
uv pip install --python .venv-serving/bin/python -r requirements-serving.txt
```

Model weights, datasets, and login use the normal Hugging Face cache and honor
`HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE`. Only prepared token chunks and
build caches live in `.cache/`. There is no offloading.

## Training

```bash
# Tiny two-GPU integration test, including final exports and serving validation.
NPROC_PER_NODE=2 bash run.sh --smoke-test --max-steps 2 --warmup-steps 0 \
  --calibration-batches 2 --save-steps 1 --eval-steps 1 \
  --gradient-accumulation-steps 1 --output-dir outputs/smoke

# Full eight-GPU run, MBS 4, global batch 32, sequence length 8192.
NPROC_PER_NODE=8 bash run.sh \
  --batch-size 4 --gradient-accumulation-steps 1 \
  --max-steps 980 --warmup-steps 100 --learning-rate 3e-6 \
  --optimizer adamw_torch_fused --compile-blocks \
  --weight-cache-max-gib 4 --activation-checkpoint-every 1 \
  --fsdp-prefetch-depth 1 --logit-chunk-size 1024 \
  --output-dir outputs/qwen38-iq1s

NPROC_PER_NODE=8 bash run.sh --batch-size 4 --gradient-accumulation-steps 1 \
  --resume-from-checkpoint outputs/qwen38-iq1s/checkpoint-50 \
  --output-dir outputs/qwen38-iq1s
```

`run.sh` changes into this directory. Paths passed to it are relative to here;
absolute paths also work. `train.py` defaults are anchored to this directory,
regardless of the caller's working directory. Use `--help` for all FSDP,
checkpointing, compilation, profiling, and export options. Only the last training
checkpoint is retained. Keep its `student-config/` and `run_config.json` beside
the checkpoint when moving a run: they record the quantization contract.

To use existing compatible environments without creating new ones:

```bash
TRAINING_PYTHON=/path/to/training/python NPROC_PER_NODE=2 bash run.sh \
  --smoke-test --max-steps 1 --warmup-steps 0 \
  --gradient-accumulation-steps 1 --serving-python /path/to/serving/python \
  --output-dir outputs/smoke
```

The launcher monitors this job's aggregate host RAM and prioritizes the active
Python environment's CUDA libraries. `HOST_MEMORY_LIMIT_GIB` defaults to 768.

## Training and serving contract

The teacher is `Qwen/Qwen3.8-27B-FP8`, kept in real FP8 storage. The frozen
external decoder defaults to
`ISTA-DASLab/GSQ-RCO-small/Qwen3.8-27B-GSQ-RCO-IQ1_S.gguf`; it is streamed through
the local GGUF loader into BF16 weights. Override `--teacher-model`,
`--decode-model`, `--decode-gguf`, or `--decode-prefix` for other compatible sources.

Only text projection masters train, with FP32 storage and BF16 compute. All text
norms, embeddings, LM head, GDN in_proj_a/b, convolution, A_log and dt_bias are
shared and frozen from the decoder. Vision is excluded from training and restored
unchanged for multimodal export. Per-rank BF16 quantized-weight caches are lazy,
nonpersistent, and invalidated after optimizer updates; the cap is in GiB.

Input globals are calibrated once across all ranks and frozen. FP8 activation
block scales remain dynamic. Q/K/V, gate/up and GDN QKV/Z share globals within
their groups, while keeping distinct matrices. Activation rounding matches native
nearest-even FP4 conversion. The handoff uses hybrid NIXL's final-prompt-token
on the decoder. KL is teacher-to-student at shifted assistant positions, with
chunked head projections and the phase mask preserved through recomputation.

Legacy checkpoints retain their old contract when resumed/exported. Changing
observer mode requires explicit recalibration. A legacy checkpoint with trained
specialized norms is not equivalent to the shared-norm model. See
[INFERENCE_AUDIT.txt](INFERENCE_AUDIT.txt) for the audit and the remaining linked
llama.cpp handoff difference; successful export does not prove cross-runtime
numerical parity.

## Automatic and manual exports

After the final resumable checkpoint, training releases its GPU models and writes:

- `prefill-nvfp4/`: real packed FP4 weights, FP8 block scales, globals, tokenizer
  and chat template; real-model exports also include frozen BF16 vision/processor assets.
- `prefill-gguf/`: one ODP GGUF per layer, with every code, scale and BF16 gate
  tensor checked after reading it back. Shared norms, embedding and head come
  from the separate decoder GGUF at inference.
- `serving-check/`: stock-vLLM logs and text/image HTTP results, including checks
  that native NVFP4 kernels run and fused globals are preserved.

```bash
.venv/bin/python export_prefiller.py \
  --checkpoint outputs/qwen38-iq1s/checkpoint-980 \
  --output-dir outputs/manual-nvfp4 --multimodal

.venv/bin/python export_gguf.py outputs/manual-nvfp4 outputs/manual-gguf --tag IQ1_S

.venv-serving/bin/python check_serving.py outputs/manual-nvfp4 \
  --gpu 0 --output outputs/manual-serving-check
```

The GGUF format follows the vendored ODP packer, with provenance/license in
`odp/`. `--gguf-tag` overrides the tag inferred from the decoder filename.
`serve_prefiller.py` is the optional legacy scale-preserving serving entry point;
new-contract exports use stock `vllm serve` directly.

`--no-export-prefiller` skips all final exports/checks. `--no-export-gguf` skips
GGUF only; `--no-verify-serving` skips the GPU serving check. Disposable
`--no-save-checkpoints` runs skip exports/checks. Failed exports or serving checks
fail the command but preserve the resumable checkpoint. Output replacement is
limited to validated artifacts belonging to the same run.

## Tests and transfer

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q
```

The CPU suite includes a two-process Gloo checkpoint test, which needs permission
to bind local sockets. The CUDA-only autocast test skips with GPUs hidden.

Copy the source files and subdirectories, including requirements, tests, notices
and licenses. `.venv/`, `.venv-serving/`, `.cache/`, `outputs/` and Python caches
are machine-local artifacts and are excluded by `.gitignore`; recreate the
virtual environments after moving. Checkpoints and prepared token caches can be
moved separately if needed. No symlink to the original repository is required.

The old measurements and repository-specific commands are preserved only as
historical records in [docs/TRAINING_HISTORY.md](docs/TRAINING_HISTORY.md) and
[HANDOFF.txt](HANDOFF.txt), not as setup instructions for this standalone copy.
