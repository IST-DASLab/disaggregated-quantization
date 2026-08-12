# `qad/bin` — entrypoints

Every script here is invoked from the **qad root**, not from inside `bin/`:

```bash
cd /lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/prefill-decode-shenanigans/qad
./bin/run_qad.sh --quantizer nvfp4
```

They each resolve `QAD_DIR` as their own parent directory, so `checkpoints/`, `results/`
and the importable packages are found regardless of your cwd. Log directories are keyed
off the **repo** root (one level above `qad/`), which is where `logs/` actually lives.

Where everything else sits:

| path | what |
|---|---|
| `qad/training/` | training library + `qad.py`, the training entrypoint |
| `qad/eval/` | eval drivers (`eval_disagg.py`, `eval_vllm.py`, `eval_transformers.py`) |
| `qad/serving/` | disaggregated-serving runtime (Nixl 1P1D pair, proxy, KV-noise connector) |
| `qad/diagnostics/` | bring-up probes and the KV-transfer gate |
| `qad/cluster_scripts/` | SLURM ops: gap-filling, pruning, watching, reaping |
| `qad/quantizers/`, `qad/export/`, `qad/tests/` | formats, checkpoint export, tests |
| `qad/checkpoints/`, `qad/results/` | data — never move these |
| `<repo>/logs/` | all SLURM logs (`train/`, `eval_disagg/`, `checks/`, …) |

Resource footprint, as declared in each script's `#SBATCH` header:

| entrypoint | nodes | GPUs | wall |
|---|---|---|---|
| `run_qad.sh` | 1 | 8 | 4 h (cluster cap — see `--chain`) |
| `run_eval_disagg.sh` | 1 | 2 | 4 h |
| `run_eval_dual.sh` | 1 | 1 | 3 h |
| `run_eval_vllm.sh`, `run_eval.sh` | 1 | 1 | 2 h |
| `run_tests.sh` | 1 | 8 (1 unless `test_checkpoint` is selected) | 30 min, interactive QoS |

---

## Python environments — read this first

Nothing GPU-related runs in the login-node interpreter. **All GPU work runs inside one
container image**, launched by `srun --container-image=...`:

```
/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
```

with `--no-container-mount-home` and `--container-mounts=/lustre:/lustre` (training also
mounts `$HOME/.netrc` so wandb can authenticate). The login node's `.venv` has no usable
CUDA driver — use it only for plotting, log parsing, and the `cluster_scripts/` helpers.

The container does **not** contain everything. The qad root plus four external trees get
layered onto `PYTHONPATH`, each for a different reason. They are deliberately *not* all
applied at once — two of them break vLLM if they reach the wrong process, so which
overlay goes where matters more than it looks.

### 1. The qad root itself
Every entrypoint puts `$QAD_DIR` on `PYTHONPATH`. That is what makes `quantizers`,
`training`, `export` and `serving` importable as top-level packages. The KV-noise
connector is loaded by vLLM through the dotted path `serving.kv_noise_connector`, so the
qad root must be present for the engines too, not just the driver.

### 2. `lm_eval_overlay` — the harness (783 MB)
```
/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/lm_eval_overlay
```
Supplies `lm_eval` and its dependencies. It goes on the **driver's** `PYTHONPATH` only.

> It ships `huggingface-hub` 1.24.0, which shadows the container's copy and makes
> `vllm serve` refuse to start (`huggingface-hub>=0.34.0,<1.0 is required`).
> `eval/eval_disagg.py` therefore **strips this entry** from the environment it hands to
> the prefill/decode servers. If you launch a server by hand, strip it yourself.

### 3. `nixl_nodeps` — the transfer library (234 MB)
```
/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/nixl_nodeps
```
`nixl` + the CUDA-matched `nixl_cu13` wheel, installed with **`--no-deps`**. Put on the
**servers'** `PYTHONPATH` by `serving/run_nixl_server.sh` (`$NIXL_PREFIX`).

> Do **not** use the sibling `nixl_overlay/` (5.2 GB). That was an install *without*
> `--no-deps`; it dragged in a second copy of torch which shadows the container's and
> breaks `vllm._C` with an undefined `at::TensorBase` symbol. It is kept only as a
> record of the failure. `diagnostics/run_nixl_check.sh` is the test that proves
> `import nixl._api` and `import vllm` can coexist in one interpreter.

### 4. `psx-luts` — the LUT extension (858 MB)
```
/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/psx-luts     # $PSX_LUTS_PATH
```
A compiled extension imported lazily by the 2-bit LUT quantizer, added to `PYTHONPATH` by
`run_qad.sh`. It must be **built** before use, or the import raises with instructions:
```bash
cd "$PSX_LUTS_PATH" && PSX_LUTS_FAST_BUILD=1 python setup.py build_ext --inplace
```
Only the `nvr2bit` / `nvfp4nvr2bit*` formats touch it; everything else runs without it.

### 5. `third_party/Liger-Kernel/src`
Added to `sys.path` directly by `training/qad.py` and `eval/eval_transformers.py` for
`LigerFusedLinearJSDLoss` (the fused distillation loss). No environment variable — it is
resolved relative to the repo root.

### Hugging Face cache and offline mode
```
HF_HOME=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
TOKENIZERS_PARALLELISM=false
```
Datasets are always read from cache (`HF_DATASETS_OFFLINE=1`). `HF_HUB_OFFLINE` is `1` for
the disaggregated path and `0` elsewhere — many parallel array tasks hitting the Hub at
once earns a 429, and offline mode is what stops it.

---

## Training — `run_qad.sh`

Submits itself with `sbatch`; stage 0 runs on the login node, creates
`logs/train/<stamp>_<tag>/`, then re-submits into it.

```bash
./bin/run_qad.sh --quantizer nvfp4                        # defaults: Qwen/Qwen3-4B
MODEL=Qwen/Qwen3-8B ./bin/run_qad.sh --quantizer nvfp4    # pick the model via env
./bin/run_qad.sh --quantizer nvfp4 --debug                # interactive QoS, 1 h
./bin/run_qad.sh --quantizer nvfp4 --chain 3              # see below
```

One node, 8 GPUs, `--time=04:00:00` (a cluster cap).

**`--chain N` is required past ~4B.** A 4B run measures 5.81 s/step ≈ 4.01 h for 2485
steps, so it TIMEOUTs a few steps from the end; 8B is roughly double. `--chain N` submits
N jobs sharing one `--job-name` under `--dependency=singleton`, so SLURM runs them one at
a time and each resumes where the last stopped (state is written every 100 steps).
Over-provision freely — a chain job that finds the run finished exits immediately.

Job names are `qad-<model>-<quantizer>`, which is what `singleton` keys on. Env knobs:
`MODEL`, `RUN_PREFIX` (default `qad3x`, namespaces the checkpoint tag), `CKPT_DIR`,
`WANDB_MODE`. Everything else is passed through to `training/qad.py` — `--lr`,
`--train-tokens`, `--lr-schedule`, `--save-every`, `--export-every`, `--resume`, …

Checkpoints land in `checkpoints/<prefix>-<model>-<quant>-<hash>/{weights,state}/`.

## Evals

`run_eval_disagg.sh` is the current path — real disaggregated serving, a prefill engine
and a decode engine on separate GPUs exchanging KV, driven through one OpenAI endpoint.
Two GPUs per job.

```bash
./bin/run_eval_disagg.sh --model Qwen/Qwen3-4B --quantizer nvfp4 \
     --run-name qad3x-Qwen-Qwen3-4B --iter 2250
./bin/run_eval_disagg.sh ... --steps 250,750,1250,2250     # job array, one element per step
./bin/run_eval_disagg.sh ... --no-think                    # thinking suppressed
./bin/run_eval_disagg.sh --model Qwen/Qwen3-4B --unquantized   # BF16 baseline (forces iter 0)
```

Flags: `--quantizer`, `--quantizer-params`, `--run-name`, `--iter` / `--steps`, `--tasks`
(default `gsm8k minerva_math500`), `--limit`, `--think` / `--no-think`, `--unquantized`,
`--tag`, `--max-gen-toks` (default 4096), `--max-model-len` (8192), `--concurrency` (512),
`--log-samples`, `--ckpt-dir`.

Results go to `results/disagg/think/<tag>/step_<N>.json` (or `nothink/`). **Thinking mode
is a separate benchmark, not a variant** — the same checkpoint differs by ~20 points on
GSM8K between modes, which is why the two trees never mix.

```bash
./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-1.7B      # every format × both modes × steps
./bin/run_eval_disagg_sweep.sh --dry-run                    # print submissions, submit none
./bin/run_eval_disagg_sweep.sh --missing-only               # only the gaps
```

One sbatch array per (format, mode); each element is one step, and each job holds two
GPUs — check the queue before launching all sizes. Narrow the grid with `--formats`,
`--steps`, `--modes`, `--tasks`, `--limit`, `--max-model-len`. Formats are given as
`Label:quantizer` pairs, e.g. `--formats "NVFP4:nvfp4 BF16:unquantized"`.

`prune_checkpoints.py` keeps any step that already has a result anywhere under
`results/` — so prune *after* the evals land, never before.

Other drivers, all sharing the same tag convention and on-disk layout:

| script | what | results |
|---|---|---|
| `run_eval_vllm.sh` | single-engine vLLM serving | `results/vllm/{think,nothink}/` |
| `run_eval.sh` | in-process `transformers` + lm-eval | `results/transformers/` |
| `run_eval_dual.sh` | dual formats via HF `generate()` | `results/dual/` |

`run_eval_dual.sh` predates real disaggregated serving: HF `generate()` does one
multi-token prompt pass then one token at a time, which is the prefill/decode split, so a
dual-format model switches format on its own. `run_eval_disagg.sh` now answers the same
question on real serving infrastructure and is preferred.

## Tests — `run_tests.sh`

```bash
./bin/run_tests.sh                       # every test file (8 GPUs: test_checkpoint needs them)
./bin/run_tests.sh gsq_lloyd             # just tests/test_gsq_lloyd.py, 1 GPU
./bin/run_tests.sh export nvfp4 dual     # several
./bin/run_tests.sh -v nvfp4              # full output instead of PASS/FAIL lines
```

Runs on an interactive allocation inside the container. `test_checkpoint.py` is
distributed (it verifies ZeRO-2 optimizer shards restore exactly) and is launched under
`torchrun`; everything else is single-process.

---

## Not in `bin/` but part of the same workflow

```bash
./cluster_scripts/submit_missing_evals.py            # dry run: what's missing
./cluster_scripts/submit_missing_evals.py --apply    # submit the gaps
./cluster_scripts/prune_checkpoints.py               # dry run: what would be dropped
./cluster_scripts/prune_checkpoints.py --apply       # actually drop them
./cluster_scripts/watch_job.sh 516588                # stream one job's milestones
./cluster_scripts/reap_stalled.sh --dry-run          # find jobs hung holding GPUs
./cluster_scripts/autoeval_watch.sh                  # submit evals as checkpoints appear (tmux)
```

`reap_stalled.sh` kills only jobs whose **results file is already on disk** — lm-eval
prints nothing while generating, so "silent for N minutes" reaps healthy jobs.

Serving and diagnostics:

```bash
./serving/run_nixl_server.sh --prefill-model DIR --decode-model DIR \
     --tokenizer Qwen/Qwen3-0.6B --ready-file /path/ready
./diagnostics/run_nixl_1p1d.sh                # end-to-end 1P1D smoke test
./diagnostics/run_kv_verify.sh                # gate: does KV actually cross?
./diagnostics/run_nixl_check.sh               # can nixl and vllm coexist in one interpreter?
./diagnostics/run_nixl_ucx_diag.sh            # UCX/CUDA registration diagnosis
```

`run_kv_verify.sh` is the gate on every disaggregated number. If the KV transfer silently
fails, the decode engine just recomputes the prompt with its own weights and returns a
fluent, plausible answer — the eval reports a "disaggregated" score that is really
homogeneous-decode, and nothing in the output looks wrong. See `docs/DISAGG.md`.
