---
name: add-quant-format
description: "Adds a new quantization format to the QAD codebase — homogeneous or prefill/decode dual — end to end: layer class, registry entry, tests, gated smoke run, training launch across model sizes, eval sweep, and plotting. Use when asked to add/implement a new quantizer, grid, or prefill/decode format (e.g. 'add a format with NVFP4 prefill and Lloyd43 decode', 'add a 3-bit grid called X')."
---

# Add a quantization format

Every step gates the next. Skipping a gate has cost hours or GPU-days at least once each;
the notes below say which.

## 1. Decide the shape before writing code

- **Homogeneous** (one format everywhere) → one checkpoint per step, `variants == [None]`.
- **Dual / prefill-decode** → `"variants": ["prefill", "decode"]`, two checkpoint dirs per
  step, served by two vLLM engines.
- **Shared vs split master**: shared = one trained weight serving both phases; split = two
  masters that diverge (2x linear FLOPs, 2x optimizer memory). If asked for both, they are
  two registry entries over one base class.

Ask only if the answer changes the work materially (e.g. trained format vs
inference-only ablation — that is a ~100x cost difference). Otherwise pick and state it.

## 2. Wire it

Files: `qad/quantizers/<family>.py` (layer), `qad/quantizers/grids.py` (any new LUT),
`qad/quantizers/__init__.py` (import + `REGISTRY`).

Reuse the existing primitives — subclass `NVFP4Linear` for anything FP4 (E2M1 rounding,
two-level block scaling, STE, packed export all come free), `BlockScaledLinear`
otherwise. Do not reimplement quantization math.

Three constraints that bite silently:

- **Names must be alphanumeric.** `_METHOD_RE` in the plotting notebook parses
  `<run>-<model>-<method>-<hash>`; a `-` or `_` in the method name breaks tag parsing.
- **`defaults` is md5-hashed into the checkpoint tag.** New defaults → new tag (correct
  for a new format). Editing an *existing* format's defaults renames every one of its
  checkpoints and orphans its results — assert the old hash still matches.
- **Verify the import actually resolves.** A `REGISTRY` entry referencing an
  unimported `apply_*` fails every test file at import. Static check:

```bash
python3 -c "
import ast,sys; t=ast.parse(open('qad/quantizers/__init__.py').read())
imp={a.asname or a.name for n in ast.walk(t) if isinstance(n,ast.ImportFrom) for a in n.names}
use={n.id for n in ast.walk(t) if isinstance(n,ast.Name) and n.id.startswith('apply_')}
print('missing:', sorted(use-imp) or 'none'); sys.exit(1 if use-imp else 0)"
```

## 3. Mixed-precision export is already supported

`export_config()` returning `None` makes `export/save.py` write a plain HF bf16 model
with no `quantization_config`; override `export_tensors` / `load_tensors` for that half.
Use it for a BF16 half, or for a pseudo-quantized half (3-bit Lloyd has no kernel, so it
ships dequantized exactly as `lloyd3bit` does).

**No eval-side change is needed.** `resolve_pair()` keys off the presence of `prefill/`
+ `decode/` directories, so a mixed BF16↔FP4 engine pair serves over NIXL unmodified
(proven: 1319+ requests, 0 proxy errors).

## 4. Phase-aware calibration

`calibrate_nvfp4` must run inside `quant_phase(prefill_mask_from_labels(labels))`. The
activation observer only ever sees its own phase; unmasked, a format whose W4A4 half is
*decode* observes nothing of its own phase and folds the other one into the static scale
baked into its checkpoint.

## 5. Test — and make the test able to fail

Add `qad/tests/test_<format>.py` following the existing `check(name, cond, detail)` style.

**The bug class that recurs: a flag passed is not a flag applied.** Three instances so
far — `--no-think` was a silent no-op that mislabeled a whole results tree; `timeout=`
was never overridden and killed 21 jobs at ~80% after 2h each; `signed=` was accepted by
`pack_nvfp4_weight` then destroyed by a `clamp(min=1e-8)`.

So assert the new setting **differs** from the old, not just that it is self-consistent —
something silently ignored still round-trips against itself perfectly:

```python
check("signed pack DIFFERS from unsigned", not torch.equal(su, ss))
```

Also assert the *contrast* the format exists for (e.g. "lloyd3bit has neither pin"), and
that pre-existing formats are untouched (their tag hashes).

## 6. Gate before spending GPU, in this order

```bash
cd qad && ./bin/run_tests.sh -v <format>                       # new gates
./bin/run_tests.sh export nvfp4 quantizers dual <format>       # regression
```

Regression must show existing golden exports **byte-identical**
(`tensor values identical  max|Δ|=0.000e+00`) and every `hash unchanged`.

Then **one short smoke run** — not optional, it is the only cover for the training loop
under the phase mask, calibration, a real dual export, and `__init__` ordering:

```bash
MODEL=Qwen/Qwen3-0.6B RUN_PREFIX=smoke WANDB_MODE=offline \
  ./bin/run_qad.sh --quantizer <name> --train-tokens 400000 --lr-schedule constant \
               --warmup-steps 2 --val-every 5 --export-every 5 --save-every 5 --debug
```

Verify on the **real** checkpoint, not the tiny test model — half sizes and the
safetensors header expose shape/scale bugs the small model hides. `safetensors` is not
installed on the login node; parse the header directly (8-byte LE length, then JSON).

Clean up: `scancel` the smoke job and `rm -rf qad/checkpoints/smoke-*`.

## 7. Train

```bash
for m in 0.6B 1.7B 4B; do
  MODEL=Qwen/Qwen3-$m RUN_PREFIX=qad3x ./bin/run_qad.sh --quantizer <name> \
    --train-tokens 100000000 --lr-schedule constant
done
```

That is the `qad3x` recipe every other format uses — keep it identical or the comparison
is meaningless. 8 GPUs each, ~4h (a 4B run measures 5.81 s/step, i.e. 4.01h for 2485
steps — it WILL hit the 4:00 wall, which is why those runs report TIMEOUT). Runs inherit
`--export-every 250`, switching to `--export-tail-every 125` past step 2000: 13
checkpoints per run, not 99. Keep that in step with `prune_checkpoints.py --tail-grid`.
NCCL teardown often reports FAILED *after* the last checkpoint lands — check the
checkpoints before assuming loss. A run is done once 2250 (the last plotted step) is
exported. `--resume auto` continues from `state/`.

Arm one `Monitor` on the jobs (`qad/cluster_scripts/watch_job.sh <jobid>`), failure-
focused. Watchers report; they never cancel.

## 8. Evaluate and plot

```bash
./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-<size> \
    --tasks "gsm8k minerva_math500" --formats "Label:<name>"
./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-<size> \
    --tasks "mmlu_pro" --formats "Label:<name>"
```

New formats are **not** in the default `FORMATS`, so `--formats` is required. `mmlu_pro`
(12032 docs) is ~4x the cost of the other two combined — submit it separately so it does
not starve the cheap benchmarks. Use `--missing-only` to gap-fill.

Add a `METHODS` entry in `notebooks/plots.ipynb` cell 1 with an **unused** colour; the
recovery curves and the tail-recovery bars pick it up automatically.

**Reading the results:** training `val_ntp` is unmasked, so it is uninformative for any
format whose quantized phase is decode — it will sit at teacher level all run. The masked
signal is on the tqdm line (`kl=`, `ntp=`); the real answer is the eval sweep. Bars built
on fewer than 5 tail steps are flagged in the cell output and are not settled plateaus.
