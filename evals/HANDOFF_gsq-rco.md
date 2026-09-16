# Handoff: evaluate ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF on MMLU-Pro / MMMU-Pro

**Status 2026-09-11 (oci-jhb-slurm-1). MMLU-Pro serving + scoring WORKS end to end.**
IQ3_S smoke: accuracy 0.714 (10/14), rc=0, results written. Remaining: MMMU-Pro harness,
then the full sweep.

The previous version of this file described the OLD cluster (`aws-cmh-slurm-1`, all
`/lustre/fsw/portfolios/adlr/...` paths, job 3650495). Those paths and job ids are dead.

`<PD>` = `/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode`

## THE ONE THING THAT MATTERS: install the plugin from git, not PyPI

vLLM 0.28.1 has **no in-tree GGUF** (`gguf` absent from `QUANTIZATION_METHODS`), so
`vllm-gguf-plugin` is mandatory. **The PyPI release 0.0.5 does NOT support this model.**
It was cut 2026-08-10; Qwen3.5 support landed 2026-08-18 in
`vllm-project/vllm-gguf-plugin` PR #98 ("[Models] support Qwen3.5/3.6 multimodal GGUF and
MTP", which adds a dedicated 430-line `weights_adapter/qwen3_5.py` providing
`Qwen35GGUFAdapter` / `Qwen35MtpGGUFAdapter`) and has never been released. Related fixes
also landed after 0.0.5: #109 (release GGUF shard tensors when materializing fused
weights) and the TP-rank fix in the merged-column/QKV GGUF loaders.

Against 0.0.5 the load fails in five escalating ways — unknown `model_type: qwen3_5`;
`Qwen3_5VisionConfig` has no `num_hidden_layers`; `Qwen3_5Config` has no `vocab_size`;
432 unmapped Gated-DeltaNet tensors; then
`'GGUFUninitializedWeightTypeParameter' object has no attribute 'output_dim'` in vLLM's
fused-linear loader. **Do not try to patch these**; they are symptoms of a stale wheel.
(An earlier pass through this file did patch them by hand and still hit the fused-linear
wall. Those patches were dead and superseded once the plugin was built from git; the stale
`gguf_overlay/` tree and the `gguf_overlay_patches/` copies have both been deleted.
`<PD>/gguf_overlay_main` — built from git per the recipe below — is the only overlay, and
`bin/gguf_to_bf16.py` still needs it on `PYTHONPATH` for the weights adapter.)

Build it like this (≈2 min, compiles `_C_gguf.abi3.so`, needs no GPU):

```bash
git clone https://github.com/vllm-project/vllm-gguf-plugin.git <PD>/vllm-gguf-plugin-src
# inside the vllm container, on a compute node:
cd <PD>/vllm-gguf-plugin-src
TORCH_CUDA_ARCH_LIST='10.0;10.3' MAX_JOBS=16 \
  pip install --no-cache-dir --no-deps --no-build-isolation . --target <PD>/gguf_overlay_main
pip install --no-cache-dir --no-deps gguf --target <PD>/gguf_overlay_main
```

`--no-deps` is not optional: a resolving install drags in a second torch and breaks
`vllm._C`. The plugin registers via the `vllm.general_plugins` entry point, which only
fires once the engine calls `load_general_plugins()` — checking `QUANTIZATION_METHODS`
before that misleadingly reports "not registered".

## Infrastructure — DONE, do not redo

- **Weights**: 4 quants + mmproj at `<PD>/models/Qwen3.8-27B-GSQ-RCO-GGUF/` (38 GB, sizes
  verified). Real files in ONE directory so vLLM's `mmproj*.gguf` auto-discovery works.
  The repo also ships 4 `-mtp` variants (speculative decoding) — deliberately not
  downloaded; not needed for accuracy.
- **Base config/tokenizer**: `<PD>/models/Qwen3.8-27B/` from the ungated `Qwen/Qwen3.8-27B`
  (config.json, tokenizer.json, chat_template.jinja, ...). The GGUF repo ships none of
  these — they are embedded per-`.gguf` — and the plugin still needs an HF config source:
  it resolves `hf_config_path > tokenizer > remote repo id > Path(gguf).parent`, and a
  LOCAL `.gguf` falls to the parent dir, which holds only `.gguf` files. Hence
  `--hf-config-path` in `serve_flags`. `model_type: qwen3_5`,
  `Qwen3_5ForConditionalGeneration`, 64 text layers (48 Gated-DeltaNet + 16 attention),
  dense, multimodal, `tie_word_embeddings: false`.
- **Container**: `<PD>/containers/vllm-nightly.sqsh` (vLLM 0.28.1rc1, torch 2.13+cu130,
  transformers 5.17.0); provenance file beside it. The old `vllm-nightly-muse.sqsh` does
  not exist here and is not needed.
- **`muse-pydeps`** overlay at `<PD>/muse-pydeps` (399 MB, imports verified).
- **`evals/` re-pointed to this cluster** (19 files): account `coreai_psx_qad`, container,
  `hf_cache`, `muse-pydeps`, `harnesses`, `models`. `results/` deliberately NOT rewritten —
  those paths are historical records.
- **`models.json` `qwen3.8-27b`**: `"server_pythonpath": "<PD>/gguf_overlay_main"` (existing
  per-model mechanism, read at `run_eval.sh:318`) and
  `serve_flags: ["--reasoning-parser","qwen3","--hf-config-path","<PD>/models/Qwen3.8-27B"]`.

### Two cluster-specific fixes in `bin/run_eval.sh`

- **`MIN_GPUS` default 1 -> 4.** The QOS enforces a whole-node floor; a 1-GPU request dies
  at submit with `sbatch: error: QOSMinGRES` and nothing in the log. Single-engine still
  serves on 1 GPU with TP=1; the other 3 idle.
- **`/scratch:/scratch` added to every `--container-mounts`.** Three sites used a quoted
  `"/lustre:/lustre"` form; without `/scratch` the job dies with
  `run_eval.sh: No such file or directory` inside the container.

`--limit`, `--interactive`, `--repeats`, `--setting`, `--mode`, `--long` are all real
`run_eval.sh` flags (the old handoff's caution about this is resolved).

## Verified working

```bash
cd <PD>/prefill-decode-shenanigans/evals
M=<PD>/models/Qwen3.8-27B-GSQ-RCO-GGUF
./bin/run_eval.sh --model qwen3.8-27b --bench mmlu_pro --tag gsq-rco-iq3s-smoke \
    --weights $M/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf --limit 8 --interactive
```
Model loading took 12.85 GiB / 64 s (matches the 11.77 GB file — the embedding stays
quantized, as it should: `token_embd` is IQ2_S in the IQ3_S arm and IQ1_M in IQ2_XS, since
these are mixed-precision imatrix GGUFs). KV cache 3.49M tokens. **accuracy 0.714 (10/14).**

## MMMU-Pro — set up and verified

Harness at `<PD>/harnesses/MMMU` pinned to `268471d` (the `MMMU_COMMIT` in
`bin/setup_harnesses.sh`), giving `mmmu-pro/prompts.yaml` and `mmmu-pro/infer/infer_gpt.py`
— the two files the drivers import. Dataset `MMMU/MMMU_Pro` in the hub cache, 2.8 GB, all
three configs (`vision` 4 parquet shards, both `standard` 2 each).

**`setup_harnesses.sh` was NOT run wholesale, deliberately.** It is `set -euo pipefail`
with no `--only`, and its dataset block downloads **gated `Idavidrein/gpqa` FIRST** — a
401 there aborts before `MMMU/MMMU_Pro` is ever fetched. It also pulls ultrachat_200k,
C-RADIOv4-H and OCRBench, none of which this task needs. Only the two MMMU steps were run,
reusing its pinned commit. (`clone_pinned` is idempotent, so running the full script later
is still fine.)

Smoke (`--bench mmmu --setting vision --limit 4`): **accuracy 0.75**, `truncated 0,
empty_content 0, unparsed 0`, rc=0.

**mmproj auto-discovery works** — verified from the generations, not just from the absence
of errors: the model transcribes specific image content (e.g. Nast's "The Only One Barred
Out" cartoon, quoting the placard text verbatim). In the `vision` setting the question is
rendered INTO the image, so reading it at all proves the projector is live; randomly
initialised vision weights could not produce this. The plugin resolves the projector
separately and explicitly excludes `mmproj`-named files from being chosen as the backbone,
so the co-located `mmproj-Qwen3.8-27B-BF16.gguf` is picked up automatically.

Note `vllm-gguf-plugin` PR #120 ("Serve Qwen3.5/3.6/3.8 text-only GGUF without an mm_proj")
is still OPEN and is NOT needed here — it addresses the opposite case, serving these models
when no projector is present. Current `main` expects an mm_proj; we have one.

## PIVOT (2026-09-11, user): dequantize to BF16, serve through NATIVE vLLM

GGUF serving is no longer the evaluation path. All four arms are dequantized to dense
BF16 HF checkpoints and served by stock vLLM with no plugin involved.

`bin/gguf_to_bf16.py` does the conversion (~9.5 min/arm, 54.7 GB out, 1184 tensors).
Two things make it correct rather than obvious:

- **transformers' own GGUF loader cannot read this model.** 5.17.0's GGUF table has
  qwen3/qwen3_moe but NOT `qwen35`, so `from_pretrained(gguf_file=...)` is not an option.
  The converter therefore reuses `vllm-gguf-plugin`'s `Qwen35GGUFAdapter.transform_weights`
  — the same code path that served the model correctly — for the Gated-DeltaNet layout
  restore (value-head retiling, `A_log -> log(-w)`, conv1d unsqueeze, in_proj_qkv value
  rows retiled while qk rows are left alone). None of that is re-derived.
- **Dequantizing BEFORE the adapter is what selects the right branch.** The adapter has a
  dense path and a packed path; the packed one SKIPS the out_proj dim=1 restore because
  reordering packed columns would corrupt them, deferring it to runtime. Feeding dense
  tensors (and emitting no `.weight_type` markers) takes the dense branch, which applies
  the full restore — correct for a checkpoint that will never see the GGUF runtime.

Faithfulness: dequantization reconstructs exactly the values the GGUF block format
encodes; it does not recover the original unquantized weights, so the quantization error
under measurement is preserved. GSQ-RCO is weight-only, so activations were always in
higher precision anyway.

**Verified three ways** before running the sweep:
1. Key set is an EXACT match against `Qwen3_5ForConditionalGeneration` — 1184 expected,
   1184 present, 0 missing, 0 extra. (Compare against the ...ForConditionalGeneration
   wrapper, not `AutoModel`: the latter is the inner `Qwen3_5Model` and legitimately
   lacks the `model.` prefix and `lm_head`.)
2. Loads natively in 51.1 GiB / 36 s, tensors bf16 and finite.
3. MMLU-Pro smoke scores **0.7142857142857143 — identical to the GGUF path's
   0.7142857142857143** on the same 14 items.

`models.json` reverted accordingly: `server_pythonpath` removed and `--hf-config-path`
dropped from `serve_flags`. The BF16 dirs are self-contained (own config.json +
tokenizer), and native vLLM already supports `qwen3_5`.

**TP=4 is a large win here and should be the default.** The QOS allocates whole 4-GPU
nodes regardless, and TP=1 left three idle. Measured on MMMU vision: GGUF+TP=1 ran
~10.5 items/min; BF16+native+TP=4 runs ~140 items/min. That is far more than the 4x from
tensor parallelism alone — native BF16 kernels beat GGUF's dequantize-on-the-fly path by
a wide margin on top of it.

## Remaining

1. **Full MMLU-Pro sweep**: running — 4 bitwidths x 12,032 items, `--repeats 1`, on
   `batch_long`. Jobs 312750/1/2/3, tags `gsq-rco-iq2xs|iq2s|iq3xxs|iq3s`.
2. **Full MMMU-Pro sweep**: running — 4 bitwidths x 1730 items (`--setting vision`,
   `--repeats 1`), jobs 313350/1/2/3, same tags, results under
   `results/qwen3.8-27b/mmmu/vision_cot/<tag>/`.
   **On plain `batch` (4 h), NOT `--long`.** The 24h batch-queue advice in
   `evals/README.md` is an artifact of the 2.4T-model evals and does not apply to this
   model (2026-09-11, user). MMMU-Pro vision is 1730 items against MMLU-Pro's 12,032, and
   every driver resumes on re-run, so a wall-clock overrun costs a resubmission, not the
   work.
3. **NVFP4 prefill / frozen GSQ-RCO decode, disaggregated** — the zero-training baseline
   for the QAD question. `bin/run_quantize.sh --scheme NVFP4` produced
   `models/Qwen3.8-27B-NVFP4` (W4A4, 19 GB, 400 quantized layers); it is paired against
   each BF16 decode arm with

   ```
   ./bin/run_eval.sh --bench mmlu_pro --model qwen3.8-27b --disagg --tp 2 --long \
       --prefill-weights $M/Qwen3.8-27B-NVFP4 \
       --decode-weights  $M/Qwen3.8-27B-GSQ-RCO-IQ3_S-bf16 --tag nvfp4p-iq3s-d
   ```

   `--disagg` alone is not enough to prove the pair worked: the driver's
   `VERDICT: PASS (kv_crossed=True, prefill_did_the_prefill=True)` line is what confirms
   the KV actually crossed rather than the decode engine quietly prefilling for itself.
   The path differs from every homogeneous run only in the weights, so
   `NIXL_ENFORCE_COMPAT=0` is set automatically (see the comment at that site — it is
   correct here and only here).

   **`--tp N` is per ENGINE, and it used to be ignored on this path.**
   `serving/run_nixl_server.sh` passed no `--tensor-parallel-size` and hardwired GPU 0 to
   prefill and GPU 1 to decode, so on a 4-GPU tray a pair ran as two TP=1 engines with
   two GPUs idle — while `serving/nixl_engine.sh` (the multi-node path) honoured TP all
   along. Fixed: TP reaches both engines, the device lists follow it (prefill `0..TP-1`,
   decode `TP..2TP-1`), and `run_eval.sh` requests `2*TP` GPUs. Verify from the engine
   logs, not from the submit banner:

   ```
   grep "\[nixl\] tp=" <logdir>/*.out            # tp=2 prefill_gpus=0,1 decode_gpus=2,3
   grep -m1 tensor_parallel_size <logdir>/prefill.log <logdir>/decode.log
   ```

4. Optionally formalize the 4 weight paths into `models.json`'s `weights` sub-dict (tags
   `gsq-rco-iq3s` etc.) so `compare_arms.py`/`report_all.py` pick them up automatically.
   Not required — the `--weights` override is sufficient.
