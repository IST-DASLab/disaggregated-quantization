# Model evaluations under disaggregated serving

Covers **Muse-Glimmer-30B** and **Qwen3.8-27B**. `models.json` is the only file that
knows anything model-specific: the checkpoint, the served name, and the serve flags
without which a model does not work at all -- Muse-Glimmer needs both of its channel
parsers, Qwen needs `--reasoning-parser qwen3` or its chain of thought lands in
`content` and gets scored as the answer. Both bash and the drivers read that one file,
so they cannot drift apart.

    ./bin/run_eval.sh --model qwen3.8-27b --bench mmmu --long
    ./bin/run_eval.sh --model muse-glimmer --bench mmmu --tag nvfp4 \
        --weights $MODELS/Muse-Glimmer-30B-NVFP4
    python3 drivers/progress.py        # what is running, how far, preliminary accuracy

### Evaluating a third-party checkpoint (e.g. the GPTQ arms)

A checkpoint someone else quantized runs like any other set of weights — the model key
supplies the serve flags, `--weights` supplies the checkpoint:

```bash
M=/lustre/fsw/portfolios/adlr/users/apanferov/models
for r in "" -r1 -r2 -r3; do
  ./bin/run_eval.sh --model muse-glimmer --bench mmmu --tag nvfp4-gptq$r --long \
      --weights $M/muse-glimmer-nvfp4a4
done
```

Four passes, because one pass has no spread and every comparison against it comes out
degenerate. Add the tag to `report.arms` in `models.json` so `compare_arms.py` picks it
up; add a contrast to `CONTRASTS` in `compare_arms.py` if it answers a question the
existing four don't.

**Verify the module set before you spend GPU time.** Comparing our RTN against their
GPTQ is an ALGORITHM comparison only if both quantize the same modules. Count the packed
weights in each and compare the NAMES, not just the totals:

```bash
srun --partition=cpu --account=adlr_psx_numerics --time=00:10:00 --ntasks=1 \
  --container-image=$CONTAINER --no-container-mount-home --container-mounts=/lustre:/lustre \
  python3 -c "
import glob
from safetensors import safe_open
def packed(d):
    ks = []
    for f in sorted(glob.glob(d + '/*.safetensors')):
        with safe_open(f, 'pt') as fh: ks.extend(fh.keys())
    return {k[:-14] for k in ks if k.endswith('.weight_packed')}
a, b = packed('$MINE'), packed('$THEIRS')
print(len(a), len(b), a == b, sorted(a ^ b)[:4])"
```

Run it **inside the container**: the login node has no `safetensors`, and a guard that
defaults the count to 0 on ImportError reports a fake mismatch that looks like a real
finding. That happened here.

This check earned itself immediately. `muse-glimmer-nvfp4a4` and
`gemma4-31b-it-nvfp4a4-GPTQ` matched ours exactly (416 and 410). `qwen3.8b-nvfp4a4-GPTQ`
did not — 400 against our 496, because it leaves the 96 Gated-DeltaNet
`in_proj_a`/`in_proj_b` gating projections in bf16. Those are 0.097% of quantized weight
elements but sit on the recurrent state-decay path, and Qwen was the model losing most
to W4A4. Without the check, "GPTQ beats RTN on Qwen" would have been partly a layer-list
difference. Our recipe now matches theirs (`models.json` ignores those two), and
`quantize_rtn.py --extra-ignore ... --suffix=-x` builds a variant to match any other
third-party selection. Pass `--suffix=VALUE` with the equals sign — argparse reads a
dash-leading value as another option.

Also worth knowing: their checkpoints may carry tensors ours don't (Gemma's ships an
explicit `lm_head` where ours relies on tied embeddings), so a size difference on disk
is not by itself evidence of a different quantization.

Results are keyed `results/<model>/<bench>/<tag>/`; without the model level two models
writing the same benchmark and tag would land in one directory and average together.

## Muse-Glimmer-30B

Disaggregated-serving evaluations of `meta-models/Muse-Glimmer-30B` on three benchmarks
from the vendor's own report: **GPQA-Diamond**, **MMMU-Pro**, and **IFBench**.

The target is a *comparison*, not a leaderboard entry: reproduce the published BF16
numbers on a single engine first, then re-run the identical harness against a
prefill/decode-disaggregated deployment and see what moves. That only works if the
single-engine run is a faithful reproduction, so most of the care here goes into not
silently diverging from the published protocol.

---

## Why these three benchmarks

The report covers more, but these three are the ones that are simultaneously (a) low
enough noise to rank two serving configurations, (b) scorable without an external LLM
judge, and (c) restartable, which a 30B reasoning model on a 4h queue requires.

| Benchmark | Items | Published BF16 | Why it survived the cut |
|---|---|---|---|
| GPQA-Diamond | 198 × N repeats | 83.5 | Deterministic scorer; repeats buy back the resolution 198 items lack |
| MMMU-Pro `vision` | 1730 | 74.0 | Exercises the vision path; the adapter is only 6.5% of parameters, so it is a text-model eval with images attached |
| IFBench | 299 | 77.0 | Programmatic constraint checkers — no judge, no ambiguity |

Deliberately excluded, and why, so these do not get re-proposed:

- **SWE-Bench Verified** — its harness builds a three-layer Docker image per instance.
  This cluster has enroot/pyxis only; there is no Docker, Podman, Apptainer, or
  Charliecloud, so the containerised execution the benchmark depends on cannot be run.
- **HLE** — scored by an LLM judge.
- **AIME-2025 / LiveCodeBench** — not in the published report, so there is no number to
  reproduce; AIME is 30 items (3.3 points each) and cannot rank anything.

MMMU-Pro was originally excluded as multimodal and re-admitted after measuring the
vision tower: **1.92B of 29.78B parameters, 6.5%** (3.6 GiB). A quantization or serving
change to the text stack is therefore still the dominant effect on this benchmark.

**Which MMMU-Pro subset the published 74.0 refers to is not stated.** The report says
only "1730 multiple choice questions … the answer choice space has been significantly
expanded", which does not disambiguate — `vision` and `standard (10 options)` are each
1730 items and both derive from the 10-option construction — and it sources the number
from Artificial Analysis rather than measuring it. The only evidence available is
measurement, and it favours `vision`: 73.4 against standard10's 72.2. The report is
treated here as vision-only, and **`vision` is the only subset run from now on**.

The report also specifies protocol details worth matching: IFBench and MMMU-Pro results
are **averaged over 4 runs**, which is what the repeat tags and `--repeats 4` do here.
Two small discrepancies are recorded rather than chased: the report calls IFBench "294
tasks" where the pinned test file has 300 unique prompts, and it reports "mean pass rate
across tasks" without saying strict or loose.

---

## Setup (once)

```bash
./bin/import_container.sh        # on a CPU node -- see the header, it will not work on a login node
./bin/setup_harnesses.sh         # on a login node -- needs network
```

`setup_harnesses.sh` pins the third-party harnesses (IFBench `db69a6f`, MMMU `268471d`),
downloads GPQA and MMMU-Pro, builds the pip overlay *inside the container*, and
pre-fetches the nltk corpora. The harness checkouts and the overlay live on lustre, not
in this repo; the pinned commits are what make them reproducible.

The model is a plain HF checkpoint at
`/lustre/fsw/portfolios/adlr/users/apanferov/models/Muse-Glimmer-30B` (56 GB).

## Running

```bash
./bin/run_muse_eval.sh --bench gpqa                          # 4 repeats x 198
./bin/run_muse_eval.sh --bench ifbench
./bin/run_muse_eval.sh --bench mmmu --long                    # vision, batch_long 24h
./bin/run_muse_eval.sh --bench gpqa --limit 4 --repeats 1 --tag smoke --interactive
```

`--interactive` uses the interactive QoS (priority 700 vs normal's 100) — for smoke
tests, where queue wait dominates the work. Not for real sweeps, which hold a GPU for
hours and should not jump the queue.

`--disagg` replaces the single engine with a **1P1D pair**: a prefill engine and a
decode engine on separate GPUs exchanging KV over vLLM's NixlConnector, fronted by one
OpenAI endpoint. The serving stack is `qad/serving/run_nixl_server.sh` +
`nixl_proxy.py`, reused rather than forked — every non-default setting in it
(`kv_buffer_device=cpu` because this container's UCX has no CUDA support,
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `UCX_TLS=all`, `kv_load_failure_policy=fail`) was
established the hard way and a copy would drift. The model-specific flags reach it
through `--extra-serve-args`.

```bash
./bin/run_muse_eval.sh --bench gpqa --disagg --tag disagg    # 2 GPUs
```

One job boots its own server, drives one benchmark, scores, and tears down. Re-running
the identical command **resumes**: every driver appends and fsyncs per item and reloads
what it already has. `--tag` names the configuration under test (`bf16`, later the
disaggregated arms) and is what keeps two configurations out of each other's files.

Results land in `muse/results/<bench>/<tag>/`; `summary.json` is versioned, raw
generations are not (see `.gitignore`). Logs go to `logs/muse/`.

---

## Results (2026-08-26)

`python3 drivers/compare_arms.py` regenerates all of this. `drivers/progress.py` shows
what is running.

MMMU-Pro `vision`, all 1730 items, 4 independent passes per arm, vendor sampling
settings. GPQA and IFBench are still collected but not reported: at 198 and 300 items
they cannot resolve a 1-2 point difference, and averaging them in destroys MMMU's signal
rather than adding to it.

| method | prefill | decode | Muse-Glimmer-30B | Qwen3.8-27B | Gemma-4-31B-it |
|---|---|---|---|---|---|
| BF16 | BF16 | BF16 | **72.90** ±0.60 * | **74.78** ±0.33 | **66.84** ±0.22 |
| `nvfp4` | W4A4 | W4A4 | 70.38 ±0.30 | 69.70 ±0.83 | 64.81 ±0.59 |
| `nvfp4a16` | W4A16 | W4A16 | 71.81 ±0.61 | 70.69 ±0.85 | 65.36 ±0.25 |
| `nvfp4pd` | **W4A4** | **W4A16** | 71.34 ±0.44 | 71.17 ±0.97 | 65.94 ±0.36 |

\* Muse's BF16 reference is `bf16-pd` (disaggregated, 4 passes); the other two use
single-engine `bf16`, with `bf16-pd` run once as a sanity check that disaggregation
serves the architecture at all. Disaggregation itself is accuracy-neutral: measured on
Muse at −0.06 [−1.85, +1.73].

### The contrasts, Holm-corrected within each model

| contrast | Muse-Glimmer | Qwen3.8-27B | Gemma-4-31B |
|---|---|---|---|
| cost of W4A4 (BF16 − nvfp4) | +2.53 ✗ | +5.09 ✗ | +2.02 ✗ |
| cost of mixed (BF16 − nvfp4pd) | +1.56 ✗ | +3.61 ✗ | +0.90 ~ |
| **NVFP4PD recovers (nvfp4 − nvfp4pd)** | **−0.97 ✗** | **−1.47 ✗** | **−1.13 ✗** |
| **mixed vs W4A16 (nvfp4a16 − nvfp4pd)** | **+0.46 ~** | **−0.48 ~** | **−0.58 ~** |

✗ = rejects at 96% after Holm; ~ = not resolved.

**The result replicates on three unrelated architectures.** W4A4 on both phases costs
2.0–5.1 points everywhere. NVFP4PD recovers a significant part of that on every model,
and is statistically indistinguishable from W4A16 on every model — so the mixed format
buys W4A16-grade accuracy while serving the compute-bound prefill half in W4A4. On Gemma
it is not even distinguishable from BF16 (+0.90, p=0.27).

Sensitivity to 4-bit varies about 2.5× across architectures (Qwen −5.1, Muse −2.5,
Gemma −2.0), so no single model's number generalises.

### How the tests work

Both arms answer the identical items, so the evidence is in the items where they
disagree and `SE(δ) = sqrt(flip_rate/N)` — measured at ~0.45 against 1.51 for the
independence formula. That is the difference between "n=1730 cannot resolve a 1.5-point
gap" and most pairs resolving. Following arXiv 2602.10144.

The primary test is an **exact paired sign-flip test**: with k repeats every per-item
difference is a multiple of 1/k, so the null distribution is a convolution of at most k
binomials and is computed exactly rather than sampled. It assumes only that the two arms
are exchangeable per item under the null, and that items are independent — checked by
clustering on the 30 subjects, design effects 0.47–1.27×, i.e. no material clustering.
It does *not* assume normality or equal variance, and it stays exact despite each δᵢ
being estimated from only 4 repeats.

Exact per-repeat McNemar is reported as a deliberately conservative secondary. It
assumes one score per example, so with 4 repeats it either has to pool (treating 6920
correlated observations as independent) or demand all four reject (discarding the
precision the repeats bought). The latter is what is reported.

More repeats have a floor: `Var(δᵢ)` splits into a between-item term that only more
ITEMS can reduce and a within-item term that shrinks as 1/k. At k=4 roughly half the
variance is already the irreducible item term, so k→∞ buys only 1.5–2.2× in SE.
Resolving `nvfp4a16 vs nvfp4pd` at its ~0.5-point gap would take ~19 repeats — feasible,
but an equivalence test against a declared margin would answer that question better.

---

## Things that will bite

**A too-small token budget scores as a weak model.** Muse-Glimmer emits channel-scoped
messages: the chain of thought goes to `reasoning_content` and the answer to `content`.
When the budget runs out mid-chain the request returns `finish_reason="length"` with
**empty `content`** and no error at all. Measured floor: a one-sentence answer to
"write one short sentence about the sea" consumes ~190–225 tokens, essentially all of it
reasoning; at `max_tokens=40` the same request returns empty. Hence `--max-tokens 32768`
and `--max-model-len 65536` by default, and `finish_reason` / `completion_tokens`
recorded on every row so truncation is a number in `summary.json` rather than an
invisible pile of wrong answers.

**Sampling must come from the server, not the harness.** The vendor settings are
`temperature 1.0 / top_p 0.95 / top_k 64`, applied by `--generation-config auto`. Every
driver here sends `max_tokens` and nothing else. Both upstream harnesses would otherwise
override them — IFBench's generator sends `temperature 0.6` and a fixed seed. The
launcher greps the server log for the line vLLM prints when it applies the model's
generation config and **exits non-zero if it is missing**, because the failure mode is
otherwise a score that is merely slightly off.

**A disaggregated run that silently isn't one.** If the KV never crosses, the decode
engine just recomputes the prompt with its own weights: the request succeeds, the text
is fluent, and the number reported is really homogeneous-decode. Nothing logs a fallback.

The A/B/C attribution test used for the quantized pairs in `qad/serving/DISAGG.md`
**cannot work here** — both engines hold identical BF16 weights, so local recompute and a
successful transfer are distributionally identical and no amount of reading completions
separates them. What does separate them is *where the prefill work happened*, which the
engines report directly:

```
decode : Avg prompt throughput: 0.0 tokens/s ... External prefix cache hit rate: 100.0%
prefill: Avg prompt throughput: 74.5 tokens/s ... Running: 0 reqs
```

`drivers/verify_disagg.py` parses both engine logs and requires all three: the decode
engine's external prefix-cache hit rate at ~100%, the prefill engine actually doing
prefill work, and the decode engine doing less prompt work than the prefill engine. It
writes `disagg_verification.json` beside the scores and exits non-zero otherwise, so a
mislabelled run cannot end quietly. Two structural guards back it up — the proxy returns
502 if the prefill leg came back without `kv_transfer_params`, and the engines run
`kv_load_failure_policy=fail` so an attempted-but-failed transfer raises rather than
falling back to local recompute.

**The disagg stack changes more than disaggregation.** `run_nixl_server.sh` pins
`VLLM_ATTENTION_BACKEND=FLASH_ATTN` (the host-buffer staging path indexes the KV tensor
in a layout FLASHINFER's HND does not use, and the mismatch shows up as a device-side
assert mid-request) and passes `--no-async-scheduling`. The single-engine arm takes
vLLM's defaults, which here means FLASHINFER. So a naive bf16-vs-disagg delta contains
both effects, not one.

Rather than assume the backend is immaterial, there is a third arm: single-engine run
with `VLLM_ATTENTION_BACKEND=FLASH_ATTN` in the environment (`--export=ALL` carries it
through sbatch), tagged `bf16fa`. If it agrees with `bf16`, the delta is attributable to
disaggregation; if it does not, `bf16fa` is the correct baseline and `bf16` is measuring
the backend.

**Both `muse_glimmer` parsers are required together.** `--tool-call-parser muse_glimmer`
and `--reasoning-parser muse_glimmer`; the model emits channel-scoped messages rather
than JSON tool calls or tagged reasoning, and one parser without the other mis-parses
every turn.

**The pip overlay must not reach the server.** It carries its own `huggingface-hub` and
`numpy`; putting it on the server's `PYTHONPATH` shadows the container's copies and vLLM
refuses to start. Only the client process gets it.

**The container must be imported on a CPU node.** Login nodes cap at 24 CPUs / 176 GB,
and `mksquashfs` spawning one compressor per core dies there with
`Out of memory (cache_alloc)`. `ENROOT_TEMP_PATH` also has to be node-local — Lustre
cannot create the overlay whiteouts enroot needs, and the failure silently produces no
image.

---

## What is in here

| Path | Role |
|---|---|
| `bin/import_container.sh` | vLLM nightly → `.sqsh` (CPU node) |
| `bin/setup_harnesses.sh` | pinned harness checkouts, datasets, pip overlay, nltk corpora |
| `bin/run_muse_eval.sh` | one job = serve + evaluate + score + tear down |
| `drivers/common.py` | shared client: per-item durability, no client-side sampling |
| `drivers/gpqa_diamond.py` | simple-evals prompt + `Answer: $LETTER` extraction, N repeats |
| `drivers/ifbench_gen.py` | generation only; scoring is IFBench's `run_eval`, unmodified |
| `drivers/ifbench_report.py` | strict/loose scores + per-constraint breakdown → `summary.json` |
| `drivers/mmmu_pro_infer.py` | MMMU-Pro prompts (imported from the harness) over the local endpoint |
| `drivers/mmmu_pro_score.py` | MMMU-Pro's own answer parser, applied to partial files too |
| `drivers/verify_disagg.py` | proves from the engine logs that the KV actually crossed |
| `drivers/report_all.py` | every `summary.json` in one table, with coverage and truncation |

Each driver's module docstring says what it replaces upstream and why; those reasons are
the non-obvious part and are not repeated here.
