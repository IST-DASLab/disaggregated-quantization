# Gemma-3 support for QAD + evals — plan and progress

Goal: QAD-train and evaluate **gemma-3-270m-it, 1b-it, 4b-it, 12b-it**, reusing the
existing pipeline rather than forking it. Text performance only — the vision tower is out
of scope. Branch: `gemma`.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/needs a decision

---

## 1. Verified facts

All measured against transformers **5.3.0** in the training container and the real HF
configs (network works from compute nodes; `HF_HOME` is the shared cache).

| model | architecture | hidden | layers | vocab | tie | heads/kv | head_dim | inter | sliding |
|---|---|---|---|---|---|---|---|---|---|
| gemma-3-270m-it | `Gemma3ForCausalLM` | 640 | 18 | 262144 | yes | 4/1 | 256 | 2048 | 512 |
| gemma-3-1b-it | `Gemma3ForCausalLM` | 1152 | 26 | 262144 | yes | 4/1 | 256 | 6912 | 512 |
| gemma-3-4b-it | `Gemma3ForConditionalGeneration` | 2560 | 34 | 262208 | yes | 8/4 | 256 | 10240 | 1024 |
| gemma-3-12b-it | `Gemma3ForConditionalGeneration` | 3840 | 48 | 262208 | yes | 16/8 | 256 | 15360 | 1024 |

* **Attention is 5:1 sliding:full.** `layer_types` alternates five `sliding_attention`
  layers to one `full_attention` (3/18, 4/26, 5/34, 8/48 full layers respectively).
* **Checkpoint key layout differs by size.** 270m/1b: `model.*`. 4b/12b:
  `language_model.model.*` plus `vision_tower.*` and `multi_modal_projector.*` (439 of the
  4b's 883 tensors are vision). **No `lm_head` tensor in any of them** — the head is tied
  to the embedding.
* **Module names match Qwen3** where it matters: `q_proj/k_proj/v_proj/o_proj`,
  `gate_proj/up_proj/down_proj`. So `FUSED_GROUPS` and `replace_linears(skip=("lm_head",))`
  transfer unchanged.
* **Four norms per layer, not two**: `input_layernorm`, `post_attention_layernorm`,
  `pre_feedforward_layernorm`, `post_feedforward_layernorm` (Gemma-2 sandwich norms).
* **Chat template**: `<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n…<end_of_turn>\n`.
  There is **no system role** — a system message is silently prepended to the first user
  turn. `role="assistant"` is accepted and renders as `model`.
* **Every dimension is divisible by 8** (DistAdamW's ZeRO-2 shard constraint) and by 16
  (the quantization block size): hidden 640/1152/2560/3840, inter 2048/6912/10240/15360,
  q_proj out 1024/1024/2048/4096, kv out 256/256/1024/2048, vocab 262144/262208.

---

## 2. Conceptual incompatibilities with how we do QAD

Ordered by risk. These are the reason this is not a one-line `MODEL=` change.

### 2.1 `Gemma3RMSNorm` is `(1 + weight)` with a **zeros** init — HIGH
```python
output = self._norm(x.float())
output = output * (1.0 + self.weight.float())     # note the +1, and the fp32 multiply
```
Any code that assumes `y = w * normed` is wrong here, and any code that assumes a norm
weight is O(1) is wrong twice (Gemma's are O(0)). This directly breaks
`full_disag.DualRMSNorm`. **Mitigated by deprecating `--full-disag`** (see 2.7), which
removes the only place we touch norms. Nothing else in QAD reads norm weights.

### 2.2 The 4b/12b wrapper is multimodal — HIGH
`qad.py` assumes `student.model` is the base transformer and `student.lm_head` the output
projection. For `Gemma3ForConditionalGeneration`, `model` is a `Gemma3Model` holding
`vision_tower` + `multi_modal_projector` + `language_model`, so:
* `student.model(input_ids=...)` runs the multimodal path, not the text stack.
* `replace_linears` would quantize the **vision tower's 12 linears per layer** as well.
* the exported checkpoint would carry a vision tower nothing serves.

**RESOLVED — and NOT by rehoming the text stack.** Rehoming the language model into a
`Gemma3ForCausalLM` shell works for training but breaks the checkpoint:
`config.text_config` serializes with `architectures: None` so the export needs hand
patching to stay servable, and the tied head then needs hand untying to be writable.

The approach taken instead (`training/models.py`): load the checkpoint as the class it
*declares itself to be*, change nothing about the module tree or the config, and resolve
the text stack through an explicit per-architecture table
(`_TEXT_PATHS = {"Gemma3ForConditionalGeneration": ("model.language_model", "lm_head")}`).
`save_pretrained`/vLLM semantics stay stock. The vision tower stays loaded and unused
(~0.42B params at 4b) — that is the price of never touching the architecture.

**Consumers must scope to `text_stack(model).base`.** Walking the whole model is wrong in
ways that do not raise: the first `nn.Embedding` in `named_modules()` order is the SigLIP
*position* embedding, not `embed_tokens`, and 163 of the wrapper's 401 `nn.Linear`s are
vision. `apply_full_disag` takes the first match and would have wrapped the wrong one.

#### 2.2b The wrapper ALSO cannot be served — `model_type` decides the attention backend
Measured 2026-08-17 (job 530471), and it forces a decision we had deferred. Serving stock
4b under `--kv-transfer-config` dies at engine init:

```
ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this
configuration. Reason: ['partial multimodal token full attention not supported']
```

`ModelConfig.is_mm_prefix_lm` (`vllm/config/model.py:1125`) returns True when
`hf_config.model_type in ("gemma3", "paligemma")` — it keys on **model_type alone**, not on
whether a vision tower is present or any image is ever passed. That sets `use_mm_prefix`
(`attention/layer.py:228`), and only **two** backends implement `supports_mm_prefix()`:
`flex_attention` and `triton_attn`. FLASH_ATTN and FlashInfer both refuse.

| model | `model_type` | architecture | FLASH_ATTN |
|---|---|---|---|
| 270m, 1b | `gemma3_text` | `Gemma3ForCausalLM` | works |
| 4b, 12b | `gemma3` | `Gemma3ForConditionalGeneration` | **rejected** |

So the two halves of the family have *contradictory* backend requirements: 270m/1b REQUIRE
FLASH_ATTN (FlashInfer asserts on head_dim=256 + block_size 16, breakage 1 below), while
4b/12b REJECT it. No single pinned backend serves all four as shipped.

**Resolution: 4b/12b must be served as a TEXT-ONLY export** carrying
`model_type: gemma3_text` + `architectures: ["Gemma3ForCausalLM"]` — i.e. structurally the
same shape as the stock 1b repo. Then `is_mm_prefix_lm` is False, FLASH_ATTN is valid, and
one backend serves the whole family. Three reasons this is right rather than a workaround:
1. The QAD checkpoint is text-only **by construction** (we train `text_stack(...).base`).
   Serving stock multimodal BF16 as the "baseline" would make baseline-vs-quantized differ
   in architecture AND attention backend, not just quantization. That is not a control.
2. It drops a 0.42B (4b) / ~1.2B (12b) vision tower that is never used.
3. The alternative — TRITON_ATTN or FLEX_ATTENTION for 4b/12b only — leaves the family
   split across backends permanently.

NOTE this is exactly the hand-patch §2.2 above flagged: `config.text_config` serializes
with `architectures: None`, so the export must set `architectures: ["Gemma3ForCausalLM"]`
explicitly. That cost is unavoidable — it is now paid at EXPORT time (where it is a
two-field config fix) rather than at LOAD time (where it corrupted the weight load).
`training/models.py` stays exactly as it is; this changes only `export/save.py`.

### 2.3 Sliding-window attention vs disaggregated serving — RESOLVED: safe by fallback
Five of every six layers use a 512/1024-token sliding window. Investigated against the
pinned vLLM clone at **d7de043d55**, byte-verified identical to the container's installed
`0.14.2.dev0+gd7de043d5`, so this is the code our jobs actually run.

**It works, but by accident rather than design.** Setting `--kv-transfer-config`
force-disables the hybrid KV cache manager (`vllm/config/vllm.py:979-995`), which rewrites
every `SlidingWindowSpec` into `FullAttentionSpec(sliding_window=N)`
(`kv_cache_utils.py:1179-1188`). All layers then share one spec → one KV cache group, no
rolling eviction (`single_type_kv_cache_manager.py:377-388`), and NixlConnector takes its
"assume global attention" path and ships every block of every layer
(`nixl_connector.py:2234-2244`) — a correct superset of what the windowed kernel reads.
Masking still happens at compute time (`gpu_model_runner.py:1925-1928`).

The connector itself has **no** sliding-window awareness: `block_window_per_layer`, the
only window-aware transfer path, is populated for `model_type == "llama4"` only
(`nixl_connector.py:1390-1410`). Our correctness rests entirely on that HMA fallback
engaging, which is why task 3.2 now asserts the startup log lines that prove it did.

**Cost: KV memory blows up.** With HMA off, the 5/6 sliding layers allocate
`max_model_len` instead of `window` — roughly 3-4x more KV at 8k context with a 1024
window. Expect to lower `--max-model-len` or raise `--gpu-memory-utilization`; an OOM is
the expected first symptom, not a bug.

**The handshake hash is irrelevant to us — do not reason about it.** Our prefill and
decode halves are deliberately *different quantizations of the same model*: that is the
entire point of the format. The compat hash includes the checkpoint path, so it can never
match for any DQ format, which is exactly why `enforce_handshake_compat: false` is already
set (`serving/run_nixl_server.sh`). Whether the hash covers the sliding window or not
changes nothing for us, because the check never runs. (For the record it does not, despite
the docstring at `nixl_connector.py:170` claiming otherwise — but that is an upstream
documentation bug, not a risk we carry.)

What DOES remain active and is our real safety net: the structural handshake asserts at
`nixl_connector.py:1717-1756` (all remote layers share one block length, block lengths
match modulo the TP ratio, region/layer counts match). Those are not disabled by
`enforce_handshake_compat: false`.

**No upstream test coverage.** `tests/v1/kv_connector/nixl_integration/` tests only
`Qwen/Qwen3-0.6B`; grep for sliding_window/hybrid/Mamba across that tree returns nothing.
So we are the first to exercise this combination and must gate it empirically.

### 2.3b Hybrid / Mamba models — hard startup failure (safe)
`MambaSpec` appears nowhere in `kv_cache_utils.py`, so unification cannot collapse a
FullAttention+Mamba mix and `kv_cache_utils.py:1198-1205` raises. The engine refuses to
start rather than transferring garbage. Upstream hybrid-SSM disagg landed in v0.20.0+,
far after our version — so Gemma-3n or any Mamba hybrid is out of scope until we upgrade.

### 2.4 262k vocab with tied embeddings — MEDIUM
The embedding is 262144×640 = 168M params on a 270M model, i.e. most of it, and it is
tied to the head. Consequences:
* `LigerFusedLinearJSDLoss` materializes logits in chunks over a 262k vocab — memory
  per chunk is 4× Qwen3's 152k. May need a smaller `chunk_size`.
* Quantizing only linears leaves the embedding in bf16, and at the small end that is
  most of the model. Measured (non-embedding linear params vs tied embedding):

  | model | linear params | embedding | embedding share |
  |---|---|---|---|
  | 270m | 0.100e9 | 0.168e9 | **63%** |
  | 1b | 0.698e9 | 0.302e9 | 30% |
  | 4b | 3.209e9 | 0.671e9 | 17% |
  | 12b | 10.758e9 | 1.007e9 | 9% |

  So at 270m the embedding is *larger than everything we quantize*, and the achievable
  footprint reduction is bounded at ~37% no matter how few bits we use. State this up
  front rather than presenting a disappointing pareto point later.
* No `lm_head` tensor exists in the source checkpoints. Our exporter writes both
  `embed_tokens.weight` and `lm_head.weight` (verified on Qwen3 4b, which is also tied);
  need to confirm vLLM accepts that for Gemma3.

### 2.5 Tooling hardcodes `Qwen-Qwen3-` — MEDIUM, mechanical
* `cluster_scripts/submit_missing_evals.py`: `f"{RUN}-Qwen-Qwen3-{model}-{quant}-*"` (×2)
  and `f"Qwen/Qwen3-{model}"`.
* `bin/run_qad.sh`: `sed 's/^Qwen-//'` when building the job name.
* `notebooks/plots.ipynb`: `_METHOD_RE = ^{RUN}-(Qwen-Qwen3-[^-]+)-…`, plus
  `MODEL_ORDER`, `LINEAR_PARAMS`, `MODEL_COLOUR`.
* Defaults `MODEL=${MODEL:-Qwen/Qwen3-*}` in most `bin/*.sh` — harmless, but they mean a
  forgotten `MODEL=` silently trains the wrong family.

### 2.6 Attention implementation — LOW, verify
`qad.py` hardcodes `attn_implementation="flash_attention_2"`. Gemma-3 sliding window is
supported by FA2 in transformers 5.3, but confirm at load rather than assume.

### 2.7 `--full-disag` is deprecated
Per user: it did not work out. Deprecating removes 2.1 as a blocker and removes the
`DualRMSNorm`/`DualEmbedding`/`DualLMHead` surface from the Gemma port entirely. Keep the
code and tests for the record; mark the registry/flag as deprecated so nobody wires it
into a Gemma sweep.

### 2.8 Non-issues (checked, no action)
* **Label masking is template-agnostic.** `training/data.py` uses a prefix-diff over
  `apply_chat_template`, so it needs no Gemma-specific markers.
* **No double-BOS.** Tokenization uses `add_special_tokens=False`, so the template's
  literal `<bos>` is tokenized exactly once.
* **Projection names and fused groups** are identical to Qwen3.
* **Shard divisibility** holds for every dimension at every size.

---

## 3. Plan

### Phase 0 — decisions (defaults chosen; override if you disagree)
- [x] **0.1** `RUN_PREFIX=gemma3` (Qwen stays `qad3x`). Tags become
      `gemma3-google-gemma-3-1b-it-<quant>-<hash>`: never collides with a Qwen glob, lets
      the notebook switch families by changing one constant, and keeps every existing
      directory and script working — **no new result trees**. This is the "reasonable
      separation" requirement, satisfied without forking the layout.
- [ ] **0.2** Formats. Default: `nvfp4` + `nvfp4a16` first to validate the path end to
      end, then fan out to the LUT/dual families that matter for the paper.
- [ ] **0.3** 12b in scope? It is 1.5× Qwen-8B, which already needs a 3-deep chain.
      Default: defer 12b until 270m/1b/4b are green.

### Phase 1 — model loading (text-only, all four sizes)
- [x] **1.1** `training/models.py`: `load_model()` loads the declared class with no
      surgery; `text_stack()` resolves `(base, head, embed)` through `_TEXT_PATHS` and
      raises on an unknown wrapper rather than guessing. 25/25 checks pass, including the
      anti-regression test that `Gemma3ForCausalLM.from_pretrained(4b)` IS detectably
      wrong. NOTE: resolving via `AutoModelForCausalLM` is itself the trap — `gemma3` maps
      to `Gemma3ForCausalLM`, which loads the 4b repo half-random without complaining.
- [ ] **1.2** Use it in `qad.py` (student + teacher) and in `eval/eval_transformers.py`.
- [ ] **1.3** Verify FA2 loads for all four sizes; fall back to `sdpa` if not.
- [ ] **1.4** Unit test: all four sizes load, expose `.model`/`.lm_head`, have no vision
      modules, and have the expected layer/linear counts.

### Phase 2 — quantization correctness
- [ ] **2.1** Confirm `replace_linears` + `FUSED_GROUPS` cover exactly the text linears
      and skip `lm_head` (count them per size).
- [ ] **2.2** Export one checkpoint per size at step 0 and confirm it round-trips
      (`test_export`-style: packed NVFP4 == `_wq`).
- [ ] **2.3** Confirm the tied embedding/head export produces something vLLM loads —
      this is where the Qwen path silently lost `quantization_config` once before.
- [~] **2.4** **TEXT-ONLY EXPORT for 4b/12b — BLOCKS ALL 4b/12b WORK, serving and QAD.**
      IMPLEMENTED and config-verified at both sizes via
      `cluster_scripts/export_text_only.py`, which exports a BF16 text-only checkpoint
      from a stock repo with no training. 4b -> 445 tensors, 12b -> 627, both with
      `model_type: gemma3_text`, `architectures: ["Gemma3ForCausalLM"]`, no
      `vision_config`, no `vision`/`multi_modal`/`language_model` tensor names, and
      `rope_local_base_freq` present. `verify_text_load` passed on both, so these are the
      checkpoint's weights and not a half-random load.
      **The acceptance test — vLLM actually serving one — is job 532091 and still
      queued.** Until it returns, the export is verified by construction only, which is
      exactly the leniency gap that let the rope defect through (a transformers round-trip
      accepts shapes vLLM rejects). Do not treat 2.4 as closed on config evidence.
      That job doubles as the 4b BF16 baseline: the baseline CANNOT be the stock repo,
      because a stock-multimodal baseline vs a text-only quantized model differs in
      architecture and attention backend on top of quantization.
      ORIGINAL REQUIREMENT, for reference:
      See §2.2b. `export/save.py` must write `model_type: gemma3_text` and
      `architectures: ["Gemma3ForCausalLM"]` (the `text_config` serializes with
      `architectures: None`) and must NOT emit `vision_config`, the vision tower, or the
      multimodal projector. Acceptance: vLLM serves the export under FLASH_ATTN with
      `--kv-transfer-config`, and its config is shape-identical to the stock 1b repo's.
      Note this is needed for the BF16 baseline too, not just for quantized checkpoints —
      a stock-multimodal baseline vs a text-only quantized model is not a control.

### Phase 3 — serving gate, ON STOCK BF16 (independent of QAD entirely)
The whole sliding-window question is answerable with `--unquantized`, which serves the
stock BF16 model on both engines. No quantizer, no checkpoint, no training — and it
produces the BF16 baselines the plots need anyway. Do this BEFORE any QAD work.

- [x] **3.0** BF16 disagg canary, `google/gemma-3-1b-it --unquantized --think` (job 530197):
      the stack came up, ran both benchmarks and wrote results. **The HMA-off fallback
      engaged** (both warnings present, GPU KV cache 6,341,968 tokens, 157 GiB available —
      no memory pressure at 1b). Scores: GSM8K `exact_match,flexible-extract` **44.4%**,
      MATH-500 `math_verify,none` **41.8%** — plausible for BF16 gemma-3-1b-it, so the
      disaggregated stack produces sane Gemma output.
- [x] **3.1** **Monolithic vs disaggregated on the SAME BF16 model** (job 530482,
      `run_eval_vllm.sh --unquantized`). This is the actual verification: 3.0 proves the
      stack runs, not that KV transfer is correct — a partly-broken transfer would still
      give a plausible-looking 44%. Single-engine and disaggregated must agree.
      **PASSED at 1b**: all three metrics within one combined stderr (see progress log).
      Rules out gross corruption; does NOT substitute for 3.2's identity oracle.
- [ ] **3.2** **Sliding-window KV gate.** Our existing A/B/C differential proves a
      transfer *happened*, not that it was *correct* — a wrong per-layer block mapping
      still makes A differ from both B and C, so it would report PASS on corrupt KV. For
      Gemma that gap matters. Run, in order:
      1. **Identity oracle** (new, and the one that actually proves correctness): serve the
         SAME checkpoint on prefill and decode, greedy, and require the output to be
         **token-identical** to the monolithic run. Any divergence is direct proof of KV
         corruption, with no weight confound.
      2. **Long prompts.** Below the window every SWA layer behaves exactly like full
         attention and the window logic is never exercised — a green test would prove
         nothing. Use >=4x the window: >2048 tokens at 270m/1b, >4096 at 4b/12b. Our Qwen
         prompts are far too short to carry over.
      3. The existing A/B/C differential, with those same long prompts.
      4. **Assert the fallback engaged**, on both engines' logs: "Turning off hybrid kv
         cache manager because `--kv-transfer-config` is set" (`config/vllm.py:984`) and
         "Hybrid KV cache manager is disabled for this hybrid model"
         (`kv_cache_utils.py:1163`), plus exactly one KV cache group. If those are absent,
         stop — the basis for 2.3's safety verdict does not hold.
      5. **Preflight config diff** of the two exported halves: same `sliding_window` and
         `layer_types`, neither with `--disable-sliding-window`, identical
         dtype/backend/max-model-len, P TP == D TP. Note this guards against OUR export
         path, not against vLLM — the two halves come from one source config so they agree
         by construction, but `save_checkpoint` has silently dropped config fields before
         (the `quantization_config` incident), so it is cheap insurance on our own code.
      **Blocking for Phase 5.**

### Phase 4 — tooling generalization
- [x] **4.1** `submit_missing_evals.py`: `FAMILIES` table + `--family`; the three
      hardcoded `Qwen/Qwen3` strings are gone. Both families dry-run clean.
      `MODES` is now per family in that same table (`gemma3` is think-only) — it was a
      module-level constant, so a Gemma sweep would have queued a full set of `nothink`
      jobs that all die at `eval_disagg.py`'s probe guard. Qwen keeps `think`+`nothink`,
      asserted.
- [x] **4.1b** `autoeval_watch.sh` counted training jobs with `grep -c '^qad-Qwen3'`.
      Training jobs are named `qad-<model>-<quant>` from `cut -d/ -f2`, so every Gemma run
      is `qad-gemma-3-4b-it-<quant>` and counted as ZERO — the watcher would conclude
      "training finished" and exit in the middle of a live Gemma sweep, while the
      complementary `grep -vc` counted those same training jobs as eval jobs. Now
      classified by EXCLUDING the four eval job names
      (`qad-eval|qad-dual-eval|qad-disagg|qad-vllm`), which is family-agnostic. Arguments
      are forwarded to `submit_missing_evals.py`, so `--family gemma3` works; one watcher
      covers one family, run a second for the other.
- [x] **4.2** `run_qad.sh`: job name is now `cut -d/ -f2` (vendor-agnostic). Qwen names
      are byte-identical to before; Gemma gets `gemma-3-4b-it` not `google-gemma-3-4b-it`.
- [x] **4.3** `notebooks/plots.ipynb` is family-agnostic. `RUN` -> `RUNS = ["qad3x",
      "gemma3"]`, both loaded into ONE grid with each (family, size) its own row.
      `_METHOD_RE`/`_BASELINE_RE` no longer assume `Qwen-Qwen3-[^-]+` (the Gemma model
      segment CONTAINS hyphens, so Gemma results were invisible rather than reported
      missing); `_size()` handles `270m -> 0.27`; `short_name()` replaces the scattered
      `.replace("Qwen-", "")`; `LINEAR_PARAMS`/`MODEL_COLOUR` carry all four Gemma sizes.
      The load-bearing change is `tasks_for(model)`: averaging is now per-model, because
      `avg_accuracy` only averages steps where EVERY plotted benchmark has a value — with
      the global task list a think-only family got an EMPTY Average curve, empty bars and
      silently vanished from the Pareto. Verified: all 12 code cells execute; a synthetic
      Gemma arm produces a full 10-point Average curve and renders in both the grid and
      the Pareto; and Qwen `models`/`baselines`/`FALLBACK`/`MODEL_ORDER` are **identical**
      to the pre-change notebook.
      Cross-family caveat recorded in the code: a Gemma Average is over think-only columns
      while a Qwen Average is over both modes, so the two Average panels are not averages
      of the same column set. Fine within a family; do not read one against the other.
- [ ] **4.4** Sweep the `bin/*.sh` defaults: every script that defaults `MODEL` should
      also default `RUN_PREFIX` consistently, so a forgotten flag fails loudly rather
      than training the wrong family into the wrong tag.

### Phase 5 — train and evaluate
- [ ] **5.1** Smoke run per size (`--train-tokens 400000 --debug`), verifying export.
- [ ] **5.2** Full QAD at 270m/1b/4b (+12b per 0.3). Chain depth from MEASURED runtime,
      not from caution: **270m needs no chain** — 2532 steps in 1h51m against a 4h cap,
      and the two-master split formats do not come close to closing that gap. A chain link
      is not free: it sits PENDING on `--dependency=singleton` holding a queue slot for the
      entire duration of the run it follows. Same conclusion previously reached for Qwen
      0.6B.
      **1b needs no chain either — measured**, 525 steps in 21 min including startup, i.e.
      ~1h40m for 2532 steps. Essentially 270m's wall clock despite 7x the linear params,
      because the 262k-vocab LM head dominates the step at both sizes. Parameter count is
      a bad predictor here; measure, do not extrapolate. Chains cancelled at both sizes.
      4b is the first size where one is plausibly needed (submitted with `--chain 2`);
      measure it too before extending to 12b.

      **NEVER CHAIN A CONFIGURATION THAT HAS NOT YET RUN.** Chains are for a run that is
      known to work and merely needs more wall clock. `--dependency=singleton` starts each
      link only after the previous one dies, so a config that fails — OOM at a new size,
      a bad flag — fails once per link, burning an allocation each time at the same place;
      a link that resumes from `state/` will crash at the identical step. Order is:
      submit ONE job, confirm it survives startup and reports a step rate, then decide the
      chain depth from that measurement. 12b was submitted `--chain 3` on a 3.35x
      parameter-ratio guess before anything at that size had ever run — wrong on both
      counts, since memory (12b is ~1.5x Qwen3-8B, which already measured 120.8 GiB/GPU)
      was the open question, not wall clock.

      **12b MEMORY: needs 2 nodes. Measured, not estimated.** GPUs are 178.35 GiB.
      * 12b single-master at world=8, mbs=8: OOM at 176.71 GiB used, failing a 1.88 GiB
        allocation. ~2% over.
      * Same at mbs=4: STILL OOM, failing a 266 MiB allocation with 174.90 GiB allocated.
        Halving the micro-batch barely moved it — **activations are not the dominant term**
        at this size; the fp32 master + bf16 student + bf16 teacher + gradients are, and
        micro-batch touches none of them. Matches the 8B datapoint scaled by parameters
        (120.8 GiB x 12.2/8.2 ~= 180 GiB).
      * **world=16 (2 nodes), mbs=4: WORKS.** ZeRO-2 (`dist_optim.py:69`) shards optimizer
        state and reduce_scatters gradients, so doubling the world halves exactly the terms
        that dominate. `mbs=4 x world=16` also keeps the global batch at 64, so 12b stays
        comparable with every other size. Ran clean to step 400, val_ntp -0.108 vs teacher.
      * **12b needs NO CHAIN at 2 nodes: 4.05 s/step x 2513 steps = 2.83 h + ~11 min
        startup ~= 3.0 h**, inside one 4 h window.

      CONSTRAINT AT world=16: `dist_optim.py:94` asserts `p.shape[0] % world_size == 0` for
      every param with numel >= 1024. Gemma-3-12b (3840 / 15360 / 262144 / 4096 / 2048) and
      Qwen3-8B (4096 / 12288 / 151936) all pass. It fails loudly on rank 0 at optimizer
      construction, so an incompatible world size cannot corrupt anything — but world size
      is not freely tunable.

      OPEN: Qwen3-8B SPLIT formats (`nvfp4pdsplit`, `nvfp4lloyd43split`,
      `nvfp4lloyd21split`) are the three missing from the 8B tree, submitted at 2 nodes and
      still queued when the cluster went into maintenance. Not a foregone conclusion: 12b
      was ~2% over on a SINGLE-master budget, while 8B-split was ~20% over (215 vs 178) and
      carries TWO replicated fp32 masters — and ZeRO-2 does not shard parameters, so the
      expensive term does not shrink with node count. May need 4 nodes, or an approach that
      attacks the replicated masters directly.

      Extrapolating step time from parameter count has also been wrong every time it was
      tried here: 1b matched 270m's wall clock despite 7x the linear params (the 262k-vocab
      LM head dominates), and the 4b estimate was low because total elapsed was divided by
      steps, folding ~55 min of startup into the per-step figure. Measure training-only
      s/step from the tqdm line, not elapsed/steps.
- [ ] **5.3** Evals via `submit_missing_evals.py --family gemma3 --apply`.
- [ ] **5.4** Add Gemma panels to the notebook.

### Phase 6 — deprecate `--full-disag`
- [ ] **6.1** Mark the flag and `quantizers/full_disag.py` deprecated in-place; leave the
      tests green. Do not extend it to Gemma.

---

## 4. Open questions

1. **Sliding window + Nixl KV transfer** (2.3) — unknown until 3.2 runs. Highest risk.
2. **Is 270m worth training?** ~60% of its parameters are the tied embedding, which we do
   not quantize, so the achievable compression is small. It is cheap, so probably still
   worth it as a scaling anchor, but the headline number will look unimpressive.
3. **Benchmark fit.** Our evals (GSM8K 5-shot, MATH-500 4-shot, MMLU-Pro 5-shot) are
   maths/reasoning-heavy. gemma-3-270m-it will score near the floor on all three, which
   makes it useless for ranking formats (cf. the AIME25 lesson). Consider adding an easier
   generative benchmark for the small end, or accept 270m as a smoke target only.

---

## 5. Progress log

*(newest last; record intermediate results here as work proceeds)*

**2026-08-17 — investigation complete, plan drafted.**
- Confirmed all architecture facts in §1 against transformers 5.3.0 and live HF configs.
- Found the four incompatibilities in §2.1–2.4; none are blockers except the untested
  sliding-window KV path (2.3).
- Confirmed §2.8 non-issues by reading `training/data.py`: label masking is a
  template-agnostic prefix-diff and tokenization already passes
  `add_special_tokens=False`, so the data pipeline needs **no** Gemma-specific work.
- Network access from compute nodes works; configs and the 4b safetensors index were
  fetched live, so model download is not a blocker.

**2026-08-17 (later) — loader written, tooling generalized.**
- `training/models.py` added. **The naive path is a trap and is now pinned by a test**:
  `Gemma3ForCausalLM.from_pretrained("google/gemma-3-4b-it")` *succeeds* and returns a
  model whose embedding, final norm, head and every `pre_feedforward_layernorm` were
  silently re-initialized — `_checkpoint_conversion_mapping` is empty for the text class,
  so every text tensor is "missing". A weight-magnitude sanity check does NOT catch this
  (random init is nonzero); `verify_text_load` compares against the raw safetensors.
- First cut of the loader was itself wrong: it did `Gemma3ForCausalLM(text_cfg)` and then
  `load_state_dict`, i.e. random-initialized ~4B params (~16 GB) purely to overwrite them,
  which stalled for >10 min at 4b. Now uses `from_pretrained(state_dict=...)`, reusing
  transformers' low-memory path. `verify_text_load` also streams shards instead of
  slurping 8 GB to compare six tensors.
- 4.1/4.2 done. Tag namespaces verified:
  `qad3x-Qwen-Qwen3-4B-nvfp4-*` (unchanged) vs `gemma3-google-gemma-3-4b-it-nvfp4-*`.
- All four Gemma checkpoints cached in `HF_HOME` (270m 549M, 1b 1.9G, 4b 8.1G, 12b).
- 12b is IN scope (named explicitly in the goal), sequenced last.
- Loader tests are running in a sub-agent; results pending.

**2026-08-17 (later still) — vLLM P/D investigated against the pinned clone.**
- Sliding-window P/D is **safe on our version**, via the HMA-off fallback, not via any
  window-aware transfer path. Full detail and file:line evidence in 2.3. The practical
  consequences are a 3-4x KV memory blow-up and a stricter gate (3.2).
- Hybrid/Mamba P/D fails loudly at startup — no silent-corruption risk, but no support
  either until vLLM >= 0.20.
- Found a real docstring/behaviour mismatch in `nixl_connector.py`: the compat hash omits
  the sliding window although line 170 says it covers it.
- vLLM clone pinned at d7de043d55 to match the container. Do not move it.
- Loader test re-run was cancelled before completing, so `training/models.py` is
  hand-verified (6 tensors byte-equal, 0 meta params, 239 linears, no vision) but its
  test file has NOT been run green end to end. Outstanding.

**Correction (user):** the sub-agent's handshake analysis was largely moot. The compat
hash keys on the checkpoint path, so it can never match for a disaggregated-quantization
pair — we disable it deliberately and always will. Reasoning about what the hash covers is
not useful for this project; the structural asserts and the empirical gate are what matter.

**2026-08-17 — version-gap check on the vLLM findings (user challenge).**
The sub-agent cited a NixlConnector compatibility matrix from *current* docs while we run
v0.14.1. Re-verified everything against the pinned clone only, then diffed against
`origin/main` to see what the matrix is actually describing.

*Independently confirmed at v0.14.1 (d7de043d55):* the HMA-off fallback
(`config/vllm.py:983`), `SlidingWindowSpec` -> `FullAttentionSpec` conversion
(`kv_cache_utils.py:1148-1188`), `block_window_per_layer` gated to
`model_type == "llama4"` (`nixl_connector.py:1391-1392`), and zero `MambaSpec` references
in `kv_cache_utils.py`.

*The matrix describes different machinery, and the gap is real:* on `origin/main` the
connector has been refactored into `kv_connector/v1/nixl/` and now declares
`class NixlBaseConnector(KVConnectorBase_V1, SupportsHMA)` (connector.py:79) — and the
llama4-only window gate is gone. So "SWA supported" on latest means genuine HMA-aware
transfer with proper windowed caches. **We do not have that.** We get the fallback.

*But the conclusion holds, on better evidence than the agent had.* The same fallback still
exists on main, and main's warning text now spells out the consequence explicitly:

    "hybrid SSM models (e.g. Jamba, Bamba) require HMA and will fail at startup
     without it; models with sliding window attention will run with REDUCED
     PERFORMANCE."

That is upstream, today, characterizing the exact path we are on as a performance
trade-off for SWA and a hard failure for SSM — matching what we observe at v0.14.1. A
search of all 301 post-v0.14.1 `kv_transfer/` commits found no SWA-specific P/D
*correctness* fix; the window/hybrid/mamba ones are overwhelmingly Mamba/SSM/MLA feature
work.

*Net:* Gemma-3 SWA over Nixl P/D is correct on our version but pays the full-length-cache
memory cost. Upgrading vLLM would buy back memory, not correctness. The empirical gate
(3.2) stands regardless, because "upstream says it should work" is not evidence that our
particular stack does.

**2026-08-17 — BF16 disagg gate started (no quantization involved).**
Key sequencing insight (user): the sliding-window KV question can be settled entirely on
STOCK BF16 models via `run_eval_disagg.sh --unquantized`, with zero QAD/quantization in
the loop — and it produces the BF16 baselines the plots need anyway.

Canary `google/gemma-3-1b-it --unquantized --think` (job 530197): **the HMA-off fallback
engages for Gemma-3, confirmed in the live prefill log** —
  * "Turning off hybrid kv cache manager"            x1
  * "Hybrid KV cache manager is disabled"            x1
  * GPU KV cache size: 6,341,968 tokens
That is task 3.2 item 4 satisfied, and it is the empirical basis for the 2.3 verdict.
(Checked too early the first time and saw zeros — the stack logs are created before they
are written. Do not read a fresh stack dir as evidence of absence.)

**Also outstanding, found by the loader work:**
- `export/save.py` cannot save a tied-embedding model loaded on CPU: safetensors refuses
  shared storage (`lm_head.weight` / `model.embed_tokens.weight`). Production exports from
  CUDA, where `.detach().cpu()` copies and breaks the sharing, so this has never bitten —
  but every Gemma size and Qwen3-0.6B/1.7B/4B are tied, so a CPU-side export path would
  hit it. Latent, not urgent.
- `training/models.py` still has NO consumers: `training/qad.py` calls
  `AutoModelForCausalLM.from_pretrained` directly and uses `student.model` /
  `student.lm_head`. Wiring it (task 1.2) means scoping `quant_entry["apply"]` and
  `replace_linears` to `text_stack(s).base`, which touches the production Qwen3 path.

**2026-08-17 — BF16 disagg verification under way, and two Gemma-specific gotchas.**

*Result:* Gemma-3-1b BF16 runs end to end through the disaggregated stack and scores
sanely (GSM8K 44.4 flexible-extract, MATH-500 41.8 math_verify). The HMA-off fallback is
confirmed engaged in the live log. Monolithic comparison (530451) in flight — that is the
number that decides whether KV transfer is correct, since a plausible score alone does not.

*Gotcha 1 — `--no-think` will HARD FAIL on Gemma.* `eval_disagg.py:306-308` asserts that a
no-think prompt still renders a `<think></think>` block and raises `SystemExit` otherwise.
That is a Qwen-specific invariant: Gemma has no thinking mode at all, so the block is never
present. **The think/nothink axis is meaningless for Gemma** — both modes render the same
prompt, and `--no-think` errors out. Gemma sweeps must run think-mode only (i.e. plain),
and `submit_missing_evals.py`'s `MODES` must become family-dependent. Add to task 4.1.

*Gotcha 2 — read the right metric key.* The raw JSON carries several metrics per task and
the obvious one is the wrong one: `gsm8k.exact_match,strict-match` was 13.3% while
`flexible-extract` (what the notebook uses) was 44.4%, and `minerva_math500.exact_match,none`
was 0.0 while `math_verify,none` was 41.8%. Judging a Gemma run by the wrong key would
manufacture a catastrophic-looking regression.

*Also observed:* 9 transient `502 decode: ReadError` responses mid-generation, all retried
successfully; the job exited FAILED (2:0) after writing complete results — the same
finish-then-die pattern already seen on the Qwen training jobs. Worth watching whether the
502 rate scales with model size, since that WOULD start corrupting scores.

**2026-08-17 — BF16 fan-out; thinking settled; two more Gemma-specific breakages.**

*Thinking is settled — Gemma has no thinking axis.* Measured on 1b and 4b: the
`enable_thinking` kwarg is **silently accepted and completely inert** — the rendered prompt
is byte-identical with it True, False or absent, and the chat template never mentions
thinking (Qwen's prompt DOES change). Accepted-but-inert is the worst kind of "supported":
nothing errors, so you would believe you had toggled something.
`--no-think` therefore hard-fails on Gemma at `eval_disagg.py:305-308`, which renders a
probe prompt and raises `SystemExit` when no `<think>` block appears. That guard (added
after a `setdefault` bug mislabelled a whole tree) makes the dangerous case impossible:
a bogus Gemma "nothink" tree cannot be produced. **Decision (user): Gemma results live in
`results/disagg/think/` and only that mode is ever run.** `--think` is the default, so it
need not be passed. The notebook's `MODES` still has to become family-aware or Gemma will
silently render nothing (`avg_accuracy` requires every task@mode) — deferred, task 4.3.

*Breakage 1 — head_dim=256 kills the single-engine path.* `run_eval_vllm.sh` let vLLM pick
its backend; vLLM picks FlashInfer, which asserts:
"There is a bug in FlashInfer block_size 16 head size 256 support" — Gemma-3 is head_dim
256 at every size, vLLM's default block_size is 16, and the engine never starts. The
disaggregated path was immune only because `run_nixl_server.sh` already pins FLASH_ATTN.
Fixed by defaulting `run_eval_vllm.sh` to `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
(`EVAL_ATTENTION_BACKEND` overrides). Second benefit: single-engine and disaggregated now
share a backend, so a monolithic-vs-disaggregated difference is a KV-transfer difference
rather than a backend A/B. Caveat: Qwen single-engine runs previously used whatever vLLM
chose, so small drift against older `results/vllm/` numbers is possible.

*Breakage 2 — proxy port collision.* The 1b mmlu_pro run died with
"[nixl] proxy DIED before ready" / `[Errno 98] Address already in use`, while both engines
came up fine. `free_ports()` binds to prove a port is free and then releases it, so there
is a TOCTOU window; the canary was still tearing down on the same node. Transient —
resubmitting worked. Worth knowing it presents as a proxy failure, not an engine one.

*Fleet outcome (superseded by the entries below):* 530460 died on the stale `TASKS`
default and was resubmitted as 530482 (COMPLETED). 530468 (1b mmlu_pro) COMPLETED.
530469/70 (270m) failed on the missing chat template, resubmitted as 530504/530505 (both
COMPLETED). 530471/72 (4b) failed on the multimodal backend conflict and are NOT
resubmittable — they need task 2.4 first. 12b likewise.

**BF16 baselines now on disk: 270m and 1b complete (gsm8k, math500, mmlu_pro). 4b and 12b
blocked on 2.4.**

### 2026-08-17 — our scores are NOT comparable to the published Gemma numbers

Raised because our BF16 1b GSM8K (44.4) sits ~18 points under the published 62.8, which
looks like a broken baseline and is not one. **The protocols differ**, and the difference
is entirely on our side of the fence:

| benchmark | Gemma 3 report, IT (Table 21) | ours (lm_eval task default) |
|---|---|---|
| GSM8K     | **0-shot**, CoT | **5-shot**, chat template |
| MATH      | **0-shot**      | **4-shot** (`minerva_math500`) |
| MMLU-Pro  | unspecified     | 5-shot |

Published IT numbers (Table 18) are 1B 62.8 / 4B 89.2 / 12B 94.4 / 27B 95.9. Note the
report gives GSM8K **two** protocols and never states which produced Table 18: Table 19
(pretrained) says 8-shot CoT, Table 21 (IT) says 0-shot CoT. Unresolved on purpose — we
deliberately did not spend GPU discriminating them, because it changes nothing below.

DECISION: keep the lm_eval defaults. Every format comparison we make is BF16 vs quantized
under one identical harness, so a constant protocol offset cancels. Chasing the published
protocol would buy a nicer-looking absolute number and cost cross-run comparability with
the entire existing Qwen results tree.

DO NOT quote our absolute scores against published ones. The gap is a harness artifact.

Corroborating: `exact_match,strict-match` is depressed for EVERY model, not just Gemma —
Qwen3-0.6B BF16 scores 16.5 strict vs 66.4 flexible — because `strict-match` requires the
few-shot `#### N` format that no instruct model emits under a chat template.
`flexible-extract` is the only meaningful GSM8K column. Same shape as the
`minerva_math500` `exact_match,none` 0.0 vs `math_verify,none` 41.8 split.

### 2026-08-17 — BF16 fleet: 4 of 6 jobs failed, two distinct root causes

Both 270m jobs (530469/530470) and both 4b jobs (530471/530472) failed; only the 1b runs
survived. Not transient, and not the same bug.

**270m — missing chat template, caused by our own cache warming.** Failure was
`ValueError: Cannot use chat template functions because tokenizer.chat_template is not
set`. The template is NOT missing upstream: `google/gemma-3-270m-it` ships
`chat_template.jinja` (1.53 kB), it was missing from our local snapshot. Cause: 270m uses
the NEWER standalone-file format, while 1b keeps the template inside
`tokenizer_config.json` and 4b/12b use `chat_template.json` — so any cache warm using
`allow_patterns=["*.json", "*.safetensors"]` silently drops it for 270m only. That exact
pattern is at `training/models.py:183` (`verify_text_load`). **ACTION: add `"*.jinja"`
there**, or the next fresh cache reproduces this.

Repaired the cache in place: all three cached templates are byte-identical
(md5 `acabb12fa812`, 1532 bytes = upstream's 1.53 kB), so Gemma 3 ships ONE template
family-wide and copying 1b's into the 270m snapshot is exact, not an approximation.
Verified by md5 after writing. Caveat: a future `snapshot_download` of 270m may not
reconcile a hand-written file — the `*.jinja` fix above is the durable one.

**4b — cannot be served at all as shipped.** See §2.2b: `model_type: gemma3` forces
`is_mm_prefix_lm`, which FLASH_ATTN refuses. This is NOT fixable by resubmitting, and it
blocks 4b/12b BF16 baselines until the text-only export exists. 270m is resubmittable now;
4b/12b are not.

Diagnosability note: engine logs DO survive — `run_eval_disagg.sh:198` passes
`--log-dir logs/eval_disagg/stack_<jobid>_<task>` on lustre. The driver's
"prefill DIED before ready" is only a summary; the real traceback is in that dir's
`prefill.log`. (`eval_disagg.py:428` defaults to node-local `/tmp` when `--log-dir` is
absent, so anything invoking `eval_disagg.py` directly WILL lose its engine logs.)

### 2026-08-17 — Phase 3.1 RESULT: monolithic and disaggregated agree on 1b BF16

Job 530482 (monolithic, FLASH_ATTN) vs 530197 (disaggregated), same stock BF16
`google/gemma-3-1b-it`, same harness, same backend:

| metric | monolithic | disaggregated | delta | combined stderr |
|---|---|---|---|---|
| gsm8k `flexible-extract`   | 44.05% | 44.43% | +0.38 pt | 1.93 pt |
| gsm8k `strict-match`       | 13.42% | 13.27% | -0.15 pt | 1.32 pt |
| minerva_math500 `math_verify` | 42.60% | 41.80% | -0.80 pt | 3.13 pt |

All three agree well within one combined standard error. Because both runs now pin
FLASH_ATTN, this is a KV-transfer comparison and not a backend A/B.

**What this does and does not establish.** It rules out a grossly broken transfer — a wrong
per-layer block mapping on a 5:1 sliding/full model would not land inside 0.8 pt. It does
NOT prove correctness: greedy decoding is only reproducible up to vLLM's batching
nondeterminism, so a *subtly* wrong transfer could hide inside a ~1-3 pt band, and these
prompts are far below the 512-token sliding window, so the window logic is barely
exercised. **3.2's identity oracle (token-identical output, prompts >=4x the window)
remains the test that actually proves it, and is still required before Phase 5.**

### 2026-08-17 — 270m BF16 lands

After the chat-template repair, 530504/530505 completed:

| task | 270m BF16 | 1b BF16 |
|---|---|---|
| gsm8k `flexible-extract` | 7.58% | 44.43% |
| minerva_math500 `math_verify` | 9.20% | 41.80% |
| mmlu_pro `custom-extract` | 5.46% | 13.92% |

All three benchmarks are GENERATIVE with extraction filters (`custom-extract` for
mmlu_pro), not multiple choice, so there is no chance floor to read these against — a low
score is just a low score. 270m is simply a weak model. Treated as a normal size.

*Breakage 3 — two eval entrypoints had a default `TASKS` that cannot run.* `run_eval.sh`
and `run_eval_vllm.sh` both defaulted to `"gsm8k math_500 aime_2025"`, but the lm_eval
overlay registers those as `minerva_math500` and `aime25`; `math_500` and `aime_2025` do
not exist. Any invocation relying on the default died at task-load with
`KeyError: "Spec 'math_500' is not a registered task/group/tag name"` — which is exactly
how 530460, the monolithic reference, failed. Latent for a long time because every caller
passed `--tasks` explicitly. Both now default to `"gsm8k minerva_math500"`, matching
`run_eval_disagg.sh` so monolithic and disaggregated score an identical task set.
530460 resubmitted as **530482**.
