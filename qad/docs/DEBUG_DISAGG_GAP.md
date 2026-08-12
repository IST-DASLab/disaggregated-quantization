# Why quantized models score ~35 points lower through disaggregated serving

## RESOLVED (2026-07-29): it is our exported `config.json`, not serving

Serving the SAME weights (symlinked `model.safetensors`, only `config.json` edited):

    served, shipped config : 18.50
    served, fixed config   : 48.50      <- in-process on the same 200 docs is 45.50

Three fields were fixed together (`dtype: float32` -> `torch_dtype: bfloat16`,
`use_cache: False` -> `True`, flat `rope_theta` restored alongside the transformers-5
`rope_parameters`). Bisect running (job 478956) to attribute the 30 points.

This explains what nothing else could: BF16 is unaffected because its config is the
stock one, and the in-process path was fine because `eval_vllm.py` passes
`dtype="bfloat16"` explicitly, overriding the bad field. Every configurable knob on the
serving side was irrelevant because the fault was in the checkpoint we handed it.

**Consequence: every served/disagg number measured so far is invalid**, including the
whole 0.6B sweep. Re-export or patch configs, then re-run.


Running log of hypotheses, results and next ideas. Newest results at the top of each
section. **Probe format: `nvfp4a16` @ Qwen3-0.6B step 2450, GSM8K.** Chosen because it
is weight-only (no activation quantization to confound), homogeneous (same checkpoint
on both engines, so no dual-format complexity), and has both a good reference and a bad
number.

## The observation

GSM8K, Qwen3-0.6B, thinking ON, step 2450, flexible-extract:

| format | disagg | single-engine | rel. loss vs own BF16 baseline |
|---|---|---|---|
| BF16 baseline | 66.03 | 64.59 | — (the two paths AGREE, Δ +1.44) |
| nvfp4a16 | 28.51 | 61.87 | disagg 56.8% / single **4.2%** |
| nvfp4 | 24.87 | 57.47 | disagg 62.3% / single 11.0% |
| nvfp4pdshared | 28.43 | (no single equivalent) | 56.9% |
| nvfp4pdsplit | 23.81 | (no single equivalent) | 63.9% |

Also, thinking OFF, full GSM8K: `nvfp4a16` in-process **63.38** vs single-server HTTP
**18.95** vs disagg **18.12**.

Two facts constrain everything:

1. **BF16 is unaffected** — served and in-process agree to 1.4 points. So serving is not
   broken in general.
2. **Single-server HTTP is as bad as disagg** (18.95 vs 18.12). So it is NOT caused by
   disaggregation, the KV transfer, or the proxy. It is *quantized weights + being
   served over HTTP*.

The dual (pd) formats have no single-engine equivalent, so until this is fixed **no pd
number can be trusted** — that is what blocks the actual research question.

## Ruled out

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| 1 | disaggregation / KV transfer / proxy | **OUT** | single-server HTTP 18.95 ≈ disagg 18.12 |
| 2 | generation cap too small | **OUT** | 512 → 18.27, 4096 → 18.12 |
| 3 | server applies checkpoint's sampling defaults | **OUT** | `--generation-config vllm` 18.88 ≈ `auto` 18.95, both models |
| 4 | malformed prompt / chat template | **OUT** | prompt dumped and verified: 5-shot + target, correct `<im_start>` counts, thinking block as expected |
| 5 | responses paired with wrong documents | **OUT** | 328 responses contain a number from their own question, 20 do not (~94%) |
| 6 | dropped/failed requests scored as wrong | **OUT** | zero `API request failed` in logs |
| 7 | wrong dtype on the served path | **OUT** | explicit `--dtype bfloat16` → 17.59 (no change) |
| 8 | different weight-loading / quantization path | **OUT** | both paths log the identical marlin FP4 path, same engine build |
| 9 | HTTP harness broken in general | **OUT** | BF16 in-process 44.35 vs HTTP 44.12 |
| 16 | quantized checkpoint's **generation_config.json** carries different defaults | **OUT** | diffed against base Qwen3-0.6B: identical except `transformers_version`. Same temperature/top_p/top_k/eos. |
| 17 | served path uses a different **dtype** (our config says `dtype: float32`, in-process forces bfloat16) | **OUT as cause** | explicit `--dtype bfloat16` on the server → 17.59, and the flag was verified applied (`'dtype': 'bfloat16'` in non-default args). Still a real EXPORT BUG: default served runs resolve `dtype=auto` → float32. |
| 10 | GPU lacks native FP4 (my claim) | **WRONG** | B200 has native FP4; the marlin warning appears only for the weight-only A16 path, where weight-only kernels are correct anyway. W4A4 shows no such warning. |

## Open hypotheses

| # | hypothesis | how to test | status |
|---|---|---|---|
| 11 | ~~Batch size selects a different (buggy) quantized kernel path.~~ **DEAD** In-process queues hundreds of prompts; the API path was measured pinned at `Running: 31 reqs`. Marlin/FP4 kernels branch on batch size; BF16 does not use them at all, which fits fact 1. | **Arm A** (job 478917): in-process only, `max_num_seqs` 256 vs 32, same 200 docs. No server involved. | big=**45.50** |
| 12 | ~~Same as 11, end-to-end on the server~~ **DEAD**: if client pacing alone moves the score on ONE server, weights/config/loading are all excluded. | **Arm B** (478922): served **32 -> 16.00**, **128 -> 19.00**, vs in-process 45.50 on the same 200 docs. Concurrency is irrelevant to accuracy. | **DEAD** |
| 13 | ~~Chunked prefill / prefix caching differ between the paths~~ **DEAD** | both paths log `enable_chunked_prefill=True` and `enable_prefix_caching=True`. | **DEAD** |
| 14 | CUDA-graph capture differs between the two paths for quantized models. | Serve with `--enforce-eager` and re-measure. Partly probed earlier on P2P NCCL, never on this axis. | not started |
| 18 | `config.json` differs from base in `use_cache: False` and the transformers-5 rope rename (`rope_parameters` instead of `rope_theta`/`rope_scaling`). A silently-defaulted rope_theta (10000 vs 1000000) would be catastrophic. | Both paths read the SAME config, so this cannot explain served-vs-in-process — but check vLLM actually picks up rope_theta=1000000 from `rope_parameters`. | low priority, would affect both paths equally |
| 19 | **CUDA graphs vs eager.** Real asymmetry: every disagg number was measured with `--enforce-eager` (set in run_nixl_server.sh per vLLM's disagg docs), every in-process number with CUDA graphs (`enforce_eager=False`). Quantized kernels under graph capture could differ. | Already answered by Arm B: its server ran WITHOUT `--enforce-eager` (CUDA graphs, same as in-process) and still scored 16-19. Arm C (478928) confirms the eager side. | **DEAD** |
| 15 | **What actually differs in the generated text.** Not really a hypothesis any more -- with batching, eager, caching, dtype, sampling, prompts and alignment all excluded, the text itself is the only remaining evidence. | **textdiff** (job 478931): same 20 docs, in-process then served, same job, dump both. | RUNNING |

## Expected outcomes

* **Arm A small ≈ 28** → hypothesis 11 confirmed; the bug is in vLLM's quantized
  low-batch path, and raising concurrency fixes accuracy *and* the ~25x throughput
  deficit at once.
* **Arm A small ≈ 62** → batch size is out; go to 15 (text diff), then 13/14.
* **Arm B 128 >> 32** → actionable regardless of mechanism: raise `--concurrency` in
  `run_eval_disagg.sh` and re-run the sweep.

## Notes / gotchas hit while debugging

* Two engines in one process fails: vLLM spawns EngineCore as a subprocess that
  re-imports the main module, so a heredoc script dies with
  `FileNotFoundError: '/workspace/<stdin>'`. One engine per process invocation.
* `HF_HUB_OFFLINE=1` breaks any path doing full snapshot resolution — the cached
  `Qwen/Qwen3-0.6B` snapshot is incomplete. It is fine for tokenizer-only paths (the
  sweep), not for in-process `VLLM(pretrained=...)`.
* Comparisons must pin the model size: an unconstrained `*nvfp4a16*` glob matched 4B
  tags and reported 86.88 for a 0.6B model.
* `vllm serve` must NOT see the lm-eval overlay on PYTHONPATH: its huggingface-hub
  1.24.0 shadows the container's and the server aborts with `ImportError:
  huggingface-hub>=0.34.0,<1.0 is required ... found 1.24.0`. Strip it for the server,
  keep it for the client. Killed arm B once (job 478918).
* The 100-doc subset is harder than the full set (a16 no-think: 46-47 on the subset vs
  63.38 full), so only compare arms that share a subset.
