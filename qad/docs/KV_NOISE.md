# Pseudo KV-cache compression

Models lossy KV storage without implementing a codec: a token's K/V is degraded **once**,
as it is written to the paged cache, and every later read sees the degraded value. Prefill
and decode are noised independently, so the two phases can be studied separately on a
single BF16 checkpoint — nothing is confounded by training.

Implementation: `kv_noise_connector.py` · gates: `tests/test_kv_noise.py`

## Rate parametrisation

```
sigma_group = 2^-kv_bits * RMS(group of 16 channels)
```

`4^-kv_bits` of distortion **power**, i.e. Shannon's `D(R) = sigma^2 * 4^-R` for a Gaussian
source. The amplitude multiplier is therefore `2^-bits`, **not** `4^-bits` — an error of one
square, invisible in any output, which is why `tests/test_kv_noise.py` asserts the measured
distortion against the rate-distortion identity rather than against itself.

| kv_bits | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| amplitude | 50% | 25% | 12.5% | 6.25% | 3.1% | 1.6% | 0.39% | 0.098% |
| vs bf16 RMS rounding floor (0.113%) | 442x | 221x | 111x | 55x | 28x | 14x | 3.5x | **0.9x** |

`>= 32` disables (and is asserted bit-identical to no connector at all). Above ~9 bits the
injected noise approaches bf16's own rounding error and the run measures the storage format
rather than the rate.

## Running a sweep

```bash
# prefill-only noise at 4 bits, decode untouched
KV_BITS_PREFILL=4 KV_BITS_DECODE=32 \
  ./bin/run_eval_disagg.sh --model Qwen/Qwen3-0.6B --unquantized \
    --tasks "gsm8k minerva_math500" --think \
    --tag Qwen-Qwen3-0.6B-unquantized-kvp4

# decode-only noise at 2 bits
KV_BITS_PREFILL=32 KV_BITS_DECODE=2 ... --tag Qwen-Qwen3-0.6B-unquantized-kvd2
```

**`--tag` is mandatory.** A sweep reuses ONE checkpoint at several rates and is
distinguished by nothing else, so without it every rate overwrites the same
`step_0000000.json` — including the shared BF16 baseline directory that every recovery
number divides by. That has happened; it silently corrupted the 0.6B think gsm8k baseline
(0.664 -> 0.10) and was only caught because the results path in the log did not match the
tag. Tag naming `...-kv[pd]<bits>` is what the plots and the progress cell parse.

Both rates travel in one config and each engine selects its own by `kv_role`, so the two
cannot be transposed by a launcher bug. Confirm from the engine logs:

```
prefill  NoisyNixlConnector[prefill] kv_bits=4.0  group=16 targets=kv (amplitude 0.0625 of RMS)
decode   NoisyNixlConnector[decode]  kv_bits=32.0 group=16 targets=kv (amplitude 0 of RMS)
```

## Three traps, all of which have actually fired

**1. CUDA graphs skip the hook.** `save_kv_layer` is a *Python* hook around the attention
op; a replayed CUDA graph runs captured kernels and never re-enters Python. vLLM captures
decode graphs by default, so on the decode engine the noise was applied for only some
steps. Measured at `kv_bits_decode=1` (50% amplitude): with graphs the model still scored
gsm8k 0.575 **and produced fluent text**; with `--enforce-eager` the same config scored
0.0. `run_nixl_server.sh` now forces eager whenever noise is active. Partially-applied
noise is worse than none — it looks like a publishable robustness result.

**2. MultiConnector cannot be used here.** The obvious composition,
`MultiConnector[NoisyKVConnector, NixlConnector]`, does not forward
`set_host_xfer_buffer_ops` to its children (it is called on the top-level connector,
`gpu_model_runner.py`), so the wrapped NixlConnector never gets its copy operation and the
decode engine dies on the first received KV with
`nixl_connector.py: assert self.copy_blocks is not None`. That path is only taken when
`kv_buffer_device=cpu`, which is what this deployment uses. Hence `NoisyNixlConnector`
is a **subclass** of NixlConnector, not a wrapper.

**3. Bash `[ -lt ]` is integer-only.** `[ "3.5" -lt 32 ]` errors *and* evaluates false, so a
fractional rate would fall through to the plain NixlConnector and run with no noise at all
— a clean baseline mislabelled as a 3.5-bit result. Compare with `awk`.

## Verifying a run actually applied noise

Never infer it from the score. In order of strength:

1. **Read the generations.** `LOG_SAMPLES=1 LIMIT=40` writes `*_samples_<task>.jsonl`.
   At 1-2 bits the text must be visibly broken; fluent output means the noise is not
   landing, whatever the score says. This is what caught trap 1.
2. The per-engine `NoisyKVConnector[phase] ... amplitude` line (above).
3. `kv_bits >= 32` must reproduce the no-connector baseline bit-identically.

## Plotting

`plot_kv_sweep()` in `notebooks/plots.ipynb`: one row per model, columns = benchmarks then
mean recovery, x = kv_bits with the gentle end on the right. Reads the results trees
directly (these runs are not checkpoints — one model, several rates, no quantizer or step).

**Reading the arms fairly.** At equal rate the two arms do not damage equal amounts of
cache. Measured on 0.6B think (`LOG_SAMPLES` probe, median words):

| task | prompt | generation | gen:prompt |
|---|---|---|---|
| gsm8k | 519 | 340 | 0.66 |
| minerva_math500 | 270 | 1102 | 4.1 |

So prefill noise dominates on gsm8k and decode noise dominates on MATH-500 for reasons of
token counts alone. A flat decode curve is not evidence that decode KV is robust.

Note also that the cache stores **post-RoPE** keys (`qwen3.py` applies `rotary_emb` before
`self.attn` writes the cache), so both arms perturb rotated keys — which is what a real KV
codec compresses.
