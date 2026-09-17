# Manuscript tables

Run from any directory with the repository's Python environment. Generators print
LaTeX `tabular` content only; they do not run training, evaluations, or plots.
Generated `.tex` files live in `notebooks/tables`. Captions, labels, sizing, and float
placement remain in `latex/main.tex`, which loads each body with
`\input{tables/NAME.tex}`. Copy/upload `notebooks/tables` as `tables/` beside the
manuscript when building the paper, just as with `notebooks/figures`.

From the repository root:

```sh
.venv/bin/python notebooks/table_generators/generate.py qadd27b_accuracy
.venv/bin/python notebooks/table_generators/generate.py all --write
.venv/bin/python notebooks/table_generators/generate.py all --check
.venv/bin/python notebooks/table_generators/generate.py qadd27b_accuracy --write
.venv/bin/python -m unittest discover -s notebooks/table_generators
```

`--check` ignores formatting whitespace and LaTeX comments, but fails on changed
values, missing files/inputs, or malformed output. `--write` saves the selected
`notebooks/tables/NAME.tex` files; neither mode edits the manuscript. Use
`--output-dir PATH` to export directly to another manuscript's `tables/` directory.
All selected generators must succeed before anything is written. No mode stages files.
Tables containing red incomplete-result markers remain explicitly incomplete;
`--check` means agreement with the sources, not that all experiments are finished.

| Table label (omit `tab:` on the command line) | Generator | Source |
|---|---|---|
| `minimal_comparison` | `make_main_table.py` | Notebook DH/PH loaders, prefill/decode CSVs, model configs |
| `bigmodels` | `make_large_model_table.py` | `evals/drivers/compare_arms.py`: per-item scores and paired significance; pinned BF16 summaries for Qwen3.8-27B |
| `qadd27b_accuracy` | `make_qadd27b_table.py` | Notebook Unsloth loader, all eight formats, mandatory step 980 |
| `qadd27b_lengths` | `make_qadd27b_lengths_table.py` | Per-item `*_lengths.npz` exports and final-step results summaries, same quantized arms as the accuracy table |
| `interop` | `make_interop_table.py` | `cross_grid.py` exported-score mode, full 3-decode by 4-prefill grids at step 980 |
| `prefill-per-model` | `make_prefill_table.py` | `qad/kernels/prefill/offload_prefill.csv`, resident and `zero-ssd`; core Qwen 3/Gemma 3 only (27B uses separate llama.cpp measurements) |
| `nvfp4-breakdown` | `make_breakdown_table.py` | `nvfp4_breakdown.csv` plus resident full-stack timings |
| `hyper`, `parallel`, `models` | `make_recorded_table.py NAME` | Explicit setup metadata in `recorded_tables.json` |
| `attn-backend` | `make_recorded_table.py attn-backend` | Historical attention timings recorded in `recorded_tables.json` |

The old `notebooks/make_main_table.py` and `evals/drivers/make_table.py` now live here.
The large-model generator requires four passes on both MMLU-Pro and MMMU-Pro
by default, preserves the FP8/MXFP4 footnotes, and sends input
diagnostics to stderr rather than contaminating the LaTeX output.
For Qwen3.8-27B, the BF16 means use four complete tracked summaries: `bf16-r1`
through `bf16-r4` for MMLU-Pro, and `bf16`, `bf16-r1` through `bf16-r3` for
MMMU-Pro. This avoids the incomplete per-item export containing only `bf16-r5`.
Missing or incomplete selected summaries fail generation; later repeats and
hardware/control tags are not substituted. Quantized-arm means and paired
significance still use per-item scores, never these aggregate summaries.

The core table may need Hugging Face model configs already cached locally or network
access to fetch them. Use `HF_HUB_OFFLINE=1` for strictly offline checking. Its accuracy
columns require the fixed five-checkpoint window, including every RULER length.
The 27B table uses the plot's baseline selection, but refuses fallback to an earlier
QADD step. It includes IQ3_S and Q3_K_XL even though the plot hides them.
The generation-length table uses those same eight quantized pairs, with one panel
per benchmark. It computes means, medians and linearly interpolated 95th percentiles
from per-item completion-token counts, including zero counts and truncated outputs;
lengths are displayed to the nearest token. Truncation rates use the matching
results summaries. Missing or partial inputs, repeated/invalid IDs, multiple passes,
or mismatched item sets fail generation. Only step 980 is used for conjugates;
earlier checkpoints and cross-format interoperability runs are not substituted.

The interoperability table reuses `evals/drivers/cross_grid.py` for tag selection,
per-item grids and exact McNemar tests. Both benchmarks require all twelve cells
and full common item coverage. Bold identifies training-matched conjugates, not
row maxima; stars compare each other cell to that matched reference (two-sided,
uncorrected `p < 0.05`). Reproduce its comparisons without raw generations with:

```sh
.venv/bin/python evals/drivers/cross_grid.py --exported --bench mmlu_pro --stats
.venv/bin/python evals/drivers/cross_grid.py --exported --bench mmmu --stats
```

Exported mode preserves the manuscript's per-item benchmark scores. In particular,
MMMU-Pro retains the official parser's per-pass seeded fallback; the driver's
original raw mode instead resets that seed per item and can differ on unparsed
answers. No raw-output or lenient-scoring claim is made for the exported grid.

## Recorded inputs and precision

Setup metadata was extracted once from the manuscript into structured rows; it is not
re-extracted at generation time. Update this explicit source when the setup changes.
Raw attention-backend microbenchmark output is not tracked in this checkout, so those
rows preserve the recorded measurements and selected backends; the generator does not
claim to recompute measurements that are unavailable.

The component CSV retains three decimal places; component times use decimal half-up
rounding to reproduce the manuscript. Ratios and totals are computed from the input
precision, not the displayed values. Negative launch/idle residuals are preserved,
not clipped. Accuracy deltas are computed before rounding either score.
