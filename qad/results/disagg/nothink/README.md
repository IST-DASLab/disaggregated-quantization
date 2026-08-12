# eval_results_disagg_nothink — thinking SUPPRESSED

Disaggregated (Nixl 1P1D) GSM8K runs from 2026-07-29 with `--no-think`, genuinely
suppressed (verified: prompts carry an empty `<think></think>` block). Scores:
dual-shared 20.24, homo-a4 19.41, homo-a16 18.12 (flexible-extract, n=1319, cap 4096).

Internally comparable, and comparable to the single-server no-think control (18.95) —
disaggregation itself costs nothing measurable. NOT comparable to
`eval_results_vllm_think/`, which is thinking-enabled.
