# eval_results_vllm_nothink — thinking SUPPRESSED

Produced with `eval_vllm.py --no-think`, which now forces `enable_thinking=False` and
verifies it in the rendered prompt (an empty `<think></think>` block must be present)
before running. Not comparable to `eval_results_vllm_think/`: the same model differs by
~20 points on GSM8K between the two modes. Treat them as separate benchmarks.
