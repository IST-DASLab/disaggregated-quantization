# Qwen3.8-27B llama.cpp TTFT

`qwen3_8_27b_llama_cpp_ttft.csv` preserves the user's supplied llama.cpp ODP
measurements verbatim: nine prompt lengths, prompt caching disabled, three
repetitions and one warmup. ODP first beats the baseline at the measured 4K
context and reaches its largest TTFT speedup, 1.78x, at 8K (12.2732 s to
6.9021 s). At 16K the speedup is 1.75x; at 32K it is 1.57x.
The Pareto figure plots `baseline_ttft_ms` and `odp_ttft_ms`, with min--max
shading. The separate `*_prompt_ms` measurements are retained, but are not TTFT
and are not substituted for it.

The supplied central TTFT statistic was not identified as a mean or median.
The baseline is Unsloth's Qwen3.8-27B IQ1_S checkpoint using llama.cpp's native
weight-only pathway. Hardware was not specified alongside the supplied CSV;
do not infer it from the older DGX Spark transformer-stack microbenchmarks.
No isolated SSD-loading-floor measurement was supplied for this run. The plot
shows context lengths from 1K onward and restores the earlier 2645.159 ms
load-only reference from `qad/kernels/prefill/load_floor_zero_ssd.csv`
(Qwen3.8-27B, NVFP4, cold-carve-out protocol). This reference is a separate
microbenchmark, not a measurement extracted from the llama.cpp TTFT run.
