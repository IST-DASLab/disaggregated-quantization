# eval_results_vllm_think — thinking ENABLED

Every result in this directory was produced with Qwen3 **thinking on**, including runs
whose command line said `--no-think`.

Why: `eval_vllm.py` suppressed thinking by wrapping `lm.tokenizer.apply_chat_template`
with `kwargs.setdefault("enable_thinking", False)`. That never took effect — lm-eval's
`VLLM` class renders the template through a path the wrapper did not intercept — so the
prompts went out WITHOUT the empty `<think></think>` block and the model opened its own.
The script printed "Thinking suppressed via tokenizer patch." regardless, which is why
the mislabelling went unnoticed.

Proof (2026-07-29, Qwen3-0.6B BF16, full datasets):

    eval_vllm.py --no-think   minerva_math500 math_verify = 0.622
    eval_vllm.py --think      minerva_math500 math_verify = 0.622   <- identical
    saved prompt: think_in_prompt=0, resp_starts_with_think=True    <- thinking-on signature

For contrast, a correctly suppressed run renders `think_in_prompt=1` (the empty block)
and the response does not start with `<think>`. The difference is large: GSM8K 0.6B BF16
scores 65.05 with thinking and 44.35 without.

`eval_vllm.py` now forces the kwarg and ASSERTS the rendered prompt actually changed,
so an ineffective patch aborts instead of producing mislabelled numbers. Thinking-off
results land in `eval_results_vllm_nothink/`.
