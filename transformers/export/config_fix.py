"""Repair config.json fields that only matter when a checkpoint is SERVED.

Deliberately torch-free so the offline repair tool can run on a login node.

`config.save_pretrained()` writes the TRAINING model's settings. Three of them are
wrong for serving, and cost 30 GSM8K points: identical weights scored 18.50 served
with the config as written and 48.50 once repaired (in-process scored 45.50 on the
same docs). Only `vllm serve` reads these verbatim -- the in-process lm-eval path
passes dtype explicitly -- which is why this only ever corrupted served and
disaggregated numbers.
"""


def fix_serving_fields(cfg: dict) -> dict:
    """Mutate cfg in place; return it for convenience."""
    # training runs in float32, so `dtype: "float32"` is written and a server
    # resolving dtype=auto loads fp32 for weights whose masters are bf16
    cfg.pop("dtype", None)
    cfg["torch_dtype"] = "bfloat16"
    # False during training; a served model wants its KV cache
    cfg["use_cache"] = True
    # transformers 5.x nests these under rope_parameters, but stock Qwen configs put
    # them at top level. A silently defaulted rope_theta is 10000 vs Qwen3's 1000000,
    # which wrecks long prompts. Write both so either reader agrees.
    rp = cfg.get("rope_parameters") or {}
    cfg.setdefault("rope_theta", rp.get("rope_theta", 1000000))
    cfg.setdefault("rope_scaling", rp.get("rope_scaling"))
    return cfg
