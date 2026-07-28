"""Backwards-compatible shim.

The ZeRO-2 optimizer grew a per-group update rule (AdamW or Lion) and moved to
dist_optim.py. `DistAdamW` is still the AdamW-by-default entry point, so existing
imports keep working unchanged.
"""

from .dist_optim import (STATE_KEYS, DistAdamW, DistOptimizer, _adamw_step,
                         _lion_step)

__all__ = ["DistAdamW", "DistOptimizer", "STATE_KEYS", "_adamw_step", "_lion_step"]
