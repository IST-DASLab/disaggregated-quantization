"""Legacy checkpoint layer-name mapping; no pipeline runtime dependency."""
import re

_LAYER_RE = re.compile(r"^(.*\blayers\.)(\d+)(\..*)$")

def remap_layer_indices(state: dict, offset: int) -> dict:
    """Shift `layers.<i>` -> `layers.<i+offset>` in every key.

    split_stack renumbers stage 1's layers to 0..k so the ModuleList and each layer's
    self-reported layer_idx stay consistent while it runs. For EXPORT that has to be
    undone: otherwise stage 1's "layers.0" is the same key as stage 0's "layers.0" and
    the merge silently drops half the model -- the exact failure the export guard was
    put there to prevent.
    """
    if not offset:
        return state
    out = {}
    for k, v in state.items():
        m = _LAYER_RE.match(k)
        out[f"{m.group(1)}{int(m.group(2)) + offset}{m.group(3)}" if m else k] = v
    return out
