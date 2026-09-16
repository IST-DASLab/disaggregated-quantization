"""Gates for quantization_config.ignore.

The failure this exists to prevent is PARTIAL COVERAGE UNDER PIPELINE PARALLELISM. The
list used to be derived from the live model, but each PP stage holds only its half of the
layers -- at 27B, rank 0 sees layers 0..31 of 64, which is 24 of the 48 GDN blocks,
exactly half. The TENSORS are correct (merge_export_state stitches both halves before
writing); only the config was written from one stage's view, so vLLM built quantized
layers for every GDN block in the other half whose weights are plain `.weight`:

    AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'

Deriving from the merged state dict is what makes stage count irrelevant.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export.save import _ignore_list_from_state  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


class FakeT:
    def __init__(self, ndim):
        self.ndim = ndim


def synth(n_layers: int, gdn_every_4th_skipped: bool = True) -> dict:
    """A state dict shaped like the 27B export: GDN blocks with a quantized out_proj /
    in_proj_qkv / in_proj_z beside unquantized in_proj_a / in_proj_b."""
    st = {}
    for i in range(n_layers):
        p = f"model.language_model.layers.{i}"
        if gdn_every_4th_skipped and i % 4 == 3:          # full-attention layer
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                st[f"{p}.self_attn.{proj}.weight_packed"] = FakeT(2)
        else:
            for proj in ("out_proj", "in_proj_qkv", "in_proj_z"):
                st[f"{p}.linear_attn.{proj}.weight_packed"] = FakeT(2)
            for proj in ("in_proj_a", "in_proj_b"):
                st[f"{p}.linear_attn.{proj}.weight"] = FakeT(2)
            st[f"{p}.linear_attn.norm.weight"] = FakeT(1)
        st[f"{p}.input_layernorm.weight"] = FakeT(1)
    st["model.language_model.embed_tokens.weight"] = FakeT(2)   # 2-D but NOT a Linear
    st["lm_head.weight"] = FakeT(2)
    return st


print("== full 64-layer export ==")
full = synth(64)
ig = _ignore_list_from_state(full)
gdn_blocks = {e.rsplit(".linear_attn", 1)[0] for e in ig if "linear_attn" in e}
check("all 48 GDN blocks covered", len(gdn_blocks) == 48, f"{len(gdn_blocks)}")
check("in_proj_a/b ignored", sum("in_proj_a" in e or "in_proj_b" in e for e in ig) == 96)
check("linear_attn + norm parents listed",
      sum(e.endswith("linear_attn") for e in ig) == 48
      and sum(e.endswith("linear_attn.norm") for e in ig) == 48)
check("lm_head ignored", "lm_head" in ig)
check("embedding NOT ignored (2-D but not a Linear)",
      "model.language_model.embed_tokens" not in ig)
check("no quantized module ignored",
      not any(f"{e}.weight_packed" in full for e in ig))

print("\n== the PP bug: a HALF model must not be the source ==")
# What deriving from one stage would have produced.
half = {k: v for k, v in full.items()
        if not k.startswith("model.language_model.layers.")
        or int(k.split(".")[3]) < 32}
ig_half = _ignore_list_from_state(half)
gdn_half = {e.rsplit(".linear_attn", 1)[0] for e in ig_half if "linear_attn" in e}
check("half model yields HALF the GDN blocks (the bug)", len(gdn_half) == 24,
      f"{len(gdn_half)} of 48 -- this is what shipped")
check("merged model yields all of them (the fix)",
      len(gdn_blocks) == 2 * len(gdn_half), f"{len(gdn_blocks)} vs {len(gdn_half)}")

print("\n== every unquantized Linear is accounted for ==")
lin_unq = {k[: -len(".weight")] for k, v in full.items()
           if k.endswith(".weight") and v.ndim == 2
           and f"{k[: -len('.weight')]}.weight_packed" not in full
           and "embed" not in k}
missing = sorted(lin_unq - set(ig))
check("no unquantized Linear left out", not missing, str(missing[:3]))

print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
