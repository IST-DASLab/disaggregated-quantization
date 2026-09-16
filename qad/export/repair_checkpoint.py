"""Repair a QAT prefill checkpoint written before the export fixes, IN PLACE.

    python export/repair_checkpoint.py <ckpt_dir>... [--base <model>] [--dry-run]

Both defects are in what was written, not in the trained weights, so a checkpoint can be
repaired without retraining:

  1. `.inner.*` fp32 training masters shipped beside the bf16 tensors the norm wrapper
     exports. The pair agrees to within one bf16 ulp -- the wrapper's weight_prefill IS
     the trained tensor, and `.inner.weight` is the original module's parameter kept only
     so functional_call has something to substitute -- so dropping `.inner` is lossless.
     vLLM otherwise fails with "no module or parameter named ... .inner".

  2. `quantization_config.ignore` said ["lm_head"] while the vision tower and the
     linear_attn gate projections were also stored unquantized, so vLLM built quantized
     layers for plain `.weight` tensors and died with "'MergedColumnParallelLinear'
     object has no attribute 'data'". The correct list is derived HERE from the tensors
     actually present: a module is quantized iff it has a `weight_packed`.

     The list also includes the non-Linear PARENT of any MIXED module -- one holding both
     a quantized and an unquantized Linear -- and that parent's norm. At 27B those are
     the 48 linear_attn blocks (quantized out_proj beside ignored in_proj_a/b) and their
     gated norms. That is not a guess about the format: with those entries the derived
     list is SET-EQUAL to the reference nvfp4 config shipped for this model (303 entries,
     zero difference in either direction); without them it is the 207 that `targets:
     ["Linear"]` alone implies. --no-mixed-parents drops back to 207.

Tokenizer/processor files are copied from --base if given: the export writes none, and
vLLM resolves the image processor from the model directory, so multimodal serving dies
with "Can't load image processor" without them.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "preprocessor_config.json", "processor_config.json",
    "chat_template.json", "chat_template.jinja", "video_preprocessor_config.json",
)


def repair(ckpt: Path, base: Path | None, mixed_parents: bool, dry: bool) -> int:
    st = ckpt / "model.safetensors"
    if not st.exists():
        print(f"  no model.safetensors in {ckpt}")
        return 1

    with safe_open(st, framework="pt") as f:
        keys = list(f.keys())
        tensors = {} if dry else {k: f.get_tensor(k) for k in keys}

    inner = [k for k in keys if k.endswith(".inner.weight") or ".inner." in k]
    kept = [k for k in keys if k not in set(inner)]

    # A module is quantized iff it shipped a packed weight.
    quantized = {k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")}
    # ...and unquantized-but-linear iff it shipped a bare .weight and is not a norm.
    # Norms are 1-D; Linears are 2-D. The header carries shapes, so use them.
    with safe_open(st, framework="pt") as f:
        shapes = {k: f.get_slice(k).get_shape() for k in keys}
    # 2-D marks a Linear... and also an EMBEDDING, whose weight is [vocab, hidden]. Both
    # embed_tokens and the vision pos_embed would otherwise be listed as unquantized
    # Linears -- harmless (targets is ["Linear"], so they cannot match) but wrong, and it
    # was the entire 209-vs-207 gap against the module-tree derivation.
    def is_embedding(mod: str) -> bool:
        leaf = mod.rpartition(".")[2]
        return "embed" in leaf

    ignore = sorted(
        k[: -len(".weight")] for k in keys
        if k.endswith(".weight") and len(shapes[k]) == 2
        and k[: -len(".weight")] not in quantized
        and ".inner." not in k
        and not is_embedding(k[: -len(".weight")])
    )

    if mixed_parents:
        # Parents holding BOTH a quantized and an ignored Linear.
        def parent(n):
            return n.rpartition(".")[0]
        mixed = {parent(q) for q in quantized} & {parent(i) for i in ignore}
        extra = sorted(mixed) + sorted(f"{m}.norm" for m in mixed)
        ignore = sorted(set(ignore) | set(extra))

    cfg_path = ckpt / "config.json"
    cfg = json.loads(cfg_path.read_text())
    old_ignore = cfg.get("quantization_config", {}).get("ignore")

    print(f"  tensors {len(keys)} -> {len(kept)}   (.inner dropped: {len(inner)})")
    print(f"  quantized modules: {len(quantized)}   ignore: "
          f"{len(old_ignore or [])} -> {len(ignore)}")
    if dry:
        return 0

    if inner:
        save_file({k: tensors[k] for k in kept}, str(st),
                  metadata={"format": "pt"})
    if "quantization_config" in cfg:
        cfg["quantization_config"]["ignore"] = ignore
        cfg_path.write_text(json.dumps(cfg, indent=2))

    n_copied = 0
    if base is not None:
        for fn in TOKENIZER_FILES:
            src = base / fn
            if src.exists() and not (ckpt / fn).exists():
                shutil.copy2(src, ckpt / fn)
                n_copied += 1
    print(f"  tokenizer/processor files copied: {n_copied}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path, nargs="+")
    p.add_argument("--base", type=Path, default=None,
                   help="model dir to copy tokenizer/processor files from")
    p.add_argument("--no-mixed-parents", action="store_true",
                   help="omit the mixed-parent entries; the reference "
                        "config for this model includes them")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    rc = 0
    for c in a.ckpt:
        print(f"{c}")
        rc |= repair(c, a.base, not a.no_mixed_parents, a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
