"""Repair `config.json` in already-exported checkpoints (weights untouched).

Exported configs inherited three fields from the TRAINING model that only matter when
the checkpoint is SERVED. Identical weights scored 18.50 GSM8K served with the config
as written and 48.50 once repaired. `vllm serve` reads these verbatim; the in-process
lm-eval path overrides dtype itself, which is why the bug only ever showed up in
served/disaggregated numbers.

    python fix_checkpoint_configs.py --dry-run          # report what would change
    python fix_checkpoint_configs.py                    # rewrite in place
    python fix_checkpoint_configs.py --filter 0.6B-nvfp4-

Only config.json is touched, so this is cheap and reversible per file (a .bak is left
next to each rewritten config).
"""
import argparse
import json
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # qad root, for export/
from export.config_fix import fix_serving_fields
# Reused rather than reimplemented: the repair MUST produce exactly what a fresh export
# now produces, or a re-exported checkpoint and a repaired one would disagree.
from export.save import _fix_rope_fields, _has_per_layer_rope


class _RopeView:
    """Adapts a raw `rope_parameters` dict to the attribute access _fix_rope_fields wants.

    That function normally reads a live transformers config object; here the config is
    only JSON on disk, and the dict is the sole field it consults.
    """

    def __init__(self, rope_parameters):
        self.rope_parameters = rope_parameters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # parent.parent: this file lives in qad/cluster_scripts/, so checkpoints/ is one level
    # up. The old default pointed at qad/cluster_scripts/checkpoints, which does not exist,
    # so the script silently scanned nothing unless --ckpt-dir was passed.
    p.add_argument("--ckpt-dir",
                   default=str(Path(__file__).resolve().parent.parent / "checkpoints"))
    p.add_argument("--filter", default="", help="only tags containing this substring")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    root = Path(args.ckpt_dir)
    changed = scanned = 0
    for cfg_path in sorted(root.glob("*/weights/step_*/**/config.json")) + sorted(root.glob("*/weights/step_*/config.json")):
        if args.filter and args.filter not in str(cfg_path):
            continue
        scanned += 1
        cfg = json.loads(cfg_path.read_text())
        before = json.dumps(cfg, sort_keys=True)
        fix_serving_fields(cfg)
        # Second serving-only config defect, same shape as the first: plain Gemma-3
        # (270m/1b) exported `rope_parameters` keyed BY LAYER TYPE, which vLLM refuses
        # before loading any weight ("rope_parameters should have a 'rope_type' key").
        # Repaired here so checkpoints already on disk become servable without retraining.
        # No-op on Qwen, whose rope_parameters is flat and already valid.
        if _has_per_layer_rope(cfg):
            _fix_rope_fields(cfg, _RopeView(cfg.pop("rope_parameters", None)))
        # A file-level repair can only recover the sliding-layer theta from the nested
        # dict it is deleting. If that dict was absent, the value is simply not in the
        # file and this tool CANNOT fix it -- say so loudly rather than reporting a clean
        # scan, because the resulting checkpoint serves silently wrong (sliding layers at
        # the full-attention 1e6) instead of failing. Re-export it instead.
        if cfg.get("model_type") == "gemma3_text" and "rope_local_base_freq" not in cfg:
            print(f"  UNREPAIRABLE {cfg_path.relative_to(root)}: no rope_local_base_freq "
                  f"and no rope_parameters to recover it from — RE-EXPORT this step")
        if json.dumps(cfg, sort_keys=True) == before:
            continue
        changed += 1
        rel = cfg_path.relative_to(root)
        if args.dry_run:
            print(f"  would fix {rel}")
            continue
        if not args.no_backup:
            shutil.copy2(cfg_path, cfg_path.with_suffix(".json.bak"))
        cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"scanned={scanned} changed={changed}{' (DRY RUN)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
