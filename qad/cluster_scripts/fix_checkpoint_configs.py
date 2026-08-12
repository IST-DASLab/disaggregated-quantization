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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
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
