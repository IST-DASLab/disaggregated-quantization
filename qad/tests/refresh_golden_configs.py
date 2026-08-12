"""Refresh tests/_export_golden/*/config.json after an INTENTIONAL export change.

    python tests/refresh_golden_configs.py            # report the diff, change nothing
    python tests/refresh_golden_configs.py --write    # apply it

Only config.json is ever rewritten, and only after every exported tensor is verified
bit-identical to the golden one. That is the whole point: if the tensors moved, the
change is not "the config format was updated" and this script refuses to bless it.
Tensor VALUES are compared rather than the file hash -- safetensors embeds metadata
(including `step`), so a file md5 reports a difference when the data is identical.

Written because the goldens went stale against fix_serving_fields (the flat
rope_theta / torch_dtype repair that fixed a 35-point serving gap) and nothing
noticed -- run_tests.sh was swallowing the failure.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from export.save import save_checkpoint
from quantizers import REGISTRY, build_quantizer_params

GOLD = Path(__file__).parent / "_export_golden"
TMP = Path("/tmp/qad_golden_refresh")
WRITE = "--write" in sys.argv


def tiny_model():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, False
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).cuda().float()


def build(name):
    m = tiny_model()
    params, _ = build_quantizer_params(name, "")
    REGISTRY[name]["apply"](m, **params)
    for mod in m.modules():
        if hasattr(mod, "act_amax"):
            mod.act_amax.fill_(3.5)
            mod._observed = True
    return m


def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    changed, refused = [], []

    for name in sorted(p.name for p in GOLD.iterdir() if p.is_dir()):
        out = TMP / name
        save_checkpoint(build(name), out, step=0)

        # Compare tensor VALUES, not the file md5: safetensors embeds metadata
        # (including `step`), so a file hash reports a difference when the data is
        # bit-identical. This is the same check test_export.py makes.
        ref = load_file(str(GOLD / name / "model.safetensors"))
        got = load_file(str(out / "model.safetensors"))
        if set(ref) != set(got):
            refused.append(f"{name}: tensor KEYS changed "
                           f"+{sorted(set(got)-set(ref))[:3]} -{sorted(set(ref)-set(got))[:3]}")
            continue
        worst = max((ref[k].float() - got[k].float()).abs().max().item() for k in ref)
        if worst != 0.0:
            refused.append(f"{name}: tensor VALUES changed, max|delta|={worst:.3e}")
            continue

        old = json.loads((GOLD / name / "config.json").read_text())
        new = json.loads((out / "config.json").read_text())
        if old == new:
            print(f"  {name:12} unchanged")
            continue

        keys = sorted(set(old) | set(new))
        diff = [(k, old.get(k, "<absent>"), new.get(k, "<absent>"))
                for k in keys if old.get(k, "<absent>") != new.get(k, "<absent>")]
        print(f"  {name:12} {len(diff)} field(s) differ (tensors byte-identical):")
        for k, a, b in diff:
            print(f"      {k}: {str(a)[:48]} -> {str(b)[:48]}")
        changed.append((name, new))

    if refused:
        print("\nREFUSED -- tensors changed, this is not a config-format update:")
        for r in refused:
            print("  " + r)
        raise SystemExit(1)

    if not changed:
        print("\ngoldens already current")
        return
    if not WRITE:
        print(f"\n{len(changed)} golden config(s) would be rewritten. Re-run with --write.")
        return
    for name, new in changed:
        (GOLD / name / "config.json").write_text(json.dumps(new, indent=2))
    print(f"\nrewrote {len(changed)} golden config(s)")


if __name__ == "__main__":
    main()
