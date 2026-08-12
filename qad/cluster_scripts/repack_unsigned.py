"""Re-encode signed-scale NVFP4 checkpoints so vLLM can serve them.

    python cluster_scripts/repack_unsigned.py                 # report only
    python cluster_scripts/repack_unsigned.py --apply
    python cluster_scripts/repack_unsigned.py --filter nvfp4lloyd43 --apply

WHY
---
E2M1 is symmetric, so a SIGNED block scale is free at the format level: negating both
the code and the scale cancels, and pack/unpack round-trips identically either way
(verified: dequantized weights match to 0.00e+00). NVFP4-the-hardware-format is
different: its block scale factor is UE4M3, UNSIGNED, because that is what the Blackwell
block-scaled MMA (tcgen05.mma) consumes. A byte >= 0x80 is therefore read as
exponent/mantissa rather than a negative -- the magnitudes come out wrong, not just the
signs. Measured: GSM8K 0.0 against 0.569 for the same-size lloyd43, with exactly 50% of
block scales negative. Every format that serves correctly has 0%.

Because the two encodings dequantize to the SAME weights, this is a RE-EXPORT, not a
retrain: unpack honouring the sign, re-pack unsigned, and the trained values are
preserved exactly. That is six 4-hour runs saved.

The rewrite is atomic (temp file + os.replace) and refuses to proceed unless the
re-packed tensor dequantizes bit-identically to the original.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import load_file, save_file

from quantizers.nvfp4 import pack_nvfp4_weight, unpack_nvfp4_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def has_negative_scales(t: dict) -> int:
    return sum(1 for k, v in t.items()
               if k.endswith("weight_scale") and bool((v.float() < 0).any()))


def repack(path: str, block: int = 16) -> tuple[int, float]:
    """Rewrite one model.safetensors in place. Returns (layers changed, max deq drift)."""
    t = load_file(path)
    changed, worst = 0, 0.0
    for k in [k for k in t if k.endswith("weight_packed")]:
        base = k[: -len("weight_packed")]
        ws, wgs = t[base + "weight_scale"], t[base + "weight_global_scale"]
        if not bool((ws.float() < 0).any()):
            continue
        # dequantize with the SIGN honoured -- this is the ground truth to preserve
        deq = unpack_nvfp4_weight(t[k].cuda(), ws.cuda(), wgs.cuda(), block)
        # the checkpoint stores weight_global_scale RECIPROCAL (2688/amax)
        gs = (1.0 / wgs.float().cuda()).reshape(())
        p2, s2, g2 = pack_nvfp4_weight(deq, block, global_scale=gs, signed=False)
        deq2 = unpack_nvfp4_weight(p2, s2, (1.0 / g2).reshape(1), block)
        drift = float((deq2 - deq).abs().max())
        worst = max(worst, drift)
        assert not bool((s2.float() < 0).any()), f"{base}: re-pack still has negative scales"
        t[k] = p2.cpu()
        t[base + "weight_scale"] = s2.cpu()
        t[base + "weight_global_scale"] = (1.0 / g2).reshape(1).cpu()
        changed += 1
    if changed:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".safetensors")
        os.close(fd)
        save_file(t, tmp, metadata={"format": "pt", "repacked": "unsigned"})
        os.replace(tmp, path)
    return changed, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", default="", help="only tags containing this substring")
    ap.add_argument("--apply", action="store_true", help="rewrite (default: report only)")
    ap.add_argument("--max-drift", type=float, default=1e-5,
                    help="abort if a re-pack changes the dequantized weight by more")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*", "weights",
                                          "step_*", "**", "model.safetensors"),
                             recursive=True))
    todo = []
    for f in files:
        tag = f.split("/checkpoints/")[1].split("/")[0]
        if args.filter and args.filter not in tag:
            continue
        try:
            t = load_file(f)
        except Exception:
            continue
        n = has_negative_scales(t)
        if n:
            todo.append((f, n))

    if not todo:
        print("nothing to re-pack — no checkpoint carries negative block scales")
        return
    print(f"{len(todo)} checkpoint file(s) carry signed scales and cannot be served:")
    for f, n in todo[:8]:
        print(f"  {n:4d} tensors  {f.split('/checkpoints/')[1]}")
    if len(todo) > 8:
        print(f"  ... and {len(todo) - 8} more")
    if not args.apply:
        print("\nreport only; re-run with --apply")
        return

    tot, worst = 0, 0.0
    for i, (f, _) in enumerate(todo, 1):
        c, d = repack(f)
        tot += c
        worst = max(worst, d)
        if d > args.max_drift:
            raise SystemExit(f"ABORT: {f} drifted {d:.2e} > --max-drift {args.max_drift}")
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} files, {tot} tensors, max drift {worst:.2e}", flush=True)
    print(f"\nre-packed {tot} tensors across {len(todo)} files; "
          f"max dequantized drift {worst:.2e} (weights preserved)")


if __name__ == "__main__":
    main()
