"""Checks for the zero-ssd (ODP carve-out) protocol.

    python test_zero_ssd.py                 # Qwen3-0.6B
    python test_zero_ssd.py --model google/gemma-3-4b-it

zero-ssd claims a timed boundary that starts with decode weights resident and ends with
them restored. Several ways that can be quietly wrong, each checked here:

  * the first block gets pre-loaded anyway, so the cold start is not measured;
  * the restoration lands somewhere other than the arena the prefill slots used, which
    would make it an extra allocation rather than a carve-out;
  * the restored bytes are not the payload's, so "restored" means nothing;
  * memory reuse perturbs the computation, which must not happen;
  * `_preloaded` survives graph capture or a previous run and block 0 is skipped.

The latency check is directional on purpose: zero-ssd must be SLOWER than ssd, because it
adds a cold first-block read and a full carve-out restoration that ssd never pays. A
zero-ssd time at or below ssd would mean one of those is not happening.
"""

import argparse
import torch

import offload_forward as OF
from offload_forward import (ResidentRunner, SSDOffloadRunner, ZeroSSDRunner, build,
                             make_templates, measure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--quant", default="nvfp4")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    blocks, cfg, kwargs_fn, kind_of = build(args.model, args.quant)
    templates = make_templates(blocks, kind_of)
    variants, variant_of = kwargs_fn(cfg, args.seq)
    x = torch.randn(1, args.seq, cfg.hidden_size, dtype=OF.DTYPE, device=OF.DEV)
    from pathlib import Path
    d = Path.home() / ".bench_offload_blocks" / args.model.replace("/", "_") / args.quant

    res = ResidentRunner(templates, blocks, kind_of=kind_of)
    ref = res.run(x.clone(), templates, graphs=None, variants=variants,
                  variant_of=variant_of)
    torch.cuda.synchronize()

    z = ZeroSSDRunner(templates, blocks, d, threads=16, kind_of=kind_of)
    ok = True
    try:
        print(f"{args.model}  {args.quant}  seq={args.seq}")
        print(f"  P (prefill bytes)   {z.prefill_bytes/2**30:8.3f} GiB over {z.n} blocks")
        print(f"  C (slot allocation) {z.carveout_bytes/2**30:8.3f} GiB over {z.n_slots} slots")
        print(f"  SSD per request     {(z.prefill_bytes+z.carveout_bytes)/2**30:8.3f} GiB")
        print(f"  payload             {'SYNTHETIC' if z.synthetic_payload else 'real'}\n")

        # arena holds decode bytes before any prefill: that is the documented start state
        t = z.verify_restored()
        print(f"  arena seeded with payload before first cycle : {'yes' if t else 'NO'}")
        ok &= t

        for rep in (1, 2):
            z.pre_rep()
            got = z.run(x.clone(), templates, graphs=None, variants=variants,
                        variant_of=variant_of)
            torch.cuda.synchronize()
            same = torch.equal(got, ref)
            rest = z.verify_restored()
            print(f"  rep {rep}: output == resident {'yes' if same else 'NO':<4}"
                  f"  arena restored {'yes' if rest else 'NO':<4}"
                  f"  _preloaded={z._preloaded}")
            ok &= same and rest and z._preloaded is None

        # no untimed preload: calling it must leave nothing in a slot
        z.preload_first()
        ok &= z._preloaded is None
        print(f"  preload_first() is a no-op                   : "
              f"{'yes' if z._preloaded is None else 'NO'}")

        ssd = SSDOffloadRunner(templates, blocks, d, threads=16, kind_of=kind_of)
        t_ssd = measure(cfg, ssd, templates, args.seq, kwargs_fn, reps=3, use_graphs=False)
        t_zero = measure(cfg, z, templates, args.seq, kwargs_fn, reps=3, use_graphs=False)
        ssd.close()
        print(f"\n  ssd       {t_ssd:8.1f} ms")
        print(f"  zero-ssd  {t_zero:8.1f} ms   ({t_zero/t_ssd:.2f}x)")
        slower = t_zero > t_ssd
        ok &= slower
        if not slower:
            print("  FAIL: zero-ssd must be slower -- it adds a cold block 0 and a "
                  f"{z.carveout_bytes/2**30:.2f} GiB restoration")
    finally:
        z.close()
    print("\n" + ("OK: carve-out protocol behaves as specified" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
