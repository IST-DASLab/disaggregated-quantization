"""The block-0 preload must change WHEN block 0 is read, never WHAT is computed.

    python test_preload.py                    # Qwen3-0.6B, seq 4096

preload_first() takes a slot out of the free pool and puts it straight into `filled`, so it
touches the exact handshake that keeps the drive from overwriting a block the GPU is still
reading. The parent class is explicit that getting that wrong is INVISIBLE at short
sequences -- a 68 ms read dwarfs a 0.2 ms compute, so nothing races -- and wide open at long
ones. So this compares against the resident runner, whose weights never move, at a length
where compute per block is not trivially short, and it runs the offload path twice to catch
a preload that is consumed once and then silently skipped on the second rep.

Bitwise equality is the bar: all three paths execute identical kernels on identical weights.
"""

import argparse
import torch

import offload_forward as OF
from offload_forward import (ResidentRunner, SSDOffloadRunner, build, make_templates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--quant", default="bf16")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    blocks, cfg, kwargs_fn, kind_of = build(args.model, args.quant)
    templates = make_templates(blocks, kind_of)
    variants, variant_of = kwargs_fn(cfg, args.seq)
    x = torch.randn(1, args.seq, cfg.hidden_size, dtype=OF.DTYPE, device=OF.DEV)
    fns = [t for t in templates]

    from pathlib import Path
    d = Path.home() / ".bench_offload_blocks" / f"_preloadtest_{args.model.replace('/', '_')}"

    res = ResidentRunner(templates, blocks, kind_of=kind_of)
    ref = res.run(x.clone(), fns, graphs=None, variants=variants, variant_of=variant_of)
    torch.cuda.synchronize()

    ssd = SSDOffloadRunner(templates, blocks, d, threads=16, kind_of=kind_of)
    try:
        print(f"{args.model}, seq={args.seq}, {len(blocks)} blocks, quant={args.quant}")
        print(f"{'path':<26}{'max abs diff':>14}{'bitwise':>10}")
        ok = True
        for tag, pre in (("ssd, no preload", False), ("ssd + preload (rep 1)", True),
                         ("ssd + preload (rep 2)", True)):
            ssd.pre_rep()
            if pre:
                ssd.preload_first()
            got = ssd.run(x.clone(), fns, graphs=None, variants=variants,
                          variant_of=variant_of)
            torch.cuda.synchronize()
            same = torch.equal(got, ref)
            ok &= same
            print(f"{tag:<26}{(got - ref).abs().max().item():>14.3e}"
                  f"{('yes' if same else 'NO'):>10}")
    finally:
        ssd.close()
    print("\n" + ("OK: preload changes only when block 0 is read"
                  if ok else "FAILED: preload changed the result"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
