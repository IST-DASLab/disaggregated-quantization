"""Sweep every CUDA launch config on every real projection shape and emit the table.

    python tune_cuda.py            # sweep and print
    python tune_cuda.py --emit     # also print the dict to paste into cuda_gemv.py

The dispatch is STATIC per shape (the pattern Marlin and AllSpark both use): a GEMV at
batch 1 is short enough that autotuning at runtime would cost more than it saves, and the
set of shapes a served model uses is known ahead of time. So the table is measured once,
here, and compiled in.

`gpl` is the parameter that actually moves: a warp covers 1024*gpl weights per step, so
picking gpl > K/1024 leaves lanes idle. The rest is a trade between amortizing the staged
x over more rows and having enough CTAs to fill the SMs.
"""

import argparse

import torch
import triton

from lloyd43.bench import QWEN3_MODELS, TUNE_SHAPES, model_linears, _setup
from lloyd43.format import FORMATS
from lloyd43.cuda_gemv import CUDA_CONFIGS, _heuristic, gemv_lloyd43_cuda
from lloyd43.reference import gemv_bf16
from lloyd43.roofline import flat_read_ms

BYTES_BF16, BYTES_L43 = 2.0, 3 / 8 + 1 / 16


def tune_shape(N, K, warmup=25, rep=100, fmt=None):
    x, wbf, packed, bs, gs = _setup(N, K, fmt=fmt)
    t_bf16 = triton.testing.do_bench(lambda: gemv_bf16(x, wbf), warmup=warmup, rep=rep)
    results = []
    for w, r, g, sx in CUDA_CONFIGS:
        fn = lambda: gemv_lloyd43_cuda(x, packed, bs, gs, K, warps=w,
                                       rows_per_warp=r, gpl=g, stage_x=sx, fmt=fmt)
        try:
            fn()
            results.append(
                (triton.testing.do_bench(fn, warmup=warmup, rep=rep), w, r, g, sx))
        except Exception:
            continue
    results.sort()
    heur = _heuristic(N, K)
    t_heur = next((t for t, *c in results if tuple(c) == heur), float("nan"))
    bpw = (fmt.bytes_per_weight if fmt is not None else BYTES_L43)
    roof = flat_read_ms(N * K * BYTES_BF16) / flat_read_ms(N * K * bpw)
    del wbf, packed, bs
    torch.cuda.empty_cache()
    return t_bf16, results, t_heur, roof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="print the dict to paste in")
    ap.add_argument("--format", nargs="+", default=["lloyd43"],
                    choices=sorted(FORMATS) + ["all"],
                    help="which grid to tune; the 2-bit kernel wants different tiles")
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}   "
          f"{len(CUDA_CONFIGS)} configs x {len(TUNE_SHAPES)} shapes\n")
    print(f"{'N':>7}{'K':>7}  {'best cfg':<16}{'us':>8}{'vs bf16':>9}"
          f"{'roof':>7}{'%roof':>7}{'vs heur':>9}")

    fams = sorted(FORMATS) if "all" in args.format else args.format
    for fam in fams:
        fmt = FORMATS[fam]
        print(f"\n########## {fam}: {fmt.bits} bits, {fmt.bytes_per_weight:.4f} B/weight")
        table = _tune_one(fmt, args)


def _tune_one(fmt, args):
    table = {}
    for N, K in TUNE_SHAPES:
        t_bf16, res, t_heur, roof = tune_shape(N, K, fmt=fmt)
        if not res:
            print(f"{N:>7}{K:>7}  no config succeeded")
            continue
        best, w, r, g, sx = res[0]
        table[(N, K)] = (w, r, g, sx)
        print(f"{N:>7}{K:>7}  w{w:<2} r{r} g{g} {'smem' if sx else 'reg '}{best*1000:8.1f}"
              f"{t_bf16/best:8.2f}x{roof:6.2f}x{100*(t_bf16/best)/roof:6.0f}%"
              f"{t_heur/best:8.2f}x")

    if args.emit:
        print(f"\n_TABLE_{fmt.name.upper()} = {{")
        for (N, K), (w, r, g, sx) in sorted(table.items()):
            print(f"    ({N:>5}, {K:>5}): ({w:>2}, {r}, {g}, {sx}),")
        print("}")
    return table


if __name__ == "__main__":
    main()
