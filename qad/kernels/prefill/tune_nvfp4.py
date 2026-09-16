"""Measure the best (kernel, chunk) plan per projection shape, and write it into
nvfp4_tuning.py.

    python tune_nvfp4.py                              # every model in offload_prefill.csv
    python tune_nvfp4.py --models Qwen/Qwen3-8B
    python tune_nvfp4.py --rows 8192 16384 32768      # narrower sweep, faster
    python tune_nvfp4.py --show                       # print the table, measure nothing
    python tune_nvfp4.py --dry-run                    # measure, print, do not write

PORTING TO ANOTHER BOX. Run it there. The table is keyed on
`torch.cuda.get_device_name(0)`, so a 5090's entries land beside GB10's rather than
replacing them, and a device with no entry falls back to `NVFP4Linear`'s own heuristic --
i.e. the untuned behaviour is the old behaviour, not a broken one.

WHAT IS TIMED. The GEMM only, from an already-quantized activation, because the
activation quantization costs the same total work however the rows are split and so cannot
change which plan wins. That also makes the measurement match `forward_prequantized`, the
path the fused GeGLU routes `down` through. The `torch.cat` that reassembles a chunked
result IS included -- it is real, it is only paid by chunked plans, and on a whole block it
hands back most of what chunking wins.

WHAT IS CHECKED. Every candidate must be `torch.equal` to the unchunked vLLM call before it
is allowed to be timed. All four paths here are bitwise identical on GB10, so a plan is a
speed choice and never an accuracy one; if that ever stops being true on some box, the
candidate is rejected loudly and the sweep continues, rather than either tuning onto a
kernel that computes something else or throwing away every shape already measured.

WHY A PLAN MUST WIN TWICE. Every candidate is timed in two passes, in opposite orders, and
a non-default plan is recorded only if it beats plain unchunked vLLM by `--margin` in BOTH.
One pass cannot tell a kernel from a dice roll on this box: two tunings run back to back
disagreed on 8 of 33 shapes, every disagreement inside a 1.05-1.15x band, and one of them
invented a chunked-flashinfer entry the other never saw. The full-model drift control says
why -- identical code, shape and sequence length measured 1832 / 1895 / 2034 / 2047 ms over
one afternoon, an 11.7% spread with no monotone trend.

What survives that test is not subtle and is perfectly reproducible: the tall-output
`gate_up` shapes at 16k and 32k rows, 1.9-2.5x over unchunked vLLM, identical in both
tunings. What it filters out never had a defensible claim to a table entry.

The shapes come from model CONFIGS, not checkpoints -- the four fused projections
(qkv / o / gate_up / down) are determined by hidden_size, intermediate_size and the head
counts, so tuning needs no weights and no downloads beyond a config.json.
"""

import argparse
import pathlib
import re
import time

import torch

import vllm._custom_ops as ops
from triton.testing import do_bench
from nvfp4_linear import ACT_AMAX, E4M3_MAX, FP4_MAX, global_encode_scale

HERE = pathlib.Path(__file__).parent
TABLE_FILE = HERE / "nvfp4_tuning.py"
BEGIN = "# --- BEGIN GENERATED TABLE"
END = "# --- END GENERATED TABLE ---"
DEV = torch.device("cuda")

# The block-scale layout is built in 128-row tiles, so every chunk candidate must be a
# multiple of 128 or the slice stops corresponding to those rows.
ALIGN = 128


def backends():
    """Callables that do the fp4 GEMM, keyed by the name stored in the table."""
    out = {"vllm": lambda a, b, sa, sb, alpha: ops.cutlass_scaled_fp4_mm(
        a, b, sa, sb, alpha, torch.bfloat16)}
    try:
        from flashinfer.gemm import mm_fp4
        out["flashinfer"] = lambda a, b, sa, sb, alpha: mm_fp4(
            a, b.T, sa, sb, alpha, torch.bfloat16, block_size=16, backend="cutlass")
    except Exception as exc:                      # noqa: BLE001 - report, do not fail
        print(f"  flashinfer unavailable, tuning vLLM only: {str(exc).splitlines()[0][:80]}")
    return out


def projection_shapes(model: str):
    """The fused projections as (name, out_features, in_features).

    Read off the BLOCK the harness actually runs when the architecture has more than one
    layer kind, rather than re-derived from hidden/head counts. Qwen3.5 is why: it fuses
    differently from the checkpoint and from every other model here, so the generic rule
    below gets two of its five shapes wrong --

        attn qkv   the config rule gives (8192, 5120); the block runs (14336, 5120),
                   because attn_output_gate makes the q half twice as wide.
        gdn in_proj  a single (16480, 5120) GEMM holding qkv|z|b|a, which the generic rule
                   does not know exists at all -- and it is on 48 of the 64 layers.

    Tuning the generic shapes would therefore optimise one GEMM the model never issues and
    leave the dominant one untuned.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    cfg = getattr(cfg, "text_config", cfg)

    if "linear_attention" in tuple(getattr(cfg, "layer_types", ()) or ()):
        import torch.nn as nn
        from qwen35_block import Qwen35FullBlock, Qwen35LinearBlock
        seen, out = set(), []
        for cls in (Qwen35LinearBlock, Qwen35FullBlock):
            with torch.device("meta"):        # structure only: no weights, no device memory
                blk = cls(cfg, dtype=torch.bfloat16, device="meta")
            for name, mod in blk.named_modules():
                if isinstance(mod, nn.Linear):
                    key = (mod.out_features, mod.in_features)
                    if key not in seen:
                        seen.add(key)
                        out.append((name.split(".")[-1], *key))
        return out

    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    heads, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or hidden // heads
    q_dim, kv_dim = heads * head_dim, kv * head_dim
    return [("qkv", q_dim + 2 * kv_dim, hidden),
            ("o", hidden, q_dim),
            ("gate_up", 2 * inter, hidden),
            ("down", hidden, inter)]


class _ShapeOnly:
    """Just enough of NVFP4Linear to ask what it WOULD do, with no weights allocated.

    Borrows the real methods rather than restating the rule, so the incumbent this tuner
    measures cannot drift away from the one the model runs.
    """

    def __init__(self, n: int, k: int):
        from nvfp4_linear import NVFP4Linear
        self.out_features, self.in_features = n, k
        self._cls = NVFP4Linear

    def chunk(self, rows: int) -> int:
        return self._cls._chunk_rows(self, rows)

    def __getattr__(self, item):
        # L2_BYTES, MAX_OUT_ELEMS, CHUNK_TABLE, _weight_bytes, _chunk_heuristic
        return getattr(self._cls, item).__get__(self) if callable(
            getattr(self._cls, item, None)) else getattr(self._cls, item)


def chunk_candidates(rows: int, n: int = 0, k: int = 0, divisors=(2,)):
    """Two plans: one call, or two halves -- plus whatever the untuned code already does.

    THE INCUMBENT IS ALWAYS A CANDIDATE. Without it, "best of what I tried" stops meaning
    "at least as good as today": on gemma-3-12b's gate_up at 32768 rows the {0, half} search
    picked flashinfer with a 16384 chunk at 63.0 ms and called it a 1.54x win, while
    _chunk_rows' own CHUNK_TABLE entry for that shape (6784) had measured 46.6 ms. A tuner
    that can regress the thing it is tuning is worse than no tuner, because the regression
    arrives wearing a speedup.

    The set was 0 plus rows//{2,3,4,6,8}, which is 12 candidates per (shape, row count) --
    3828 timings a tune once each is run in both orders, ~20 minutes of GPU. The finer
    divisors did not earn that. Chunk granularity is a second-order knob by the repo's own
    measurement: worth ~1.28x on an isolated projection but ~1.03x on a whole block, because
    the `torch.cat` that reassembles the slices hands most of it back. What actually matters
    is the binary -- does this shape need splitting at all, and which kernel runs it.

    The cost is real and worth stating: the previous table chose rows//6 for one shape at
    16384 (2688) and rows//8 for another at 32768 (4096), so those now pick between one call
    and two halves instead. Tuning drops to ~7 minutes, which is what makes re-tuning on a
    new box a routine step rather than a project.
    """
    out = [0]
    for d in divisors:
        c = (rows // d) // ALIGN * ALIGN
        if ALIGN <= c < rows and c not in out:
            out.append(c)
    if n and k:
        inc = _ShapeOnly(n, k).chunk(rows)
        if ALIGN <= inc < rows and inc not in out:
            out.append(inc)
    return out


def run_plan(kernel, x_fp4, x_bs, w_fp4, w_bs, alpha, rows, chunk):
    if not chunk or chunk >= rows:
        return kernel(x_fp4, w_fp4, x_bs, w_bs, alpha)
    return torch.cat([kernel(x_fp4[i:i + chunk], w_fp4, x_bs[i:i + chunk], w_bs, alpha)
                      for i in range(0, rows, chunk)], 0)


def timed(fn, warmup_ms: int = 25, rep_ms: int = 200):
    """Median ms from triton.testing.do_bench.

    Hand-rolled event timing got this wrong twice over, and both errors pushed the same way.

    IT CLEARS L2 BEFORE EVERY RUN, which is the whole point here. Timing one GEMM 7 times
    back to back leaves its weight hot in this box's 24 MiB L2 -- but the collapse this
    table exists to route around only happens when the weight MISSES L2 (nvfp4_linear's
    L2_BYTES test), and in the real model 48 layers of distinct weights evict each other
    every forward. So a cache-hot microbenchmark measures a regime the model never runs in,
    and was flattering the fp4 kernels: 1.9-2.5x at the bench against 1.24x for the same
    projection inside the captured layer.

    IT AMORTIZES. do_bench sizes the loop to fill `rep_ms` and times the batch, instead of
    wrapping a single call in two syncs and an event pair -- ~5-10 us of overhead that is a
    real fraction of a 0.2 ms GEMM, and the reason the small shapes read as noise.

    rep_ms is raised from the 100 ms default because the largest shapes here take ~90 ms a
    call, which would leave a single iteration to take a median of.
    """
    return do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")


def tune_shape(n: int, k: int, rows: int, kernels, divisors=(2,)):
    """Best (backend, chunk, ms) for one (shape, rows), and the reference time."""
    x = torch.randn(rows, k, dtype=torch.bfloat16, device=DEV) * (ACT_AMAX / 6)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.05
    x_gs = torch.tensor((FP4_MAX * E4M3_MAX) / ACT_AMAX, dtype=torch.float32, device=DEV)
    w_gs = global_encode_scale(w)
    alpha = ((1.0 / x_gs) * (1.0 / w_gs)).to(torch.float32)
    x_fp4, x_bs = ops.scaled_fp4_quant(x, x_gs, is_sf_swizzled_layout=True,
                                       backend="cutlass")
    w_fp4, w_bs = ops.scaled_fp4_quant(w, w_gs, is_sf_swizzled_layout=True,
                                       backend="cutlass")
    del x, w

    ref = run_plan(kernels["vllm"], x_fp4, x_bs, w_fp4, w_bs, alpha, rows, 0)

    # Build the candidate list first, bitwise-checking as we go, and time nothing yet.
    plans = []
    for name, kernel in kernels.items():
        for chunk in chunk_candidates(rows, n, k, divisors):
            def go(kernel=kernel, chunk=chunk):
                return run_plan(kernel, x_fp4, x_bs, w_fp4, w_bs, alpha, rows, chunk)
            try:
                got = go()
            except Exception as exc:              # noqa: BLE001 - shape may be unsupported
                print(f"      {name}/{chunk or '-'}: skipped ({str(exc).splitlines()[0][:60]})")
                continue
            # Rejected, not fatal: a candidate that disagrees is dropped loudly and the
            # sweep continues. Dying here would throw away every shape already measured,
            # and the guard's job is to keep a wrong kernel out of the table, not to end
            # the run.
            if not torch.equal(got, ref):
                print(f"      {name}/{chunk or '-'}: REJECTED, not bitwise equal to "
                      f"unchunked vLLM at N={n} K={k} rows={rows}")
                del got
                continue
            del got
            plans.append((name, chunk, go))

    # ORDER BIAS. Timing the candidates once, in a fixed order, measures the clock ramp as
    # much as the kernels: the first plan timed after the allocation and quantization above
    # runs at a lower SM clock than the tenth. Because `vllm` is yielded first, the plain
    # unchunked call -- the baseline every comparison is made against -- was always the one
    # paying that ramp, biasing every shape toward whatever ran later. The first table built
    # this way put 11 of its 18 flashinfer picks at exactly rows=4096, all in a 1.05-1.12x
    # band, which is not how a kernel behaves.
    #
    # So: warm the device on the reference plan, then time every candidate TWICE, in
    # opposite orders, and keep the faster. Position in the order can then only cost a
    # candidate in one of its two passes.
    for _ in range(10):
        run_plan(kernels["vllm"], x_fp4, x_bs, w_fp4, w_bs, alpha, rows, 0)
    torch.cuda.synchronize()

    passes = []
    for sequence in (plans, list(reversed(plans))):
        this = {}
        for name, chunk, go in sequence:
            this[(name, chunk)] = timed(go)
        passes.append(this)

    del x_fp4, x_bs, w_fp4, w_bs, ref
    torch.cuda.empty_cache()
    best = {k: min(p[k] for p in passes) for k in passes[0]}
    results = sorted((t, name, chunk) for (name, chunk), t in best.items())
    return results, passes


def pick(results, passes, margin: float, base_key=("vllm", 0)):
    """Fastest plan that beats the incumbent by `margin` in EVERY pass, else the incumbent.

    `base_key` is the plan the untuned code would run -- vLLM at whatever chunk _chunk_rows
    picks -- NOT plain unchunked vLLM. Measuring the margin against unchunked would let a
    candidate that is slower than today's behaviour be recorded as an improvement, simply
    because today's behaviour was never the thing it was compared with.

    A single pass cannot tell a kernel from a dice roll on this box. Two tunings run back to
    back disagreed on 8 of 33 shapes -- one gained a chunked-flashinfer entry the other never
    saw -- and every disagreement sat in the 1.05-1.15x band. The full-model drift control
    says why: the same code, same shape, same sequence length measured 1832 / 1895 / 2034 /
    2047 ms across one afternoon, an 11.7% spread with no monotone trend, so a 5% "win"
    measured once is indistinguishable from the box's mood.

    Requiring the win to repeat in both passes is a cheap stability test that a lucky
    measurement cannot pass. It keeps everything that matters: every decision worth >=1.2x
    -- the tall-output gate_up shapes at 16k and 32k rows -- reproduced identically across
    both tunings.
    """
    if base_key not in passes[0]:
        base_key = ("vllm", 0)
    if base_key not in passes[0]:
        return results[0]
    for ms, be, chunk in results:                  # fastest first
        if (be, chunk) == base_key:
            break
        if all(p[(be, chunk)] * margin <= p[base_key] for p in passes):
            return ms, be, chunk
    # The incumbent's OWN chunk, not 0. Writing ("vllm", 0) here recorded "one unchunked
    # call" whenever the incumbent won -- and because a table entry overrides _chunk_rows
    # entirely, that did not fall back to the incumbent, it DISABLED it. Gemma-3-12b's
    # gate_up incumbent is vllm/6784 at ~46 ms; the table said unchunked at ~97 ms, and the
    # model went 6049 -> 8989 ms at 32768, a 48% regression wearing the label "no change".
    return min(p[base_key] for p in passes), base_key[0], base_key[1]


def render(tables: dict) -> str:
    lines = [BEGIN + " (tune_nvfp4.py rewrites everything between these markers) ---",
             "# device -> {(out_features, in_features): ((max_rows, backend, chunk), ...)}",
             "# chunk == 0 means \"one call, no chunking\". A lookup takes the first rule "
             "whose max_rows",
             "# is >= the row count being run, so the rules must stay sorted by max_rows.",
             "DEVICE_TABLES = {"]
    for device in sorted(tables):
        lines.append(f"    {device!r}: {{")
        for (n, k) in sorted(tables[device]):
            rules = tables[device][(n, k)]
            body = ", ".join(f"({r}, {b!r}, {c})" for r, b, c in rules)
            lines.append(f"        ({n}, {k}): ({body},),")
        lines.append("    },")
    lines.append("}")
    lines.append(END)
    return "\n".join(lines)


def write_table(tables: dict):
    text = TABLE_FILE.read_text()
    start, end = text.index(BEGIN), text.index(END) + len(END)
    TABLE_FILE.write_text(text[:start] + render(tables) + text[end:])
    print(f"\nwrote {TABLE_FILE}")


def load_existing():
    import importlib
    import nvfp4_tuning
    importlib.reload(nvfp4_tuning)
    return {d: dict(v) for d, v in nvfp4_tuning.DEVICE_TABLES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: every model in offload_prefill.csv")
    ap.add_argument("--rows", nargs="+", type=int,
                    default=[2048, 4096, 8192, 16384, 32768],
                    help="row counts to tune at; a lookup rounds up to the nearest")
    ap.add_argument("--source", default="offload_prefill.csv")
    ap.add_argument("--divisors", nargs="+", type=int, default=[2],
                    help="chunk candidates are rows//d for each d here, plus one unchunked "
                         "call and whatever _chunk_rows would pick. Default [2] is the cheap "
                         "sweep; pass more (e.g. 2 3 4 6 8 12 16) to tune one model harder")
    ap.add_argument("--margin", type=float, default=1.05,
                    help="a non-default plan must beat unchunked vLLM by this factor to be "
                         "recorded; below it the difference is noise and the table would "
                         "pin whichever way the dice fell (default 1.05)")
    ap.add_argument("--show", action="store_true", help="print the table and exit")
    ap.add_argument("--dry-run", action="store_true", help="measure but do not write")
    args = ap.parse_args()

    tables = load_existing()
    if args.show:
        for device, shapes in tables.items():
            print(f"\n{device}")
            for (n, k), rules in sorted(shapes.items()):
                print(f"  ({n:>6}, {k:>6})  " +
                      "  ".join(f"<={r}: {b}/{c or '-'}" for r, b, c in rules))
        return

    models = args.models
    if models is None:
        import pandas as pd
        models = list(pd.read_csv(HERE / args.source)["model"].unique())

    device = torch.cuda.get_device_name(0)
    kernels = backends()
    print(f"{device}   kernels: {', '.join(kernels)}   rows: {args.rows}\n")

    shapes = {}
    for model in models:
        for name, n, k in projection_shapes(model):
            shapes.setdefault((n, k), []).append(f"{model.split('/')[-1]} {name}")

    table = dict(tables.get(device, {}))
    t0 = time.perf_counter()
    for (n, k), users in sorted(shapes.items()):
        print(f"({n:>6}, {k:>6})  {', '.join(users[:3])}"
              f"{' ...' if len(users) > 3 else ''}")
        rules = []
        for rows in sorted(args.rows):
            results, passes = tune_shape(n, k, rows, kernels,
                                          divisors=tuple(args.divisors))
            if not results:
                continue
            inc_chunk = _ShapeOnly(n, k).chunk(rows)
            inc_key = ("vllm", 0 if inc_chunk >= rows else inc_chunk)
            best_ms, best_be, best_chunk = pick(results, passes, args.margin, inc_key)
            base = next((ms for ms, be, c in results if be == "vllm" and c == 0), None)
            # TWO baselines, because only one of them is the decision.
            #
            # `vllm unchunked` is the naive reference. `incumbent` is the fastest plan
            # vLLM alone can reach -- chunked or not -- which is what NVFP4Linear already
            # did before this table existed, since _chunk_rows fires on exactly the shapes
            # where the unchunked kernel collapses. Quoting the win against the unchunked
            # number turns "2.5x" into a headline for a change worth ~1.0x end to end:
            # Qwen3-4B's gate_up scored 1.86x that way and moved its model 1904 -> 1925 ms.
            incumbent = next((ms for ms, be, c in results if (be, c) == inc_key), base)
            rules.append((rows, best_be, best_chunk))
            print(f"    rows={rows:>6}  best {best_be}/{best_chunk or '-':<6} "
                  f"{best_ms:7.2f} ms   vs best vllm {incumbent:7.2f} ms "
                  f"({incumbent / best_ms:.2f}x)   [vs vllm unchunked {base:7.2f} ms "
                  f"({base / best_ms:.2f}x)]", flush=True)
        # Collapse runs of identical plans: (2048, vllm, 0), (4096, vllm, 0) -> one rule.
        collapsed = []
        for rows, be, chunk in rules:
            if collapsed and collapsed[-1][1] == be and collapsed[-1][2] == chunk:
                collapsed[-1] = (rows, be, chunk)
            else:
                collapsed.append((rows, be, chunk))
        # A shape's rule list is REPLACED, not merged, so tuning a subset of row counts
        # silently discards the rules for every other row count -- and because a lookup
        # rounds UP, the shorter lengths then resolve to the longest tuned plan, which can
        # be a different kernel family entirely. Measured here: re-tuning only rows=32768
        # dropped `down`'s flashinfer/1024 rule and left 8192 resolving to vllm/2048,
        # worth ~7.5 ms per layer (13% of that model's pass). Warn rather than merge:
        # merging would keep stale rules from an older, possibly slower, kernel build.
        prev = tables.get(device, {}).get((n, k))
        if prev is not None and len(prev) > len(collapsed):
            lost = sorted({r for r, _, _ in prev} - {r for r, _, _ in collapsed})
            print(f"    WARNING: replacing {len(prev)} rules with {len(collapsed)}; "
                  f"rules for rows {lost} are being discarded. Re-run with the FULL "
                  f"--rows set to keep them.", flush=True)
        table[(n, k)] = tuple(collapsed)
    print(f"\ntuned {len(shapes)} shapes in {time.perf_counter() - t0:.0f}s")

    tables[device] = table
    if args.dry_run:
        print(render(tables))
    else:
        write_table(tables)


if __name__ == "__main__":
    main()
