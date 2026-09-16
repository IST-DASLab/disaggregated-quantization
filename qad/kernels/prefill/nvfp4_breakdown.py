"""Where NVFP4 prefill time goes, attributed INSIDE the real captured graph.

    python nvfp4_breakdown.py                    # Gemma3-12B at 16k
    python nvfp4_breakdown.py --model qwen8b --seq 16384
    python nvfp4_breakdown.py --dump             # every kernel, unbucketed

WHY THIS IS NOT A SUM OF ISOLATED MEASUREMENTS
----------------------------------------------
The first version of this script timed each component separately at the shape the model
runs and added them up. Two rows never closed: the parts missed 21% of Qwen3-8B's real bf16
layer but only 7% of its NVFP4 layer, which made the residual appear to speed up 4.34x.
Nothing in that residual -- RMSNorm, RoPE, residual adds -- differs between the arms, so a
4.34x there is not physics, it is the model being wrong.

The kernel names say why. In bf16 the norm is its own pass over the activation:

    triton_per_fused__fused_rms_norm_0

In NVFP4 inductor folds it into the quantization that follows it:

    triton_per_fused__fused_rms_norm_scaled_fp4_quant_view_zeros_0

So the norm's traffic is real and unavoidable in bf16, and largely free in NVFP4 because
something already had to touch that tensor. Timing components in isolation cannot see this:
it charges both arms the same standalone norm, and then neither arm's parts add up.

This version profiles the real compiled+captured graph and buckets every kernel that
actually ran. The parts sum to device-busy time BY CONSTRUCTION, and the only residual is
the launch/idle gap between kernels, which is reported rather than hidden.

The cost is that fused kernels cannot always be split: when the norm and the quantization
are one kernel there is no honest way to divide its microseconds, so it is reported under a
joint bucket rather than guessed at. That is a real property of the compiled graph, not a
limitation of the measurement.
"""

import argparse
import pathlib
from collections import OrderedDict

import torch
from torch.profiler import ProfilerActivity, profile

import offload_forward as OF

MODELS = {
    "gemma12b": "google/gemma-3-12b-it",
    "gemma4b": "google/gemma-3-4b-it",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen4b": "Qwen/Qwen3-4B",
    "qwen06b": "Qwen/Qwen3-0.6B",
}

# Bucketing is by kernel name, in this order -- first match wins. GEMMs are split out
# first and then labelled positionally (qkv, o, gate_up, down), which is the order the
# block runs them in; everything else is classified by what inductor named the fusion.
PROJ_ORDER = ("qkv", "o", "gate_up", "down")


def classify(name: str) -> str:
    """Bucket a kernel by what it IS, never by a name it merely mentions.

    Inductor names a fused triton kernel after every op the fusion touches, INCLUDING ops
    it only feeds. The q_norm/k_norm + RoPE kernel is called

        triton_red_fused__fused_rms_norm__scaled_dot_product_flash_attention_add_cat_mul_neg_...

    so a substring test for "flash" bills the norms to attention -- which it did, in both
    arms, inflating attention and hiding the same microseconds from norms+RoPE. Real
    attention on this path is pytorch_flash::flash_fwd_kernel, which is not a triton kernel.
    So: triton and native elementwise kernels are classified on their own content first, and
    only non-triton kernels may be attention or a GEMM.
    """
    n = name.lower()
    if n.startswith("triton_") or "elementwise" in n or "memcpy" in n or "fill" in n:
        if "geglu" in n or "silu" in n or "gelu" in n:
            return "mlp_act"
        # Includes the fp4 quantization inductor folded into the norm it follows; there is
        # no way to split that kernel, so the bucket carries both. See the module docstring.
        return "norm_rope_resid"
    if "flash" in n or "fmha" in n:
        return "attention"
    # The GEMM names actually seen on this box, from `--dump`:
    #   bf16  nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN      (cuBLAS)
    #   bf16  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_...>
    #   fp4   _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversal...  (vLLM AND
    #         FlashInfer -- both dispatch CUTLASS, so one rule covers the per-shape choice
    #         made in nvfp4_tuning.py)
    # "cutlass" and "device_kernel" are matched explicitly rather than relying on the
    # mangled name happening to contain "gemm": a kernel that lost that substring would
    # fall to "other", which does not merely mislabel it but breaks proj_name() downstream,
    # since that walks `kinds` looking for the first "proj" after attention.
    if ("gemm" in n or "nvjet" in n or "gemvx" in n
            or "cutlass" in n or "device_kernel" in n):
        return "proj"
    if "geglu" in n:                      # our own _geglu_nvfp4_kernel, not a triton one
        return "mlp_act"
    if "cvt_fp16_to_fp4" in n:
        return "act_quant"
    return "other"


def profile_layer(fn, x, kw, reps=20):
    """(bucketed device us, total device us, wall us per replay) for one captured layer."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x, **kw)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(x, **kw)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    # wall clock, the number the end-to-end prefill benchmark actually reports
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    wall = e0.elapsed_time(e1) * 1e3 / reps

    # Profile MANY replays, not one: a single profiled iteration is noisy enough to make
    # attention -- bit-identical work in both arms -- look 10% apart, and to push the
    # launch/idle gap negative against the averaged wall clock.
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    evs = sorted((e for e in p.events() if e.device_time_total > 0),
                 key=lambda e: e.time_range.start)
    assert evs, "profiler saw no kernels; the attribution would be meaningless"

    # One replay's worth of kernels, to attribute the GEMMs; the profile covers `reps`
    # replays, so every bucket is divided by reps at the end.
    per_iter = len(evs) // reps
    assert per_iter * reps == len(evs), (
        f"{len(evs)} kernels over {reps} replays is not a whole number per replay; "
        "the attribution below would be wrong")

    # GEMMs are attributed STRUCTURALLY, not positionally. NVFP4 chunks gate_up and down
    # (see nvfp4_linear.CHUNK_TABLE), so a projection can be several GEMM launches, and
    # labelling "the Nth GEMM" silently drops the extra chunks -- which it did, hiding 11%
    # of the NVFP4 layer in unprinted proj4/proj5 buckets while bf16 (never chunked) summed
    # to 100%. Attention and the GeGLU kernel are unmissable landmarks, so:
    #   GEMMs before attention        -> qkv
    #   first GEMM after attention    -> o          (never chunked at these shapes)
    #   remaining GEMMs before GeGLU  -> gate_up
    #   GEMMs after GeGLU             -> down
    kinds = [classify(e.name) for e in evs[:per_iter]]
    i_attn = next(i for i, k in enumerate(kinds) if k == "attention")
    i_act = next((i for i, k in enumerate(kinds) if k == "mlp_act"), per_iter)

    def proj_name(i):
        if i < i_attn:
            return "qkv"
        if i > i_act:
            return "down"
        first_after = next(j for j in range(i_attn + 1, per_iter) if kinds[j] == "proj")
        return "o" if i == first_after else "gate_up"

    buckets, kernels = OrderedDict(), []
    for i, e in enumerate(evs):
        j = i % per_iter
        kind = kinds[j]
        if kind == "proj":
            kind = proj_name(j)
        buckets[kind] = buckets.get(kind, 0.0) + e.device_time_total / reps
        if i < per_iter:
            kernels.append((e.name, e.device_time_total, kind))
    # The whole point of in-graph attribution is that the parts add up. Check it.
    att = [k for k, kk in zip(evs[:per_iter], kinds) if kk == "attention"]
    assert all("flash" in k.name.lower() or "fmha" in k.name.lower() for k in att), \
        "a non-attention kernel landed in the attention bucket: " + \
        str([k.name[:60] for k in att])
    total = sum(buckets.values())
    assert abs(total - sum(e.device_time_total for e in evs) / reps) < 1e-6, "lost a kernel"

    # NOTHING EXPENSIVE MAY SIT IN "other". The buckets reconcile to device time by
    # construction, so an unrecognized kernel does not show up as a missing total -- it
    # shows up as a projection reading low, which looks like a result rather than a bug.
    # That matters now that the fp4 GEMM is a per-shape choice (nvfp4_tuning.py): a
    # FlashInfer kernel whose name does not match classify()'s GEMM patterns would land
    # here AND break the structural attribution downstream of it. Fail instead of writing.
    stray = buckets.get("other", 0.0)
    assert stray <= 0.02 * total, (
        f"{stray:.1f} us ({stray / total:.1%}) of the layer is in the 'other' bucket -- "
        "classify() does not recognize these kernels, so the projection attribution below "
        "is wrong: " +
        str([n[:70] for n, _, k in kernels if k == "other"][:6]))
    return buckets, sum(buckets.values()), wall, kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(MODELS))
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--dump", action="store_true", help="print every kernel and its bucket")
    ap.add_argument("--out", default="nvfp4_breakdown.csv",
                    help="CSV to merge into, keyed on (model, seq_len, component); "
                         "the paper table is generated from this, not from stdout")
    ap.add_argument("--run", default="", help="tag appended to the row, e.g. a repeat id")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    name = MODELS[args.model]
    blocks, cfg, kwargs_fn, _kind_of = OF.build(name, "none")
    variants, variant_of = kwargs_fn(cfg, args.seq)
    L = len(variant_of)

    # One representative layer per attention variant, weighted by how many the model has.
    reps_of = {}
    for li, vi in enumerate(variant_of):
        reps_of.setdefault(vi, li)
    keep = {li: blocks[li] for li in reps_of.values()}
    counts = {vi: variant_of.count(vi) for vi in reps_of}
    blocks.clear()
    torch.cuda.empty_cache()

    print(f"{torch.cuda.get_device_name(0)}   {name}, S={args.seq}, {L} layers")
    print("variants: " + ", ".join(f"{counts[vi]}x variant{vi}" for vi in sorted(counts)))

    x = torch.randn(1, args.seq, cfg.hidden_size, dtype=OF.DTYPE, device=OF.DEV)
    out = {}
    for arm in ("bf16", "nvfp4"):
        if arm == "nvfp4":
            from nvfp4_linear import NVFP4Linear, convert
            for b in keep.values():
                convert(b)
            # Which kernel each projection will actually run is a per-shape, per-length
            # decision now (nvfp4_tuning.py). Say it out loud: a breakdown that does not
            # record which plan produced it cannot be compared against another one later,
            # and "gate_up got faster" is unreadable without knowing what changed under it.
            seen = {}
            for b in keep.values():
                for pname, mod in b.named_modules():
                    if isinstance(mod, NVFP4Linear):
                        seen[pname] = mod._plan(args.seq)
            print("  fp4 plans at S=%d: " % args.seq
                  + ", ".join(f"{p}={be}/{c if c < args.seq else '-'}"
                              for p, (be, c) in sorted(seen.items())))
        agg, wall_tot = {}, 0.0
        for vi, li in sorted(reps_of.items()):
            fn = torch.compile(keep[li], fullgraph=True, dynamic=False)
            b, tot, wall, kernels = profile_layer(fn, x, variants[vi])
            w = counts[vi] / L
            for k, v in b.items():
                agg[k] = agg.get(k, 0.0) + v * w
            wall_tot += wall * w
            if args.dump:
                print(f"\n--- {arm}, variant{vi} ({counts[vi]} layers) ---")
                for kn, us, kind in kernels:
                    print(f"  {us:9.1f}us  [{kind:>15}]  {kn[:80]}")
        out[arm] = (agg, wall_tot)
        torch.cuda.empty_cache()

    order = ["qkv", "o", "gate_up", "down", "mlp_act", "act_quant", "norm_rope_resid",
             "attention", "other"]
    label = {"qkv": "qkv", "o": "o", "gate_up": "gate_up", "down": "down",
             "mlp_act": "MLP non-linearity", "act_quant": "activation quant (standalone)",
             "norm_rope_resid": "norms + RoPE + residual", "attention": "attention",
             "other": "other"}
    b16, w16 = out["bf16"]
    b4, w4 = out["nvfp4"]
    # A bucket missing from `order` prints as nothing and the columns quietly stop summing
    # to 100%. That is exactly how the chunked-GEMM bug survived a smoke test.
    stray = (set(b16) | set(b4)) - set(order)
    assert not stray, f"buckets not in the print order, would be dropped: {sorted(stray)}"

    print(f"\n{'component':<32}{'bf16 ms':>9}{'fp4 ms':>9}{'speedup':>9}{'% fp4':>8}")
    for k in order:
        a, c = b16.get(k, 0.0) / 1e3, b4.get(k, 0.0) / 1e3
        if a == 0 and c == 0:
            continue
        sp = f"{a/c:>8.2f}x" if a and c else f"{'-':>9}"
        print(f"{label[k]:<32}{a:>9.2f}{c:>9.2f}{sp}{100*c/(w4/1e3):>7.0f}%")

    d16, d4 = sum(b16.values()) / 1e3, sum(b4.values()) / 1e3
    print(f"{'-'*67}")
    print(f"{'device busy (sum of kernels)':<32}{d16:>9.2f}{d4:>9.2f}{d16/d4:>8.2f}x")
    print(f"{'launch/idle gap':<32}{w16/1e3-d16:>9.2f}{w4/1e3-d4:>9.2f}")
    print(f"{'LAYER (wall clock)':<32}{w16/1e3:>9.2f}{w4/1e3:>9.2f}{w16/w4:>8.2f}x")
    print(f"{'MODEL (x%d layers)' % L:<32}{L*w16/1e6:>8.2f}s{L*w4/1e6:>8.2f}s{w16/w4:>8.2f}x")

    untouched = sum(b4.get(k, 0.0) for k in
                    ("attention", "mlp_act", "act_quant", "norm_rope_resid", "other")) / 1e3
    proj4 = sum(b4.get(k, 0.0) for k in PROJ_ORDER) / 1e3
    proj16 = sum(b16.get(k, 0.0) for k in PROJ_ORDER) / 1e3
    print(f"\nprojections alone                       {proj16/proj4:>6.2f}x")
    print(f"everything else ({untouched:5.2f} ms of the fp4 layer)  "
          f"{100*untouched/(w4/1e3):>4.0f}% of it")
    print(f"ceiling if the projections were free    {(w16/1e3)/untouched:>6.2f}x")

    if args.out:
        _merge_csv(args.out, name, args.seq, args.run, b16, b4, w16, w4, order, label)


def _merge_csv(path, model, seq, run, b16, b4, w16, w4, order, label):
    """Merge this (model, seq) breakdown into the CSV, keyed on (model, seq, component).

    Written so the paper table is generated from a tracked file rather than from a terminal
    scrollback, and so a re-run visibly overwrites the row it supersedes.
    """
    import csv
    f = pathlib.Path(path)
    fields = ["model", "seq_len", "component", "bf16_ms", "nvfp4_ms", "run"]
    rows = list(csv.DictReader(open(f))) if f.exists() else []
    idx = {(r["model"], r["seq_len"], r["component"]): r for r in rows}
    out = [(label[k], b16.get(k, 0.0) / 1e3, b4.get(k, 0.0) / 1e3) for k in order
           if b16.get(k) or b4.get(k)]
    out.append(("LAYER wall clock", w16 / 1e3, w4 / 1e3))
    for comp, a, c in out:
        r = {"model": model, "seq_len": str(seq), "component": comp,
             "bf16_ms": f"{a:.3f}", "nvfp4_ms": f"{c:.3f}", "run": run}
        k = (model, str(seq), comp)
        if k in idx:
            idx[k].update(r)
        else:
            rows.append(r); idx[k] = r
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {f} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
