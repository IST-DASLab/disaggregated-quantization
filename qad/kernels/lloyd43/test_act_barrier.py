"""Prove the dead-end activation quant executes, and that it costs nothing extra.

    python test_act_barrier.py

A quantization whose result is discarded is textbook dead code. If dynamo, inductor, or
vLLM's act_quant_fusion pass removes it, the *aq arms silently measure the plain
weight-only number and report it as the non-disaggregated one. A null timing delta and a
deleted op look identical from the outside, so timing cannot establish this -- the profiler
has to see the kernel.

The op is kept alive by a declared mutation of x that never happens (see cuda_gemv). That
buys ordering and DCE-immunity for free, but functionalization clones the mutated input
unless inductor's reinplacing pass removes the clone -- which would bill the arm for a copy
the format never performs. So there are three things to check:

  1. the quant kernel still runs, under fullgraph compile and inside a CUDA graph;
  2. NO extra copy kernel appeared beside it;
  3. a control -- the same work as a PURE op with its result ignored IS eliminated, which
     proves the check can fail and that the declared mutation is what preserves it.

Each case runs in its own process. Successive torch.profiler sessions in one process were
observed to return empty event lists after the first few, which reads exactly like "the op
was optimized away" -- the failure this test exists to detect. One process per measurement
removes that ambiguity.
"""

import argparse
import json
import subprocess
import sys

import torch
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

DEV = "cuda"
L = 4                                   # four linears, as in one decoder layer
K = 4096
QUANT_KERNEL = "fp4"                    # vllm::cvt_fp16_to_fp4
COPY_KERNEL = ("copy", "clone")


@torch.library.custom_op("lloyd43_test::pure_quant", mutates_args=())
def pure_quant(x: Tensor, global_scale: Tensor) -> Tensor:
    """The control: identical work, no declared mutation, result ignored by the caller."""
    import vllm._custom_ops as ops
    q, _ = ops.scaled_fp4_quant(x.reshape(-1, x.shape[-1]), global_scale,
                                is_sf_swizzled_layout=True, backend="cutlass")
    return q


@pure_quant.register_fake
def _(x, global_scale):
    return torch.empty((x.shape[0], x.shape[-1] // 2), dtype=torch.uint8, device=x.device)


def run_case(case: str, v2: bool = True) -> dict:
    from lloyd43.cuda_gemv import act_quant_barrier

    # vLLM compiles with enable_auto_functionalized_v2=False (see its compilation config),
    # and that is the mode the *aq arms actually run in. It matters: under v1,
    # functionalizing a mutating custom op goes through auto_functionalized, which CLONES
    # the mutated input unless inductor's reinplacing pass removes the clone. A per-linear
    # clone of the activation is precisely the fake cost this measurement must not contain,
    # so the no-copy check has to run in vLLM's mode, not in PyTorch's default.
    torch._inductor.config.enable_auto_functionalized_v2 = v2

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    gs = torch.tensor(6 * 448 / 10.0, dtype=torch.float32, device=DEV)
    x = torch.randn(1, K, device=DEV, dtype=torch.bfloat16)
    w = [torch.randn(K, K, device=DEV, dtype=torch.bfloat16) * 0.02 for _ in range(L)]

    def barrier(t):
        for wi in w:
            act_quant_barrier(t, gs)      # mutation declared; t is never written
            t = t @ wi.T
        return t

    def baseline(t):
        for wi in w:
            t = t @ wi.T
        return t

    def control(t):
        for wi in w:
            pure_quant(t, gs)             # pure, result ignored -> should vanish
            t = t @ wi.T
        return t

    if case == "numerics":
        return {"equal": bool(torch.equal(baseline(x), barrier(x)))}

    fn = {"eager": barrier, "compiled": torch.compile(barrier, fullgraph=True,
                                                      dynamic=False),
          "baseline": torch.compile(baseline, fullgraph=True, dynamic=False),
          "control": torch.compile(control, fullgraph=True, dynamic=False),
          "cudagraph": torch.compile(barrier, fullgraph=True, dynamic=False)}[case]

    call = lambda: fn(x)
    if case == "cudagraph":
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(x)
        call = g.replay

    for _ in range(5):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        call()
        torch.cuda.synchronize()
    names = [e.name.lower() for e in p.events() if e.device_time_total > 0]
    assert names, "profiler saw no kernels at all; result would be meaningless"
    return {"quant": sum(QUANT_KERNEL in n for n in names),
            "copy": sum(any(c in n for c in COPY_KERNEL) for n in names),
            "total": len(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--no-v2", action="store_true",
                    help="run with enable_auto_functionalized_v2=False, as vLLM does")
    args = ap.parse_args()
    if args.case:
        print("RESULT " + json.dumps(run_case(args.case, v2=not args.no_v2)))
        return 0

    r = {}
    for case in ("numerics", "eager", "compiled", "cudagraph", "baseline", "control"):
        cmd = [sys.executable, __file__, "--case", case]
        if args.no_v2:
            cmd.append("--no-v2")
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"{case}: subprocess failed\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
            return 1
        r[case] = json.loads(line[0][7:])

    print(f"numerics unchanged            : "
          f"{'bitwise identical' if r['numerics']['equal'] else 'CHANGED'} over {L} linears")
    for case, label in (("eager", "eager"), ("compiled", "fullgraph"),
                        ("cudagraph", "in CUDA graph")):
        print(f"quant kernels, {label:<15}: {r[case]['quant']}  (expect {L})")
    print(f"copy kernels, with barrier    : {r['compiled']['copy']}  "
          f"(baseline without it: {r['baseline']['copy']})")
    print(f"quant kernels, pure + ignored : {r['control']['quant']}  "
          f"(control: eliminated if 0)")

    alive = all(r[c]["quant"] == L for c in ("eager", "compiled", "cudagraph"))
    no_clone = r["compiled"]["copy"] <= r["baseline"]["copy"]
    ok = alive and no_clone and r["numerics"]["equal"]
    if not r["numerics"]["equal"]:
        print("\nFAILED: the barrier changed the result; it is not a no-op")
    elif not alive:
        print("\nFAILED: barrier optimized away; the *aq arms would be fraudulent")
    elif not no_clone:
        print("\nFAILED: functionalization left a clone; the arm would be billed for a "
              "copy the format never performs")
    elif r["control"]["quant"] == L:
        print("\nNote: the control was NOT eliminated, so this run does not demonstrate "
              "that the declared mutation is what keeps the op alive.")
    else:
        print("\nOK: survives compile+capture, adds no copy, and dies without the mutation")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
