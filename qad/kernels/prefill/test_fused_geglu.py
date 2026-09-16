"""Tests for the fused GeGLU+NVFP4 path, including a control that the tests can fail.

    python test_fused_geglu.py            # wiring, compile, capture
    python test_fused_geglu.py --mutate   # ... and prove the checks catch a broken kernel

WHY THE MUTATION CONTROL EXISTS
-------------------------------
The first version of the block-level check scaled every parameter by 0.05. That put every
activation far below the static ACT_AMAX=10 the W4A4 path assumes, so every block scale
underflowed e4m3 to zero, the whole MLP quantized to nothing, and the test compared zero
against zero. It reported "bitwise identical" for a kernel computing SiLU instead of GeLU.

A test that cannot fail is worse than no test, because it is read as evidence. So this file
carries both halves: the checks, and a control that deliberately breaks the kernel five ways
and asserts the checks notice. `test_wiring` also refuses to run at all if the MLP branch
output is ~zero, which is what made the original failure invisible.

The tie-rule mutation is EXPECTED to pass. Flipping `>=` to `>` at 0.75 changes only which
of two equidistant fp4 codes is chosen; that is the exact class of difference
`fused_geglu_quant._selftest` documents as tolerable, and it costs nothing numerically.
Every mutation that changes a result is caught.
"""

import argparse
import copy
import pathlib
import subprocess
import sys

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).parent
SRC = HERE / "fused_geglu_quant.py"

# (anchor, replacement, expected_to_be_caught)
MUTATIONS = {
    "tie rule flipped": ("code = tl.where(a >= 0.75, 2, code)",
                         "code = tl.where(a > 0.75, 2, code)", False),
    "scale off by 2%": ("bs = (amax * gs / 6.0).to(tl.float8e4nv)",
                        "bs = (amax * gs / 6.0 * 1.02).to(tl.float8e4nv)", True),
    "nibble order swapped": ("packed = (lo | (hi << 4)).to(tl.uint8)",
                             "packed = (hi | (lo << 4)).to(tl.uint8)", True),
    # Anchored on the ACT == 0 branch, not on a fused one-liner: when the kernel gained its
    # `ACT` constexpr (GeLU for Gemma, SiLU for Qwen3) the single expression
    # `y = 0.5 * gate * (...) * up` became `act = ...` followed by `y = act * up`, and this
    # anchor stopped matching. Since the loop asserted on a missing anchor, the suite died
    # here and the last two mutations silently stopped running -- including this one, which
    # exists because an early wiring test reported "bitwise identical" for a kernel that
    # computed SiLU instead of GeLU.
    "gelu -> silu": ("act = 0.5 * gate * (1.0 + libdevice.tanh(inner))",
                     "act = gate * tl.sigmoid(gate)", True),
    "skip bf16 rounding": ("    y = y.to(tl.bfloat16).to(tl.float32)", "    pass", True),
}


def test_wiring(seq_len: int = 2048, verbose: bool = True) -> bool:
    """Fused and unfused must be equally close to bf16, and the path must compile+capture.

    Default parameter init on purpose: weights ~ U(-1/sqrt(K), 1/sqrt(K)) against N(0,1)
    input puts activations at order 1, inside the ACT_AMAX the quantizer assumes. Scaling
    them down is what produced the all-zero MLP that made this test vacuous.
    """
    from transformers import AutoConfig
    import gemma3_block as G
    from nvfp4_linear import convert

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = getattr(AutoConfig.from_pretrained("google/gemma-3-4b-it"), "text_config")

    ref = G.Gemma3Block(cfg, dtype=torch.bfloat16, device="cuda").eval()
    fused, split = copy.deepcopy(ref), copy.deepcopy(ref)
    convert(fused)
    convert(split)
    split.fuse_mlp_quant = False

    variants, variant_of = G.attention_kwargs(cfg, seq_len, dtype=torch.bfloat16,
                                              device="cuda")
    kw = variants[variant_of[0]]
    x = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")

    seen = []
    orig_rms = F.rms_norm
    F.rms_norm = lambda t, s, w, e: (seen.append(t), orig_rms(t, s, w, e))[1]
    try:
        seen.clear(); o_fused = fused(x, **kw).float(); m_fused = seen[-1]
        seen.clear(); o_split = split(x, **kw).float(); m_split = seen[-1]
    finally:
        F.rms_norm = orig_rms
    o_bf16 = ref(x, **kw).float()

    # THE GUARD. Without it a degenerate all-zero MLP passes everything.
    if min(m_fused.float().norm().item(), m_split.float().norm().item()) < 1e-3:
        if verbose:
            print("MLP output is ~zero: activations underflowed the static scale, "
                  "this test would prove nothing")
        return False

    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    r_fb, r_sb = rel(o_fused, o_bf16), rel(o_split, o_bf16)
    ok = r_fb <= r_sb * 1.10 + 1e-6

    fn = torch.compile(fused, fullgraph=True, dynamic=False)
    fn(x, **kw)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x, **kw)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap = fn(x, **kw)
    g.replay()
    torch.cuda.synchronize()

    if verbose:
        print(f"MLP branch norm     : fused {m_fused.float().norm():.2f}  "
              f"unfused {m_split.float().norm():.2f}")
        print(f"fused   vs bf16     : {r_fb:.4e}")
        print(f"unfused vs bf16     : {r_sb:.4e}   "
              f"({'equal' if ok else 'FUSED IS WORSE'})")
        print(f"fullgraph + capture : ok, out {tuple(cap.shape)}")
    return ok


def run_mutations() -> bool:
    """Break the kernel five ways; assert the checks notice the ones that matter."""
    orig = SRC.read_text()
    check = ("import sys; sys.path.insert(0, '.'); "
             "import fused_geglu_quant as F; print('SELF', F._selftest())")
    all_good = True
    print(f"{'mutation':<24}{'kernel selftest':>17}{'wiring':>9}   verdict")
    try:
        for name, (anchor, repl, should_catch) in MUTATIONS.items():
            # A stale anchor is a failure of THIS mutation, not a reason to abandon the
            # ones after it. Asserting here meant a kernel edit that moved one line
            # disabled every later mutation without ever saying so -- the suite printed
            # three green rows and a traceback, which reads like a crash rather than like
            # lost coverage.
            if anchor not in orig:
                print(f"{name:<24}{'-':>17}{'-':>9}   "
                      f"*** ANCHOR MISSING, update this test: {anchor[:40]}... ***")
                all_good = False
                continue
            SRC.write_text(orig.replace(anchor, repl))
            st = subprocess.run([sys.executable, "-c", check], capture_output=True,
                                text=True, cwd=HERE)
            self_ok = "SELF True" in st.stdout
            wr = subprocess.run([sys.executable, __file__, "--quiet"],
                                capture_output=True, text=True, cwd=HERE)
            wiring_ok = wr.returncode == 0
            caught = not (self_ok and wiring_ok)
            verdict = ("caught" if caught else "tolerated (equidistant ties, by design)")
            if caught != should_catch:
                verdict = f"*** UNEXPECTED: {'caught' if caught else 'MISSED'} ***"
                all_good = False
            print(f"{name:<24}{('pass' if self_ok else 'FAIL'):>17}"
                  f"{('pass' if wiring_ok else 'FAIL'):>9}   {verdict}")
    finally:
        SRC.write_text(orig)
        assert SRC.read_text() == orig, "FAILED TO RESTORE THE KERNEL SOURCE"
    print("\nkernel source restored")
    return all_good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", action="store_true",
                    help="also run the control that proves these checks can fail")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ok = test_wiring(verbose=not args.quiet)
    if not args.quiet:
        print("wiring:", "OK" if ok else "FAILED")
    if ok and args.mutate:
        print()
        ok = run_mutations()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
