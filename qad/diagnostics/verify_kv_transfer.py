"""Prove that KV actually crosses from the prefill server to the decode server.

This is the gate on every disaggregated number. If the transfer silently fails, the
decode engine just recomputes the prompt with its OWN weights and returns a fluent,
plausible answer — the eval reports a "disaggregated" score that is really
homogeneous-decode, and nothing in the output looks wrong.

The test does not trust logs or metrics, only behaviour. Three stacks are run over
identical prompts with greedy decoding:

    A  prefill=W4A4   decode=W4A16     <- the disaggregated pair under test
    B  prefill=W4A16  decode=W4A16     <- homogeneous decode-only
    C  prefill=W4A4   decode=W4A4      <- homogeneous prefill-only

and the completions are compared token for token.

    A == B  ->  FAIL. The prompt is being recomputed by the decode weights, i.e. the
                KV never arrived. This is exactly the silent-fallback signature,
                because a fallback makes A degenerate into B.
    A == C  ->  FAIL in the other direction: decode appears to be running the prefill
                weights, so the pair is not actually split.
    A != B and A != C  ->  PASS. A's output depends on BOTH sets of weights, which is
                only possible if the prefill-built KV reached the decode engine.

Greedy decoding (temperature 0) makes the comparison exact, so a difference is
signal rather than sampling noise.

    python verify_kv_transfer.py --a4 <nvfp4 dir> --a16 <nvfp4a16 dir> \\
        --tokenizer Qwen/Qwen3-0.6B
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# parent.parent, not parent: this file lives in qad/diagnostics/, so .parent is
# diagnostics/ -- but the driver it shells out to is qad/eval/eval_disagg.py. It used to
# sit beside this file and the reference was never updated when it moved, so the gate
# died with "can't open file .../diagnostics/eval_disagg.py" before running any stack.
_QAD = Path(__file__).resolve().parent.parent
_EVAL_DISAGG = _QAD / "eval" / "eval_disagg.py"
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def collect(prefill: Path, decode: Path, tokenizer: str, out: Path,
            port_base: int, label: str) -> list:
    """Run one stack via eval_disagg.py --probe and return its completions."""
    log(f"=== stack {label}: prefill={prefill.name} decode={decode.name} ===")
    cmd = [sys.executable, str(_EVAL_DISAGG),
           "--prefill-model", str(prefill), "--decode-model", str(decode),
           "--tokenizer", tokenizer, "--probe", str(out),
           "--port-base", str(port_base),
           "--log-dir", str(out.parent / f"logs_{label}")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        log(f"stack {label} FAILED (rc={r.returncode})")
        print(r.stdout[-3000:]); print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"stack {label} did not run")
    return json.loads(out.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a4", required=True, help="W4A4 (nvfp4) model dir")
    p.add_argument("--a16", required=True, help="W4A16 (nvfp4a16) model dir")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--work-dir", default="/tmp/kvverify")
    p.add_argument("--port-base", type=int, default=8700)
    args = p.parse_args()

    a4, a16 = Path(args.a4), Path(args.a16)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    # Different port bases: a lingering socket from a torn-down stack would otherwise
    # make the next one bind-fail or, worse, talk to the previous engine.
    A = collect(a4,  a16, args.tokenizer, work / "A_disagg.json",  args.port_base + 0,  "A_disagg")
    B = collect(a16, a16, args.tokenizer, work / "B_homo_a16.json", args.port_base + 10, "B_homo_a16")
    C = collect(a4,  a4,  args.tokenizer, work / "C_homo_a4.json",  args.port_base + 20, "C_homo_a4")

    same_ab = sum(a["completion"] == b["completion"] for a, b in zip(A, B))
    same_ac = sum(a["completion"] == c["completion"] for a, c in zip(A, C))
    n = len(A)

    print("\n" + "=" * 74)
    print(f"prompts compared: {n}")
    print(f"  A(disagg W4A4->W4A16) identical to B(homogeneous W4A16): {same_ab}/{n}")
    print(f"  A(disagg W4A4->W4A16) identical to C(homogeneous W4A4) : {same_ac}/{n}")
    for i, (a, b, c) in enumerate(zip(A, B, C)):
        print(f"\n[{i}] {a['prompt'][:70]}")
        print(f"   A: {a['completion'][:100]!r}")
        print(f"   B: {b['completion'][:100]!r}")
        print(f"   C: {c['completion'][:100]!r}")

    print("\n" + "=" * 74)
    if same_ab == n:
        print("VERDICT: FAIL — A is identical to homogeneous-W4A16 on every prompt.")
        print("  The decode engine is recomputing the prompt with its own weights;")
        print("  the KV transfer is NOT happening. Any 'disaggregated' number from")
        print("  this stack would actually be homogeneous-decode.")
        raise SystemExit(1)
    if same_ac == n:
        print("VERDICT: FAIL — A is identical to homogeneous-W4A4 on every prompt,")
        print("  so generation is not running on the decode weights at all.")
        raise SystemExit(1)
    print("VERDICT: PASS — A differs from both homogeneous stacks, so its output")
    print("  depends on the prefill weights AND the decode weights. KV crossed.")


if __name__ == "__main__":
    main()
