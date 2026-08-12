"""Extrapolate full-benchmark wall time from a --limit calibration run.

    python estimate_eval_time.py 479792 479793 479794 479795

Why not just scale total elapsed: a job's elapsed time is startup + generation, and
startup does not grow with the document count. Engine load, NIXL handshake and CUDA
graph capture cost several fixed minutes (capture alone is ~104s at max_num_seqs=512),
so scaling elapsed by 12032/350 inflates the estimate by roughly that fixed cost times
34. The two phases are separated here and only the generation half is scaled.

Generation start/end come from the proxy log's own timestamps -- the first and last
"prefill ok" -- which bracket exactly the phase that scales with document count.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # cluster_scripts/ -> qad -> repo
LOGS = ROOT / "logs" / "eval_disagg"
FULL_DOCS = {"mmlu_pro": 12032, "gsm8k": 1319, "minerva_math500": 500}
WALL_MIN = 240  # --time=04:00:00 in run_eval_disagg.sh

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


def sacct(job: str, field: str) -> str:
    out = subprocess.run(["sacct", "-j", job, "-n", "--format=" + field, "-P"],
                         capture_output=True, text=True).stdout.splitlines()
    return out[0].strip() if out else ""


def elapsed_min(s: str) -> float:
    if not s:
        return 0.0
    d, _, rest = s.partition("-")
    if not rest:
        rest, d = d, "0"
    p = [float(x) for x in rest.split(":")]
    sec = p[-1] + (p[-2] * 60 if len(p) > 1 else 0) + (p[-3] * 3600 if len(p) > 2 else 0)
    return (float(d) * 86400 + sec) / 60


def gen_window(job: str):
    """(minutes spent generating, requests served) from the job's proxy log."""
    for d in LOGS.glob(f"stack_{job}_*"):
        p = d / "proxy.log"
        if not p.exists():
            continue
        first = last = None
        n = 0
        for line in open(p, errors="ignore"):
            if "prefill ok" not in line:
                continue
            m = TS.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            first = first or t
            last = t
            n += 1
        if first and last:
            return (last - first).total_seconds() / 60, n
    return None, 0


def job_meta(job: str):
    """model / quantizer / think / tasks / limit, straight from the driver banner."""
    for f in LOGS.glob(f"*/{job}*.out"):
        txt = f.read_text(errors="ignore")
        m = re.search(r"\[disagg-eval\] model=(\S+).*?think=(\d) tasks=(.+)", txt)
        if m:
            return m.group(1), m.group(2) == "1", m.group(3).strip()
    return "?", None, "?"


def docs_evaluated(job: str, tasks: str) -> int:
    """Documents this run scored, counting ONLY the tasks it was asked to run.

    step_<N>.json is merged, not overwritten, so a file can carry gsm8k and
    minerva_math500 from an earlier sweep alongside this run's mmlu_pro. Summing all of
    n-samples charged 1819 foreign documents to an mmlu_pro job and put its rate out by
    an order of magnitude.
    """
    want = tasks.split()
    for f in LOGS.glob(f"*/{job}*.out"):
        for path in re.findall(r"results -> (\S+)", f.read_text(errors="ignore")):
            try:
                ns = json.load(open(path)).get("n-samples", {})
            except Exception:
                continue
            return sum(v.get("effective", 0) for k, v in ns.items()
                       if any(k == t or k.startswith(t + "_") for t in want))
    return 0


def main() -> None:
    jobs = sys.argv[1:]
    if not jobs:
        raise SystemExit(__doc__)

    print(f"{'job':>9} {'model':22} {'mode':8} {'docs':>6} {'startup':>8} {'gen':>8} "
          f"{'docs/min':>9} {'FULL est':>9} {'vs wall':>9}")
    for job in jobs:
        state = sacct(job, "State").split()[0] if sacct(job, "State") else "?"
        total = elapsed_min(sacct(job, "Elapsed"))
        model, think, tasks = job_meta(job)
        gen, reqs = gen_window(job)
        docs = docs_evaluated(job, tasks)
        if not docs and reqs:
            # n-samples was dropped by the pre-fix merge, and these calibration runs
            # were written before that fix. The proxy's request count is the same
            # quantity here: with timeout=3600 there are no retries, so one request is
            # one document (verified: 0 TimeoutErrors across every post-fix job).
            docs = reqs

        if gen is None or not docs:
            print(f"{job:>9} {model[:22]:22} {'?':8} {'-':>6} {'-':>8} {'-':>8} {'-':>9} "
                  f"{'-':>9} {'-':>9}   {state} (no proxy/result data yet)")
            continue

        startup = max(0.0, total - gen)
        rate = docs / gen if gen > 0 else 0
        full_docs = sum(FULL_DOCS.get(t, 0) for t in tasks.split())
        full = startup + (full_docs / rate if rate else 0)
        flag = "OK" if full < WALL_MIN * 0.85 else ("TIGHT" if full < WALL_MIN else "OVER WALL")
        if state != "COMPLETED":
            flag = state[:9]
        print(f"{job:>9} {model[:22]:22} {'think' if think else 'nothink':8} {docs:>6} "
              f"{startup:>7.1f}m {gen:>7.1f}m {rate:>9.1f} {full:>8.0f}m {flag:>9}")

    print(f"\n  FULL est = startup + full_docs/rate, against a {WALL_MIN}-min wall.")
    print("  Startup is measured, not scaled: it is fixed cost (engine load, NIXL "
          "handshake, CUDA graph capture) and does not grow with document count.")
    print("  A sweep runs 10 steps x 6 formats x 2 modes per size; each job is one step.")


if __name__ == "__main__":
    main()
