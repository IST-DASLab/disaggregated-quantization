"""Submit evals for every exported checkpoint that has no result and no job in flight.

    python cluster_scripts/submit_missing_evals.py                 # report only
    python cluster_scripts/submit_missing_evals.py --apply
    python cluster_scripts/submit_missing_evals.py --apply --models 8B
    python cluster_scripts/submit_missing_evals.py --apply --formats lloyd21

WHY THIS EXISTS
---------------
Gap-filling a live sweep by hand is a loop: list the newly exported steps, subtract the
ones that already have results, submit the rest. `run_eval_disagg_sweep.sh --missing-only`
does the middle part, but it has one blind spot that has caused a real duplicate
submission: **a step whose job is still QUEUED has no result file yet, so it looks
exactly like a gap.** Re-running --missing-only against a live queue re-submits work that
is already running. Doing it by hand instead means maintaining the in-flight set in your
head, which is the same bug with a slower failure mode.

So this script tracks what it submitted. Every submission appends its SLURM array job id
and the steps it covers to a ledger; on the next run any ledger entry whose job is still
PENDING/RUNNING is subtracted from the gaps. That makes the script safe to call
repeatedly, at any moment, against a queue in any state -- which is the whole point.

WHAT COUNTS AS A GAP
--------------------
  * the checkpoint exists on disk, and
  * the step is on the PLOTTED grid (step 0, or a multiple of 250 up to 2250 -- the
    --export-tail-every 125 points are deliberately skipped, no figure draws them), and
  * no result JSON reports every task in the group, and
  * no ledger job covering it is still in the queue.

Step 0 is included: it is the calibrated-PTQ / RTN baseline that the recovery curves are
read against.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "cluster_scripts", ".eval_submissions.jsonl")
SWEEP = os.path.join(ROOT, "bin", "run_eval_disagg_sweep.sh")
# One entry per model family. `hf` is the repo-id template and `models` the sizes; the
# checkpoint/result tag is always "<run>-<hf id with / -> ->-<quant>-<hash>", which is what
# run_qad.sh builds, so Qwen and Gemma tags can never collide and share one directory tree.
# `modes` is per family and NOT cosmetic. Qwen3 has a real thinking switch, so both modes
# are meaningful. Gemma-3 has none: `enable_thinking` is accepted and completely inert (the
# rendered prompt is byte-identical with it True, False or absent), and `--no-think` HARD
# FAILS on the probe guard in eval_disagg.py. Submitting "nothink" for Gemma would queue a
# full sweep of jobs that all die on startup.
FAMILIES = {
    "qad3x":  {"hf": "Qwen/Qwen3-{}",        "models": ["0.6B", "1.7B", "4B", "8B"],
               "modes": ["think", "nothink"]},
    "gemma3": {"hf": "google/gemma-3-{}-it", "models": ["270m", "1b", "4b", "12b"],
               "modes": ["think"]},
}
FAMILY = "qad3x"                     # overridden by --family
RUN = FAMILY
MODELS = FAMILIES[FAMILY]["models"]
MODES = FAMILIES[FAMILY]["modes"]


def hf_id(model: str) -> str:
    return FAMILIES[RUN]["hf"].format(model)


def tag_prefix(model: str, quant: str) -> str:
    """The glob every checkpoint/result directory for this (family, model, quant) matches."""
    return f"{RUN}-{hf_id(model).replace('/', '-')}-{quant}-*"
GRID = [0] + list(range(250, 2251, 250))
TASK_GROUPS = ["gsm8k minerva_math500", "mmlu_pro"]

# quantizer -> sweep label. Only formats the plots draw; adding one here is all that is
# needed for this script to start covering it.
LABELS = {
    "nvfp4": "NVFP4", "nvfp4a16": "NVFP4A16",
    "lloyd3bit": "W3A16-Lloyd", "lloyd43": "W3A16-Lloyd43", "lloyd21": "W3A16-Lloyd21",
    "nvr2bit": "NVR2BIT",
    "nvfp4pdshared": "NVFP4-pd-shared", "nvfp4pdsplit": "NVFP4-pd-split",
    "nvfp4prefill": "NVFP4-prefill", "nvfp4decode": "NVFP4-decode",
    "nvfp4lloyd43shared": "NVFP4-Lloyd43-shared",
    "nvfp4lloyd43split": "NVFP4-Lloyd43-split",
    "nvfp4lloyd43upcast": "NVFP4-Lloyd43-upcast",
    "nvfp4lloyd21upcast": "NVFP4-Lloyd21-upcast",
    "nvfp4lloyd21split": "NVFP4-Lloyd21-split",
    # Non-disaggregated controls: the upcast weight served on BOTH phases.
    "nvfp4lloyd43upcastboth": "NVFP4-Lloyd43-upcastboth",
    "nvfp4lloyd21upcastboth": "NVFP4-Lloyd21-upcastboth",
    "nvfp4nvr2bitupcast": "NVFP4-NVR2BIT-upcast",
    "nvfp4nvr2bitsplit": "NVFP4-NVR2BIT-split",
}


def exported_steps(model: str, quant: str) -> set:
    """Grid steps with a checkpoint on disk."""
    dirs = glob.glob(os.path.join(ROOT, "checkpoints", tag_prefix(model, quant), "weights"))
    out = set()
    for d in dirs:
        for x in os.listdir(d):
            m = re.match(r"step_(\d+)$", x)
            if m and int(m.group(1)) in GRID:
                out.add(int(m.group(1)))
    return out


def landed_steps(model: str, quant: str, mode: str, tasks: str) -> set:
    """Steps whose result JSON reports EVERY task in the group.

    Mirrors missing_steps() in run_eval_disagg_sweep.sh, including the subject-leaf rule:
    mmlu_pro reports as mmlu_pro plus mmlu_pro_<subject> keys, and a file holding only
    gsm8k must not count as covering mmlu_pro.
    """
    want = tasks.split()
    tree = os.path.join(ROOT, "results", "disagg", mode)
    out = set()
    for d in glob.glob(os.path.join(tree, tag_prefix(model, quant))):
        for f in glob.glob(os.path.join(d, "step_*.json")):
            try:
                res = json.load(open(f)).get("results", {})
            except Exception:
                continue
            if all(any(k == t or k.startswith(t + "_") for k in res) for t in want):
                out.add(int(re.search(r"step_(\d+)", os.path.basename(f)).group(1)))
    return out


def active_jobs() -> set:
    try:
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%A"],
                           capture_output=True, text=True, timeout=60)
        return {x.strip() for x in r.stdout.split() if x.strip()}
    except Exception:
        return set()


def inflight() -> dict:
    """(model, quant, mode, tasks) -> set of steps covered by a job still in the queue."""
    if not os.path.exists(LEDGER):
        return {}
    live = active_jobs()
    out = {}
    for line in open(LEDGER):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if str(e.get("jobid")) not in live:
            continue
        k = (e["model"], e["quant"], e["mode"], e["tasks"])
        out.setdefault(k, set()).update(e["steps"])
    return out


def record(jobid, model, quant, mode, tasks, steps) -> None:
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps({"jobid": jobid, "model": model, "quant": quant,
                             "mode": mode, "tasks": tasks, "steps": sorted(steps),
                             "t": int(time.time())}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="submit (default: report only)")
    ap.add_argument("--family", default=FAMILY, choices=sorted(FAMILIES),
                    help="model family; also the RUN_PREFIX the tags were built with")
    ap.add_argument("--models", nargs="*", default=None,
                    help="sizes to cover (default: every size in --family)")
    ap.add_argument("--formats", nargs="*", default=sorted(LABELS))
    ap.add_argument("--force", action="store_true",
                    help="bypass the empty-ledger bootstrap guard")
    # So autoeval_watch.sh can iterate every family without hardcoding a second copy of
    # the list that would drift the moment a family is added here.
    ap.add_argument("--list-families", action="store_true",
                    help="print the family names, one per line, and exit")
    args = ap.parse_args()
    if args.list_families:
        print("\n".join(sorted(FAMILIES)))
        return
    # --family selects the tag namespace, so it has to land before anything globs a
    # checkpoint or result directory.
    global RUN, MODELS, MODES
    RUN = args.family
    MODELS = FAMILIES[RUN]["models"]
    MODES = FAMILIES[RUN]["modes"]
    if args.models is None:
        args.models = MODELS

    flying = inflight()
    # BOOTSTRAP GUARD. The ledger only knows about jobs THIS script submitted, so on a
    # first run against a queue that already has eval jobs in it, every one of those
    # steps looks like a gap and would be submitted twice. That is precisely the failure
    # this script exists to prevent, so refuse rather than reproduce it. Either wait for
    # the queue to drain (after which the ledger stays authoritative forever), narrow the
    # run with --models/--formats to work nobody has queued, or accept it with --force.
    if args.apply and not flying and not args.force:
        live_evals = [j for j in active_jobs()]
        if len(live_evals) > 0 and not os.path.exists(LEDGER):
            print(f"  REFUSING: {len(live_evals)} job(s) are already in the queue and the "
                  f"ledger is empty, so\n  their steps are indistinguishable from real "
                  f"gaps and would be submitted twice.\n"
                  f"  Either wait for the queue to drain (then the ledger is "
                  f"authoritative from now on),\n  or narrow with --models/--formats to "
                  f"work you know is unqueued AND pass --force.")
            raise SystemExit(2)

    n_jobs = n_steps = 0
    for model in args.models:
        for quant in args.formats:
            if quant not in LABELS:
                print(f"  skip {quant}: not in LABELS (add it to cover this format)")
                continue
            have = exported_steps(model, quant)
            if not have:
                continue
            for tasks in TASK_GROUPS:
                for mode in MODES:
                    gaps = (have
                            - landed_steps(model, quant, mode, tasks)
                            - flying.get((model, quant, mode, tasks), set()))
                    if not gaps:
                        continue
                    steps = ",".join(str(s) for s in sorted(gaps))
                    tag = f"{model:5} {quant:22} {mode:8} {tasks.split()[0]:16}"
                    n_jobs += 1
                    n_steps += len(gaps)
                    if not args.apply:
                        print(f"  GAP  {tag} {steps}")
                        continue
                    # --run is NOT optional. This script globs checkpoints with the
                    # family-aware tag prefix, but the sweep it shells out to defaults to
                    # RUN=qad3x on its own -- so without this a --family gemma3 scan finds
                    # the Gemma checkpoints correctly and then submits jobs that look for
                    # qad3x-google-gemma-3-270m-it-<quant>-* and find nothing. Same class
                    # of bug as the --full-disag eval-tag incident: the gap scan and the
                    # submitted job must resolve the SAME tag.
                    cmd = [SWEEP, "--run", RUN,
                           "--model", hf_id(model), "--tasks", tasks,
                           "--formats", f"{LABELS[quant]}:{quant}", "--steps", steps,
                           "--modes", mode]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    m = re.search(r"Submitted batch job (\d+)", r.stdout)
                    if m:
                        record(m.group(1), model, quant, mode, tasks, gaps)
                        print(f"  SENT {tag} {steps}  -> job {m.group(1)}")
                    else:
                        print(f"  FAIL {tag} {steps}\n{r.stdout[-400:]}{r.stderr[-400:]}")
    verb = "submitted" if args.apply else "would submit"
    print(f"\n  {verb} {n_jobs} array(s) covering {n_steps} (step, mode, task-group) point(s)")
    if not args.apply and n_jobs:
        print("  report only; re-run with --apply")


if __name__ == "__main__":
    main()
