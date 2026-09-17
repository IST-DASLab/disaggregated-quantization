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

# __file__ is NOT usable here: CPython resolves it to an absolute path via os.getcwd()
# during interpreter startup, before any code in this file runs -- os.getcwd(), like
# bash's `pwd -P`, is always the PHYSICAL cwd (a raw getcwd() syscall), so __file__
# already has the /lustre->/scratch symlink resolved no matter how this process was
# invoked. A /scratch-rooted ROOT builds a /scratch SWEEP path
# (run_eval_disagg_sweep.sh) that every downstream script inherits through its own
# self-location logic, ending in a container trying to bash a path it can't see (only
# /lustre is mounted). sys.argv[0], unlike __file__, is left exactly as typed on the
# command line -- combine it with PWD (bash's LOGICAL cwd, the string from the
# caller's last cd, not a syscall) instead. os.path.join drops PWD if argv[0] is
# already absolute, so this is correct either way -- but the caller must actually have
# cd'd to a /lustre path (matches what invokes this: autoeval_watch.sh's own
# SELF_DIR).
_cwd = os.environ.get("PWD") or os.getcwd()
ROOT = os.path.dirname(os.path.dirname(os.path.normpath(os.path.join(_cwd, sys.argv[0]))))
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
# ONE checkpoint prefix for every family. A separate `gemma3-` checkpoint namespace was a
# relic of the early Gemma ablations; all models now train under run_qad.sh's default
# RUN_PREFIX, so a family key selects MODELS AND MODES ONLY and no longer namespaces
# checkpoints. This used to be implicit (the family name WAS the prefix), which made a
# whole family silently invisible: the Gemma sweep here trained as qad3x-google-gemma-3-*
# while the scanner globbed gemma3-google-gemma-3-* and reported
# "would submit 0 ... covering 0 point(s)" for all 12 runs -- a clean, successful-looking
# exit that evaluates nothing. Historical gemma3-prefixed dirs under results/ are old-cluster
# scores whose checkpoints no longer exist; they are left as orphaned history.
PREFIX = "qad3x"
FAMILY = "qad3x"                     # overridden by --family
RUN = FAMILY
MODELS = FAMILIES[FAMILY]["models"]
MODES = FAMILIES[FAMILY]["modes"]


def hf_id(model: str) -> str:
    return FAMILIES[RUN]["hf"].format(model)


def tag_prefix(model: str, quant: str) -> str:
    """The glob every checkpoint/result directory for this (family, model, quant) matches."""
    return f"{PREFIX}-{hf_id(model).replace('/', '-')}-{quant}-*"
GRID = [0] + list(range(250, 2251, 250))
TASK_GROUPS = ["gsm8k minerva_math500", "mmlu_pro"]

# quantizer -> sweep label. Only formats the plots draw; adding one here is all that is
# needed for this script to start covering it.
LABELS = {
    "nvfp4": "NVFP4", "nvfp4a16": "NVFP4A16",
    # lloyd21 is W2A16, not W3A16 -- four grid levels, not eight (quantizers/__init__.py's
    # "lloyd21": "the lloyd43 construction at 2 BITS"). It read W3A16 here, which would
    # have put the wrong bit-width on every plot legend that reads LABELS.
    "lloyd3bit": "W3A16-Lloyd", "lloyd43": "W3A16-Lloyd43", "lloyd21": "W2A16-Lloyd21",
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
}

# Excluded from the DEFAULT --formats, not from LABELS: still coverable with an explicit
# --formats nvfp4 (etc.) if that ever changes. 2026-09-09 (user): these three already
# have a separate, conventional eval plan and existing results outside this pipeline --
# autoeval_watch.sh picking them up as "gaps" the moment their checkpoints export would
# duplicate/interfere with that, not fill a real gap.
# Derived from RULER_FORMATS below: every format on the RULER plan is excluded from the
# DISAGG default, which is exactly what the 2026-09-09 note above asks for -- they have
# a separate eval plan, and picking them up as disagg "gaps" the moment a checkpoint
# exports would submit gsm8k/math500/mmlu_pro jobs nobody asked for. Kept as two names
# because they are two questions ("do not disagg-sweep this" vs "do RULER this"); they
# merely happen to coincide while the sweep is RULER-only.
DEFAULT_EXCLUDED_FORMATS = None   # set below, once RULER_FORMATS exists

# Formats RULER autosubmits. Kept SEPARATE from DEFAULT_EXCLUDED_FORMATS -- those two were
# the same set only while the sweep was the three phase-isolation arms. They answer
# different questions: EXCLUDED is "do not send this to the disagg sweep", RULER_FORMATS is
# "do send this to RULER". Anything listed here must also be in LABELS (the submit path
# reads LABELS[quant] for the sweep label) and must actually have checkpoints, or it is
# simply skipped.
RULER_FORMATS = [
    "nvfp4",
    "nvfp4a16",                                 # W4A16 weight-only, the no-A4 control
    "lloyd43", "lloyd21",                       # W3A16 / W2A16 weight-only LUT baselines
    "nvfp4prefill", "nvfp4decode",              # phase-isolation ablation
    "nvfp4pdshared", "nvfp4pdsplit",            # NVFP4 prefill / NVFP4A16 decode
    "nvfp4lloyd43upcastboth", "nvfp4lloyd43upcast", "nvfp4lloyd43split",   # LUT3 decode
    "nvfp4lloyd21upcastboth", "nvfp4lloyd21upcast", "nvfp4lloyd21split",   # LUT2 decode
]
DEFAULT_EXCLUDED_FORMATS = set(RULER_FORMATS)

# ===================================================================================
# RULER autosubmit. "These three already have a separate, conventional eval plan" ABOVE
# refers to this: the plan is RULER, run through bin/run_eval_ruler.sh rather than the
# disagg sweep. Same GRID (2026-09-09, user: "all steps from 0 to 2250 evaluated every
# 250 steps, same as other evals"), same checkpoint-discovery (exported_steps() below is
# reused as-is -- checkpoint scanning does not care which eval will read the checkpoint),
# but everything downstream differs enough to need its own path rather than reusing
# run_eval_disagg_sweep.sh's:
#   * run_eval_ruler.sh has no array/multi-step mode -- one job is one (model, quant,
#     step, mode) point, not a --steps a,b,c array the way SWEEP covers a whole gap set
#     in one sbatch call. So this submits one job per missing point, not one job per gap.
#   * results/ruler/<mode>/<tag>/step_N.json's shape is per-task-per-length metrics, not
#     landed_steps()'s flat task-name match, and "complete" means covering the model's
#     FULL seqlen sweep, not just having every task key present (a job that only ran a
#     subset of lengths -- e.g. an interrupted retry -- would otherwise look done).
#   * modes are NOT the family's disagg modes: RULER's own module docstring/run script
#     fixed this earlier -- Qwen is nothink-only (RULER's 128-token generation budget
#     has no room for a <think> trace) and Gemma is think-only (no thinking axis at
#     all), never "both" the way FAMILIES[...]["modes"] lists for disagg.
# ===================================================================================
RULER_SWEEP = os.path.join(ROOT, "bin", "run_eval_ruler.sh")
RULER_MODES = {"qad3x": ["nothink"], "gemma3": ["think"]}
# The 13 RULER-paper tasks eval_ruler.py runs (its own RULER_TASKS default) -- duplicated
# here rather than imported so this script stays free of qad/transformers imports; it is
# only ever used to check "does this result file have every task", not to run anything.
RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_vt", "ruler_cwe", "ruler_fwe",
    "ruler_qa_squad", "ruler_qa_hotpot",
]
# model size -> max_position_embeddings. Hardcoded rather than resolved live via
# AutoConfig: that needs transformers + the HF cache (a CPU container job, per
# run_eval_ruler.sh's own login-mode resolution), which this script deliberately avoids
# needing just to scan for gaps. Measured against the real configs for every model this
# sweep covers; a new model size added here needs its ceiling added too, or
# ruler_seqlens() below raises KeyError rather than silently mis-sizing the sweep.
RULER_MAX_CONTEXT = {
    "0.6B": 40960, "1.7B": 40960, "4B": 40960, "8B": 40960,
    "270m": 32768, "1b": 32768, "4b": 131072, "12b": 131072,
}


# Must match run_eval_ruler.sh's RULER_MIN_SEQLEN / RULER_TOP_SEQLEN exactly. If they
# drift, ruler_landed_steps() judges every result partial and the gap scan resubmits the
# entire corpus on every poll, forever.
#
# 4096 enabled 2026-09-12 -- see run_eval_ruler.sh for the full reasoning. It is only
# safe because the submit path below now asks ruler_missing_seqlens() what a given STEP
# lacks instead of passing this whole list: a point holding 8k/16k/32k gets a 4096-only
# job. Passing the full sweep would recompute lengths already on disk, and since RULER
# regenerates its documents every run those scores would change underneath any figure
# already drawn from them. If that per-step logic is ever removed, this must go back
# to 8192 in the same change.
RULER_MIN_SEQLEN = 4096
RULER_TOP_SEQLEN = 32768


def ruler_seqlens(model: str) -> list:
    """Mirrors run_eval_ruler.sh's own auto-sweep exactly: powers of 2 from
    RULER_MIN_SEQLEN up to min(model ceiling, RULER_TOP_SEQLEN), dropping the top value
    if it lands EXACTLY on the model's ceiling (no margin left there for RULER's
    generation budget + chat-template overhead otherwise -- confirmed live, see
    eval_ruler.py's max_model_len clamp and its own comment)."""
    cap = min(RULER_MAX_CONTEXT[model], RULER_TOP_SEQLEN)
    out, n = [], RULER_MIN_SEQLEN
    while n <= cap:
        out.append(n)
        n *= 2
    # A top length landing exactly ON the model's ceiling used to be dropped here, since
    # vLLM cannot serve a prompt with no room above it for generation + chat template.
    # eval_ruler.py now handles that by running the largest length that DOES fit and
    # reporting it under the nominal label (31744 -> "32768"), so gemma-3-270m/1b get a
    # 32K column like everything else instead of stopping at 16384.
    return out


def ruler_step_coverage(model: str, quant: str, mode: str) -> dict:
    """{step: set(seqlens already measured)} for this run's result JSONs.

    Exists so a partially covered step can be TOPPED UP with only the lengths it lacks.
    Re-running a length already on disk is not just wasted GPU time: RULER generates its
    documents fresh per run, so a recomputed score is a different sample of the same
    distribution and the number silently moves under any figure already drawn from it.
    A file missing whole TASKS is treated as covering nothing, since the per-length
    columns of a truncated run cannot be trusted either.
    """
    tree = os.path.join(ROOT, "results", "ruler", mode)
    out = {}
    for d in glob.glob(os.path.join(tree, tag_prefix(model, quant))):
        for f in glob.glob(os.path.join(d, "step_*.json")):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            if not set(RULER_TASKS) <= set(data.get("results", {})):
                continue
            step = int(re.search(r"step_(\d+)", os.path.basename(f)).group(1))
            out[step] = out.get(step, set()) | set(data.get("seqlens", []))
    return out


def ruler_missing_seqlens(model: str, coverage: dict, step: int) -> list:
    """The lengths this step still needs, in sweep order -- never the ones it has."""
    have = coverage.get(step, set())
    return [s for s in ruler_seqlens(model) if s not in have]


def ruler_landed_steps(model: str, quant: str, mode: str) -> set:
    """Steps whose result JSON covers every RULER task AND the model's full seqlen
    sweep -- a file from a partial/interrupted run (fewer tasks, or fewer lengths than
    the model's own ruler_seqlens()) must not count as done, or a retry never happens.
    """
    want_seqlens = set(ruler_seqlens(model))
    tree = os.path.join(ROOT, "results", "ruler", mode)
    out = set()
    for d in glob.glob(os.path.join(tree, tag_prefix(model, quant))):
        for f in glob.glob(os.path.join(d, "step_*.json")):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            res = data.get("results", {})
            have_seqlens = set(data.get("seqlens", []))
            if set(RULER_TASKS) <= set(res) and want_seqlens <= have_seqlens:
                out.add(int(re.search(r"step_(\d+)", os.path.basename(f)).group(1)))
    return out


# The container-mode driver banner run_eval_ruler.sh prints, same regex used by
# notebooks/plots.ipynb's RULER tracking cell -- kept in sync by hand, not imported,
# since the notebook cell has no importable module either.
RULER_BANNER = re.compile(
    r"\[ruler-eval\] model=(\S+?)\s+(?:quantizer=(\S+)\s+iter=(\S+)\s+|UNQUANTIZED \(BF16\)\s+)"
    r"think=(\d)")


def ruler_inflight() -> dict:
    """(model, quant, mode) -> steps a currently RUNNING qad-ruler job is covering.

    NOT ledger-based, unlike inflight() above. Two reasons: run_eval_ruler.sh has no
    array mode, so every job is exactly one (model, quant, step, mode) point, directly
    readable from that one job's own banner line -- no need to remember "which steps did
    THIS job cover" the way an array submission requires. And more importantly: RULER
    jobs were being submitted by hand (this file's own predecessor) for a while before
    this autosubmit path existed, so a ledger would start empty against a queue that
    already has real jobs in it -- exactly the bootstrap-guard scenario the disagg path
    refuses on, except here there is a clean fix (read the live jobs directly) instead of
    just refusing.

    PENDING jobs are not covered: their .out file does not exist yet, so which step they
    target is unknowable until they start. That is a bounded, self-healing gap (worst
    case a step gets submitted twice; eval_ruler.py merges into the same file rather than
    clobbering), not a correctness issue worth blocking on.
    """
    out = {}
    try:
        # -O StdOut, NOT -o "%o". squeue's %o is the COMMAND (here the run_eval_ruler.sh
        # path), not the stdout file -- so the old spec made this open the SCRIPT and
        # search it for a run banner, which never matched. ruler_inflight() therefore
        # returned {} unconditionally, even for RUNNING jobs, and every re-run resubmitted
        # work already in progress. -O keeps the same "jobid state path" shape, so the
        # split(None, 2) below is unchanged; %j in the path is still expanded by hand.
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r",
                            "-n", "qad-ruler", "-O", "JobID:24,State:16,StdOut:512"],
                           capture_output=True, text=True, timeout=60)
    except Exception:
        return out
    hf_to_model = {FAMILIES[fam]["hf"].format(m): (fam, m)
                  for fam in FAMILIES for m in FAMILIES[fam]["models"]}
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "RUNNING":
            continue
        stdout_path = parts[2].replace("%j", parts[0])
        try:
            txt = open(stdout_path, errors="ignore").read()
        except OSError:
            continue
        m = RULER_BANNER.search(txt)
        if not m:
            continue
        model_hf, quant, it, think = m.groups()
        fam_model = hf_to_model.get(model_hf)
        if not fam_model or fam_model[0] != RUN:
            continue
        _, model = fam_model
        mode = "think" if think == "1" else "nothink"
        step = int(it) if it not in (None, "") else 0
        out.setdefault((model, quant, mode), set()).add(step)
    return out


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


# Same job-name convention as autoeval_watch.sh's EVAL_JOBS: the fixed --job-name each
# eval entrypoint's #SBATCH header declares (training jobs are qad-<model>-<quant>, never
# one of these four literals).
# "qad-ruler" belongs here: run_eval_ruler.sh submits under that name, and while it was
# missing active_jobs() could never see a RULER job, so ledger entries for RULER always
# looked "no longer in flight". Same drift as autoeval_watch.sh's EVAL_JOBS regex.
EVAL_JOB_NAMES = {"qad-eval", "qad-dual-eval", "qad-disagg", "qad-vllm", "qad-ruler"}


def active_jobs() -> set:
    """Job IDs of this user's currently-queued EVAL jobs (not training).

    Filtering by name matters here, not just cosmetically: every ledger entry IS an
    eval job by construction (record() is only called after a SWEEP submission), so an
    unfiltered active_jobs() would silently never match a training job id anyway in
    inflight() -- but the bootstrap guard below uses this list to ask "is anything
    already queued that could collide with what I'm about to submit," and training jobs
    obviously cannot collide with an eval-step gap. Without this filter the guard
    refused unconditionally the moment ANY training job was in the queue -- which is
    every time autoeval_watch.sh runs concurrently with training, its entire purpose.
    """
    try:
        # %i, not %A: %A is documented as the array's base job id, but SLURM actually
        # gives each array TASK its own unique numeric id the moment it leaves PENDING
        # for RUNNING, and %A reports THAT per-task id instead once it exists -- e.g.
        # array 3645019's task _0 shows as %A=3645030 while running, not 3645019.
        # record() stores the BASE id (from sbatch's "Submitted batch job N" at
        # submission time), so comparing against %A silently stopped matching the
        # instant any element started running: the ledger entry looked no-longer-in-
        # flight, and a later poll resubmitted the exact steps still in progress. %i
        # gives the stable "BASEID_INDEX" display form regardless of task state; split
        # off the base.
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
                           capture_output=True, text=True, timeout=60)
        out = set()
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in EVAL_JOB_NAMES:
                out.add(parts[0].split("_")[0])
        return out
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
    ap.add_argument("--formats", nargs="*",
                    default=sorted(set(LABELS) - DEFAULT_EXCLUDED_FORMATS))
    ap.add_argument("--ruler-formats", nargs="*", default=list(RULER_FORMATS),
                    help="formats to autosubmit through RULER instead of the disagg "
                         "sweep (default: RULER_FORMATS above); "
                         "pass an empty list to skip RULER entirely")
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
                    cmd = [SWEEP, "--run", PREFIX,
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

    # -----------------------------------------------------------------------------
    # RULER. Separate loop, separate job model (see the block comment above
    # RULER_SWEEP): one job per (model, quant, step, mode) point, not one array per gap
    # set, and modes come from RULER_MODES, not the family's disagg MODES.
    # -----------------------------------------------------------------------------
    ruler_flying = ruler_inflight()
    # Ledger-based cover, UNIONED with the live-banner scan above. ruler_inflight() can
    # only see RUNNING jobs (a PENDING job has no stdout to read a banner from), and its
    # docstring calls that "bounded, self-healing". That assumption breaks completely the
    # moment jobs cannot start: under a cluster maintenance reservation all 224 submitted
    # RULER jobs sat PENDING for hours, so nothing was ever "in flight" and a second run
    # duplicated the ENTIRE sweep (observed: 448 queued jobs, exactly 2x224). Worse,
    # autoeval_watch.sh calls --apply every INTERVAL, so it would have re-submitted the
    # whole grid on every poll. The ledger records at SUBMIT time, so it covers PENDING.
    ruler_led = inflight()
    n_jobs_r = n_steps_r = 0
    ruler_modes = RULER_MODES[RUN]
    for model in args.models:
        for quant in args.ruler_formats:
            if quant not in LABELS:
                print(f"  skip {quant}: not in LABELS (add it to cover this format)")
                continue
            have = exported_steps(model, quant)
            if not have:
                continue
            for mode in ruler_modes:
                gaps = (have
                        - ruler_landed_steps(model, quant, mode)
                        - ruler_flying.get((model, quant, mode), set())
                        - ruler_led.get((model, quant, mode, "ruler"), set()))
                if not gaps:
                    continue
                tag = f"{model:5} {quant:22} {mode:8} {'ruler':16}"
                n_jobs_r += len(gaps)
                n_steps_r += len(gaps)
                if not args.apply:
                    print(f"  GAP  {tag} {','.join(str(s) for s in sorted(gaps))}")
                    continue
                run_name = f"{PREFIX}-{hf_id(model).replace('/', '-')}"
                # Per STEP, not per run: a step holding 8k/16k/32k and lacking only 4096
                # must be topped up with 4096 alone. Passing the full sweep would
                # recompute the three it already has, and since RULER regenerates its
                # documents every run those scores would change -- silently moving
                # numbers under figures already drawn from them.
                coverage = ruler_step_coverage(model, quant, mode)
                for step in sorted(gaps):
                    seqlens = ruler_missing_seqlens(model, coverage, step)
                    if not seqlens:
                        continue
                    cmd = [RULER_SWEEP, "--model", hf_id(model), "--quantizer", quant,
                           "--run-name", run_name, "--iter", str(step),
                           "--seqlens", ",".join(str(s) for s in seqlens),
                           "--max-context", str(RULER_MAX_CONTEXT[model]),
                           "--think" if mode == "think" else "--no-think"]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    m = re.search(r"Submitted batch job (\d+)", r.stdout)
                    if m:
                        # tasks="ruler" keys these apart from the disagg task-groups in
                        # the shared ledger. One point per entry: run_eval_ruler.sh has no
                        # array mode, so a job is exactly one (model, quant, step, mode).
                        record(m.group(1), model, quant, mode, "ruler", [step])
                        print(f"  SENT {tag} {step} [{','.join(str(s) for s in seqlens)}]"
                              f"  -> job {m.group(1)}")
                    else:
                        print(f"  FAIL {tag} {step}\n{r.stdout[-400:]}{r.stderr[-400:]}")
    verb = "submitted" if args.apply else "would submit"
    print(f"\n  RULER: {verb} {n_jobs_r} job(s) covering {n_steps_r} (step, mode) point(s)")
    if not args.apply and n_jobs_r:
        print("  report only; re-run with --apply")


if __name__ == "__main__":
    main()
