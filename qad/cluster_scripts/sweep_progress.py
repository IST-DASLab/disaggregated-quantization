#!/usr/bin/env python3
"""Sweep progress & ETA: what every eval job is doing, and when the sweep finishes.

    ./cluster_scripts/sweep_progress.py                 # RULER (the default sweep)
    ./cluster_scripts/sweep_progress.py --kind disagg   # gsm8k/math500/mmlu_pro instead
    ./cluster_scripts/sweep_progress.py --watch 60      # refresh every 60s
    ./cluster_scripts/sweep_progress.py --top 40        # show more running jobs

Reads SLURM plus the eval logs on disk and reports, per running job, WHICH STAGE OF THE
PIPELINE it is in and how far through generation it is -- then an ETA for the whole
sweep.

Stages come from markers the driver prints, so "no output" is never mistaken for
"stalled": lm-eval is silent for the whole generation phase, which is most of a job's
life. Progress inside that phase is read from lm-eval's own tqdm counter instead, with
the proxy's completed-request count as a fallback.

ETA uses the OBSERVED median runtime of already-COMPLETED jobs of the same kind rather
than a guess, and adds a queue-drain term for PENDING jobs based on how many run at once.

This began as a cell in notebooks/plots.ipynb. It is a script because it answers an
operational question, not a figure one: it wants to be run from a shell on the login
node, repeatedly, while a sweep is in flight -- which a notebook cell makes awkward.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# This script lives in qad/cluster_scripts/, so the repo root is two levels up. Derived
# from __file__ rather than assumed to be the CWD: the notebook cell used Path("..")
# because it ran from notebooks/, and that is exactly the assumption that breaks the
# moment the same code is invoked from anywhere else.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# The two sweeps differ in more than a job name, so each one carries its own tuple.
# RULER is NOT an array sweep -- run_eval_ruler.sh submits one job per point -- which
# changes both the squeue parsing and the stack-dir naming below.
# ---------------------------------------------------------------------------
def kinds_for(root):
    qad = root / "qad"
    return {
        "ruler": {
            "job_name":   "qad-ruler",
            "logs":       root / "logs" / "eval_ruler",
            "results":    qad / "results" / "ruler",
            "banner_tag": "ruler-eval",
            "gen_marker": "ruler: tasks",   # eval_ruler.py:207, not "lm_eval: tasks"
            "arrays":     False,
        },
        "disagg": {
            "job_name":   "qad-disagg",
            "logs":       root / "logs" / "eval_disagg",
            "results":    qad / "results" / "disagg",
            "banner_tag": "disagg-eval",
            "gen_marker": "lm_eval: tasks",
            "arrays":     True,
        },
    }

# quantizer slug -> the label used everywhere else (plots, sweep --formats)
QUANT_LABEL = {
    "unquantized": "BF16", "nvfp4a16": "NVFP4A16", "nvfp4": "NVFP4",
    "nvfp4prefill": "NVFP4-prefill", "nvfp4decode": "NVFP4-decode",
    "nvfp4pdshared": "NVFP4-pd-shared", "nvfp4pdsplit": "NVFP4-pd-split",
    "lloyd3bit": "W3A16-Lloyd", "lloyd43": "Lloyd43", "lloyd21": "Lloyd21",
    "nvfp4lloyd43split": "L43-split", "nvfp4lloyd43upcast": "L43-upcast",
    "nvfp4lloyd43upcastboth": "L43-upcastboth",
    "nvfp4lloyd21split": "L21-split", "nvfp4lloyd21upcast": "L21-upcast",
    "nvfp4lloyd21upcastboth": "L21-upcastboth",
}
# longest slug first: "nvfp4" is a prefix of "nvfp4a16", both pd variants and every
# lloyd variant, so a shortest-first scan labels the entire sweep "NVFP4".
_SLUGS = sorted(QUANT_LABEL, key=len, reverse=True)

# Pseudo KV-cache compression runs (kv_noise_connector.py) tag themselves
# ...-kvp<bits> / ...-kvd<bits>: the rate applied to the PREFILL or the DECODE engine's
# KV cache, where sigma = 2^-bits * RMS(group of 16), i.e. 4^-bits of distortion power.
# The rate is NOT in the driver banner, only in the tag, so it has to be carried
# separately or enrich() below silently relabels every arm as plain "BF16" and the whole
# sweep looks like one repeated baseline.
# \d+(?:\.\d+)? -- fractional rates (3.5, 4.5) are swept too.
KV_RE = re.compile(r"-kv([pd])(\d+(?:\.\d+)?)(?:[-/_]|$)")

# RULER generates 500 fresh docs per task per requested length, fixed by the task spec.
RULER_TASKS = 13
RULER_DOCS_PER_TASK = 500
# Document counts for the disagg benchmarks, which are fixed-size datasets.
TASK_DOCS = {"gsm8k": 1319, "minerva_math500": 500, "aime25": 30,
             "mmlu_pro": 12032, "mmlu_flan_cot_zeroshot": 1531}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=60).stdout
    except Exception:
        return ""


def to_min(t):
    """SLURM elapsed (D-HH:MM:SS / HH:MM:SS / MM:SS) -> minutes."""
    if not t or t in ("0:00",):
        return 0.0
    d, _, rest = t.partition("-")
    if not rest:
        rest, d = d, "0"
    try:
        p = [float(x) for x in rest.split(":")]
    except ValueError:
        return 0.0
    sec = p[-1] + (p[-2] * 60 if len(p) > 1 else 0) + (p[-3] * 3600 if len(p) > 2 else 0)
    return (float(d) * 86400 + sec) / 60


def parse_kv(text):
    """'P4' / 'D2' for a KV-noise arm, else '' -- prefill or decode, at that bit rate."""
    m = KV_RE.search(text or "")
    return f"{m.group(1).upper()}{m.group(2)}" if m else ""


def parse_tag(text):
    """(model size, format label) from a log-dir label or a results-tree tag.

    The KV rate is folded INTO the label so the aggregate tables, which key on
    (size, format), keep the arms apart instead of summing them into one row.
    """
    m = re.search(r"Qwen3-([0-9.]+B)", text or "")
    if not m:
        m = re.search(r"gemma-3-([0-9.]+[bm])-it", text or "")
    size = m.group(1) if m else "?"
    kv = parse_kv(text)
    suffix = f"+kv{kv}" if kv else ""
    for slug in _SLUGS:
        if re.search(rf"[-/]{slug}([-/_]|$)", text or ""):
            return size, QUANT_LABEL[slug] + suffix
    return size, ("?" + suffix if suffix else "?")


def bar(frac, width=14):
    filled = int(round(min(max(frac, 0.0), 1.0) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def collect(cfg, top):
    JOB_NAME = cfg["job_name"]
    LOGS = cfg["logs"]

    # ---- 1. what SLURM thinks. StdOut is included because it resolves size/format for
    #         PENDING jobs too, which have not written a log line yet ----------------
    # -r expands array elements. Without it SLURM collapses pending ones into ranges
    # ("479272_[3-9]" is ONE line), so a 204-element sweep reports ~30 pending and the
    # ETA below comes out ~6x short. Harmless for RULER, which has no arrays.
    #
    # ArrayJobID/ArrayTaskID, NOT JobID: for an array element JobID is a distinct id
    # (array 479274 task 2 -> JobID 479313) while the .out file is named 479274_2.out.
    # Keying on JobID silently matched nothing, so 33 generating jobs displayed as
    # "queued/starting". JobID is still read -- older stack dirs are named after it, and
    # for RULER it IS the id the stack dir uses.
    _W = [12, 10, 12, 14, 14, 250]

    def _split(line):
        out, pos = [], 0
        for w in _W:
            out.append(line[pos:pos + w].strip())
            pos += w
        return out

    rows = []
    for line in sh(f'squeue -u $(whoami) -h -r -n {JOB_NAME} '
                   f'-O "ArrayJobID:12,ArrayTaskID:10,JobID:12,State:14,'
                   f'TimeUsed:14,StdOut:250"').splitlines():
        if not line.strip():
            continue
        ajid, atid, jobid, state, elapsed, stdout = _split(line)
        jid = f"{ajid}_{atid}" if atid and atid != "N/A" else ajid
        label = os.path.basename(os.path.dirname(stdout))
        size, fmt = parse_tag(label)
        rows.append({"id": jid, "array": ajid, "taskid": (atid if atid != "N/A" else "0"),
                     "jobid": jobid, "state": state, "elapsed": elapsed,
                     "size": size, "fmt": fmt, "kv": parse_kv(label),
                     "mode": "?", "step": "?", "task": "?", "tasks_raw": "", "seqlens": ""})

    # ---- 2. stage + exact config of each job, from its own log ---------------------
    # ordered most-advanced first; the first marker found wins
    STAGES = [
        ("results ->",         "done (writing)"),
        (cfg["gen_marker"],    "generating"),
        ("thinking",           "engines up"),
        ("stack ready",        "stack ready"),
        ("starting stack",     "engines loading"),
        (cfg["banner_tag"] + "]", "container up"),
    ]
    log_index = {}
    for f in glob.glob(str(LOGS / "*" / "*.out")):
        base = os.path.basename(f)
        m = re.match(r"^(\d+)_(\d+)\.out$", base)
        if m:
            jid = f"{m.group(1)}_{m.group(2)}" if m.group(2) != "4294967294" else m.group(1)
            log_index.setdefault(jid, f)
        elif (m := re.match(r"^(\d+)\.out$", base)):       # RULER: one job, no array id
            log_index.setdefault(m.group(1), f)

    BANNER = re.compile(
        rf"\[{cfg['banner_tag']}\] model=(\S+?)\s+"
        rf"(?:quantizer=(\S+)\s+iter=(\S+)\s+|UNQUANTIZED \(BF16\)\s+)"
        rf"think=(\d)\s+(?:tasks=|seqlens=)(.+)")

    def enrich(r):
        """Fill stage/mode/step/task from the driver banner -- authoritative, unlike the
        dir name."""
        f = log_index.get(r["id"])
        if not f:
            r["stage"] = "queued/starting"
            return r
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            r["stage"] = "?"
            return r
        if (m := BANNER.search(txt)):
            model, quant, it, think, trailing = m.groups()
            r["size"], _ = parse_tag(model)
            # The banner is authoritative for size/format but knows nothing about the KV
            # rate, so re-attach it from the log-dir name; otherwise every kvp/kvd arm is
            # relabelled "BF16" here and 40 distinct runs collapse into one row.
            r["fmt"] = (QUANT_LABEL.get(quant, quant) if quant else "BF16") + (
                f"+kv{r['kv']}" if r.get("kv") else "")
            r["step"] = it or "0"
            r["mode"] = "think" if think == "1" else "nothink"
            if cfg["job_name"] == "qad-ruler":
                r["seqlens"] = trailing.strip()
                r["task"] = f"{len(r['seqlens'].split(','))}L"   # e.g. "4L" = 4 lengths
            else:
                r["tasks_raw"] = trailing.strip()
                r["task"] = r["tasks_raw"].replace("mmlu_flan_cot_zeroshot", "mmlu_cot")
        r["stage"] = next((name for marker, name in STAGES if marker in txt), "container up")
        return r

    rows = [enrich(r) for r in rows]
    running = [r for r in rows if r["state"] == "RUNNING"]
    pending = [r for r in rows if r["state"] != "RUNNING"]

    # ---- 3. live generation progress ----------------------------------------------
    # lm-eval prints NOTHING between "tasks" and its final score, so the per-job median
    # is the only ETA available without this: a job 90% done and one 10% done look
    # identical.
    def expected_docs(r):
        if cfg["job_name"] == "qad-ruler":
            n_len = len([s for s in (r.get("seqlens") or "").split(",") if s.strip()])
            return RULER_TASKS * RULER_DOCS_PER_TASK * n_len or None
        n = sum(TASK_DOCS.get(t, 0) for t in (r.get("tasks_raw") or "").split())
        return n or None

    def proxy_done(r):
        """Requests prefilled so far, or None if this job has no proxy log yet.

        The proxy logs one "prefill ok" per request, which is the one live counter there
        is. It measures the PREFILL side, which runs ahead of decode by up to the
        in-flight window, so it is a slight over-estimate and saturates at 100% while the
        last decodes drain. Leading indicator, not a completion signal.

        Several namings are tried: disagg stack dirs are keyed on ARRAY_JOB_ID so they
        match the .out basename (older ones on the element id), while RULER's are a bare
        stack_<jobid> with no task suffix.
        """
        candidates = [LOGS / f"stack_{r['array']}_{r['taskid']}",
                      LOGS / f"stack_{r['jobid']}_{r['taskid']}",
                      LOGS / f"stack_{r['jobid']}",
                      LOGS / f"stack_{r['array']}"]
        for d in candidates:
            try:
                with open(d / "proxy.log", errors="ignore") as fh:
                    return sum(1 for ln in fh if "prefill ok" in ln)
            except OSError:
                continue
        return None

    def lmeval_progress(jid):
        """(done, total, timeouts) from lm-eval's own tqdm bar on stderr.

        Strictly better than counting proxy lines: tqdm counts DOCUMENTS, while the proxy
        counts REQUESTS, and a retried request is a second request for the same document.
        With timeouts running at 20-35% that gap is what pushed the bar past 100%.
        """
        f = log_index.get(jid)
        if not f:
            return None, None, 0
        try:
            with open(f[:-4] + ".err", "rb") as fh:          # tail only; these get large
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 262144))
                txt = fh.read().decode("utf-8", "replace")
        except OSError:
            return None, None, 0
        # Anchor on the "Requesting API" bar. lm-eval runs a preprocessing tqdm FIRST that
        # completes in seconds, so an unanchored N/M match reads that one and reports a job
        # 8 minutes in as 100% done -- with the bar hitting the top before generation starts.
        m = re.findall(r"Requesting API.*?(\d+)/(\d+) \[", txt.replace("\r", "\n"))
        done, total = (int(m[-1][0]), int(m[-1][1])) if m else (None, None)
        return done, total, txt.count("TimeoutError")

    for r in rows:
        if r["state"] != "RUNNING":
            r["done"] = r["total"] = r["frac"] = None
            r["timeouts"] = 0
            continue
        done, total, r["timeouts"] = lmeval_progress(r["id"])
        if done is None:                    # no tqdm line yet -> fall back to the proxy
            done, total = proxy_done(r), expected_docs(r)
        r["done"], r["total"] = done, total
        r["frac"] = (done / total) if (done and total) else None

    # ---- 4. ETA from OBSERVED runtimes of finished jobs ----------------------------
    done_min = []
    for line in sh('sacct -S now-12hours -u $(whoami) -n '
                   '--format=JobID,JobName,State,Elapsed -P').splitlines():
        f = line.split("|")
        if len(f) == 4 and JOB_NAME in f[1] and f[2] == "COMPLETED" \
                and re.match(r"^\d+(_\d+)?$", f[0]):
            done_min.append(to_min(f[3]))
    done_min.sort()
    median = done_min[len(done_min) // 2] if done_min else None

    print("jobs: " + (", ".join(f"{k}={v}" for k, v in
                                sorted(Counter(r["state"] for r in rows).items()))
                      or "none queued"))
    if median:
        print(f"observed median job runtime: {median:.0f} min  "
              f"(n={len(done_min)} completed in last 12h)")
    else:
        print("no completed jobs yet - ETA unavailable until the first one finishes")

    # ---- 5. queue broken down by model size and format -----------------------------
    if rows:
        q = defaultdict(lambda: [0, 0])
        for r in rows:
            q[(r["size"], r["fmt"])][0 if r["state"] == "RUNNING" else 1] += 1
        print(f"\nqueue by model / format:\n  {'size':6} {'format':19} {'run':>4} {'pend':>5}")
        for (size, fmt), (nr, np_) in sorted(q.items()):
            print(f"  {size:6} {fmt:19} {nr:4d} {np_:5d}")

    # ---- 6. results already on disk, by size / format / thinking mode ---------------
    # The two trees are shaped differently: disagg splits think/nothink by a _think
    # SUFFIX on the tree dir, RULER by a think/ or nothink/ dir name.
    disk = defaultdict(Counter)
    for f in glob.glob(str(cfg["results"] / "*" / "*" / "step_*.json")):
        tree = Path(f).parts[-3]
        size, fmt = parse_tag(Path(f).parts[-2])
        mode = "think" if (tree.endswith("_think") or tree == "think") else "nothink"
        disk[(size, fmt)][mode] += 1
    if disk:
        print(f"\nresults on disk:\n  {'size':6} {'format':19} {'think':>6} {'nothink':>8}")
        for (size, fmt), c in sorted(disk.items()):
            print(f"  {size:6} {fmt:19} {c['think']:6d} {c['nothink']:8d}")
        print(f"  {'':6} {'TOTAL':17} {sum(c['think'] for c in disk.values()):6d} "
              f"{sum(c['nothink'] for c in disk.values()):8d}")
    else:
        print("\nresults on disk: none yet")

    # ---- 7. the running jobs themselves --------------------------------------------
    if running:
        print("\nstage tally:", dict(Counter(r["stage"] for r in running)))

        # ETA is computed per job BEFORE sorting, so it lives on the row and can be
        # sorted or inspected. The rate comes from the job's own elapsed time, hence the
        # need for a real sample first; "*" marks the median-based guess used until then.
        # Extrapolating from a job that has barely started is worthless: 1 doc of 12032
        # at 3 min projects to ~42000 min. Below this much progress, use the median.
        MIN_FRAC = 0.05
        for r in running:
            el = to_min(r["elapsed"])
            if r["frac"] and r["frac"] >= MIN_FRAC and el > 2:
                r["eta_min"] = max(0.0, (r["total"] - r["done"]) / (r["done"] / el))
                r["eta_src"] = ""
            else:
                r["eta_min"] = (max(0, median - el) if median else None)
                r["eta_src"] = "*"

        # Sort by REAL progress, most complete first, so the jobs about to free a GPU are
        # at the top. Anything not yet generating has frac None and sinks to the bottom as
        # a block rather than being interleaved by elapsed time -- a job 2 min into
        # loading engines otherwise sorts above one 90% through generating.
        order = sorted(running, reverse=True,
                       key=lambda x: (x["frac"] is not None, x["frac"] or 0.0,
                                      to_min(x["elapsed"])))

        print(f"\nrunning jobs ({len(running)}):\n  {'id':>12}  {'size':5} {'format':19} "
              f"{'mode':8} {'step':>6} {'task':10} {'stage':16} {'elapsed':>8}  "
              f"{'progress':16} {'eta':>7} {'t/o':>6}")
        for r in order[:top]:
            eta = f"~{r['eta_min']:.0f}m{r['eta_src']}" if r["eta_min"] is not None else "?"
            prog = (f"{bar(r['frac'])} {r['frac'] * 100:3.0f}%" if r["frac"] is not None
                    else (f"{r['done']} reqs" if r["done"] else "-"))
            print(f"  {r['id']:>12}  {r['size']:5} {r['fmt']:17} {r['mode']:8} "
                  f"{r['step']:>6} {r['task'][:10]:10} {r['stage']:16} "
                  f"{to_min(r['elapsed']):7.1f}m  {prog:16} {eta:>7} "
                  f"{(r['timeouts'] or ''):>6}")
        if len(running) > top:
            n_gen = sum(1 for r in order[top:] if r["frac"] is not None)
            print(f"  ... {len(running) - top} more ({n_gen} generating, "
                  f"{len(running) - top - n_gen} not yet)")
        print("  progress = lm-eval's own doc counter; * = median-based ETA (no sample yet)")
        to = sum(r["timeouts"] or 0 for r in running)
        if to:
            print(f"  t/o = client TimeoutErrors ({to} across running jobs). Each one "
                  f"discards a finished generation and resends it.")

    # ---- 8. sweep ETA ---------------------------------------------------------------
    # Prefer the CURRENT sweep's own pace over the sacct median. The median covers the
    # last 12h, so right after switching benchmarks it describes the previous one -- it
    # read 25 min while these mmlu_pro jobs were each projecting 150+. Live projections
    # come from jobs actually running now, so they track the benchmark in flight.
    projected = sorted(to_min(r["elapsed"]) / r["frac"]
                       for r in running if r["frac"] and r["frac"] >= 0.05)
    job_min = projected[len(projected) // 2] if projected else median
    src_lbl = (f"live, n={len(projected)} running" if projected
               else f"sacct median, n={len(done_min)}")

    if job_min and rows:
        conc = max(len(running), 1)
        waves = (len(pending) + conc - 1) // conc
        longest_running = max((job_min - to_min(r["elapsed"]) for r in running), default=0)
        eta_total = max(0, longest_running) + waves * job_min
        print(f"\ntypical job: {job_min:.0f} min  ({src_lbl})")
        print(f"sweep ETA: ~{eta_total:.01f} min "
              f"({len(running)} running, {len(pending)} pending, ~{conc} at a time, "
              f"{waves} more wave(s))")
        # include the weekday: these sweeps routinely run past midnight, and a bare
        # "13:22" on a 28-hour ETA reads as a time that has already passed
        print(f"           finishing around "
              f"{time.strftime('%a %H:%M', time.localtime(time.time() + eta_total * 60))}"
              f"  ({eta_total / 60:.1f} h)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("ruler", "disagg"), default="ruler",
                   help="which sweep to report on (default: ruler)")
    p.add_argument("--top", type=int, default=25,
                   help="how many running jobs to list (default: 25)")
    p.add_argument("--watch", type=int, metavar="SECONDS",
                   help="refresh every SECONDS until interrupted")
    # logs/ and results samples are gitignored, so a checkout that is not the one the
    # sweep actually ran from (a worktree, say) sees no logs at all and reports every
    # job as "queued/starting". Point this at the checkout the jobs were submitted from.
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="repo root to read logs/ and results/ from "
                        "(default: two levels above this script)")
    args = p.parse_args()

    cfg = kinds_for(args.root.resolve())[args.kind]
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")        # clear + home, so it reads as a dashboard
            print(f"=== {cfg['job_name']} @ {time.strftime('%a %H:%M:%S')} ===")
        collect(cfg, args.top)
        if not args.watch:
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    sys.exit(main())
