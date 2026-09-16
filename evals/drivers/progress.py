"""What is running right now, and how far along it is.

    python3 progress.py
    python3 progress.py --all        # finished jobs from this session's logs too

`squeue` says RUNNING, which is not the same as WORKING: these jobs spend the first
5-15 minutes loading 20-56 GB of weights and print nothing, and a stalled generation
looks identical to a healthy one from the outside. This reads the per-item progress the
drivers already emit and turns it into a count with a rate and an ETA.

WHERE THE NUMBERS COME FROM
---------------------------
Three sources, most specific first, because each fails in a different way:

  * the tqdm record in the job's .err -- gives done/total, elapsed, remaining and rate
    in one line, and is the only source that knows the rate;
  * the output file's line count -- ground truth for what is durably on disk, and the
    only one that survives a driver restart, since run_resumable reloads what it
    finished and tqdm starts its bar again from the remainder;
  * the job's own header line -- model, benchmark and tag, so a row can be named even
    before any generation has happened.

PRELIMINARY ACCURACY is scored from the rows already on disk, with the SAME parser the
final scorer uses -- MMMU-Pro's own parse_multi_choice_response, seeded identically,
GPQA's own extraction regex, and RULER's own string_match_all/string_match_part. A
second, simpler extractor written for speed here would drift from the real one and the
preliminary number would stop predicting the final one, which is its only purpose. If
those parsers cannot be imported the column shows `-` rather than a number computed a
different way.

RULER differs from the others in three ways, all handled above: its metrics give PARTIAL
credit, so accuracy accumulates floats rather than bools; its run_resumable header is
per (length, task) -- 65 of them into one file -- so the absolute total comes from
ruler_infer's own sweep line instead; and its final number is a mean over TASK means
while this column pools documents, so the two coincide only at equal coverage and this
one drifts while a sweep is mid-flight.

A RULER job also has a phase the others do not: `preparing`. RULER generates its
haystacks before it generates any tokens, and at 32k/64k that runs for hours with an
idle GPU and no output file -- shown as `preparing` with the task and length in flight,
because a job sitting at `loading` for three hours looks exactly like a hung one.

It is a running mean over the items finished FIRST, and those are the fast ones: short
prompts, short reasoning, less truncation. Expect it to drift as the slow tail lands --
on MMMU it typically reads a point or two high early on. Useful for "is this arm roughly
where the others are", not for a number to quote.

A job with none of the three is still listed, with its phase (queued / loading) rather
than a fabricated percentage. Silence is a state worth showing, not a blank.

Stdlib only, on purpose: this runs on a login node where the pip overlay is not on the
path, and a progress tool that needs its own environment is one more thing to debug at
the moment you least want to.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVALS = os.path.dirname(HERE)
REPO = os.path.dirname(EVALS)
LOGS = os.path.join(REPO, "logs", "evals")

# `desc:  45%|####  | 780/1730 [12:34<15:20,  1.03it/s]` -- also matches the s/it form
# that slow benchmarks print, which is the same information inverted.
TQDM = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:?]+),\s*([\d.]+)(it/s|s/it)")
HEADER = re.compile(r"model=(\S+)\s+bench=(\S+)\s+tag=(\S+)")
# run_resumable's own line: "1730 total | 412 done | 1318 to generate | 32 workers -> P"
RESUMABLE = re.compile(r"(\d+) total \| (\d+) done \| (\d+) to generate .*-> (\S+)")
# ruler_infer's sweep header: "ruler sweep: 13 tasks x 5 lengths = 32500 total -> P".
# RULER is the one bench whose run_resumable header is PER (length, task) -- 65 of them,
# all appending to one raw.jsonl -- so "last total wins" would pair one task's 500 with
# a file holding every task's rows. This line carries the only absolute total there is.
RULER_SWEEP = re.compile(r"ruler sweep: \d+ tasks x \d+ lengths = (\d+) total -> (\S+)")

OVERLAY = "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-pydeps"
HARNESS = os.environ.get(
    "MMMU_HARNESS",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/MMMU/mmmu-pro")
_SCORERS = {}


def scorer(bench):
    """The benchmark's own answer parser, or None if it cannot be imported here.

    Imports are deferred and cached: this tool must stay usable on a login node with
    nothing installed, so a missing pandas costs the accuracy column and nothing else.
    """
    if bench in _SCORERS:
        return _SCORERS[bench]
    fn = None
    try:
        if bench == "mmmu":
            import random
            for p in (OVERLAY, HARNESS):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import ast
            from evaluate import (get_multi_choice_info,
                                  parse_multi_choice_response)

            def fn(row):                                          # noqa: F811
                i2a, ch = get_multi_choice_info(
                    ast.literal_eval(str(row["options"])))
                # Same seed as mmmu_pro_score.py: the parser guesses randomly on an
                # unparseable response, and an unseeded guess would make this column
                # disagree with the final score for no reason.
                random.seed(20260825)
                return parse_multi_choice_response(
                    row.get("content") or "", ch, i2a) == row["answer"]
        elif bench == "ocrbench":
            # Deliberately unsupported. OCRBench scoring is 15-20 minutes of TEDS and
            # uncapped edit distance -- a "preliminary accuracy" that takes 20 minutes
            # to compute is not preliminary, and it would run on every invocation of a
            # progress tool. A cheaper approximation would be a different number from
            # the final one, which is exactly what this column must never be. The
            # progress count, rate and ETA below all still work.
            fn = None
        elif bench == "mmlu_pro":
            # Cheap enough to run on every invocation: one regex per row, no model and
            # no harness import. Seeded per call so the preliminary number is a
            # deterministic function of the rows on disk, matching the final scorer.
            import random as _r
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            from mmlu_pro_infer import predict

            def fn(row, _rng=_r.Random(20260827)):                 # noqa: F811
                return predict(row.get("content"), _rng)[0] == row["gold"]
        elif bench == "ruler":
            # RULER's OWN metric functions, imported from the clone exactly as
            # ruler_score.py does -- string_match_all for retrieval/tracking/extraction,
            # string_match_part for qa. They return a PERCENTAGE for a list of rows, so
            # a single row is scored as a one-element batch and divided back down; that
            # yields partial credit (a multi-answer niah row can be 0.4), which is why
            # preliminary_accuracy accumulates floats rather than bools.
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            import ruler_score as _rs
            root = os.environ.get(
                "RULER_HARNESS",
                "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/"
                "apanferov/prefill_decode/harnesses/RULER")
            import yaml
            with open(os.path.join(root, "scripts/synthetic.yaml")) as fh:
                by_name = yaml.safe_load(fh)
            metrics = _rs._load_module(
                os.path.join(root, "scripts/eval/synthetic/constants.py"),
                "ruler_eval_constants_progress").TASKS

            def fn(row):                                          # noqa: F811
                m = metrics[by_name[row["task"]]["task"]]["metric_fn"]
                pred = _rs.postprocess_pred(row.get("content") or "")
                return m([pred], [row["outputs"]]) / 100.0
        elif bench == "gpqa":
            if HERE not in sys.path:
                sys.path.insert(0, HERE)
            from gpqa_diamond import ANSWER_RE

            def fn(row):                                          # noqa: F811
                m = ANSWER_RE.findall(row.get("content") or "")
                return bool(m) and m[-1].upper() == row["gold"]
    except Exception:                                             # noqa: BLE001
        fn = None
    _SCORERS[bench] = fn
    return fn


def preliminary_accuracy(paths, bench):
    """Accuracy over the rows currently on disk, or None.

    Takes a LIST because a sharded arm's rows are spread over several files and the
    grouped row has to pool them. Averaging the shards' own percentages would weight a
    shard that is 10% done the same as one that is 90% done.
    """
    fn = scorer(bench)
    if isinstance(paths, str):
        paths = [paths]
    paths = [q for q in (paths or []) if q and os.path.exists(q)]
    if fn is None or not paths:
        return None
    right = n = 0
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:                             # noqa: BLE001
                        continue   # torn final line: the file is being appended to now
                    try:
                        # float, not bool: RULER's metrics give PARTIAL credit (a row
                        # with 5 expected answers and 2 found scores 0.4). bool() would
                        # round every partial row up to 1 and inflate the column, which
                        # is the one thing it must never do. Lossless for the
                        # bool-returning scorers, where float(True) is exactly 1.0.
                        right += float(fn(row))
                        n += 1
                    except Exception:                             # noqa: BLE001
                        continue
        except OSError:
            continue
    return 100.0 * right / n if n else None


def squeue():
    """Running/pending jobs as {jobid: (state, elapsed, name)}. Empty if slurm is out."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""),
             "-o", "%i|%T|%M|%j"], capture_output=True, text=True, timeout=30).stdout
    except Exception:                                                # noqa: BLE001
        return {}
    jobs = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            jobs[parts[0].strip()] = (parts[1].strip(), parts[2].strip(),
                                      parts[3].strip())
    return jobs


def log_dirs():
    """{jobid: logdir} for every job that has written a log here."""
    out = {}
    for path in glob.glob(os.path.join(LOGS, "*", "*.out")):
        out[os.path.basename(path)[:-4]] = os.path.dirname(path)
    return out


def tail(path, nbytes=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def read_job(jobid, d):
    """Everything knowable about one job from its log directory."""
    out = tail(os.path.join(d, f"{jobid}.out"))
    err = tail(os.path.join(d, f"{jobid}.err"))
    info = {"model": "?", "bench": "?", "tag": "?", "phase": "queued",
            "done": None, "total": None, "eta": "", "rate": "", "path": None}

    m = HEADER.search(out)
    if m:
        info["model"], info["bench"], info["tag"] = m.groups()
        info["phase"] = "loading"
    elif "quantize" in d:
        info["phase"] = "quantizing"
        info["bench"] = "quantize"
        parts = os.path.basename(d).split("_quantize_")
        if len(parts) > 1:
            info["model"] = parts[1].rsplit("_", 1)[0]
            info["tag"] = parts[1].rsplit("_", 1)[-1]

    if "stack up on" in out:
        info["phase"] = "generating"

    # RULER first: its sweep header is absolute, so it must not be overwritten by the
    # per-task run_resumable headers that follow it (see RULER_SWEEP).
    rsweep = RULER_SWEEP.findall(out)
    if rsweep:
        total, path = rsweep[-1]
        info["total"], info["path"] = int(total), path
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    info["done"] = sum(1 for _ in f)
            except OSError:
                pass
    # "ruler prepare:" is emitted by run_eval.sh BEFORE any generation, and at 32k/64k
    # that phase runs for hours with nothing else to show. It must be checked OUTSIDE
    # the rsweep branch above: the sweep header comes from ruler_infer.py, which does
    # not start until preparation has finished, so inside that branch this could never
    # fire during the only phase it describes. Reporting "loading" there would be wrong
    # -- it is neither queued nor generating, and a user watching a job sit at "loading"
    # for three hours would reasonably assume a stall.
    if info["done"] is None and "ruler prepare:" in out:
        info["phase"] = "preparing"
        m = re.findall(r"ruler prepare: (\S+) @ (\d+)", out)
        if m:
            info["eta"] = f"{m[-1][0]}@{m[-1][1]}"   # the task/length in flight now

    # Last resumable header wins: a GPQA job walks rep0..rep3 and prints one per repeat.
    rs = RESUMABLE.findall(out)
    if rs and not rsweep:
        total, _done, _todo, path = rs[-1]
        info["total"] = int(total)
        info["path"] = path
        info["repeat"] = f"{len(rs)}/{len(rs)}" if len(rs) > 1 else ""
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    info["done"] = sum(1 for _ in f)
            except OSError:
                pass

    t = TQDM.findall(err)
    if t:
        _pct, done, total, _elapsed, remaining, rate, unit = t[-1]
        # tqdm's total is THIS ATTEMPT'S REMAINDER, not the benchmark. After a resume it
        # reads 1139 where the benchmark is 1730, and pairing it with the file's
        # absolute line count reported 898/1139 for a run that was really 898/1730 --
        # understating a resumed job and overstating how close it is to done. The
        # resumable header's total is absolute, so it wins whenever it exists; tqdm is
        # consulted only for the rate and the ETA, which nothing else knows.
        if info["total"] is None:
            info["total"] = int(total)
        if info["done"] is None:
            info["done"] = int(done)
        info["eta"] = remaining
        info["rate"] = f"{rate}{unit}"

    if info["done"] is not None and info["total"]:
        info["phase"] = "generating"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="include jobs no longer in the queue")
    ap.add_argument("--no-acc", action="store_true",
                    help="skip preliminary scoring (it reads every output file)")
    args = ap.parse_args()

    live = squeue()
    dirs = log_dirs()
    # Only jobs this harness launched. An interactive `srun` shows up in squeue as
    # `python3` or `bash` with no log directory, and listing it as `?/?/python3` is
    # noise in the one view meant to answer "what is running".
    # "muse-" stays in the list only so jobs submitted before the prefix was renamed
    # keep showing while they drain; drop it once the queue has turned over. The name
    # check exists for PENDING jobs, which have no log directory yet -- once a job runs
    # it is matched by `j in dirs` regardless of what it is called.
    live = {j: v for j, v in live.items()
            if j in dirs or v[2].startswith(("eval-", "quantize-", "muse-"))}
    ids = sorted(set(live) | (set(dirs) if args.all else set()),
                 key=lambda x: (x not in live, x))
    if not ids:
        print("nothing running, and no logs under logs/evals/")
        return

    rows = []
    for jid in ids:
        d = dirs.get(jid)
        info = read_job(jid, d) if d else {
            "model": "?", "bench": "?", "tag": (live.get(jid) or ("", "", "?"))[2],
            "phase": "queued", "done": None, "total": None, "eta": "", "rate": ""}
        state, elapsed = (live.get(jid) or ("DONE", "-", ""))[:2]
        rows.append((jid, state, elapsed, info))

    w = max(len(f"{r[3]['model']}/{r[3]['bench']}/{r[3]['tag']}") for r in rows)
    acc_col = "" if args.no_acc else f" {'acc*':>6}"
    print(f"{'job':>9} {'state':<9} {'time':>7}  {'setup':<{w}}  "
          f"{'progress':>13}  {'rate':>9} {'eta':>8}{acc_col}")
    for jid, state, elapsed, i in rows:
        setup = f"{i['model']}/{i['bench']}/{i['tag']}"
        prog = (f"{i['done']}/{i['total']}"
                if i["done"] is not None and i["total"] else i["phase"])
        if args.no_acc:
            accs = ""
        else:
            acc = preliminary_accuracy(i["path"], i["bench"]) if i.get("path") else None
            accs = f" {acc:6.2f}" if acc is not None else f" {'-':>6}"
        print(f"{jid:>9} {state:<9} {elapsed:>7}  {setup:<{w}}  "
              f"{prog:>13}  {i['rate']:>9} {i['eta']:>8}{accs}")

    n_live = sum(1 for r in rows if r[1] not in ("DONE",))
    stuck = [r for r in rows if r[1] == "RUNNING" and r[3]["phase"] == "loading"]
    print(f"\n{n_live} in the queue"
          + (f"; {len(stuck)} running but not yet generating (loading weights)"
             if stuck else ""))
    if not args.no_acc:
        print("* accuracy over the rows finished SO FAR, which are the fast items -- "
              "it drifts as the slow tail lands")


if __name__ == "__main__":
    main()
