#!/bin/bash
# Kill jobs that have finished their work but are hung holding GPUs.
#
#   ./cluster_scripts/reap_stalled.sh                 # act
#   ./cluster_scripts/reap_stalled.sh --dry-run       # report only
#   ./cluster_scripts/reap_stalled.sh --stall-min 20  # require 20 min of silence (default 10)
#   ./cluster_scripts/reap_stalled.sh --idle-min 45   # ALSO reap silent-but-unfinished jobs (off by default)
#
# WHY NOT JUST "no log output for N minutes"
# -----------------------------------------
# Because that reaps healthy jobs. lm-eval prints NOTHING while generating: a job can
# sit silent for 30+ minutes and be perfectly busy. Observed here repeatedly -- four
# jobs at 14 minutes of silence were all mid-generation.
#
# The failure this targets is specific and has a precise signature: vLLM engines
# regularly hang at teardown ("destroy_process_group() was not called before program
# exit"), AFTER the run has printed its results. Those jobs are pure waste -- the work
# is already on disk and they keep two GPUs until the wall clock kills them. Six were
# reaped by hand in one session, holding 10 GPUs between them.
#
# So the default rule requires BOTH:
#   1. the driver log contains a completion marker (results written / final score), and
#   2. nothing has been written to it for --stall-min minutes.
# Condition 1 is what makes this safe: a job that has not finished is never touched.
#
# --idle-min adds an opt-in second rule for jobs that are silent and have NOT finished.
# That one can kill healthy work, so it is off by default and should be set well above
# the longest plausible generation phase.

set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$SELF_DIR")")"   # cluster_scripts/ -> qad -> repo (logs/ lives here)

USER_NAME=${USER_NAME:-$(whoami)}
STALL_MIN=${STALL_MIN:-10}
IDLE_MIN=${IDLE_MIN:-0}        # 0 = disabled
NAME_FILTER=${NAME_FILTER:-}
DRY=0

while (($# > 0)); do
  case "$1" in
    --stall-min) STALL_MIN="$2"; shift 2 ;;
    --idle-min)  IDLE_MIN="$2";  shift 2 ;;
    --name)      NAME_FILTER="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# NOTE this script only ever reaps EVAL jobs by its default rule: the completion marker it
# keys on ("results -> <path>") is printed by eval_disagg.py alone, so a training job never
# satisfies condition 1 no matter how long it hangs. Reaping finished TRAINING runs is a
# different question with a different answer -- see reap_past_grid.sh, which stops them on
# step count rather than on silence, because a training run that has produced every plotted
# checkpoint is worth stopping whether or not it has gone quiet.

# THE RULE: kill only a job whose RESULTS FILE IS ON DISK.
#
# Log text is not evidence. Two earlier versions of this script were wrong:
#   * "no output for N minutes" reaped healthy jobs -- lm-eval prints nothing while
#     generating, so a busy job looks identical to a dead one.
#   * "log contains a result line" reaped a live job mid-way through its second stage,
#     because the first stage had already printed a score.
# The artifact itself settles it: if the results JSON exists, the work is done and a
# still-RUNNING job is only holding GPUs (vLLM routinely hangs in teardown after
# writing). If no results file exists, the job is never touched, whatever its log says.
#
# The driver prints "results -> <path>"; every such path must exist for the job to
# count as finished. A job that prints no such line (ad-hoc scripts, multi-stage
# benchmarks) is therefore never reaped by the default rule.

now=$(date +%s)
now=$(date +%s)
reaped=0; checked=0

while read -r id name elapsed; do
  [ -z "$id" ] && continue
  [ -n "$NAME_FILTER" ] && [[ "$name" != *"$NAME_FILTER"* ]] && continue
  checked=$((checked + 1))

  # THE JOB'S OWN LOG, OR NOTHING.
  #
  # This used to glob "${base}_*.out" -- every element of the array -- and then take
  # the newest with `ls -t | head -1`. So element 6 read element 1's log, found the
  # "results -> step_0000500.json" element 1 had legitimately written, confirmed that
  # file existed, and killed element 6 mid-generation. It reaped 8 live jobs at once
  # this way (479275_2..9), each blamed on the same step_0000500.json, destroying
  # ~40 min x 8 jobs x 2 GPUs of work.
  #
  # One element's log can never speak for another. Match the exact id only; a job with
  # no log of its own is left alone.
  f=$(ls -t "$ROOT"/logs/*/*/"${id}".out 2>/dev/null | head -1)
  if [ -z "$f" ]; then
    # non-array jobs are written as <jobid>_4294967294.out by the %A_%a pattern
    f=$(ls -t "$ROOT"/logs/*/*/"${id}"_4294967294.out 2>/dev/null | head -1)
  fi
  if [ -z "$f" ]; then
    # training jobs land in logs/train/<stamp>_<tag>/<name>_<jobid>.out -- the job id is a
    # SUFFIX there, not the whole basename, so the two patterns above never match one.
    f=$(ls -t "$ROOT"/logs/train/*/*_"${id}".out 2>/dev/null | head -1)
  fi
  [ -z "$f" ] && continue

  age_min=$(( (now - $(stat -c %Y "$f")) / 60 ))

  # every "results -> <path>" the driver announced must actually exist on disk
  mapfile -t paths < <(grep -ahoE "results -> [^ ]+" "$f" 2>/dev/null | sed 's/^results -> //')
  have_results=0
  if [ "${#paths[@]}" -gt 0 ]; then
    have_results=1
    for pth in "${paths[@]}"; do
      [ -f "$pth" ] || have_results=0
    done
  fi

  if [ "$have_results" = 1 ]; then
    if [ "$age_min" -ge "$STALL_MIN" ]; then
      echo "REAP $id ($name) elapsed=$elapsed idle=${age_min}m -- results on disk (${paths[-1]##*/}), hung holding GPUs"
      [ "$DRY" = 0 ] && scancel "$id" 2>/dev/null
      reaped=$((reaped + 1))
    fi
  elif [ "$IDLE_MIN" -gt 0 ] && [ "$age_min" -ge "$IDLE_MIN" ]; then
    echo "REAP $id ($name) elapsed=$elapsed idle=${age_min}m -- unfinished but silent past --idle-min"
    [ "$DRY" = 0 ] && scancel "$id" 2>/dev/null
    reaped=$((reaped + 1))
  fi
done < <(squeue -u "$USER_NAME" -h -t RUNNING -o "%i %j %M" 2>/dev/null)

echo "reap: checked=$checked reaped=$reaped stall_min=$STALL_MIN idle_min=$IDLE_MIN$([ "$DRY" = 1 ] && echo ' (DRY RUN)')"
