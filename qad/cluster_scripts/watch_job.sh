#!/bin/bash
# Stream a SLURM job's milestones and failures, one line per event, exiting when the
# job reaches a terminal state. Written for the Monitor tool: each stdout line becomes
# a notification, and the process ending ends the watch.
#
#   ./cluster_scripts/watch_job.sh 488298                    # milestones + failures
#   ./cluster_scripts/watch_job.sh 488298 'val_ntp|my-thing'  # extra patterns
#   ./cluster_scripts/watch_job.sh 479272_3                   # array elements too
#   LOGDIR=logs/train/2026..._x ./cluster_scripts/watch_job.sh 488298   # explicit logs
#
# WHY
# ---
# `squeue` says a job is RUNNING, which is not the same as WORKING. These jobs go quiet
# for long stretches -- lm-eval prints nothing while generating, training logs to wandb
# rather than stdout -- so the useful signal is the few lines that mark real progress,
# plus every way the job can end.
#
# SILENCE IS NOT SUCCESS. The failure patterns matter as much as the milestones: a
# watcher that greps only the happy path stays quiet through a crash, and quiet looks
# exactly like "still running". The terminal state is ALWAYS emitted, so a watch ends
# with a verdict instead of trailing off.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

JOB="${1:?usage: watch_job.sh <jobid> [extra-regex]}"
EXTRA="${2:-}"
INTERVAL="${INTERVAL:-30}"
LOGDIR="${LOGDIR:-}"

# Progress that means something HAPPENED, not merely that the process is alive.
MILESTONES='checkpoint →|checkpoint ->|Final *\||Teacher val_ntp|^Steps:|^Model:|results -> |JOB_COMPLETE|stack ready|Graph capturing finished'
# Every way the work can die. Deliberately broad -- a missed failure reads as silence.
FAILURES='Traceback|Error|error:|ERROR|FAILED|assert|Killed|OOM|out of memory|CUDA error|NCCL.*(error|timeout)|srun: error|Segmentation fault|No space left'

PAT="($MILESTONES|$FAILURES)"
[ -n "$EXTRA" ] && PAT="($MILESTONES|$FAILURES|$EXTRA)"

# Both streams: python tracebacks land on stderr, and reading only stdout is exactly
# the blind spot this exists to close.
find_logs() {
    if [ -n "$LOGDIR" ]; then
        ls "$LOGDIR"/*.out "$LOGDIR"/*.err 2>/dev/null
        return
    fi
    local out
    out=$(scontrol show job "$JOB" 2>/dev/null \
          | grep -oE 'StdOut=[^[:space:]]+' | cut -d= -f2- | head -1)
    # a pattern like %A_%a means SLURM has not expanded it yet -- fall back to a search
    if [ -n "$out" ] && [ "${out//%/}" = "$out" ]; then
        echo "$out"
        [ -f "${out%.out}.err" ] && echo "${out%.out}.err"
        return
    fi
    local base=${JOB%%_*}
    ls -t "$ROOT"/logs/*/*/"${JOB}".out "$ROOT"/logs/*/*/"${JOB}".err \
          "$ROOT"/logs/*/*/*"${base}"*.out "$ROOT"/logs/*/*/*"${base}"*.err 2>/dev/null | head -4
}

declare -A off
last_state=""
echo "watch $JOB | armed (milestones + failures, every ${INTERVAL}s)"

while true; do
    st=$(sacct -j "$JOB" -n --format=State -P 2>/dev/null | head -1 | cut -d' ' -f1)
    if [ -n "$st" ] && [ "$st" != "$last_state" ]; then
        echo "watch $JOB | state -> $st"
        last_state="$st"
    fi

    while read -r f; do
        [ -f "$f" ] || continue
        cur=$(wc -c < "$f" 2>/dev/null || echo 0)
        prev=${off[$f]:-0}
        if [ "$cur" -gt "$prev" ]; then
            # only bytes added since the last pass, so each line is reported once
            while IFS= read -r line; do
                echo "watch $JOB | ${line:0:200}"
            done < <(tail -c +$((prev + 1)) "$f" 2>/dev/null | tr '\r' '\n' \
                     | grep -aE "$PAT" | grep -avE 'FutureWarning|pynvml' | tail -12)
            off[$f]=$cur
        fi
    done < <(find_logs)

    case "$st" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED)
            echo "watch $JOB | DONE $st (elapsed $(sacct -j "$JOB" -n --format=Elapsed -P 2>/dev/null | head -1))"
            exit 0 ;;
    esac
    sleep "$INTERVAL"
done
