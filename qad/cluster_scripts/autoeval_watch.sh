#!/bin/bash
# Submit evals as checkpoints appear, until training is done and nothing is left.
# Built for a tmux window: start it, detach, come back to a finished sweep.
#
#   tmux new -s autoeval
#   cd .../qad && ./cluster_scripts/autoeval_watch.sh
#   <ctrl-b d>            # detach; reattach later with: tmux attach -t autoeval
#
#   INTERVAL=600 ./cluster_scripts/autoeval_watch.sh        # poll every 10 min
#   ./cluster_scripts/autoeval_watch.sh --formats nvr2bit   # narrow to one format
#
# WHY A LOOP RATHER THAN "WAIT FOR STEP 2250"
# -------------------------------------------
# Waiting for the final checkpoint serialises everything: mmlu_pro takes ~2h per step, so
# firing all of it at the end adds hours of wall-clock that could have overlapped with
# training. This submits each grid step as it lands, so evaluation finishes shortly after
# training instead of starting then.
#
# SAFE TO RESTART, AND SAFE TO RUN TWICE. submit_missing_evals.py keeps a ledger of what
# it submitted and subtracts any job still in the queue, so a step that is already running
# is never re-queued -- which is the failure this exists to avoid, not just a nicety.
#
# It exits ONLY when both are true: no training jobs remain, AND a gap scan comes back
# empty. Either alone is a trap -- gaps are empty early on simply because nothing has
# exported yet, and training ending does not mean the last steps were submitted.
set -uo pipefail
cd "$(dirname "$(realpath "$0")")/.."
INTERVAL=${INTERVAL:-120}
GRACE=${GRACE:-2}            # consecutive empty scans required after training ends
say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }

say "watching; polling every ${INTERVAL}s. detach with ctrl-b d"
empty=0
while :; do
    train=$(squeue -u "$(whoami)" -h -o "%j" 2>/dev/null | grep -c '^qad-Qwen3' || true)
    out=$(python3 cluster_scripts/submit_missing_evals.py --apply "$@" 2>&1)
    sent=$(printf '%s' "$out" | grep -c '^  SENT' || true)
    covering=$(printf '%s' "$out" | grep -oE 'covering [0-9]+' | head -1 | awk '{print $2}')
    [ -n "$sent" ] && [ "$sent" -gt 0 ] && printf '%s' "$out" | grep '^  SENT'
    evals=$(squeue -u "$(whoami)" -h -r -o "%j" 2>/dev/null | grep -vc '^qad-Qwen3' || true)
    say "training=$train  eval_jobs=$evals  submitted=${sent:-0} (${covering:-0} points)"

    if [ "$train" -eq 0 ] && [ "${sent:-0}" -eq 0 ]; then
        empty=$((empty + 1))
        if [ "$empty" -ge "$GRACE" ]; then
            say "training finished and $GRACE consecutive scans found no gaps."
            say "$evals eval job(s) still draining; nothing left to submit. done."
            exit 0
        fi
    else
        empty=0
    fi
    sleep "$INTERVAL"
done
