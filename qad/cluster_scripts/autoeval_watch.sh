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
#   ./cluster_scripts/autoeval_watch.sh --family gemma3     # Gemma-3 instead of Qwen3
#
# Every argument is forwarded verbatim to submit_missing_evals.py, so --family/--models/
# --formats all work here. NOTE the training/eval job counts below are family-agnostic but
# the SUBMISSION side is not: one watcher covers one --family. Run a second watcher (in its
# own tmux window) to cover the other family concurrently -- they share the ledger, which
# is keyed per (model, quant, mode, task-group), so they will not collide.
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

# Training vs eval jobs are told apart by EXCLUDING the four eval job names rather than by
# matching a model prefix. The old test was `grep '^qad-Qwen3'`, which is wrong the moment a
# second family exists: run_qad.sh names training jobs `qad-<model>-<quant>` from
# `cut -d/ -f2`, so Gemma runs are `qad-gemma-3-4b-it-<quant>` and would count as ZERO
# training jobs -- the loop would then see "training finished" and exit in the middle of a
# live sweep, while simultaneously counting those same jobs as eval jobs in the `-vc`.
# Keep this list in sync with the #SBATCH --job-name lines in bin/run_eval*.sh.
EVAL_JOBS='^qad-(eval|dual-eval|disagg|vllm)$'

# EVERY family is scanned unless the caller names one. The job counts above are
# family-agnostic while submit_missing_evals.py takes a single --family defaulting to
# qad3x, so a plain invocation used to see Gemma training jobs (and therefore never exit)
# while scanning Qwen only (and therefore never submit) -- a watcher that looks perfectly
# healthy and does nothing. Sweeping all families by default removes the footgun; pass
# --family <name> to narrow deliberately.
case " $* " in
    *" --family "*|*" --family="*) FAMILIES="" ;;   # caller chose; forward args verbatim
    *) FAMILIES=$(python3 cluster_scripts/submit_missing_evals.py --list-families) ;;
esac
[ -n "$FAMILIES" ] && say "families: $(echo "$FAMILIES" | tr '\n' ' ')"

say "watching; polling every ${INTERVAL}s. detach with ctrl-b d"
empty=0
while :; do
    train=$(squeue -u "$(whoami)" -h -o "%j" 2>/dev/null \
            | grep '^qad-' | grep -cvE "$EVAL_JOBS" || true)
    sent=0; covering=0
    for fam in ${FAMILIES:-__caller__}; do
        if [ "$fam" = "__caller__" ]; then
            out=$(python3 cluster_scripts/submit_missing_evals.py --apply "$@" 2>&1)
        else
            out=$(python3 cluster_scripts/submit_missing_evals.py --apply --family "$fam" "$@" 2>&1)
        fi
        s=$(printf '%s' "$out" | grep -c '^  SENT' || true)
        c=$(printf '%s' "$out" | grep -oE 'covering [0-9]+' | head -1 | awk '{print $2}')
        [ "${s:-0}" -gt 0 ] && printf '%s' "$out" | grep '^  SENT' | sed "s/^/  [$fam]/"
        sent=$((sent + ${s:-0}))
        covering=$((covering + ${c:-0}))
    done
    evals=$(squeue -u "$(whoami)" -h -r -o "%j" 2>/dev/null | grep -cE "$EVAL_JOBS" || true)
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
