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
# cd+pwd (bash builtins, no -P) rather than realpath: realpath calls getcwd(), which
# resolves the /lustre->/scratch symlink -- see run_qad.sh's SELF_DIR for the full
# story. Poisons everything downstream too: submit_missing_evals.py's ROOT is
# os.path.abspath(__file__) relative to THIS process's cwd, so a /scratch cwd here
# makes it build a /scratch-rooted run_eval_disagg_sweep.sh path, which propagates
# into run_eval_disagg.sh's own self-location and finally into the sbatch script the
# container tries to bash -- invisible there, since only /lustre is mounted.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SELF_DIR/.."
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
# `ruler` IS on this list: run_eval_ruler.sh submits --job-name=qad-ruler, and it was
# added after this regex was written. While it was missing, every running RULER job was
# counted as a TRAINING job by the `grep -vE` below -- so `train` never reached 0 and the
# watcher would never exit, while also under-counting eval_jobs. Exactly the drift this
# comment warns about.
EVAL_JOBS='^qad-(eval|dual-eval|disagg|vllm|ruler)$'

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

# RULER-ONLY, by default. `--formats` with NO values hands argparse an empty list, so
# submit_missing_evals.py's DISAGG loop iterates nothing -- a HARD guarantee, rather than
# relying on "no other format has checkpoints on disk". That was only accidentally true:
# the disagg default format list is every LABELS entry minus the excluded ones, so the
# moment any of those finishes training the watcher would start submitting
# gsm8k/math500/mmlu_pro jobs nobody asked for. RULER selection is unaffected -- it has
# its own --ruler-formats list. Set RULER_ONLY=0 to restore the disagg sweep.
# Placed BEFORE "$@" so an explicit caller --formats still wins (argparse: last wins).
RULER_ONLY=${RULER_ONLY:-1}
ONLY_ARGS=()
if [ "$RULER_ONLY" = 1 ]; then
    ONLY_ARGS=(--formats)
    say "RULER-ONLY mode (set RULER_ONLY=0 to also sweep disagg)"
fi

say "watching; polling every ${INTERVAL}s. detach with ctrl-b d"
empty=0
while :; do
    # Checkpoints land on the nextgen quota but are DISCOVERED through qad/checkpoints,
    # and run_qad.sh does not create that back-link (the tag ends in a hash computed in
    # qad.py, not reconstructible there). Without this, a newly launched run is invisible
    # to the gap scan: it trains and exports while the watcher cheerfully reports nothing
    # to do. Eight nvfp4a16 runs reached step 1500 that way. Re-run every poll, not once
    # at startup, so runs launched mid-watch are picked up too.
    ./cluster_scripts/link_checkpoints.sh 2>/dev/null | grep '^linked ' | sed 's/^/  /'

    train=$(squeue -u "$(whoami)" -h -o "%j" 2>/dev/null \
            | grep '^qad-' | grep -cvE "$EVAL_JOBS" || true)
    sent=0; covering=0
    for fam in ${FAMILIES:-__caller__}; do
        if [ "$fam" = "__caller__" ]; then
            out=$(python3 cluster_scripts/submit_missing_evals.py --apply ${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"} "$@" 2>&1)
        else
            out=$(python3 cluster_scripts/submit_missing_evals.py --apply --family "$fam" ${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"} "$@" 2>&1)
        fi
        s=$(printf '%s' "$out" | grep -c '^  SENT' || true)
        c=$(printf '%s' "$out" | grep -oE 'covering [0-9]+' | head -1 | awk '{print $2}')
        [ "${s:-0}" -gt 0 ] && printf '%s' "$out" | grep '^  SENT' | sed "s/^/  [$fam]/"
        # FAIL/REFUSING (and anything else, e.g. a traceback) were previously swallowed
        # entirely -- a watcher stuck on the bootstrap guard, or crashing every poll,
        # printed "submitted=0" forever with no way to tell why from this log.
        printf '%s' "$out" | grep -E '^  FAIL|^  REFUSING' | sed "s/^/  [$fam]/"
        if [ "${s:-0}" -eq 0 ] && ! printf '%s' "$out" | grep -qE '^  (GAP|would submit|submitted) '; then
            printf '%s' "$out" | sed "s/^/  [$fam][unexpected] /"
        fi
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
