#!/bin/bash
# Stop TRAINING jobs that have produced every checkpoint anything downstream reads.
#
#   ./cluster_scripts/reap_past_grid.sh --dry-run     # report only
#   ./cluster_scripts/reap_past_grid.sh               # act
#   ./cluster_scripts/reap_past_grid.sh --step 1251   # different cutoff
#   ./cluster_scripts/reap_past_grid.sh --name 270m   # only jobs whose name contains this
#
# THE RULE: last logged step >= --step (default 2251) -> kill. No silence requirement.
#
# WHY NO STALL CHECK, unlike reap_stalled.sh
# ------------------------------------------
# That script targets jobs that are HUNG: the work is done and the process will not exit,
# so silence is the evidence. This targets jobs that are still perfectly healthy and simply
# have nothing left to produce that anyone consumes. Waiting for them to go quiet would
# hold 8 GPUs each for the remainder of a run whose output is discarded.
#
# WHY 2251 AND NOT 2250
# ---------------------
# The figures read step 0 and every multiple of 250 up to 2250 (the GRID in
# submit_missing_evals.py); training continues to ~2532 and the tail is never drawn. Steps
# are logged only AFTER the step completes, and checkpoints are exported ON the multiple,
# so "the log has reached 2251" is proof that the step-2250 export already happened. That
# makes the step number self-validating -- no directory scan required, and no window where
# a job is killed one step before writing the checkpoint that mattered.
#
# WHAT IS LOST: steps past 2250, including the final one. Nothing plots them, and the
# notebook's tail average runs over 1250..2250. If you want a "final model" artifact for a
# format, it will be step 2250 rather than 2532.
#
# TRAINING JOBS ONLY. Eval jobs are excluded by name -- they have no step counter and are
# reap_stalled.sh's business.

set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$SELF_DIR")")"   # cluster_scripts/ -> qad -> repo (logs/ here)

USER_NAME=${USER_NAME:-$(whoami)}
STEP=${STEP:-2251}
NAME_FILTER=${NAME_FILTER:-}
DRY=0

while (($# > 0)); do
  case "$1" in
    --step)    STEP="$2";        shift 2 ;;
    --name)    NAME_FILTER="$2"; shift 2 ;;
    --dry-run) DRY=1;            shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

reaped=0; checked=0
while read -r id name elapsed; do
  [ -z "$id" ] && continue
  # Eval job names, from the #SBATCH --job-name lines in bin/run_eval*.sh.
  case "$name" in qad-disagg|qad-vllm|qad-eval|qad-dual-eval) continue ;; esac
  [ -n "$NAME_FILTER" ] && [[ "$name" != *"$NAME_FILTER"* ]] && continue

  # Training logs are logs/train/<stamp>_<tag>/<jobname>_<jobid>.out -- the id is a SUFFIX,
  # which is why reap_stalled.sh's <id>.out patterns never matched a training job.
  f=$(ls -t "$ROOT"/logs/train/*/*_"${id}".out 2>/dev/null | head -1)
  [ -z "$f" ] && continue
  checked=$((checked + 1))

  # Anchored to line start so "step" inside any other message cannot be read as progress.
  last_step=$(grep -oE "^step +[0-9]+" "$f" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -z "$last_step" ] && continue

  if [ "$last_step" -ge "$STEP" ]; then
    echo "REAP $id ($name) elapsed=$elapsed step=$last_step -- past $STEP, every plotted checkpoint written"
    [ "$DRY" = 0 ] && scancel "$id" 2>/dev/null
    reaped=$((reaped + 1))
  fi
done < <(squeue -u "$USER_NAME" -h -t RUNNING -o "%i %j %M" 2>/dev/null)

echo "reap_past_grid: checked=$checked reaped=$reaped step=$STEP$([ "$DRY" = 1 ] && echo ' (DRY RUN)')"
