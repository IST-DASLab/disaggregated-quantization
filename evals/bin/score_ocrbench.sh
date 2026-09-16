#!/bin/bash
# Score OCRBench v2 generations on the CPU partition, one job per result directory.
#
#   ./bin/score_ocrbench.sh                 # every arm with generations but no summary
#   ./bin/score_ocrbench.sh --force         # re-score even where summary.json exists
#   ./bin/score_ocrbench.sh path/to/dir ... # named directories
#
# WHY THIS IS NOT DONE INSIDE THE EVAL JOB, as it is for every other benchmark.
# OCRBench scoring is 15-20 minutes of single-threaded CPU: uncapped O(n^2) edit
# distance over full-page transcriptions, plus TEDS tree edit distance on tables. Run
# inline it holds a B200 idle for all of it -- 4-5 GPU-hours per model across 4 arms x
# 4 repeats, for work no GPU touches. MMMU and IFBench score in seconds and stay inline.
#
# Scoring is idempotent and reads only raw.jsonl, so deferring it is free. Each job gets
# a private copy of the harness (see ocrbench_score.py), so running these in parallel is
# safe -- which it would NOT be otherwise, since spotting_metric wipes hardcoded
# relative paths on every call.

#SBATCH --job-name=ocrbench-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
PYDEPS=${PYDEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-pydeps}
HARNESSES=${HARNESSES:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses}
ACCOUNT=${ACCOUNT:-coreai_psx_qad}
FORCE=0

DIRS=()
while (($# > 0)); do
    case "$1" in
        --force) FORCE=1; shift ;;
        -*) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
        *) DIRS+=("$1"); shift ;;
    esac
done

if [ -z "${SLURM_JOB_ID:-}" ]; then
    if ((${#DIRS[@]} == 0)); then
        # Every ocrbench result directory that has generations. Skipping the ones
        # already scored is what makes this safe to run repeatedly while a sweep lands.
        while IFS= read -r raw; do
            d="$(dirname "$raw")"
            [ "$FORCE" = 0 ] && [ -f "$d/summary.json" ] && continue
            # Only arms whose generation finished. Scoring a partial file wastes 15-20
            # minutes AND writes a summary that makes the directory look done, so the
            # real scoring is skipped later -- a silent partial result, which is the
            # worst outcome available.
            if [ ! -f "$d/generation_complete" ]; then
                echo "  skip (still generating): $d"
                continue
            fi
            DIRS+=("$d")
        done < <(find "$MUSE_ROOT/results" -path "*/ocrbench/*" -name raw.jsonl 2>/dev/null)
    fi
    if ((${#DIRS[@]} == 0)); then
        echo "nothing to score (use --force to re-score)"; exit 0
    fi
    LOGS="$REPO_ROOT/logs/evals/$(date +%Y%m%d_%H%M%S)_ocrbench_score"
    mkdir -p "$LOGS"
    echo "${#DIRS[@]} directory(ies) to score -> $LOGS"
    for d in "${DIRS[@]}"; do
        echo "  $d"
        SCORE_DIR="$d" MUSE_ROOT="$MUSE_ROOT" REPO_ROOT="$REPO_ROOT" \
        CONTAINER="$CONTAINER" PYDEPS="$PYDEPS" HARNESSES="$HARNESSES" \
        sbatch --export=ALL --account="$ACCOUNT" \
            --partition=cpu --qos=cpu-normal --time=02:00:00 \
            --job-name="ocrb-score-$(basename "$(dirname "$d")")-$(basename "$d")" \
            --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
            "$_SELF_DIR/$(basename "${BASH_SOURCE[0]}")" >/dev/null
    done
    exit 0
fi

srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
    --container-mounts=/scratch:/scratch,/lustre:/lustre --export=ALL bash -lc "
export PYTHONPATH='$PYDEPS'
export OCRBENCH_HARNESS='$HARNESSES/MultimodalOCR/OCRBench_v2'
python3 '$MUSE_ROOT/drivers/ocrbench_score.py' '$SCORE_DIR'
"
