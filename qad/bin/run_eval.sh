#!/bin/bash
# Evaluate a QAD checkpoint via lm-eval.
#
# This script is dual-mode:
#   HOST mode  (scontrol exists): uses srun to re-invoke itself inside the container.
#   CONTAINER  (no scontrol):     runs Python eval directly.
# This avoids bash -c quoting issues and pyxis re-execution problems with array jobs.
#
# Submit with a job array:
#   sbatch --array=0,25,50,75,100,125,150,175,200 run_eval.sh --quantizer ste3bit
#   sbatch run_eval.sh --quantizer ste3bit --iter 50          # single step
#   sbatch run_eval.sh --unquantized                          # BF16 baseline

#SBATCH --job-name=qad-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Constants (set before any mode-split so both modes see them)
# ---------------------------------------------------------------------------
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/nemo-26.02.sqsh
HF_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache
LM_EVAL_OVERLAY=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/lm_eval_overlay
LOG_KIND=eval_transformers   # log subfolder: logs/<LOG_KIND>/<timestamp>_<tag>/

# ---------------------------------------------------------------------------
# STAGE 0: pre-submit (login node). Create nested per-submission log dir and
# re-submit into it (SLURM can't create --output dirs; mkdir-in-job is too late).
# Invoke directly, e.g.:
#   ./bin/run_eval.sh --array=0,25,... --quantizer ste3bit --tasks "gsm8k ..."
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    SELF="$(realpath "$0")"
    ROOT="$(dirname "$SELF")/../.."   # script lives in qad/bin/
    STAMP="$(date +%Y%m%d_%H%M%S)"
    SB_ARRAY=""; TAG="run"; PASS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --array=*)     SB_ARRAY="$1";              shift ;;
            --array)       SB_ARRAY="--array=$2";      shift 2 ;;
            --quantizer=*) TAG="${1#--quantizer=}"; PASS+=("$1");      shift ;;
            --quantizer)   TAG="$2";                PASS+=("$1" "$2"); shift 2 ;;
            --unquantized) TAG="baseline";          PASS+=("$1");      shift ;;
            *)             PASS+=("$1");                                 shift ;;
        esac
    done
    LOGDIR="$ROOT/logs/${LOG_KIND}/${STAMP}_${TAG}"
    mkdir -p "$LOGDIR"
    echo "logs → $LOGDIR"
    exec sbatch $SB_ARRAY \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SELF" "${PASS[@]}"
fi

# ---------------------------------------------------------------------------
# HOST mode: resolve paths and re-invoke via srun inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null; then
    # `exit` after the first match is REQUIRED for job arrays: the last array element
    # has JobId == the array base id, for which `scontrol show job` prints every array
    # record, yielding a multi-line SCRIPT_PATH and a bash "No such file" (exit 127).
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")     # qad/bin
    QAD_DIR=$(dirname "$SCRIPT_DIR")         # qad -- anchors checkpoints/ and imports
    mkdir -p "$QAD_DIR/logs"

    # Export everything the container mode needs
    export SCRIPT_DIR QAD_DIR HF_CACHE LM_EVAL_OVERLAY
    export MODEL=${MODEL:-Qwen/Qwen3-4B}
    export CKPT_DIR=${CKPT_DIR:-$QAD_DIR/checkpoints}

    srun \
        --ntasks=1 \
        --container-image="$CONTAINER" \
        --no-container-mount-home \
        --container-mounts="/scratch:/scratch,/lustre:/lustre,$HOME/.netrc:/root/.netrc" \
        --export=ALL \
        bash "$SCRIPT_PATH" "$@"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode: scontrol not found — run the eval directly
# ---------------------------------------------------------------------------
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:${PYTHONPATH:-}

# Install lm_eval into overlay on first run; reuse on subsequent jobs
if [ ! -d "$LM_EVAL_OVERLAY/lm_eval" ]; then
    echo "Installing lm_eval overlay..."
    pip install lm_eval --target "$LM_EVAL_OVERLAY" --quiet --no-deps
    pip install lm_eval --target "$LM_EVAL_OVERLAY" --quiet
fi

# Defaults
MODEL=${MODEL:-Qwen/Qwen3-4B}
CKPT_DIR=${CKPT_DIR:-$QAD_DIR/checkpoints}
QUANTIZER=${QUANTIZER:-ste3bit}
# See run_eval_vllm.sh: "math_500"/"aime_2025" are not registered in the lm_eval overlay
# (it calls them minerva_math500/aime25), so this default hard-failed at task load.
TASKS=${TASKS:-"gsm8k minerva_math500"}
RUN_NAME=""
BATCH_SIZE=16
ITER=${SLURM_ARRAY_TASK_ID:-""}
UNQUANTIZED=0
THINK=1        # thinking ON by default, matching run_eval_disagg.sh
# Generations are logged BY DEFAULT: the scores alone cannot answer questions that
# come up later (length, refusals, format failures, repetition loops), and re-running
# a sweep to recover them costs far more than the disk. They land beside the results
# as step_<N>_samples_<task>.jsonl and are gitignored -- ~1 MB per file, which would
# add gigabytes to the repo. Pass --no-log-samples to opt out.
LOG_SAMPLES=${LOG_SAMPLES:-1}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iter=*)        ITER="${1#--iter=}";              shift ;;
        --iter)          ITER="$2";                        shift 2 ;;
        --quantizer=*)   QUANTIZER="${1#--quantizer=}";    shift ;;
        --quantizer)     QUANTIZER="$2";                   shift 2 ;;
        --tasks=*)       TASKS="${1#--tasks=}";            shift ;;
        --tasks)         TASKS="$2";                       shift 2 ;;
        --run-name=*)    RUN_NAME="${1#--run-name=}";      shift ;;
        --run-name)      RUN_NAME="$2";                    shift 2 ;;
        --batch-size=*)  BATCH_SIZE="${1#--batch-size=}";  shift ;;
        --batch-size)    BATCH_SIZE="$2";                  shift 2 ;;
        --model=*)       MODEL="${1#--model=}";            shift ;;
        --model)         MODEL="$2";                       shift 2 ;;
        --unquantized)   UNQUANTIZED=1;                    shift ;;
        # Thinking is ON by default, matching run_eval_disagg.sh.
        --no-think)      THINK=0;                          shift ;;
        --think)         THINK=1;                          shift ;;
        --log-samples)   LOG_SAMPLES=1;                    shift ;;
        --no-log-samples) LOG_SAMPLES=0;                   shift ;;
        *)               shift ;;
    esac
done

if [ "$UNQUANTIZED" = "0" ] && [ -z "$ITER" ]; then
    echo "ERROR: pass --iter N, submit as a job array, or use --unquantized" >&2
    exit 1
fi

ARGS=(
    --model "$MODEL"
    --ckpt-dir "$CKPT_DIR"
    --quantizer "$QUANTIZER"
    --tasks $TASKS
    --batch-size "$BATCH_SIZE"
)
[ -n "$ITER" ]         && ARGS+=(--iter "$ITER")
[ "$UNQUANTIZED" = 1 ] && ARGS+=(--unquantized)
# Pass the flag EXPLICITLY either way. Emitting nothing for one branch and relying on
# the Python default is how --no-think became a silent no-op once already.
if [ "$THINK" = 1 ]; then ARGS+=(--think); else ARGS+=(--no-think); fi
[ "$LOG_SAMPLES" = 1 ] && ARGS+=(--log-samples)
[ -n "$RUN_NAME" ]     && ARGS+=(--run-name "$RUN_NAME")

exec python "$QAD_DIR/eval/eval_transformers.py" "${ARGS[@]}"
