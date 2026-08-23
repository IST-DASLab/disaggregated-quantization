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

#SBATCH --job-name=qad-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --account=adlr_psx_numerics
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# Constants (set before any mode-split so both modes see them)
# ---------------------------------------------------------------------------
CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
HF_CACHE=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
LM_EVAL_OVERLAY=/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/lm_eval_overlay
LOG_KIND=eval_vllm   # log subfolder: logs/<LOG_KIND>/<timestamp>_<tag>/

# ---------------------------------------------------------------------------
# STAGE 0: pre-submit (login node, no SLURM allocation yet).
# SLURM can't create --output directories and mkdir-in-job is too late (slurmd
# opens the log file before the script runs), so the nested dir must exist
# BEFORE sbatch. When invoked directly (not via sbatch) we create it and
# re-submit into it. Invoke as:
#   ./bin/run_eval_vllm.sh --array=0,25,... --quantizer ste4bit --tasks "gsm8k ..."
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    SELF="$(realpath "$0")"
    ROOT="$(dirname "$SELF")/../.."   # script lives in qad/bin/
    STAMP="$(date +%Y%m%d_%H%M%S)"
    SB_ARRAY=""; SB_DEP=""; TAG="run"; PASS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --array=*)     SB_ARRAY="$1";              shift ;;
            --array)       SB_ARRAY="--array=$2";      shift 2 ;;
            # Queue behind a training job: --dependency=afterany:<jobid>. Use afterany,
            # NOT afterok — the trainer exits non-zero on a benign NCCL teardown SIGABRT
            # after everything is written, so afterok would never fire.
            --dependency=*) SB_DEP="$1";               shift ;;
            --dependency)  SB_DEP="--dependency=$2";   shift 2 ;;
            # --steps 250,750,1000,...  submits an array indexed 0..n-1 and maps each
            # index to a step below. Needed because SLURM caps array indices at
            # MaxArraySize (1001 here), so step numbers >1000 cannot be array IDs.
            --steps)       SB_ARRAY="--array=0-$(( $(echo "$2" | tr ',' ' ' | wc -w) - 1 ))"
                           PASS+=("--steps" "$2");     shift 2 ;;
            --quantizer=*) TAG="${1#--quantizer=}"; PASS+=("$1");      shift ;;
            --quantizer)   TAG="$2";                PASS+=("$1" "$2"); shift 2 ;;
            --unquantized) TAG="baseline";          PASS+=("$1");      shift ;;
            *)             PASS+=("$1");                                 shift ;;
        esac
    done
    LOGDIR="$ROOT/logs/${LOG_KIND}/${STAMP}_${TAG}"
    mkdir -p "$LOGDIR"
    echo "logs → $LOGDIR"
    exec sbatch $SB_ARRAY $SB_DEP \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        "$SELF" "${PASS[@]}"
fi

# ---------------------------------------------------------------------------
# HOST mode: resolve paths and re-invoke via srun inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null; then
    # NOTE: `exit` after the first match is REQUIRED for job arrays. SLURM gives the
    # LAST array element a JobId equal to the array's base job id, and for that id
    # `scontrol show job` prints EVERY array record — without `exit` the awk returns
    # one Command= line per record and SCRIPT_PATH becomes a multi-line string, so
    # bash fails with "No such file or directory" (exit 127). That silently killed
    # the final step of every sweep (step 200 here, 225 on the 4B runs).
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")     # qad/bin
    QAD_DIR=$(dirname "$SCRIPT_DIR")         # qad -- anchors checkpoints/ and imports
    mkdir -p "$QAD_DIR/logs"

    # Export everything the container mode needs
    export SCRIPT_DIR QAD_DIR HF_CACHE LM_EVAL_OVERLAY
    export MODEL=${MODEL:-Qwen/Qwen3-4B}
    export RUN_PREFIX=${RUN_PREFIX:-qad}   # must match the training RUN_PREFIX
    export CKPT_DIR=${CKPT_DIR:-$QAD_DIR/checkpoints}
    export HF_HUB_OFFLINE HF_DATASETS_OFFLINE

    srun \
        --ntasks=1 \
        --container-image="$CONTAINER" \
        --no-container-mount-home \
        --container-mounts="/lustre:/lustre,$HOME/.netrc:/root/.netrc" \
        --export=ALL \
        bash "$SCRIPT_PATH" "$@"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode: scontrol not found — run the eval directly
# ---------------------------------------------------------------------------
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
# Attention backend: FLASH_ATTN, matching serving/run_nixl_server.sh. Two reasons.
# (1) REQUIRED for Gemma-3: head_dim=256 with vLLM's default block_size=16 trips a
#     FlashInfer assertion ("There is a bug in FlashInfer block_size 16 head size 256
#     support") and the engine never starts. The disaggregated path never hit this
#     because it already pins FLASH_ATTN.
# (2) It makes single-engine and disaggregated numbers COMPARABLE. With vLLM free to pick
#     per path, a monolithic-vs-disaggregated difference could be a backend difference
#     rather than a KV-transfer one -- which is exactly the comparison we rely on.
# Note this changes the backend for runs that previously let vLLM choose (Qwen3, where
# head_dim=128 made FlashInfer viable); small numeric drift against older single-engine
# results is possible. Override with EVAL_ATTENTION_BACKEND if ever needed.
export VLLM_ATTENTION_BACKEND="${EVAL_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:${PYTHONPATH:-}
# The eval datasets (gsm8k, math500, aime25, mmlu[57 subjects], mmlu_pro) are cached.
# Read them from cache with NO hub API calls — otherwise many parallel array tasks
# collectively trip HF's 1000-req/5-min rate limit (429) enumerating MMLU subjects.
# Keep the HUB online though: vLLM resolves the tokenizer via snapshot_download, and
# full HF_HUB_OFFLINE trips IncompleteSnapshotError on the cached model snapshot
# (missing trivial files like README/LICENSE). The per-job model call is tiny and
# never rate-limits.
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

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
# Must be forwarded: the checkpoint tag ends in a hash of the quantizer params, so an
# arm trained with non-default hyperparameters (e.g. gsqlloyd3bit logit_lr) lives in a
# different directory. Dropping this silently resolves to the DEFAULTS hash, and the
# eval dies with "No HF checkpoint" — or worse, evaluates the wrong run.
QUANT_PARAMS=${QUANT_PARAMS:-""}
# Registered task names, verified against the lm_eval overlay. "math_500"/"aime_2025"
# were NOT registered -- the overlay calls them minerva_math500/aime25 -- so this default
# hard-failed with KeyError: "Spec 'math_500' is not a registered task/group/tag name"
# before a single token was generated. It survived unnoticed because every caller passed
# --tasks explicitly. Matches run_eval_disagg.sh so monolithic and disaggregated runs
# score the identical task set; mmlu_pro is run separately (much slower).
TASKS=${TASKS:-"gsm8k minerva_math500"}
RUN_NAME="${RUN_PREFIX:-qad}-$(echo ${MODEL:-Qwen/Qwen3-4B} | tr '/' '-')"
BATCH_SIZE=16
ITER=${SLURM_ARRAY_TASK_ID:-""}
STEPS=""
UNQUANTIZED=0
THINK=1        # thinking ON by default (matches results/vllm/think/); --no-think disables
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
        --quantizer-params=*) QUANT_PARAMS="${1#--quantizer-params=}"; shift ;;
        --quantizer-params)   QUANT_PARAMS="$2";           shift 2 ;;
        --tasks=*)       TASKS="${1#--tasks=}";            shift ;;
        --tasks)         TASKS="$2";                       shift 2 ;;
        --run-name=*)    RUN_NAME="${1#--run-name=}";      shift ;;
        --run-name)      RUN_NAME="$2";                    shift 2 ;;
        --batch-size=*)  BATCH_SIZE="${1#--batch-size=}";  shift ;;
        --batch-size)    BATCH_SIZE="$2";                  shift 2 ;;
        --model=*)       MODEL="${1#--model=}";            shift ;;
        --model)         MODEL="$2";                       shift 2 ;;
        --unquantized)   UNQUANTIZED=1;                    shift ;;
        # Thinking is OFF BY DEFAULT (every eval here is run that way).
        # --no-think is still accepted so older command lines keep working.
        --no-think)      THINK=0;                          shift ;;
        --think)         THINK=1;                          shift ;;
        --limit)         LIMIT="$2";                       shift 2 ;;
        --steps)         STEPS="$2";                       shift 2 ;;
        --log-samples)   LOG_SAMPLES=1;                    shift ;;
        --no-log-samples) LOG_SAMPLES=0;                   shift ;;
        *)               shift ;;
    esac
done

# Map the array index onto the requested step list (see --steps above).
if [ -n "$STEPS" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    ITER=$(echo "$STEPS" | cut -d, -f$((SLURM_ARRAY_TASK_ID + 1)))
    echo "array index $SLURM_ARRAY_TASK_ID -> step $ITER"
fi

if [ "$UNQUANTIZED" = "0" ] && [ -z "$ITER" ]; then
    echo "ERROR: pass --iter N, submit as a job array, or use --unquantized" >&2
    exit 1
fi

# vLLM does its own continuous batching — no --batch-size.
ARGS=(
    --model "$MODEL"
    --ckpt-dir "$CKPT_DIR"
    --quantizer "$QUANTIZER"
    --tasks $TASKS
)
[ -n "$QUANT_PARAMS" ] && ARGS+=(--quantizer-params "$QUANT_PARAMS")
[ -n "$ITER" ]         && ARGS+=(--iter "$ITER")
[ "$UNQUANTIZED" = 1 ] && ARGS+=(--unquantized)
[ "$THINK" = 1 ]       && ARGS+=(--think)
[ -n "${LIMIT:-}" ]     && ARGS+=(--limit "$LIMIT")
[ "$LOG_SAMPLES" = 1 ] && ARGS+=(--log-samples)
[ -n "$RUN_NAME" ]     && ARGS+=(--run-name "$RUN_NAME")

exec python "$QAD_DIR/eval/eval_vllm.py" "${ARGS[@]}"
