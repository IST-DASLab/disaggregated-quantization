#!/bin/bash
# GSM8K evaluation of the prefill/decode dual formats, via transformers generate().
#
# vLLM cannot serve a dual-format model in one process (the two halves are meant for
# separate workers), so the DEPLOYMENT-accuracy question is answered here instead:
# HF generate() does one multi-token pass over the prompt and then one token at a
# time, which is exactly the prefill/decode split, so the model switches format on
# its own and the decode phase attends to a KV cache built by the prefill format.
#
# Every configuration goes through the same loader and the same lm-eval invocation,
# so a difference in the numbers comes from the model rather than the harness.
#
#   ./bin/run_eval_dual.sh --steps 250,750,...  --model Qwen/Qwen3-0.6B
#   ./bin/run_eval_dual.sh --steps 2450 --configs dual-shared,homo-a4   # subset
#
# Configurations (label:run_prefix:quantizer:extra-args):
#   dual-shared   nvfp4pdshared, dual inference               <- the method
#   dual-split    nvfp4pdsplit,  dual inference               <- the method
#   dual-shared-pl / dual-split-pl   same, --include-prefill-loss runs
#   homo-a4       nvfp4    trained AND evaluated W4A4 everywhere   <- baseline
#   homo-a16      nvfp4a16 trained AND evaluated W4A16 everywhere  <- baseline
#   sim-dual      nvfp4 checkpoint driven in DUAL inference (diagnostic)
#                 -> separates dual TRAINING from dual INFERENCE
#
# The baselines are homogeneous end to end: a model trained for one format and run in
# that same format, which is what a dual model has to beat to be worth deploying.
# Nothing is coerced into a phase it was not trained for.
#
# NOT COMPARABLE TO THE vLLM SWEEP. These runs cap generation at --max-gen-toks 512
# (ample for GSM8K with --no-think), while results/vllm/think/ is produced at 4096
# because AIME/MATH need it. A truncated answer scores 0, so the cap is part of the
# measurement. Everything in results/dual/ shares one cap and is internally
# consistent; do not put these numbers on the same axis as the vLLM ones.

#SBATCH --job-name=qad-dual-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --account=adlr_psx_numerics

CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
HF_CACHE=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
LM_EVAL_OVERLAY=/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/lm_eval_overlay
LOG_KIND=eval_dual

# label:prefix:quantizer:extra
# DEFAULT set: the two mixed formats and the two homogeneous baselines they must beat.
CONFIGS_DEFAULT="dual-shared:qad3x:nvfp4pdshared: \
dual-split:qad3x:nvfp4pdsplit: \
homo-a4:qad3x:nvfp4: \
homo-a16:qad3x:nvfp4a16:"

# SIDELINED: still selectable with --configs, never run by default.
#   sim-dual  a homogeneously-trained W4A4 checkpoint driven through dual inference.
#             It answered its question (dual inference alone recovers most of the
#             W4A4->W4A16 gap) and is a diagnostic, not a deployable artifact.
#   *-pl      the --include-prefill-loss ablation. Settled: it costs 0.9-2.6 points
#             at both 0.6B and 1.7B, so it is never the configuration to ship.
CONFIGS_EXTRA="dual-shared-pl:qad3xpl:nvfp4pdshared: \
dual-split-pl:qad3xpl:nvfp4pdsplit: \
sim-dual:qad3x:nvfp4:--dual"

CONFIGS_ALL="$CONFIGS_DEFAULT $CONFIGS_EXTRA"

# ---------------------------------------------------------------------------
# STAGE 0: pre-submit (login node). Build the array over configs x steps.
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    SELF="$(realpath "$0")"; ROOT="$(dirname "$SELF")/../.."   # script lives in qad/bin/
    STAMP="$(date +%Y%m%d_%H%M%S)"
    STEPS=""; MODEL="Qwen/Qwen3-0.6B"; CONFIGS="$CONFIGS_DEFAULT"; LIMIT=""; BS=64; PASS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --steps)   STEPS="$2";  shift 2 ;;
            --model)   MODEL="$2";  shift 2 ;;
            --limit)   LIMIT="$2";  shift 2 ;;   # smoke-test the harness cheaply
            --batch-size) BS="$2";  shift 2 ;;
            --configs) SEL="$2"; CONFIGS=""
                       for want in ${SEL//,/ }; do
                           for c in $CONFIGS_ALL; do
                               [ "${c%%:*}" = "$want" ] && CONFIGS="$CONFIGS $c"
                           done
                       done
                       shift 2 ;;
            *)         PASS+=("$1"); shift ;;
        esac
    done
    [ -z "$STEPS" ] && { echo "ERROR: --steps required (e.g. --steps 250,750,2450)" >&2; exit 1; }
    NS=$(echo "$STEPS" | tr ',' ' ' | wc -w); NC=$(echo $CONFIGS | wc -w)
    N=$((NS * NC))
    LOGDIR="$ROOT/logs/${LOG_KIND}/${STAMP}_$(echo $MODEL | cut -d/ -f2)"
    mkdir -p "$LOGDIR"
    echo "configs=$NC steps=$NS -> $N tasks;  logs → $LOGDIR"
    # Array index maps to (config, step); SLURM caps indices at MaxArraySize, and
    # step numbers alone would exceed it.
    # Put the values in the ENVIRONMENT and propagate with a bare --export=ALL.
    # Never inline them into --export: that list is COMMA-separated, so
    # --export=ALL,EV_STEPS="250,750,1000" silently sets EV_STEPS=250 and treats
    # 750/1000 as further variable NAMES. It fails without any error, and every
    # array task then evaluates the same step and overwrites one result file.
    export EV_STEPS="$STEPS" EV_MODEL="$MODEL" EV_CONFIGS="$CONFIGS"
    export EV_LIMIT="$LIMIT" EV_BS="$BS"
    exec sbatch --array=0-$((N - 1)) \
        --output="$LOGDIR/%A_%a.out" --error="$LOGDIR/%A_%a.err" \
        --export=ALL \
        "$SELF" "${PASS[@]}"
fi

# ---------------------------------------------------------------------------
# HOST mode: re-invoke inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null; then
    # `exit` after the first match is REQUIRED: for the last array element SLURM
    # reports every array record under the base job id, and without it SCRIPT_PATH
    # becomes multi-line and the task dies with exit 127.
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    SCRIPT_DIR=$(dirname "$SCRIPT_PATH")     # qad/bin
    QAD_DIR=$(dirname "$SCRIPT_DIR")         # qad -- anchors checkpoints/ and results
    export SCRIPT_DIR QAD_DIR HF_CACHE LM_EVAL_OVERLAY EV_STEPS EV_MODEL EV_CONFIGS EV_LIMIT EV_BS
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/lustre:/lustre,$HOME/.netrc:/root/.netrc" --export=ALL \
        bash "$SCRIPT_PATH" "$@"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode
# ---------------------------------------------------------------------------
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:${PYTHONPATH:-}
# Datasets from cache only: many parallel array tasks enumerating the same dataset
# collectively trip HF's rate limit. Hub stays online for tokenizer resolution.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0

set -- $EV_CONFIGS
NC=$#
IDX=${SLURM_ARRAY_TASK_ID:-0}
CFG=$(eval echo \${$((IDX % NC + 1))})
STEP=$(echo "$EV_STEPS" | cut -d, -f$((IDX / NC + 1)))

LABEL=$(echo "$CFG" | cut -d: -f1)
PREFIX=$(echo "$CFG" | cut -d: -f2)
QUANT=$(echo "$CFG" | cut -d: -f3)
EXTRA=$(echo "$CFG" | cut -d: -f4)
# The single-format controls (nvfp4 / nvfp4a16) were only ever trained under the
# qad3x prefix — there is no qad3xpl copy of them, and the prefill-loss ablation
# applies to the dual runs, not to the controls. Pinning them here keeps a
# --configs selection that mixes arms from resolving a checkpoint that does not exist.
case "$QUANT" in
    nvfp4|nvfp4a16) PREFIX="qad3x" ;;
esac
RUN_NAME="$PREFIX-$(echo "$EV_MODEL" | tr '/' '-')"

echo "task $IDX -> config=$LABEL step=$STEP quantizer=$QUANT run=$RUN_NAME ${EXTRA}"

exec python "$QAD_DIR/eval/eval_transformers.py" \
    --model "$EV_MODEL" \
    --quantizer "$QUANT" \
    --run-name "$RUN_NAME" \
    --ckpt-dir "$QAD_DIR/checkpoints" \
    --iter "$STEP" \
    --tasks gsm8k \
    ${EXTRA:+$EXTRA} \
    --tag-suffix="-$LABEL" \
    --no-think \
    --batch-size "${EV_BS:-64}" \
    ${EV_LIMIT:+--limit $EV_LIMIT} \
    --output-dir "$QAD_DIR/results/dual"
