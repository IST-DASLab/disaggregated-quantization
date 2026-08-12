#!/bin/bash
# Gate on every disaggregated number: does KV actually cross from the prefill server
# to the decode server? Runs verify_kv_transfer.py, which compares a W4A4->W4A16 pair
# against both homogeneous stacks and demands that it differ from each. See that
# file's docstring for why a passing-looking eval proves nothing on its own.
#
#   ./diagnostics/run_kv_verify.sh                       # defaults: 0.6B nvfp4 / nvfp4a16 @ 2450
#   ./diagnostics/run_kv_verify.sh --step 1250
#
# Two GPUs on one node: prefill on 0, decode on 1.

#SBATCH --job-name=kv-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --account=adlr_psx_numerics

CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
HF_CACHE=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
LM_EVAL_OVERLAY=/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/lm_eval_overlay

KV_STEP="${KV_STEP:-0002450}"
KV_A4_RUN="${KV_A4_RUN:-qad3x-Qwen-Qwen3-0.6B-nvfp4-99914b93}"
KV_A16_RUN="${KV_A16_RUN:-qad3x-Qwen-Qwen3-0.6B-nvfp4a16-99914b93}"
KV_TOKENIZER="${KV_TOKENIZER:-Qwen/Qwen3-0.6B}"

while (($# > 0)); do
  case "$1" in
    --step)      KV_STEP="$2";      shift 2 ;;
    --a4-run)    KV_A4_RUN="$2";    shift 2 ;;
    --a16-run)   KV_A16_RUN="$2";   shift 2 ;;
    --tokenizer) KV_TOKENIZER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# LOGIN mode: submit
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    LOGS="$(dirname "$(dirname "$(dirname "$SELF")")")/logs/checks"   # diagnostics/ -> qad -> repo
    mkdir -p "$LOGS"
    # Values go through the environment, NOT through --export=VAR=...: sbatch splits
    # that list on commas, so any value containing one silently becomes a variable
    # name and the real value is lost.
    export KV_STEP KV_A4_RUN KV_A16_RUN KV_TOKENIZER
    exec sbatch --export=ALL \
        --output="$LOGS/kvverify_%j.out" --error="$LOGS/kvverify_%j.err" \
        "$SELF"
fi

# ---------------------------------------------------------------------------
# HOST mode: re-invoke inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null && [ -z "${KV_IN_CONTAINER:-}" ]; then
    # `exit` after the first match is REQUIRED (see run_eval_dual.sh): scontrol can
    # print several records and a multi-line path kills the task with exit 127.
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")   # qad/diagnostics
    export QAD_DIR=$(dirname "$SCRIPT_DIR")       # qad
    export HF_CACHE LM_EVAL_OVERLAY KV_IN_CONTAINER=1
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/lustre:/lustre,$HOME/.netrc:/root/.netrc" --export=ALL \
        bash "$SCRIPT_PATH"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode
# ---------------------------------------------------------------------------
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
# The overlay is needed by verify/eval_disagg (lm-eval, requests) but is stripped
# from the SERVERS' PYTHONPATH inside eval_disagg.py: its huggingface-hub 1.24.0
# shadows the container's and vLLM refuses to start.
export PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:${PYTHONPATH:-}

CKPT="$QAD_DIR/checkpoints"
A4="$CKPT/$KV_A4_RUN/weights/step_$KV_STEP"
A16="$CKPT/$KV_A16_RUN/weights/step_$KV_STEP"
for d in "$A4" "$A16"; do
    [ -d "$d" ] || { echo "ERROR: missing checkpoint $d" >&2; exit 1; }
done
WORK="$(dirname "$QAD_DIR")/logs/checks/kvverify"
echo "[kv-verify] a4=$A4"
echo "[kv-verify] a16=$A16"
echo "[kv-verify] work=$WORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec python "$SCRIPT_DIR/verify_kv_transfer.py" \
    --a4 "$A4" --a16 "$A16" \
    --tokenizer "$KV_TOKENIZER" \
    --work-dir "$WORK"
