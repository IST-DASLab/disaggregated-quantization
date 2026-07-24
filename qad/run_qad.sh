#!/bin/bash
#SBATCH --job-name=qad-qwen3-4b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=adlr_psx_numerics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# STAGE 0: pre-submit (login node). Create nested log dir logs/train/<stamp>_<tag>/
# and re-submit into it (SLURM can't create --output dirs; mkdir-in-job is too
# late). Invoke directly, e.g.:
#   ./run_qad.sh --quantizer ste4bit            # normal
#   ./run_qad.sh --quantizer ste4bit --debug    # interactive QoS
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    SELF="$(realpath "$0")"
    ROOT="$(dirname "$SELF")/.."
    STAMP="$(date +%Y%m%d_%H%M%S)"
    TAG="run"; QOS_ARGS=(); PASS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --debug)       QOS_ARGS=(--qos=interactive --time=1:00:00);   shift ;;
            --quantizer=*) TAG="${1#--quantizer=}"; PASS+=("$1");      shift ;;
            --quantizer)   TAG="$2";                PASS+=("$1" "$2"); shift 2 ;;
            *)             PASS+=("$1");                                 shift ;;
        esac
    done
    LOGDIR="$ROOT/logs/train/${STAMP}_${TAG}"
    mkdir -p "$LOGDIR"
    echo "logs → $LOGDIR"
    exec sbatch "${QOS_ARGS[@]}" \
        --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
        "$SELF" "${PASS[@]}"
fi

# ---- under SLURM allocation ----
EXTRA_ARGS=("$@")   # --debug already consumed in STAGE 0

SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
HF_CACHE=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
MODEL=${MODEL:-Qwen/Qwen3-4B}
CKPT_DIR=${CKPT_DIR:-$SCRIPT_DIR/checkpoints}   # absolute so the container CWD doesn't matter
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

srun \
    --ntasks-per-node=1 \
    --container-image="$CONTAINER" \
    --no-container-mount-home \
    --container-mounts="/lustre:/lustre,$HOME/.netrc:/root/.netrc" \
    bash -c "
        export HF_HOME=$HF_CACHE
        export TOKENIZERS_PARALLELISM=false
        export PYTHONPATH=$SCRIPT_DIR:\$PYTHONPATH
        export WANDB_MODE=${WANDB_MODE:-online}
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        python -m torch.distributed.run \
            --nproc_per_node=8 \
            --nnodes=$SLURM_NNODES \
            --node_rank=\$SLURM_PROCID \
            --master_addr=$MASTER_ADDR \
            --master_port=29500 \
            $SCRIPT_DIR/qad.py \
                --model $MODEL \
                --run-name qad-\$(echo $MODEL | tr '/' '-') \
                --ckpt-dir $CKPT_DIR \
                --global-batch-size 64 \
                \$@
    " -- "${EXTRA_ARGS[@]}"
