#!/bin/bash
#SBATCH --job-name=qad
# ^ fallback only: stage 0 always passes --job-name=qad-<model>-<quantizer>, which is
#   what --dependency=singleton keys on. Do not point tooling at this literal.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=adlr_psx_numerics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# STAGE 0: pre-submit (login node). Create nested log dir logs/train/<stamp>_<tag>/
# and re-submit into it (SLURM can't create --output dirs; mkdir-in-job is too
# late). Invoke directly, e.g.:
#   ./bin/run_qad.sh --quantizer ste4bit            # normal
#   ./bin/run_qad.sh --quantizer ste4bit --debug    # interactive QoS
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    SELF="$(realpath "$0")"
    # This script lives in qad/bin/, so the repo root is two levels up.
    ROOT="$(dirname "$SELF")/../.."
    STAMP="$(date +%Y%m%d_%H%M%S)"
    TAG="run"; QOS_ARGS=(); PASS=(); CHAIN=1
    while [ $# -gt 0 ]; do
        case "$1" in
            --debug)       QOS_ARGS=(--qos=interactive --time=1:00:00);   shift ;;
            # --time is capped at 04:00:00 by the cluster, and that is NOT enough past
            # ~4B: a 4B run measures 5.81 s/step, i.e. 4.01h for 2485 steps, so it
            # TIMEOUTs a few steps from the end. 8B is roughly double. --chain N submits
            # N jobs sharing one --job-name and holding --dependency=singleton, so SLURM
            # runs them strictly one at a time and each continues where the last stopped
            # via --resume auto (state/ is written every --save-every=100 steps, so a
            # timeout costs at most 100 steps). Over-provision freely: a chain job that
            # finds the run already finished exits at once with "Nothing to do: state at
            # step N >= total" rather than retraining anything.
            --chain)       CHAIN="$2";                                    shift 2 ;;
            --chain=*)     CHAIN="${1#--chain=}";                         shift ;;
            --nodes)       QOS_ARGS+=(--nodes="$2");                      shift 2 ;;
            --quantizer=*) TAG="${1#--quantizer=}"; PASS+=("$1");      shift ;;
            --quantizer)   TAG="$2";                PASS+=("$1" "$2"); shift 2 ;;
            *)             PASS+=("$1");                                 shift ;;
        esac
    done
    # Singleton is keyed on (user, job-name), so the NAME must identify this run. The
    # old hardcoded qad-qwen3-4b was shared by every training job, which under singleton
    # would have serialised the entire cluster's worth of runs against each other.
    # Naming it per (model, quantizer) also stops two jobs from ever writing one state/
    # directory concurrently, which would corrupt the resume point.
    MODEL_TAG=$(echo "${MODEL:-Qwen/Qwen3-4B}" | tr '/' '-' | sed 's/^Qwen-//')
    JOB_NAME="qad-${MODEL_TAG}-${TAG}"
    LOGDIR="$ROOT/logs/train/${STAMP}_${TAG}"
    mkdir -p "$LOGDIR"
    echo "logs → $LOGDIR"
    echo "job  → $JOB_NAME   (chain of $CHAIN, --dependency=singleton)"
    for _i in $(seq 1 "$CHAIN"); do
        sbatch --job-name="$JOB_NAME" --dependency=singleton "${QOS_ARGS[@]}" \
            --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
            "$SELF" "${PASS[@]}"
    done
    exit 0
fi

# ---- under SLURM allocation ----
EXTRA_ARGS=("$@")   # --debug already consumed in STAGE 0

# `exit` after the first match: `scontrol show job` can print more than one record
# (always the case for the last element of a job array), which would otherwise make
# SCRIPT_PATH a multi-line string and fail with bash "No such file" (exit 127).
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")     # qad/bin
QAD_DIR=$(dirname "$SCRIPT_DIR")         # qad -- anchors checkpoints/, wandb/, imports

CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
HF_CACHE=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
MODEL=${MODEL:-Qwen/Qwen3-4B}
# RUN_PREFIX namespaces the checkpoint tag (<prefix>-<model>-<quant>-<hash>).
# Use a fresh prefix for a new training recipe so earlier runs are never overwritten.
RUN_PREFIX=${RUN_PREFIX:-qad3x}
CKPT_DIR=${CKPT_DIR:-$QAD_DIR/checkpoints}   # absolute so the container CWD doesn't matter
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

srun \
    --ntasks-per-node=1 \
    --container-image="$CONTAINER" \
    --no-container-mount-home \
    --container-mounts="/lustre:/lustre,$HOME/.netrc:/root/.netrc" \
    bash -c "
        export HF_HOME=$HF_CACHE
        export TOKENIZERS_PARALLELISM=false
        # psx-luts carries the luts extension nvr2bit imports lazily. NO BACKTICKS:
        # this whole block is a double-quoted bash -c string, so backticks are
        # command substitution and even a COMMENT gets executed.
        export PYTHONPATH=$QAD_DIR:${PSX_LUTS_PATH:-/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/psx-luts}:\$PYTHONPATH
        export WANDB_MODE=${WANDB_MODE:-online}
        # wandb creates its run directory under \$WANDB_DIR, which defaults to the
        # CWD — and inside the container that is not a writable path, so wandb.init()
        # blocks until it times out. This hits offline mode too (it needs the same
        # local dir), which is how three jobs ended up training with logging disabled
        # after burning 6 minutes each on two 180s timeouts. Point it somewhere real.
        export WANDB_DIR=$QAD_DIR
        export RUN_PREFIX=$RUN_PREFIX
        cd $QAD_DIR
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        python -m torch.distributed.run \
            --nproc_per_node=8 \
            --nnodes=$SLURM_NNODES \
            --node_rank=\$SLURM_PROCID \
            --master_addr=$MASTER_ADDR \
            --master_port=29500 \
            $QAD_DIR/training/qad.py \
                --model $MODEL \
                --run-name $RUN_PREFIX-\$(echo $MODEL | tr '/' '-') \
                --ckpt-dir $CKPT_DIR \
                --global-batch-size 64 \
                \$@
    " -- "${EXTRA_ARGS[@]}"
