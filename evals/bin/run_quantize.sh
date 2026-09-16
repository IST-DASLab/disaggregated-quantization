#!/bin/bash
# Run quantize_rtn.py on a GPU node, inside the eval container.
#
#   ./bin/run_quantize.sh --scheme NVFP4A16
#   ./bin/run_quantize.sh --scheme NVFP4 --interactive
#
# One GPU, one job, one checkpoint. The 30B loads in bf16 (~60 GB) and llm-compressor
# rewrites layer by layer, so a single B200 is ample and there is nothing to shard.
#
# The quantizer runs with $QUANTDEPS on PYTHONPATH -- llm-compressor plus a
# compressed-tensors NEWER than the container's, since llm-compressor 0.13 needs an API
# 0.17 does not have. That upgrade is confined to this overlay: vLLM keeps reading
# checkpoints with the container's own copy, which is what makes serving the result a
# real check that the two agree on the format rather than a formality.

#SBATCH --job-name=quantize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad

set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
HF_CACHE=${HF_CACHE:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache}
QUANTDEPS=${QUANTDEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-quantdeps}
RADIODEPS=${RADIODEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-radiodeps}

MODEL=${MODEL:-muse-glimmer}
SCHEME=${SCHEME:-NVFP4}
NSAMPLES=${NSAMPLES:-64}
SEQLEN=${SEQLEN:-2048}
OVERWRITE=${OVERWRITE:-0}
EXTRA_IGNORE=${EXTRA_IGNORE:-}
SUFFIX=${SUFFIX:-}
QOS=${QOS:-normal}
# One GPU is all the quantizer uses; MIN_GPUS is what the scheduler must be ASKED for.
# See run_eval.sh -- a QOS with MinTRES gres/gpu=N rejects a smaller request outright.
# 4 on oci-jhb-slurm-1: the GPU nodes are 4-GPU trays and the QOS enforces a whole-node
# floor, so a 1-GPU request dies at submit with "sbatch: error: QOSMinGRES" and nothing in
# the log. The quantizer still uses exactly one GPU; the other three simply idle.
MIN_GPUS=${MIN_GPUS:-4}
# --long: the 4h default sizes a 30B, which loads once and rewrites layer by layer on one
# GPU. A sequential run over a 2.4T model reads the whole 4.89 TB checkpoint off Lustre as
# it onloads each of 92 layers, so the wall clock is storage-bound and 4h is a guess with
# nothing behind it. batch_long exists for exactly this.
LONG=${LONG:-0}

while (($# > 0)); do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --scheme)      SCHEME="$2";      shift 2 ;;
        --nsamples)    NSAMPLES="$2";    shift 2 ;;
        --seqlen)      SEQLEN="$2";      shift 2 ;;
        --overwrite)   OVERWRITE=1;      shift ;;
        --extra-ignore) EXTRA_IGNORE="$2"; shift 2 ;;
        --suffix)      SUFFIX="$2";      shift 2 ;;
        --interactive) QOS=interactive;  shift ;;
        --long)        LONG=1;           shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$_SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
    LOGS="$REPO_ROOT/logs/evals/$(date +%Y%m%d_%H%M%S)_quantize_${MODEL}_${SCHEME}"
    mkdir -p "$LOGS"
    export MUSE_ROOT REPO_ROOT CONTAINER HF_CACHE QUANTDEPS RADIODEPS MODEL SCHEME \
           NSAMPLES SEQLEN OVERWRITE QOS LOGS EXTRA_IGNORE SUFFIX
    echo "logs -> $LOGS (qos=$QOS)"
    if [ "$LONG" = 1 ]; then PART=batch_long; TIME=24:00:00
    else                     PART=batch;      TIME=04:00:00; fi
    exec sbatch --export=ALL --partition="$PART" --time="$TIME" --qos="$QOS" \
        --gpus-per-node="$MIN_GPUS" --job-name="quantize-${MODEL}-${SCHEME}" \
        --output="$LOGS/%j.out" --error="$LOGS/%j.err" "$SELF"
fi

if command -v scontrol &>/dev/null && [ -z "${MUSE_IN_CONTAINER:-}" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export MUSE_IN_CONTAINER=1
    # Node-local NVMe for the weights that do not fit in host RAM: 28 TB, read+write
    # during the job, cleared between jobs, so there is nothing to clean up. Created
    # HERE rather than inside the container, where $USER is root and /raid/scratch/root
    # is neither this user's nor distinguishable from anyone else's.
    export QUANT_OFFLOAD_DIR=${QUANT_OFFLOAD_DIR:-/raid/scratch/$USER/quant_offload}
    mkdir -p "$QUANT_OFFLOAD_DIR" || {
        echo "ERROR: cannot create $QUANT_OFFLOAD_DIR on $(hostname)" >&2; exit 1; }
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/scratch:/scratch,/lustre:/lustre,/raid:/raid" --export=ALL bash "$SCRIPT_PATH"
    exit $?
fi

# RADIODEPS is APPENDED, and it holds exactly open_clip, ftfy and wcwidth -- three
# packages the container does not have, installed --no-deps so the overlay cannot
# shadow anything the container does have. QUANTDEPS stays first because its
# compressed-tensors upgrade has to win.
#
# Nemotron's vision tower is C-RADIOv4-H, whose remote hf_model.py imports open_clip
# for an adaptor this checkpoint never builds (adaptor_names is null). It is still a
# hard requirement, because transformers' check_imports IMPORTS every module named in
# a remote file before it will load it, reachable at runtime or not. Stubbing it would
# have worked and would have left a fake package that silently satisfies a real import
# the day some other model does use an adaptor.
# muse-mambadeps appended: nemotron_h remote code hard-raises
#   'mamba-ssm is required by the Mamba model but cannot be imported'
# on import, so the Super and Ultra cannot even be LOADED for quantization without it.
# Built from source (aarch64, no wheels exist) with --no-deps, so it adds only
# causal_conv1d and mamba_ssm and shadows nothing llm-compressor needs.
MAMBADEPS=${MAMBADEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-mambadeps}
export PYTHONPATH="$QUANTDEPS:$RADIODEPS:$MAMBADEPS"
export HF_HOME=$HF_CACHE
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
echo "node=$(hostname) model=$MODEL scheme=$SCHEME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA=(); [ "$OVERWRITE" = 1 ] && EXTRA=(--overwrite)
EX=(); [ -n "$EXTRA_IGNORE" ] && EX=(--extra-ignore $EXTRA_IGNORE)
# --suffix=VALUE, not --suffix VALUE: the useful suffixes start with a dash
# ("-nogate") and argparse reads a dash-leading value as another option, failing with
# "argument --suffix: expected one argument" -- which reads like the variable was empty.
SF=(); [ -n "$SUFFIX" ] && SF=("--suffix=$SUFFIX")
python3 "$MUSE_ROOT/bin/quantize_rtn.py" --model "$MODEL" --scheme "$SCHEME" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" "${EXTRA[@]}" "${EX[@]}" "${SF[@]}"
RC=$?
echo "=== rc=$RC ==="
exit $RC
