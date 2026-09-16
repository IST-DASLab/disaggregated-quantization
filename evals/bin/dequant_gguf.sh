#!/bin/bash
# Dequantize a directory of GGUFs into BF16 HF checkpoints, all in parallel on one node.
#
#   ./bin/dequant_gguf.sh --gguf-dir <models>/Qwen3.8-27B-unsloth-GGUF \
#       --mmproj mmproj-BF16.gguf --prefix Qwen3.8-27B-unsloth
#
# Wraps bin/gguf_to_bf16.py, which does the actual conversion and is where the
# correctness argument lives (it reuses vllm-gguf-plugin's own Qwen35GGUFAdapter for the
# Gated-DeltaNet layout restore rather than re-deriving it).
#
# NO GPU. The conversion is numpy/torch on the host: it streams one tensor at a time and
# flushes 4 GB safetensors shards, so it is bounded by disk and single-core CPU, not by
# memory. Measured on the GSQ-RCO arms: ~9.5 min and ~55 GB per model.
#
# ONE NODE, N BACKGROUND PROCESSES -- not one job per model. Slurm hands out whole nodes
# (76 cores, 350 GB here), and a busy eval sweep already holds the per-user node budget,
# so an array of single-threaded jobs is throttled to ~2 concurrent by
# QOSMaxNodePerUserLimit. Eight conversions on one node are independent and fit easily.
#
# Idempotent: a model whose output already has a weight index is skipped, so a rerun
# after a partial pass converts only what is missing.

#SBATCH --job-name=dequant-gguf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=coreai_psx_qad

set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
P=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode
CONTAINER=${CONTAINER:-$P/containers/vllm-nightly.sqsh}
# Holds BOTH vllm_gguf_plugin (the weights adapter) and a gguf package new enough to
# dequantize IQ1_S/IQ1_M. Built from git main -- the PyPI wheel predates Qwen3.5 support.
GGUF_OVERLAY=${GGUF_OVERLAY:-$P/gguf_overlay_main}

GGUF_DIR=${GGUF_DIR:-}
MMPROJ=${MMPROJ:-mmproj-BF16.gguf}
BASE=${BASE:-$P/models/Qwen3.8-27B}
PREFIX=${PREFIX:-}
TIME=${TIME:-06:00:00}
PART=${PART:-cpu}

while (($# > 0)); do
    case "$1" in
        --gguf-dir) GGUF_DIR="$2"; shift 2 ;;
        --mmproj)   MMPROJ="$2";   shift 2 ;;
        --base)     BASE="$2";     shift 2 ;;
        --prefix)   PREFIX="$2";   shift 2 ;;
        --time)     TIME="$2";     shift 2 ;;
        --partition) PART="$2";    shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -d "$GGUF_DIR" ] || { echo "ERROR: --gguf-dir not a directory: $GGUF_DIR" >&2; exit 2; }
[ -n "$PREFIX" ]   || { echo "ERROR: --prefix is required" >&2; exit 2; }
[ -s "$GGUF_DIR/$MMPROJ" ] || { echo "ERROR: no mmproj at $GGUF_DIR/$MMPROJ" >&2; exit 2; }
[ -d "$BASE" ] || { echo "ERROR: --base not a directory: $BASE" >&2; exit 2; }

# Backbones only: the mmproj is the vision tower and rides along with each conversion,
# and an imatrix file is calibration data, not a model.
mapfile -t GGUFS < <(find "$GGUF_DIR" -maxdepth 1 -name '*.gguf' \
                     ! -name "$MMPROJ" ! -name 'imatrix*' -printf '%f\n' | sort)

TODO=()
for G in "${GGUFS[@]}"; do
    TAG=${G%.gguf}; TAG=${TAG#Qwen3.8-27B-}
    OUT="$(dirname "$GGUF_DIR")/${PREFIX}-${TAG}-bf16"
    [ -s "$OUT/model.safetensors.index.json" ] && continue
    TODO+=("$G")
done

if [ -z "${SLURM_JOB_ID:-}" ]; then
    if [ "${#TODO[@]}" = 0 ]; then
        echo "nothing to do: every GGUF in $GGUF_DIR already has a BF16 checkpoint"
        exit 0
    fi
    LOGS="$REPO_ROOT/logs/evals/$(date +%Y%m%d_%H%M%S)_dequant_$(basename "$GGUF_DIR")"
    mkdir -p "$LOGS"
    export MUSE_ROOT REPO_ROOT CONTAINER GGUF_OVERLAY GGUF_DIR MMPROJ BASE PREFIX LOGS
    echo "${#TODO[@]} models to convert; logs -> $LOGS"
    printf '  %s\n' "${TODO[@]}"
    exec sbatch --export=ALL --partition="$PART" --time="$TIME" \
        --job-name="dequant-$(basename "$GGUF_DIR")" \
        --output="$LOGS/%j.out" --error="$LOGS/%j.err" \
        "$_SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
fi

echo "=== converting ${#TODO[@]} models on $(hostname) ==="
srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
     --container-mounts=/scratch:/scratch,/lustre:/lustre --export=ALL bash -c "
set -uo pipefail
export PYTHONPATH='$GGUF_OVERLAY'
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for G in ${TODO[*]}; do
  TAG=\${G%.gguf}; TAG=\${TAG#Qwen3.8-27B-}
  OUT='$(dirname "$GGUF_DIR")'/'$PREFIX'-\$TAG-bf16
  ( python3 '$MUSE_ROOT/bin/gguf_to_bf16.py' --gguf '$GGUF_DIR'/\$G \
        --mmproj '$GGUF_DIR/$MMPROJ' --base '$BASE' --out \$OUT \
    > '$LOGS'/dequant_\$TAG.log 2>&1 ) &
done
wait
"

RC=0
for G in "${TODO[@]}"; do
    TAG=${G%.gguf}; TAG=${TAG#Qwen3.8-27B-}
    OUT="$(dirname "$GGUF_DIR")/${PREFIX}-${TAG}-bf16"
    # The index is written LAST by gguf_to_bf16.py, after every shard is flushed and
    # renamed, so its presence is the only safe completion signal -- a directory full of
    # shards can still be a conversion that died partway.
    if [ -s "$OUT/model.safetensors.index.json" ] && [ -s "$OUT/config.json" ]; then
        echo "=== ok: $TAG -> $(du -sh "$OUT" | cut -f1)"
    else
        echo "ERROR: $TAG did not produce a complete checkpoint at $OUT" >&2; RC=1
    fi
done
exit $RC
