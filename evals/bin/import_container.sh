#!/bin/bash
# Import the vLLM nightly image to .sqsh. MUST run on a CPU node, not a login node:
# login nodes cap at 24 CPUs / TasksMax=6600 / 176 GB, and mksquashfs spawning one
# compressor per core there dies with "Out of memory (cache_alloc)".
# ENROOT_TEMP_PATH must also be node-local -- Lustre cannot create the overlay
# whiteouts enroot needs ("failed to create opaque ovlfs whiteout: Operation not
# permitted"), which silently produces no image at all.
set -euo pipefail
for d in /raid /tmp /var/tmp; do
    if mkdir -p "$d/$USER-enroot/cache" 2>/dev/null; then T="$d/$USER-enroot"; break; fi
done
echo "node=$(hostname)  cpus=$(nproc)  temp=$T  mem=$(free -g | awk '/^Mem:/{print $2}')G"
export ENROOT_TEMP_PATH="$T" ENROOT_CACHE_PATH="$T/cache"
export ENROOT_SQUASH_OPTIONS="-comp lz4 -noD -processors $(( $(nproc) / 2 ))"
# enroot resolves the manifest list for the node it runs on, so the image matches the
# node's architecture. An image copied from a cluster of a different architecture cannot
# be reused: pyxis starts it and /bin/bash dies with "Exec format error".
CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
mkdir -p "$(dirname "$CONTAINER")"
rm -f "$CONTAINER"
enroot import -o "$CONTAINER" docker://vllm/vllm-openai:nightly
ls -la "$CONTAINER"
rm -rf "$T"
