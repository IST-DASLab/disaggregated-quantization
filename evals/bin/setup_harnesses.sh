#!/bin/bash
# One-time setup for the Muse-Glimmer evals. Run from a LOGIN node (it needs network).
#
#   ./bin/setup_harnesses.sh
#
# Three things, none of which belong in the repo:
#
# 1. The third-party harness checkouts, pinned to the commits these drivers were written
#    against. They are cloned rather than vendored -- together they are 343MB, mostly
#    git history and a paper PDF -- but the pins are here so the prompt construction and
#    the answer parsers are reproducible, which is the part that decides the score.
#
# 2. A pip overlay for packages the vLLM image lacks (datasets, and the IFBench scorer's
#    absl/nltk/emoji/syllapy stack). Built INSIDE the container, because wheels compiled
#    for the login node's Python are not loadable by the container's 3.12. It is
#    installed with --target rather than into the image so the image stays immutable and
#    a single overlay serves every job.
#
# 3. The nltk corpora. IFBench's instructions_util calls nltk.download(quiet=True) at
#    import; on a compute node there is no network, so that call fails SILENTLY and only
#    surfaces much later as a LookupError from inside sent_tokenize -- after the GPU
#    hours are already spent on generation. Downloading here makes the failure impossible.
set -euo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESSES=${HARNESSES:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses}
PYDEPS=${PYDEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-pydeps}
CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
HF_CACHE=${HF_CACHE:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache}
ACCOUNT=${ACCOUNT:-coreai_psx_qad}

IFBENCH_COMMIT=db69a6f05689830b0068b8f1529ebcfd2f3b164c
MMMU_COMMIT=268471d0d488258990025331c7528359c324aa25
OCRBENCH_COMMIT=6af7d948ab4d4bdf625af36c884fb7da19075d1a
# Where the OCRBench v2 IMAGES live. They are the one thing here that cannot be fetched:
# the authors distribute them via Google Drive, and the HF mirror (ling99/OCRBench_v2) is
# a lossy re-upload -- absent `eval` keys, dict answers stringified, image_shape missing
# entirely -- which silently changes what several of its scorers do. Copy this directory
# from a machine that already has it; the script warns rather than failing without it.
OCRBENCH_DATA=${OCRBENCH_DATA:-$HARNESSES/ocrbench_v2_data/OCRBench_v2}

mkdir -p "$HARNESSES"

clone_pinned () {  # $1=url $2=dir $3=commit
    if [ ! -d "$2/.git" ]; then
        git clone "$1" "$2"
    fi
    git -C "$2" fetch --all --quiet || true
    git -C "$2" checkout --quiet "$3"
    echo "  $2 @ $(git -C "$2" rev-parse --short HEAD)"
}

echo "=== harness checkouts ==="
clone_pinned https://github.com/allenai/IFBench.git "$HARNESSES/IFBench" "$IFBENCH_COMMIT"
clone_pinned https://github.com/MMMU-Benchmark/MMMU.git "$HARNESSES/MMMU" "$MMMU_COMMIT"
clone_pinned https://github.com/Yuliang-Liu/MultimodalOCR.git "$HARNESSES/MultimodalOCR" "$OCRBENCH_COMMIT"

echo "=== datasets ==="
# Downloaded with the CONTAINER's `hf`, on a cpu node, not with the login node's. A login
# node is not guaranteed to have huggingface_hub at all -- this cluster's does not -- and
# the version that writes the cache should be the version that later reads it.
#
# GPQA is gated; HF_TOKEN must be set and the licence accepted on the hub, or the
# download 401s. MMMU-Pro is ~2.8GB of images.
srun --account="$ACCOUNT" --partition=cpu --time=02:00:00 --ntasks=1 --cpus-per-task=8 \
    --container-image="$CONTAINER" --no-container-mount-home \
    --container-mounts=/scratch:/scratch,/lustre:/lustre \
    bash -lc "
set -e
export HF_HOME='$HF_CACHE'
hf download Idavidrein/gpqa --repo-type dataset --quiet
hf download MMMU/MMMU_Pro --repo-type dataset --quiet
hf download TIGER-Lab/MMLU-Pro --repo-type dataset --quiet
# Calibration set for the W4A4 activation scales. Only the first train shard is read, but
# the repo has no smaller unit to ask for.
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --quiet
# Not weights -- the REMOTE CODE for Nemotron's C-RADIOv4-H vision tower. Its config's
# auto_map points at this repo, and transformers imports it while building the model, so
# a compute node with no network cannot load Nemotron at all without it cached first.
hf download nvidia/C-RADIOv4-H --include '*.py' --include '*.json' --quiet
"
echo "  ok"

echo "=== OCRBench v2 images ==="
if [ -f "$OCRBENCH_DATA/OCRBench_v2.json" ]; then
    echo "  $OCRBENCH_DATA ($(du -sh "$OCRBENCH_DATA" | cut -f1))"
else
    echo "  MISSING: $OCRBENCH_DATA"
    echo "  Google Drive only -- copy it from a machine that has it, e.g."
    echo "    rsync -aHP <host>:$HARNESSES/ocrbench_v2_data/OCRBench_v2 $HARNESSES/ocrbench_v2_data/"
    echo "  Everything except --bench ocrbench works without it."
fi

echo "=== pip overlays (pinned, inside the container) ==="
# bin/install_overlays.sh, not an inline pip install: there are four overlays now, they
# need DIFFERENT flags (muse-radiodeps is --no-deps so it cannot shadow the container's
# numpy/scipy), and they install from evals/requirements/*.txt so a rebuild on another
# cluster reproduces the versions these results were measured with rather than whatever
# PyPI serves today.
CONTAINER="$CONTAINER" ACCOUNT="$ACCOUNT" \
    PYDEPS="$PYDEPS" QUANTDEPS="${QUANTDEPS:-}" RADIODEPS="${RADIODEPS:-}" \
    NIXL_PREFIX="${NIXL_PREFIX:-}" \
    "$_SELF_DIR/install_overlays.sh" --root "$(dirname "$PYDEPS")" || exit 1

echo "=== nltk corpora ==="
# Into the directory ifbench/instructions.py points NLTK_DATA at, so the scorer finds
# them whether or not the launcher exports NLTK_DATA itself.
NLTK_DIR="$HARNESSES/IFBench/ifbench/.nltk_data"
mkdir -p "$NLTK_DIR"
PYTHONPATH="$PYDEPS" python3 -c "
import nltk, sys
ok = True
for r in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng',
          'averaged_perceptron_tagger']:
    got = nltk.download(r, download_dir='$NLTK_DIR', quiet=True)
    print(('  OK  ' if got else '  FAIL'), r)
    ok &= bool(got)
sys.exit(0 if ok else 1)
"

echo "=== done ==="
