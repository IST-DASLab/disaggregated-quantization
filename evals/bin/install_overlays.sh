#!/bin/bash
# Rebuild the three pip overlays from pinned requirements, inside the container.
#
#   ./bin/install_overlays.sh --root /lustre/.../prefill_decode
#   ./bin/install_overlays.sh --root ... --only muse-radiodeps
#   ./bin/install_overlays.sh --root ... --verify        # import-check what exists, install nothing
#
# The overlays are `pip install --target` trees layered onto the container's site-packages
# via PYTHONPATH. They are NOT venvs: the container owns torch, vllm, transformers and
# numpy, and the overlays add only what it lacks.
#
# INSIDE THE CONTAINER, NOT ON THE LOGIN NODE. The wheels must match the container's
# Python 3.12 and its numpy/torch ABI. Installing from a login node's interpreter produces
# a tree that imports on the login node and fails on a compute node.
#
# THE FLAGS ARE NOT INTERCHANGEABLE, and each overlay uses different ones:
#
#   muse-pydeps     plain --target. Scorer dependencies the container lacks.
#   muse-quantdeps  --no-deps, and it carries a compressed-tensors NEWER than the
#                   container's, because llm-compressor 0.13 needs an API 0.17 does not
#                   have. That upgrade is confined to this overlay so vLLM keeps reading
#                   checkpoints with the container's own copy -- which is what makes
#                   serving a quantized checkpoint a real test that the two agree on the
#                   format rather than a formality.
#
#                   --no-deps because `pip --target` resolves against an EMPTY tree, not
#                   against the container's site-packages: compressed-tensors declares
#                   torch>=2.10, so resolution downloads PyPI's torch into the overlay,
#                   where run_quantize.sh's PYTHONPATH="$QUANTDEPS:$RADIODEPS" puts it in
#                   front of the container's CUDA build. The requirements file is a frozen
#                   closure, so there is nothing for the resolver to work out anyway -- and
#                   with it off, the file installs as measured. Resolution additionally
#                   REJECTS that frozen set: fsspec is pinned at 2026.7.0 and datasets
#                   5.0.1 caps it at 2026.6.0, an upgrade that happened in place on the
#                   original tree and that only a freeze can reproduce.
#   muse-radiodeps  --no-deps, and this one matters. It exists only because transformers'
#                   check_imports IMPORTS every module named in a remote modeling file,
#                   reachable at runtime or not: C-RADIOv4-H names open_clip for an
#                   adaptor Nemotron never builds, and the audio path names librosa. With
#                   dependency resolution on, pip pulls numpy and scipy into the overlay,
#                   where they SHADOW the container's -- every package here is absent from
#                   the container by design, so nothing is overridden.
#   nixl-nodeps     --no-deps, and the ONLY overlay that goes on a SERVER's PYTHONPATH --
#                   run_nixl_server.sh prepends $NIXL_PREFIX, which is why the note in
#                   run_eval.sh about keeping the overlay off the server is specifically
#                   about muse-pydeps. NixlConnector needs the `nixl` shim plus the
#                   CUDA-matched `nixl-cuNN` wheel, neither of which is in the container;
#                   without --no-deps pip pulls a second torch, which shadows the
#                   container's and breaks vllm._C with an undefined at::TensorBase
#                   symbol. Only --disagg runs touch it.
set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$(dirname "$_SELF_DIR")/requirements"

ROOT=""; ONLY=""; VERIFY=0
PYDEPS=${PYDEPS:-}; QUANTDEPS=${QUANTDEPS:-}; RADIODEPS=${RADIODEPS:-}
CONTAINER=${CONTAINER:-}
ACCOUNT=${ACCOUNT:-coreai_psx_qad}
while (($# > 0)); do
  case "$1" in
    --root)   ROOT="$2";   shift 2 ;;
    --only)   ONLY="$2";   shift 2 ;;
    --verify) VERIFY=1;    shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROOT" ] || { echo "ERROR: --root is required (where the overlays live)" >&2; exit 2; }
[ -n "$CONTAINER" ] || { echo "ERROR: export CONTAINER=/path/to/.sqsh first" >&2; exit 2; }

# name : extra pip flags : one module to import as proof it works
SPECS=(
  "muse-pydeps::absl,langdetect,nltk,immutabledict,emoji,syllapy,datasets"
  "muse-quantdeps:--no-deps:llmcompressor,compressed_tensors"
  "muse-radiodeps:--no-deps:open_clip,librosa,ftfy,scipy"
  "nixl-nodeps:--no-deps:nixl._api"
)

# A mistyped --only would otherwise match nothing, install nothing, and exit 0 saying
# "done" -- the failure mode that looks exactly like success.
if [ -n "$ONLY" ]; then
  case "$ONLY" in
    muse-pydeps|muse-quantdeps|muse-radiodeps|nixl-nodeps) ;;
    *) echo "ERROR: --only must be muse-pydeps|muse-quantdeps|muse-radiodeps|nixl-nodeps" \
            "(got '$ONLY')" >&2
       exit 2 ;;
  esac
fi

for spec in "${SPECS[@]}"; do
  IFS=: read -r name flags mods <<< "$spec"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  # Honour an explicit PYDEPS/QUANTDEPS/RADIODEPS if the caller set one, so the three
  # overlays need not be siblings. run_eval.sh and run_quantize.sh read those variables
  # independently; deriving the path from --root alone would build the right trees in the
  # wrong place the moment someone points one elsewhere.
  case "$name" in
    muse-pydeps)    dest="${PYDEPS:-$ROOT/$name}" ;;
    muse-quantdeps) dest="${QUANTDEPS:-$ROOT/$name}" ;;
    muse-radiodeps) dest="${RADIODEPS:-$ROOT/$name}" ;;
    nixl-nodeps)    dest="${NIXL_PREFIX:-$ROOT/$name}" ;;
    *)              dest="$ROOT/$name" ;;
  esac
  req="$REQ/$name.txt"
  [ -f "$req" ] || { echo "ERROR: no $req" >&2; exit 1; }

  if [ "$VERIFY" = 1 ]; then
    echo "=== verify $name ==="
    srun --account="$ACCOUNT" --partition=cpu --time=00:10:00 --ntasks=1 --cpus-per-task=4 \
      --container-image="$CONTAINER" --no-container-mount-home \
      --container-mounts=/scratch:/scratch,/lustre:/lustre \
      bash -lc "PYTHONPATH='$dest' python3 -c \"
import importlib
for m in '$mods'.split(','):
    mod = importlib.import_module(m)
    print('  ok', m, getattr(mod, '__version__', ''))\"" || exit 1
    continue
  fi

  echo "=== install $name  ($(wc -l < "$req") pinned packages, flags: ${flags:-none}) ==="
  srun --account="$ACCOUNT" --partition=cpu --time=00:30:00 --ntasks=1 --cpus-per-task=16 \
    --container-image="$CONTAINER" --no-container-mount-home \
    --container-mounts=/scratch:/scratch,/lustre:/lustre \
    bash -lc "
set -e
rm -rf '$dest'; mkdir -p '$dest'
pip install --no-cache-dir --target '$dest' $flags -r '$req'
PYTHONPATH='$dest' python3 -c \"
import importlib
for m in '$mods'.split(','):
    importlib.import_module(m)
print('  imports OK')\"
" || { echo "FAILED: $name" >&2; exit 1; }
done
echo "done. Overlays under $ROOT"
