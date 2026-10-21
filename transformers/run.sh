#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# TRAINING_PYTHON may point to any compatible environment; the default is local.
TRAINING_PYTHON="${TRAINING_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$TRAINING_PYTHON" ]]; then
    echo "Training Python not found: $TRAINING_PYTHON. Install requirements.txt in .venv or set TRAINING_PYTHON." >&2
    exit 1
fi
exec "$TRAINING_PYTHON" "$ROOT/launch.py" --nproc-per-node="${NPROC_PER_NODE:-8}" \
    --host-memory-limit-gib="${HOST_MEMORY_LIMIT_GIB:-768}" "$@"
