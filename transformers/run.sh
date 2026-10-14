#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/transformers/.cache/uv}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Use the existing UV environment; never create a second environment.
exec uv run --no-project --python "$ROOT/.venv/bin/python" \
    python "$ROOT/transformers/launch.py" --nproc-per-node="${NPROC_PER_NODE:-8}" \
    --host-memory-limit-gib="${HOST_MEMORY_LIMIT_GIB:-768}" "$@"
