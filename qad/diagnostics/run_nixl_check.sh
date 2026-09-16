#!/bin/bash
# Diagnostic: can `nixl` be installed such that BOTH `import nixl._api` and
# `import vllm` work in the SAME interpreter?
#
# This is the single biggest risk for NixlConnector on this cluster. A previous
# attempt did `pip install nixl --target .../nixl_overlay` WITHOUT --no-deps; that
# dragged in a second copy of torch, and putting the dir on PYTHONPATH shadowed the
# container's torch so vLLM's C extension failed to link:
#
#   ImportError: vllm/_C.abi3.so: undefined symbol:
#     _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
#
# So the whole point here is: install with --no-deps into a FRESH dir, then prove the
# two imports coexist. Nothing else gets built until this passes.
#
#   ./diagnostics/run_nixl_check.sh          # submits itself with sbatch
#
# One GPU is requested because `import vllm` touches the CUDA platform layer.

#SBATCH --job-name=nixl-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=00:40:00
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad

CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/nemo-26.02.sqsh
HF_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache
# Fresh dir. The old .../prefill-decode/nixl_overlay is poison (full torch inside)
# and must never go on PYTHONPATH.
NIXL_PREFIX="${NIXL_PREFIX:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/nixl_nodeps}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    LOGS="$(dirname "$(dirname "$(dirname "$SELF")")")/logs/checks"   # diagnostics/ -> qad -> repo
    mkdir -p "$LOGS"
    # Never --export=ALL,VAR=...: sbatch splits that list on commas.
    export NIXL_PREFIX
    exec sbatch --export=ALL \
        --output="$LOGS/nixlcheck_%j.out" --error="$LOGS/nixlcheck_%j.err" \
        "$SELF"
fi

if command -v scontrol &>/dev/null && [ -z "${NIXL_IN_CONTAINER:-}" ]; then
    # `exit` after the first match is REQUIRED; without it scontrol's multi-record
    # output yields a multi-line path and the task dies with exit 127.
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")   # qad/diagnostics
    export QAD_DIR=$(dirname "$SCRIPT_DIR")       # qad
    export HF_CACHE NIXL_PREFIX NIXL_IN_CONTAINER=1
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/scratch:/scratch,/lustre:/lustre,$HOME/.netrc:/root/.netrc" --export=ALL \
        bash "$SCRIPT_PATH"
    exit $?
fi

export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH   # start from the container's own environment, nothing shadowing it

echo "### 0. is nixl already in the container? ###"
python -c "import nixl; print('nixl already present:', nixl.__file__)" 2>&1 | tail -3
python -c "import nixl._api as a; print('nixl._api already present:', a.__file__)" 2>&1 | tail -3

echo
echo "### 1. baseline: does vllm import cleanly with an empty PYTHONPATH? ###"
python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -5

echo
echo "### 2. pip install nixl --no-deps --target $NIXL_PREFIX ###"
# `nixl` itself is a 10 kB SHIM: its only Requires-Dist are nixl-cu12 / nixl-cu13,
# and nixl/__init__.py picks one at import time from torch.version.cuda. Installing
# only `nixl` therefore gives
#     ImportError: torch reports CUDA 13 but nixl-cu13 is not installed
# so the CUDA-matched wheel has to be installed alongside it. Both --no-deps: the
# unpinned install is what previously dragged in a second torch and broke vllm._C.
CUDA_MAJOR=$(python -c "import torch;print((torch.version.cuda or '12').split('.')[0])")
echo "torch CUDA major = $CUDA_MAJOR -> nixl-cu${CUDA_MAJOR}"
rm -rf "$NIXL_PREFIX"
mkdir -p "$NIXL_PREFIX"
pip install --no-cache-dir --no-deps --target "$NIXL_PREFIX" \
    "nixl==1.3.1" "nixl-cu${CUDA_MAJOR}==1.3.1" 2>&1 | tail -20
echo "pip exit=$?"
echo "--- top level of $NIXL_PREFIX ---"
ls "$NIXL_PREFIX" 2>&1 | head -40
echo "--- anything that would shadow torch/numpy? (must be EMPTY) ---"
ls "$NIXL_PREFIX" 2>/dev/null | grep -E '^(torch|numpy|nvidia|triton)' || echo "(none - good)"

echo
echo "### 3. declared requirements of the nixl wheels ###"
for m in "$NIXL_PREFIX"/nixl*-*.dist-info/METADATA; do
  echo "--- $m ---"
  grep -i -E '^(Name|Version|Requires-Dist):' "$m" 2>&1 | head -30
done

echo
echo "### 4. import nixl._api ALONE, with the overlay on PYTHONPATH ###"
PYTHONPATH="$NIXL_PREFIX" python -c "
import nixl, nixl._api as api
print('nixl at', nixl.__file__)
print('nixl._api at', api.__file__)
print('NixlWrapper:', api.nixl_agent)
" 2>&1 | tail -25

echo
echo "### 5. THE TEST: import nixl._api AND vllm in the same process ###"
PYTHONPATH="$NIXL_PREFIX" python -c "
import nixl._api as api
import torch
print('torch', torch.__version__, 'from', torch.__file__)
import vllm
print('vllm', vllm.__version__)
import vllm._C
print('vllm._C OK')
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector
print('NixlConnector import OK ->', NixlConnector)
print('COEXIST_OK')
" 2>&1 | tail -30

echo
echo "### 6. reverse order (vllm first, then nixl) ###"
PYTHONPATH="$NIXL_PREFIX" python -c "
import vllm, vllm._C
import nixl._api as api
print('COEXIST_REVERSE_OK')
" 2>&1 | tail -15

echo
echo "### 7. can a nixl agent actually be constructed (needs UCX at runtime)? ###"
PYTHONPATH="$NIXL_PREFIX" UCX_TLS=all UCX_NET_DEVICES=all python -c "
from nixl._api import nixl_agent, nixl_agent_config
a = nixl_agent('probe', nixl_agent_config(backends=['UCX']))
print('agent created; plugins =', a.get_plugin_list())
print('AGENT_OK')
" 2>&1 | tail -25

echo
echo "### 8. what does vLLM's NixlConnector expect in the env? ###"
python - <<'PY' 2>&1 | head -40
import inspect, re
try:
    from vllm.distributed.kv_transfer.kv_connector.v1 import nixl_connector as nc
    src = inspect.getsource(nc)
    print('nixl_connector.py at', nc.__file__, len(src.splitlines()), 'lines')
    for k in sorted(set(re.findall(r'VLLM_[A-Z0-9_]+', src))):
        print('  env:', k)
    for k in sorted(set(re.findall(r'kv_load_failure_policy|do_remote_decode|do_remote_prefill|remote_engine_id|remote_block_ids|remote_host|remote_port|tp_size', src))):
        print('  key:', k)
except Exception as e:
    print('could not introspect:', type(e).__name__, e)
PY
echo
echo "### DONE ###"
