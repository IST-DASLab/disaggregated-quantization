#!/bin/bash
# Why does NIXL's UCX backend refuse to register GPU memory, and what fixes it?
#
# With UCX_NET_DEVICES=all the backend now constructs, but vLLM dies one step later
# in register_kv_caches:
#   W ucx_utils.cpp:622] 8 NVIDIA GPU(s) were detected, but UCX CUDA support was not
#                        found! GPU memory is not supported.
#   E ucx_utils.cpp:576] VRAM memory is detected as host by UCX. UCX is likely not
#                        configured with CUDA/ROCm support. VRAM registration cannot
#                        proceed.
#   nixl_cu13._bindings.nixlBackendError: NIXL_ERR_BACKEND
#
# The wheel DOES ship a cuda transport (nixl_cu13.libs/ucx/libuct_cuda.so.0.0.0), so
# this is a module-discovery/loading problem, not a missing feature. This script runs
# the same minimal test -- create a UCX-backed nixl agent, register a CUDA tensor --
# under several candidate environments and reports which ones work.
#
# FINDING (job 476932): none of the variants below fix it. The UCX_LOG_LEVEL=debug
# run is the one that explains why:
#     module.c:72   ucs library path: /opt/hpcx/ucx/lib/libucs.so.0
#     module.c:162  ignoring 'ucs_module_global_init' (...) from libuct_cuda.so.0,
#                   expected in libuct_cuda_gdrcopy.so.0
#     ucp_context.c:2073  no memory domain supports registering cuda memory
# Every UCX debug line is printed TWICE: there are two UCX cores live in the process
# -- the container's HPCX 2.24.1 (/opt/hpcx/ucx, pulled in independently of nixl) and
# the one vendored inside the nixl-cu13 wheel under mangled sonames
# (libucp-1da16952.so.0.0.0 etc.). The cuda transport registers its memory domain
# into one core's component registry while the ucp_context that nixl uses belongs to
# the other, so no CUDA memory domain is visible and VRAM looks like host memory.
# Pointing UCX_MODULE_DIR at either tree only makes it worse (variants B/D hang or
# segfault in ucs_topo_cleanup, because then one core loads the other's modules).
#
# WORKAROUND IN USE: NIXL_KV_BUFFER_DEVICE=cpu in run_nixl_server.sh, which stages
# the KV through host memory and needs no CUDA support in UCX at all.
# A real fix would mean a nixl built against the container's HPCX UCX rather than the
# PyPI wheel's vendored copy.
#
#   ./diagnostics/run_nixl_ucx_diag.sh

#SBATCH --job-name=nixl-ucx-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --account=adlr_psx_numerics

CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
HF_CACHE=/lustre/fsw/portfolios/adlr/users/apanferov/hf_cache
NIXL_PREFIX="${NIXL_PREFIX:-/lustre/fsw/portfolios/adlr/users/apanferov/prefill-decode/nixl_nodeps}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    LOGS="$(dirname "$(dirname "$(dirname "$SELF")")")/logs/checks"   # diagnostics/ -> qad -> repo
    mkdir -p "$LOGS"
    export NIXL_PREFIX
    exec sbatch --export=ALL \
        --output="$LOGS/nixlucx_%j.out" --error="$LOGS/nixlucx_%j.err" "$SELF"
fi

if command -v scontrol &>/dev/null && [ -z "${NIXL_IN_CONTAINER:-}" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")   # qad/diagnostics
    export QAD_DIR=$(dirname "$SCRIPT_DIR")       # qad
    export HF_CACHE NIXL_PREFIX NIXL_IN_CONTAINER=1
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/lustre:/lustre,$HOME/.netrc:/root/.netrc" --export=ALL \
        bash "$SCRIPT_PATH"
    exit $?
fi

export HF_HOME=$HF_CACHE
LIBS="$NIXL_PREFIX/nixl_cu13.libs"

echo "### container UCX-related environment (inherited) ###"
env | grep -i -E '^(UCX|HPCX|OMPI|LD_LIBRARY_PATH)' | sort

echo
echo "### is there another UCX in the container? ###"
ls -d /opt/hpcx/ucx 2>/dev/null && ls /opt/hpcx/ucx/lib/ucx 2>/dev/null | head
find /usr /opt -name 'libuct_cuda*' -maxdepth 6 2>/dev/null | head

echo
echo "### driver libs the cuda transport needs ###"
for l in libcuda.so.1 libnvidia-ml.so.1; do
    echo -n "$l: "; ldconfig -p | grep -m1 "$l" || echo MISSING
done

echo
echo "### does libuct_cuda.so actually load? ###"
python - <<PY
import ctypes
for p in ["$LIBS/libucs-3bed3ff8.so.0.0.0", "$LIBS/ucx/libuct_cuda.so.0.0.0"]:
    try:
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        print("loaded OK:", p)
    except OSError as e:
        print("FAILED  :", p, "->", e)
PY

# The actual test, parameterised by environment.
cat > /tmp/nixl_vram_test.py <<'PY'
import os, sys
import torch
from nixl._api import nixl_agent, nixl_agent_config
t = torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda:0")
a = nixl_agent("vramtest", nixl_agent_config(backends=["UCX"]))
try:
    reg = a.register_memory([(t.data_ptr(), t.numel(), 0, "")], "VRAM")
    print("VRAM_REGISTER_OK", reg is not None)
except Exception as e:
    print("VRAM_REGISTER_FAIL", type(e).__name__, e)
    sys.exit(1)
PY

run_variant() {   # label, then env assignments as extra args
    local label="$1"; shift
    echo
    echo "=========== VARIANT: $label ==========="
    echo "  env: $*"
    # Hard timeout: some of these configurations do not fail, they HANG (UCX probing
    # devices that are not there), and a hang would eat the whole allocation.
    ( export PYTHONPATH="$NIXL_PREFIX" UCX_TLS=all UCX_NET_DEVICES=all
      timeout -s KILL 120 env "$@" python /tmp/nixl_vram_test.py 2>&1 \
        | grep -v FutureWarning | tail -25
      echo "  (variant exit: ${PIPESTATUS[0]}; 137 = killed by the 120s timeout)" )
}

run_variant "A baseline (UCX_TLS=all, UCX_NET_DEVICES=all)" X=1
run_variant "B UCX_MODULE_DIR -> wheel's ucx plugin dir" "UCX_MODULE_DIR=$LIBS/ucx"
run_variant "C LD_LIBRARY_PATH prepends wheel libs" "LD_LIBRARY_PATH=$LIBS:$LIBS/ucx:${LD_LIBRARY_PATH:-}"
run_variant "D B+C" "UCX_MODULE_DIR=$LIBS/ucx" "LD_LIBRARY_PATH=$LIBS:$LIBS/ucx:${LD_LIBRARY_PATH:-}"
run_variant "E UCX_TLS=cuda_copy,cuda_ipc,tcp + module dir" \
    "UCX_MODULE_DIR=$LIBS/ucx" "UCX_TLS=cuda_copy,cuda_ipc,tcp"
# F is a long shot: the container's HPCX UCX 2.24.1 does have CUDA modules, but they
# link the UNMANGLED libuct.so.0 while the wheel ships a mangled libuct-882d6e12.so,
# so this would put two UCT cores in one process. Worth one datapoint, not a plan.
run_variant "F UCX_MODULE_DIR -> container HPCX ucx modules" \
    "UCX_MODULE_DIR=/opt/hpcx/ucx/lib/ucx"
# G: same as A but restricted to the one network device that actually exists, in
# case 'all' is what makes UCX probe forever.
run_variant "G UCX_NET_DEVICES=eth0" "UCX_NET_DEVICES=eth0"

echo
echo "### UCX debug log for the baseline variant (module loading) ###"
( export PYTHONPATH="$NIXL_PREFIX" UCX_TLS=all UCX_NET_DEVICES=all UCX_LOG_LEVEL=debug
  timeout -s KILL 180 python /tmp/nixl_vram_test.py 2>&1 \
    | grep -i -E "module|cuda|memtype|md " | head -60 )

echo "### DONE ###"
