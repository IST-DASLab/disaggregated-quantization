#!/bin/bash
# End-to-end Nixl 1P1D smoke test: bring up run_nixl_server.sh, push a handful of
# greedy completions through the proxy, then dump the evidence.
#
#   ./diagnostics/run_nixl_1p1d.sh                 # homogeneous: W4A16 on both sides
#   ./diagnostics/run_nixl_1p1d.sh --mode hetero   # W4A4 prefill -> W4A16 decode
#
# Homogeneous first on purpose: it separates "the plumbing works" from "the two
# checkpoints differ", so a failure has one cause instead of two.
#
# Everything lands in ROOT/logs/checks/nixl1p1d_<jobid>/ -- /tmp is node-local and
# invisible once the allocation ends.

#SBATCH --job-name=nixl-1p1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad

CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/nemo-26.02.sqsh
HF_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache
NIXL_PREFIX="${NIXL_PREFIX:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/nixl_nodeps}"

NIXL_MODE="${NIXL_MODE:-homo}"          # homo | hetero
NIXL_STEP="${NIXL_STEP:-0002450}"
NIXL_A4_RUN="${NIXL_A4_RUN:-qad3x-Qwen-Qwen3-0.6B-nvfp4-99914b93}"
NIXL_A16_RUN="${NIXL_A16_RUN:-qad3x-Qwen-Qwen3-0.6B-nvfp4a16-99914b93}"
NIXL_TOKENIZER="${NIXL_TOKENIZER:-Qwen/Qwen3-0.6B}"
NIXL_PROBE_LIMIT="${NIXL_PROBE_LIMIT:-2}"
# cuda | cpu -- see the kv_buffer_device comment in run_nixl_server.sh
NIXL_KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cpu}"

while (($# > 0)); do
  case "$1" in
    --mode)      NIXL_MODE="$2";      shift 2 ;;
    --step)      NIXL_STEP="$2";      shift 2 ;;
    --tokenizer) NIXL_TOKENIZER="$2"; shift 2 ;;
    --limit)     NIXL_PROBE_LIMIT="$2"; shift 2 ;;
    --kv-buffer-device) NIXL_KV_BUFFER_DEVICE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# LOGIN mode: submit
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    LOGS="$(dirname "$(dirname "$(dirname "$SELF")")")/logs/checks"   # diagnostics/ -> qad -> repo
    mkdir -p "$LOGS"
    # Values go through the environment, never through --export=ALL,VAR=...: sbatch
    # splits that list on commas and silently corrupts anything containing one.
    export NIXL_MODE NIXL_STEP NIXL_A4_RUN NIXL_A16_RUN NIXL_TOKENIZER \
           NIXL_PROBE_LIMIT NIXL_PREFIX NIXL_KV_BUFFER_DEVICE NIXL_ATTENTION_BACKEND NIXL_ENFORCE_COMPAT
    exec sbatch --export=ALL \
        --job-name="nixl-1p1d-$NIXL_MODE" \
        --output="$LOGS/nixl1p1d_%j.out" --error="$LOGS/nixl1p1d_%j.err" \
        "$SELF"
fi

# ---------------------------------------------------------------------------
# HOST mode: re-invoke inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null && [ -z "${NIXL_IN_CONTAINER:-}" ]; then
    # `exit` after the first match is REQUIRED: scontrol prints several records and a
    # multi-line path kills the task with exit 127.
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")   # qad/diagnostics
    export QAD_DIR=$(dirname "$SCRIPT_DIR")       # qad
    export HF_CACHE NIXL_PREFIX NIXL_IN_CONTAINER=1
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/scratch:/scratch,/lustre:/lustre,$HOME/.netrc:/root/.netrc" --export=ALL \
        bash "$SCRIPT_PATH"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode
# ---------------------------------------------------------------------------
set -uo pipefail
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=0
# NOTE: the lm_eval overlay is deliberately NOT on PYTHONPATH anywhere in this
# script. The probe needs only requests + transformers, both in the container, and
# the overlay's huggingface-hub 1.24.0 would make `vllm serve` refuse to start.
export PYTHONPATH="$QAD_DIR"

CKPT="$QAD_DIR/checkpoints"
A4="$CKPT/$NIXL_A4_RUN/weights/step_$NIXL_STEP"
A16="$CKPT/$NIXL_A16_RUN/weights/step_$NIXL_STEP"
case "$NIXL_MODE" in
  homo)   PREFILL="$A16"; DECODE="$A16" ;;
  hetero) PREFILL="$A4";  DECODE="$A16" ;;
  # Third arm for the A/B/C comparison (see verify_kv_transfer.py): the hetero pair
  # must differ from BOTH homogeneous stacks. Matching `homo` would mean the decode
  # engine recomputed the prompt with its own weights, i.e. the KV never crossed;
  # matching `homo_a4` would mean generation is not running on the decode weights.
  homo_a4) PREFILL="$A4"; DECODE="$A4" ;;
  *) echo "ERROR: --mode must be homo, hetero or homo_a4" >&2; exit 2 ;;
esac
for d in "$PREFILL" "$DECODE"; do
  [ -d "$d" ] || { echo "ERROR: missing checkpoint $d" >&2; exit 1; }
done

WORK="$(dirname "$QAD_DIR")/logs/checks/nixl1p1d_${SLURM_JOB_ID}_${NIXL_MODE}"
mkdir -p "$WORK"
echo "[1p1d] mode=$NIXL_MODE"
echo "[1p1d] prefill=$PREFILL"
echo "[1p1d] decode =$DECODE"
echo "[1p1d] work=$WORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Ports from the OS, never arithmetic: derived ports collide on a shared node and a
# collision on a side-channel port shows up as an opaque bind error inside the engine.
read -r PREFILL_PORT DECODE_PORT PROXY_PORT PREFILL_SIDE_PORT DECODE_SIDE_PORT <<<"$(python - <<'PY'
import socket
socks = []
for _ in range(5):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 0))          # 0.0.0.0, not loopback: these get bound on the
    socks.append(s)                 # node's interface address
print(" ".join(str(s.getsockname()[1]) for s in socks))
for s in socks:
    s.close()
PY
)"
export PREFILL_PORT DECODE_PORT PROXY_PORT PREFILL_SIDE_PORT DECODE_SIDE_PORT
export LOG_DIR="$WORK"
echo "[1p1d] ports: prefill=$PREFILL_PORT decode=$DECODE_PORT proxy=$PROXY_PORT side=$PREFILL_SIDE_PORT/$DECODE_SIDE_PORT"

READY="$WORK/ready"
rm -f "$READY"
# New session so the whole tree (vllm spawns children) dies with one kill; otherwise
# two engines keep the GPUs for the rest of the allocation.
setsid bash "$QAD_DIR/serving/run_nixl_server.sh" \
    --prefill-model "$PREFILL" --decode-model "$DECODE" \
    --tokenizer "$NIXL_TOKENIZER" --ready-file "$READY" \
    >"$WORK/stack.log" 2>&1 &
STACK_PID=$!
cleanup() { kill -- -"$STACK_PID" 2>/dev/null; kill "$STACK_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

DEADLINE=$((SECONDS + 1500))
while [ ! -f "$READY" ]; do
    if ! kill -0 "$STACK_PID" 2>/dev/null; then
        echo "[1p1d] STACK DIED before ready" >&2; break
    fi
    if [ "$SECONDS" -gt "$DEADLINE" ]; then
        echo "[1p1d] TIMEOUT waiting for stack" >&2; break
    fi
    sleep 5
done

RC=1
if [ -f "$READY" ]; then
    echo "[1p1d] stack ready on proxy port $(cat "$READY")"
    python "$SCRIPT_DIR/nixl_probe.py" --port "$(cat "$READY")" \
        --tokenizer "$NIXL_TOKENIZER" --out "$WORK/probe.json" \
        --limit "$NIXL_PROBE_LIMIT"
    RC=$?
fi

echo
echo "================ stack.log (tail) ================"; tail -40 "$WORK/stack.log"
echo "================ proxy.log ======================="; tail -60 "$WORK/proxy.log" 2>/dev/null
echo "================ prefill.log (tail) =============="; tail -60 "$WORK/prefill.log" 2>/dev/null
echo "================ decode.log (tail) ==============="; tail -60 "$WORK/decode.log" 2>/dev/null
echo
echo "================ NIXL evidence grep =============="
# These are the lines that distinguish "connector actually engaged" from "engine
# started and quietly ignored the config".
grep -n -i -E "NIXL|nixl_connector|side channel|kv_transfer|kv_load_failure|Handshake|remote_engine|Error|Traceback" \
    "$WORK/prefill.log" "$WORK/decode.log" 2>/dev/null | head -80

echo "[1p1d] probe rc=$RC"
exit $RC
