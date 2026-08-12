#!/bin/bash
# Bring up a 1P1D disaggregated vLLM pair + proxy, inside one SLURM allocation.
#
#   prefill server  GPU 0   <step>/prefill   kv_producer   (W4A4)
#   decode  server  GPU 1   <step>/decode    kv_consumer   (W4A16)
#   proxy           CPU     p2p_proxy.py     -> one OpenAI endpoint
#
# Adapted from ../run_server.sh with four changes for this environment:
#   * P2pNcclConnector instead of NixlConnector — `nixl` is NOT installed in the
#     container, so the Nixl config fails at startup. P2P NCCL needs no extra
#     package and both servers share a node.
#   * no `module load` / venv activation: everything runs inside the container.
#   * no --hf-overrides: that yarn-rope override is for long-context Qwen3-8B and is
#     wrong for the models here.
#   * ports derived from a base so several evals can share a node without colliding.
#
# The engines are started WITHOUT proxy_ip/proxy_port, so they run no registration
# ping thread and `http_port` is not required: p2p_proxy.py encodes both ZMQ
# addresses directly in the request id (see its module docstring).
#
#   ./run_disagg_server.sh --prefill-model DIR --decode-model DIR \
#        --tokenizer Qwen/Qwen3-0.6B --ready-file /path/ready
#
# Serves until killed. Writes --ready-file once all three are answering.

set -uo pipefail

PREFILL_MODEL=""; DECODE_MODEL=""; TOKENIZER=""
SERVED_NAME="${SERVED_NAME:-model}"
PORT_BASE="${PORT_BASE:-8500}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
READY_FILE=""
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
# Async scheduling runs the scheduler ahead of the executing step. A KV connector's
# metadata and forward context are per-step, and nothing in this vLLM build disables
# async scheduling when a connector is configured (the only mention of it in config is
# a log line). With P2pNcclConnector the consumer then waits in recv_tensor() on an
# unbounded condition variable and the decode engine hangs with no error at all. Off
# by default here; set DISAGG_ASYNC_SCHED=1 to restore vLLM's choice.
# vLLM's own connector docs run both disaggregated instances with --enforce-eager.
# With CUDA graphs the decode step is captured while start_load_kv writes the borrowed
# KV from Python each step; the transferred tensors then arrive intact (verified by
# matching checksums on both ends) yet the output is garbage. Eager mode removes that
# interaction. DISAGG_CUDA_GRAPHS=1 to re-enable capture.
EAGER_FLAG="--enforce-eager"
[ -n "${DISAGG_CUDA_GRAPHS:-}" ] && EAGER_FLAG=""

ASYNC_SCHED_FLAG="--no-async-scheduling"
[ -n "${DISAGG_ASYNC_SCHED:-}" ] && ASYNC_SCHED_FLAG=""

# qad/p2p_trace.py is REQUIRED, not optional. Stock P2pNcclConnector cannot work in
# this vLLM build: the engine appends a per-server random salt to the request id, so
# producer and consumer key the same KV under different names and the decode engine
# blocks forever in recv_tensor() with no error. The subclass normalizes that key.
# Uses vLLM's kv_connector_module_path hook, so no vLLM file is patched.
# DISAGG_STOCK_CONNECTOR=1 restores the broken stock connector (to re-demonstrate the
# bug); DISAGG_TRACE_VERBOSE=1 additionally prints every key (grep TRACE_KV).
KV_CONNECTOR="P2pNcclConnectorStableId"
# Plain JSON, no backslashes: the surrounding literal's \" already yield real quotes,
# and expansion does not re-process escapes.
KV_MODULE_JSON=',"kv_connector_module_path":"p2p_trace"'
if [ -n "${DISAGG_STOCK_CONNECTOR:-}" ]; then
  KV_CONNECTOR="P2pNcclConnector"; KV_MODULE_JSON=""
  echo "[disagg] WARNING: stock connector - expect the decode engine to hang"
fi
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):${PYTHONPATH:-}"
[ -n "${DISAGG_TRACE_VERBOSE:-}" ] && export DISAGG_TRACE_VERBOSE

NCCL_CHANNELS="${DISAGG_NCCL_CHANNELS:-16}"
PRODUCER_BUF="${DISAGG_PRODUCER_BUF:-1e1}"

while (($# > 0)); do
  case "$1" in
    --prefill-model) PREFILL_MODEL="$2"; shift 2 ;;
    --decode-model)  DECODE_MODEL="$2";  shift 2 ;;
    --tokenizer)     TOKENIZER="$2";     shift 2 ;;
    --served-name)   SERVED_NAME="$2";   shift 2 ;;
    --port-base)     PORT_BASE="$2";     shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --ready-file)    READY_FILE="$2";    shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -z "$PREFILL_MODEL" ] && { echo "ERROR: --prefill-model required" >&2; exit 2; }
[ -z "$DECODE_MODEL" ]  && { echo "ERROR: --decode-model required" >&2; exit 2; }
[ -z "$TOKENIZER" ]     && { echo "ERROR: --tokenizer required" >&2; exit 2; }

# Ports may be supplied explicitly (eval_disagg.py binds them first to prove they are
# free) or derived from PORT_BASE as a fallback for manual runs. Derivation alone is
# not safe on a shared node: the KV ports are ZMQ binds, and a collision surfaces as
# "Address already in use" inside the engine core, which reads like a vLLM bug.
PREFILL_PORT="${PREFILL_PORT:-$((PORT_BASE + 0))}"
DECODE_PORT="${DECODE_PORT:-$((PORT_BASE + 1))}"
PROXY_PORT="${PROXY_PORT:-$((PORT_BASE + 2))}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-$((PORT_BASE + 100))}"
DECODE_KV_PORT="${DECODE_KV_PORT:-$((PORT_BASE + 200))}"

# The KV addresses the proxy advertises must be byte-identical to what the engines
# bind. P2pNcclEngine binds its ZMQ router to `get_ip():kv_port`, and vllm's get_ip()
# prefers $VLLM_HOST_IP and otherwise probes a route -- a real interface address, never
# loopback. If the proxy says 127.0.0.1 while the engine bound to 10.x.y.z, the
# producer connects to nothing and the request hangs with no error anywhere. So pin
# VLLM_HOST_IP here and hand the SAME string to the proxy; then they cannot disagree.
KV_HOST="${VLLM_HOST_IP:-$(python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))          # no packet is sent; picks the default route
    print(s.getsockname()[0])
finally:
    s.close()
PY
)}"
[ -z "$KV_HOST" ] && { echo "ERROR: could not determine KV host IP" >&2; exit 1; }
export VLLM_HOST_IP="$KV_HOST"

# DISAGG_DEBUG=1 turns on vLLM debug logging, which is the only way to see the
# connector's `tensor_id` values. Producer and consumer must agree on
# `<request_id>#<layer_name>` exactly; if they do not, recv_tensor() waits on a
# condition variable with NO timeout and the decode engine blocks forever with
# nothing logged. Comparing the two sides' ids is what turns that silence into a
# diagnosis. Off by default: it is very verbose.
if [ -n "${DISAGG_DEBUG:-}" ]; then
  export VLLM_LOGGING_LEVEL=DEBUG
  echo "[disagg] DEBUG logging enabled (tensor_id tracing)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS=()
cleanup() {
  ((${#PIDS[@]})) && kill "${PIDS[@]}" >/dev/null 2>&1
  ((${#PIDS[@]})) && wait "${PIDS[@]}" >/dev/null 2>&1
  [ -n "$READY_FILE" ] && rm -f "$READY_FILE"
}
trap cleanup EXIT INT TERM

wait_for() {   # name port -- poll /v1/models, and fail fast if a server died
  local name=$1 port=$2
  for _ in $(seq 1 "$SERVER_READY_TIMEOUT_S"); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[disagg] ${name} ready on ${port}"; return 0
    fi
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "[disagg] ${name} DIED before ready" >&2; return 1; }
    done
    sleep 1
  done
  echo "[disagg] ${name} timed out after ${SERVER_READY_TIMEOUT_S}s" >&2; return 1
}

echo "[disagg] prefill=$PREFILL_MODEL"
echo "[disagg] decode =$DECODE_MODEL"
echo "[disagg] ports: prefill=$PREFILL_PORT decode=$DECODE_PORT proxy=$PROXY_PORT" \
     "kv=$PREFILL_KV_PORT/$DECODE_KV_PORT on $KV_HOST"

# kv_port is the connector's ZMQ address; the proxy puts it in the request id.
CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
vllm serve "$PREFILL_MODEL" \
  --port "$PREFILL_PORT" --served-model-name "$SERVED_NAME" \
  --tokenizer "$TOKENIZER" --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-request-id-headers ${ASYNC_SCHED_FLAG} ${EAGER_FLAG} \
  --kv-transfer-config "{\"kv_connector\":\"${KV_CONNECTOR}\"${KV_MODULE_JSON},\"kv_role\":\"kv_producer\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"kv_buffer_size\":\"${PRODUCER_BUF}\",\"kv_connector_extra_config\":{\"send_type\":\"PUT_ASYNC\",\"http_port\":\"${PREFILL_PORT}\",\"nccl_num_channels\":\"${NCCL_CHANNELS}\"}}" \
  >"${LOG_DIR:-/tmp}/prefill.log" 2>&1 &
PIDS+=("$!")

CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
vllm serve "$DECODE_MODEL" \
  --port "$DECODE_PORT" --served-model-name "$SERVED_NAME" \
  --tokenizer "$TOKENIZER" --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-request-id-headers ${ASYNC_SCHED_FLAG} ${EAGER_FLAG} \
  --kv-transfer-config "{\"kv_connector\":\"${KV_CONNECTOR}\"${KV_MODULE_JSON},\"kv_role\":\"kv_consumer\",\"kv_port\":\"${DECODE_KV_PORT}\",\"kv_buffer_size\":\"8e9\",\"kv_connector_extra_config\":{\"send_type\":\"PUT_ASYNC\",\"http_port\":\"${DECODE_PORT}\",\"nccl_num_channels\":\"${NCCL_CHANNELS}\"}}" \
  >"${LOG_DIR:-/tmp}/decode.log" 2>&1 &
PIDS+=("$!")

wait_for prefill "$PREFILL_PORT" || exit 1
wait_for decode  "$DECODE_PORT"  || exit 1

python "$SCRIPT_DIR/p2p_proxy.py" \
  --port "$PROXY_PORT" \
  --prefill-host 127.0.0.1 --prefill-port "$PREFILL_PORT" --prefill-kv-port "$PREFILL_KV_PORT" \
  --decode-host  127.0.0.1 --decode-port  "$DECODE_PORT"  --decode-kv-port  "$DECODE_KV_PORT" \
  --kv-host "$KV_HOST" \
  >"${LOG_DIR:-/tmp}/proxy.log" 2>&1 &
PIDS+=("$!")
wait_for proxy "$PROXY_PORT" || exit 1

echo "[disagg] all up; proxy on $PROXY_PORT"
if [ -n "$READY_FILE" ]; then mkdir -p "$(dirname "$READY_FILE")"; echo "$PROXY_PORT" > "$READY_FILE"; fi
wait
