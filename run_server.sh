#!/usr/bin/env bash
# Disaggregated prefill/decode vLLM pair + the toy proxy in front of them.
# Everything is configured through the env vars below, e.g.
#   PREFILL_GPU=0 DECODE_GPU=1 ./run_server.sh
set -euo pipefail

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-8B Qwen3-8B}"  # space-separated aliases
PREFILL_MODEL_NAME="${PREFILL_MODEL_NAME:-Qwen/Qwen3-8B}"
DECODE_MODEL_NAME="${DECODE_MODEL_NAME:-Qwen/Qwen3-8B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-8B}"

PREFILL_GPU="${PREFILL_GPU:-5}"
DECODE_GPU="${DECODE_GPU:-6}"
PREFILL_PORT="${PREFILL_PORT:-8500}"
DECODE_PORT="${DECODE_PORT:-8600}"
PROXY_PORT="${PROXY_PORT:-8595}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LENGTH}"
HF_OVERRIDES="${HF_OVERRIDES:-}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.25}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.25}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1200}"

export VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE="${VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE:-2200000}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export UCX_NET_DEVICES=all

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enforce_handshake_compat":false}}'
PROXY_SERVER="${PROXY_SERVER:-$(dirname "${BASH_SOURCE[0]}")/toy_proxy_server.py}"

read -r -a SERVED_MODEL_NAME_ARGS <<<"$SERVED_MODEL_NAME"
EXTRA_ARGS=()
if [[ -n "$VLLM_DTYPE" ]]; then EXTRA_ARGS+=(--dtype "$VLLM_DTYPE"); fi
if [[ -n "$HF_OVERRIDES" ]]; then EXTRA_ARGS+=(--hf-overrides "$HF_OVERRIDES"); fi

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

# wait_for <name> <url>: poll until the endpoint answers, or a child dies.
wait_for() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    if curl -sf "$2" >/dev/null; then echo "$1 ready"; return 0; fi
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "$1 failed: pid $pid died" >&2; return 1; }
    done
    sleep 1
  done
  echo "$1 failed: timeout" >&2
  return 1
}

# serve <role> <model> <gpu> <port> <nixl_port> <gpu_mem_util> [extra vllm args...]
serve() {
  local role=$1 model=$2 gpu=$3 port=$4 nixl=$5 mem=$6
  shift 6
  echo "Starting $role: $model"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl" \
  vllm serve "$model" \
    --port "$port" \
    --served-model-name "${SERVED_MODEL_NAME_ARGS[@]}" \
    --tokenizer "$TOKENIZER_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LENGTH" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$mem" \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$KV_CONFIG" \
    "${EXTRA_ARGS[@]}" "$@" &
  PIDS+=("$!")
  wait_for "$role" "http://localhost:$port/health"
}

serve prefill "$PREFILL_MODEL_NAME" "$PREFILL_GPU" "$PREFILL_PORT" \
  "${PREFILL_NIXL_SIDE_CHANNEL_PORT:-5610}" "$PREFILL_GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --allow-deprecated-quantization

serve decode "$DECODE_MODEL_NAME" "$DECODE_GPU" "$DECODE_PORT" \
  "${DECODE_NIXL_SIDE_CHANNEL_PORT:-5611}" "$DECODE_GPU_MEMORY_UTILIZATION"

echo "Starting proxy on port $PROXY_PORT"
python "$PROXY_SERVER" \
  --port "$PROXY_PORT" \
  --prefiller-hosts localhost --prefiller-ports "$PREFILL_PORT" \
  --decoder-hosts localhost --decoder-ports "$DECODE_PORT" &
PIDS+=("$!")
wait_for proxy "http://localhost:$PROXY_PORT/healthcheck"

echo "prefill, decode, proxy all success"
wait
