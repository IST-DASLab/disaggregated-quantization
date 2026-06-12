#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=4,5

PIDS=()

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-9B Qwen3.5-9B}"

# PREFILL_MODEL_NAME="${PREFILL_MODEL_NAME:-/nfs/scistore19/alistgrp/mkleineg/MatGPTQ-dev/EvoPress-matgptq/.tmp/Qwen3-8B-8bit}"
# DECODE_MODEL_NAME="${DECODE_MODEL_NAME:-/nfs/scistore19/alistgrp/mkleineg/MatGPTQ-dev/EvoPress-matgptq/.tmp/Qwen3-8B-4bit}"
# TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3-8B}"

#PREFILL_MODEL_NAME="${PREFILL_MODEL_NAME:-RedHatAI/Qwen3.5-9B-FP8-dynamic}"
#DECODE_MODEL_NAME="${DECODE_MODEL_NAME:-RedHatAI/Qwen3.5-9B-quantized.w4a16}"
PREFILL_MODEL_NAME="${PREFILL_MODEL_NAME:-Qwen/Qwen3.5-9B}"
DECODE_MODEL_NAME="${DECODE_MODEL_NAME:-Qwen/Qwen3.5-9B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-Qwen/Qwen3.5-9B}"

PREFILL_GPU="${PREFILL_GPU:-4}"
DECODE_GPU="${DECODE_GPU:-5}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-112640}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
HF_OVERRIDES="${HF_OVERRIDES:-{\"rope_parameters\":{\"rope_type\":\"yarn\",\"rope_theta\":1000000,\"factor\":4.0,\"original_max_position_embeddings\":32768}}}"
VLLM_DTYPE="${VLLM_DTYPE:-}"

PREFILL_PORT="${PREFILL_PORT:-8500}"
DECODE_PORT="${DECODE_PORT:-8600}"
PROXY_PORT="${PROXY_PORT:-8595}"
PREFILL_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_CHANNEL_PORT:-5610}"
DECODE_NIXL_SIDE_CHANNEL_PORT="${DECODE_NIXL_SIDE_CHANNEL_PORT:-5611}"

PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.25}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.25}"
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE="${VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE:-2200000}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1200}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# PROXY_SERVER="${PROXY_SERVER:-/nfs/scistore19/alistgrp/mkleineg/vllm-sandbox/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py}"
PROXY_SERVER="${PROXY_SERVER:-./toy_proxy_server.py}"
READY_FILE="${READY_FILE:-}"

# module load cuda/13

# source /nfs/scistore19/alistgrp/mkleineg/vllm-sandbox/.venv/bin/activate

usage() {
  cat <<'EOF'
Usage:
  ./run_server_qwen3.sh [options]

Options:
  --prefill-model-name <name>   model used by the prefill server.
  --decode-model-name <name>    model used by the decode server.
  --served-model-name <name>    served model name. Quote a space-separated list
                                to register aliases, e.g. "Qwen/Qwen3-8B Qwen3-8B".
  --tokenizer <name>            tokenizer name.
  --prefill-gpu <ids>           CUDA_VISIBLE_DEVICES for prefill.
  --decode-gpu <ids>            CUDA_VISIBLE_DEVICES for decode.
  --tensor-parallel-size <n>    vLLM tensor parallel size.
  --max-model-length <n>        vLLM max model length.
  --max-num-seqs <n>            max number of sequences for prefill.
  --max-num-batched-tokens <n>  max batched tokens for prefill.
  --dtype <dtype>               optional vLLM dtype, for example float16.
  --hf-overrides <json>         optional vLLM HF overrides JSON. Use '' to disable.
  --proxy-port <port>           proxy server port.
  -h, --help                    show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --prefill-model-name) PREFILL_MODEL_NAME="${2:?missing value for --prefill-model-name}"; shift 2 ;;
    --decode-model-name) DECODE_MODEL_NAME="${2:?missing value for --decode-model-name}"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="${2:?missing value for --served-model-name}"; shift 2 ;;
    --tokenizer) TOKENIZER_NAME="${2:?missing value for --tokenizer}"; shift 2 ;;
    --prefill-gpu) PREFILL_GPU="${2:?missing value for --prefill-gpu}"; shift 2 ;;
    --decode-gpu) DECODE_GPU="${2:?missing value for --decode-gpu}"; shift 2 ;;
    --tensor-parallel-size) TENSOR_PARALLEL_SIZE="${2:?missing value for --tensor-parallel-size}"; shift 2 ;;
    --max-model-length) MAX_MODEL_LENGTH="${2:?missing value for --max-model-length}"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="${2:?missing value for --max-num-seqs}"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="${2:?missing value for --max-num-batched-tokens}"; shift 2 ;;
    --dtype) VLLM_DTYPE="${2:?missing value for --dtype}"; shift 2 ;;
    --hf-overrides) HF_OVERRIDES="${2-}"; shift 2 ;;
    --proxy-port) PROXY_PORT="${2:?missing value for --proxy-port}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MAX_NUM_BATCHED_TOKENS" ]]; then
  MAX_NUM_BATCHED_TOKENS="$MAX_MODEL_LENGTH"
fi

cleanup() {
  if ((${#PIDS[@]} > 0)); then
    kill "${PIDS[@]}" >/dev/null 2>&1 || true
    wait "${PIDS[@]}" >/dev/null 2>&1 || true
  fi
  if [[ -n "$READY_FILE" ]]; then
    rm -f "$READY_FILE"
  fi
}
trap cleanup EXIT

wait_for_server() {
  local name=$1
  local port=$2

  for _ in $(seq 1 "$SERVER_READY_TIMEOUT_S"); do
    if wget -qO- \
      --header="Content-Type: application/json" \
      --post-data='{"model":"Qwen3.5-9B","prompt":"hi","max_tokens":1}' \
      "http://localhost:${port}/v1/completions" \
      >/dev/null 2>&1; then
      echo "${name} success"
      return 0
    fi

    for pid in "${PIDS[@]}"; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        echo "${name} failed"
        return 1
      fi
    done

    sleep 1
  done

  echo "${name} failed"
  return 1
}

DTYPE_ARGS=()
if [[ -n "$VLLM_DTYPE" ]]; then
  DTYPE_ARGS=(--dtype "$VLLM_DTYPE")
fi

HF_OVERRIDE_ARGS=()
if [[ -n "$HF_OVERRIDES" ]]; then
  HF_OVERRIDE_ARGS=(--hf-overrides "$HF_OVERRIDES")
fi

read -r -a SERVED_MODEL_NAME_ARGS <<<"$SERVED_MODEL_NAME"

echo "Starting prefill: ${PREFILL_MODEL_NAME}"
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE="$VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE" \
VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S" \
CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$PREFILL_MODEL_NAME" \
  --port "$PREFILL_PORT" \
  --served-model-name "${SERVED_MODEL_NAME_ARGS[@]}" \
  --tokenizer "$TOKENIZER_NAME" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LENGTH" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
  "${DTYPE_ARGS[@]}" \
  "${HF_OVERRIDE_ARGS[@]}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enforce_handshake_compat":false}}' &
PIDS+=("$!")
wait_for_server "prefill" "$PREFILL_PORT"

echo "Starting decode: ${DECODE_MODEL_NAME}"
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE="$VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE" \
VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S" \
CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$DECODE_MODEL_NAME" \
  --port "$DECODE_PORT" \
  --served-model-name "${SERVED_MODEL_NAME_ARGS[@]}" \
  --tokenizer "$TOKENIZER_NAME" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LENGTH" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
  "${DTYPE_ARGS[@]}" \
  "${HF_OVERRIDE_ARGS[@]}" \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"enforce_handshake_compat":false}}' &
PIDS+=("$!")
wait_for_server "decode" "$DECODE_PORT"

echo "Starting proxy on port ${PROXY_PORT}"
python "$PROXY_SERVER" \
  --port "$PROXY_PORT" \
  --prefiller-hosts localhost \
  --prefiller-ports "$PREFILL_PORT" \
  --decoder-hosts localhost \
  --decoder-ports "$DECODE_PORT" &
PIDS+=("$!")
wait_for_server "proxy" "$PROXY_PORT"

echo "prefill, decode, proxy all success"
if [[ -n "$READY_FILE" ]]; then
  mkdir -p "$(dirname "$READY_FILE")"
  touch "$READY_FILE"
fi

wait