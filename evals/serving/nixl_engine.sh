#!/bin/bash
# ONE engine of a disaggregated pair, spread over N nodes. Launched with one task per
# node by run_eval.sh, once for the prefill engine and once for the decode engine:
#
#   ROLE=kv_producer srun --nodelist=<prefill nodes> --ntasks-per-node=1 nixl_engine.sh
#   ROLE=kv_consumer srun --nodelist=<decode  nodes> --ntasks-per-node=1 nixl_engine.sh
#
# It is the multi-node counterpart of run_nixl_server.sh, which places both engines on one
# host and so cannot serve a model that needs more than one node per engine. The
# per-engine settings are the same; what differs is that each engine is itself a
# multi-node vLLM instance, so every rank above 0 within an engine needs --headless.
#
# NODE_RANK IS WITHIN THE ENGINE, not within the job. The srun step for each engine gets
# its own --nodelist, so SLURM_NODEID counts 0..NNODES-1 inside that step, which is
# exactly what vLLM wants. Deriving it from the job's node list instead would give the
# decode engine ranks NNODES..2*NNODES-1 and it would wait forever for a rank 0 that
# belongs to the other engine.
set -uo pipefail

ROLE=${ROLE:?ROLE must be kv_producer or kv_consumer}
ENGINE=${ENGINE:-engine}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SERVED_NAME=${SERVED_NAME:-model}
TP=${TP:?TP is required}
NNODES=${NNODES:?NNODES is required}
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR is required}
MASTER_PORT=${MASTER_PORT:-29501}
PORT=${PORT:-8000}
SIDE_PORT=${SIDE_PORT:-5600}
KV_HOST=${KV_HOST:?KV_HOST is required}
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-cuda}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}
LOG_DIR=${LOG_DIR:-/tmp}
NIXL_PREFIX=${NIXL_PREFIX:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/nixl_nodeps}

NODE_RANK=${SLURM_NODEID:-0}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${ENGINE}_rank${NODE_RANK}.log"

export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-eth0}}

# NVLINK, NOT RDMA. The site default is UCX_TLS=tcp; cluster support confirmed it may be
# overridden per job and that KV must go over NVLink here, because RDMA hits a 2 MB UAR
# BAR limit (NVBug 6261319, PF_LOG_BAR_SIZE=1) that a KV transfer overruns:
#   mlx5dv_devx_alloc_uar(device=mlx5_0, type=WC) failed: Cannot allocate memory
# Only racks from the recent scale-up carry the 256 MB setting. The verbs transports
# (rc/ud/dc) are deliberately absent so a transfer that cannot use NVLink fails rather
# than silently falling back onto the broken path.
export UCX_TLS="${NIXL_UCX_TLS:-cuda_ipc,cuda_copy,sm,self,tcp}"
export UCX_CUDA_IPC_ENABLE_MNNVL="${NIXL_UCX_CUDA_IPC_ENABLE_MNNVL:-y}"

# Hybrid SSM / linear-attention models transfer recurrent state through NixlConnector's
# 3-read conv path, which refuses to start without the DS layout. Inert for pure attention
# models, and Qwen3.5 / Kimi-linear both need it.
export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"

export PYTHONPATH="$NIXL_PREFIX${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import nixl._api" 2>/dev/null || {
    echo "ERROR: nixl not importable with PYTHONPATH=$PYTHONPATH" >&2
    echo "       ./evals/bin/install_overlays.sh --root \$ROOT --only nixl-nodeps" >&2
    exit 1; }

COMPAT=""
if [ "${NIXL_ENFORCE_COMPAT:-1}" = "0" ]; then
    COMPAT=',"kv_connector_extra_config":{"enforce_handshake_compat":false}'
fi
KV_CFG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$ROLE\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\",\"kv_load_failure_policy\":\"fail\"$COMPAT}"

HEADLESS=()
[ "$NODE_RANK" -gt 0 ] && HEADLESS=(--headless)

echo "[$ENGINE] rank=$NODE_RANK/$NNODES host=$(hostname) role=$ROLE tp=$TP" \
     "api=$PORT side=$SIDE_PORT kv_host=$KV_HOST ${HEADLESS[*]:-(serves)}"
echo "[$ENGINE] UCX_TLS=$UCX_TLS MNNVL=$UCX_CUDA_IPC_ENABLE_MNNVL buffer=$KV_BUFFER_DEVICE"

# --enable-cumem-allocator registers the KV cache as VMM, which the cuda_ipc/MNNVL path
# requires; vLLM's NixlConnector guide pairs it with UCX_CUDA_IPC_ENABLE_MNNVL for
# GB-series NVLink systems. Without it the transfer has no registrable memory and falls
# back or fails at handshake.
VLLM_NIXL_SIDE_CHANNEL_HOST="$KV_HOST" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_PORT" \
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --nnodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-cumem-allocator \
  --enable-request-id-headers --no-async-scheduling \
  --kv-transfer-config "$KV_CFG" \
  --host 0.0.0.0 --port "$PORT" \
  "${HEADLESS[@]}" \
  ${EXTRA_SERVE_ARGS} \
  > "$LOG" 2>&1
RC=$?
echo "[$ENGINE] rank=$NODE_RANK exited rc=$RC"
exit $RC
