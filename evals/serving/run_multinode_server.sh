#!/bin/bash
# Serve ONE model across N nodes, inside one SLURM allocation, using vLLM's own
# multi-node path. Launched with one task per node:
#
#   srun --nodes=$NNODES --ntasks=$NNODES --ntasks-per-node=1 \
#        --container-image=$CONTAINER ... serving/run_multinode_server.sh
#
# RANK 0 SERVES; EVERY OTHER RANK MUST BE PASSED --headless. The command is otherwise
# identical and --node-rank comes from SLURM rather than from an argument, so the ranks
# cannot be misnumbered by a caller looping over hostnames.
#
# --headless is NOT optional and NOT inferred. serve.py does contain a follower branch
# keyed on `parallel_config.node_rank_within_dp > 0`, but that is not what decides this:
# the CLI checks `args.headless` first (entrypoints/cli/serve.py:66) and asserts
# `not args.headless` on the API-server path (:263). Without the flag a follower builds a
# full APIServer and EngineCore, joins the TP group correctly -- world_size and the global
# ranks all look right in the log -- and then dies on
#   AssertionError: collective_rpc should not be called on follower node
# which reads like a vLLM bug rather than a missing flag. Measured: rank 1 failed this way
# while rank 0 loaded its weights and waited for a peer that was already dead.
#
# WHY mp AND NOT ray. vLLM selects the mp backend by itself once --nnodes > 1 on CUDA
# (config/parallel.py), and mp needs no cluster daemon to start, survive, or be torn down
# inside a batch job. `nnodes > 1` is REJECTED for any backend outside
# (mp, uni, external_launcher), so this is not a free choice.
#
# GPUS PER NODE IS DERIVED, NOT CONFIGURED: vLLM computes world_size / nnodes itself, so
# TP=16 over 4 nodes is 4 GPUs each. If that quotient does not match the GPUs Slurm gave
# each node, the engine fails at init rather than silently using fewer -- which is why
# this script does not try to second-guess it.
set -uo pipefail

# MODEL_PATH, not MODEL: run_eval.sh's $MODEL is the models.json KEY, and this script is
# invoked with its environment. Reusing the name would serve a checkpoint named
# "gemma-4-31b-it" and fail on a path that does not exist.
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SERVED_NAME=${SERVED_NAME:-model}
TP=${TP:?TP (tensor-parallel-size) is required}
NNODES=${NNODES:?NNODES is required}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}
LOG_DIR=${LOG_DIR:-/tmp}
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR is required: the routable IP of rank 0}
MASTER_PORT=${MASTER_PORT:-29501}

# From Slurm, not from a flag. With --ntasks-per-node=1 this is the node's index in the
# step, which is exactly vLLM's node rank, and taking it from the environment means the
# ranks cannot be misnumbered by a caller looping over hostnames.
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}

# 2.5 TB of weights off Lustre does not load in 600s, and the default turns a slow load
# into a timeout that reads like a hang. The vendor recipe raises it for the same reason.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}

# GLOO, NOT NCCL, is the gap. This cluster's nodes already export NCCL_SOCKET_IFNAME=eth0
# plus NCCL_IB_HCA over the eight RoCE devices, so NCCL is correctly configured and is
# deliberately NOT touched here. torch.distributed's gloo group -- which carries the CPU
# side of the handshake -- has no such site default, and left alone it picks an interface
# by itself: on a node with eth0 plus eth1-eth8 (the RoCE NICs) that choice is a coin
# flip, and the wrong one hangs the rendezvous with no error at all.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-eth0}}

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/server_rank${NODE_RANK}.log"

HEADLESS=()
[ "$NODE_RANK" -gt 0 ] && HEADLESS=(--headless)

echo "[mn] rank=$NODE_RANK/$NNODES host=$(hostname) tp=$TP master=$MASTER_ADDR:$MASTER_PORT" \
     "${HEADLESS[*]:-(serves the API)}"
echo "[mn] GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
echo "[mn] log -> $LOG"

# --host 0.0.0.0 on rank 0: the client may run in a different container on the same node,
# and a server bound to 127.0.0.1 inside its own namespace is not reachable from it.
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --nnodes "$NNODES" \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --host 0.0.0.0 --port "$PORT" \
  "${HEADLESS[@]}" \
  ${EXTRA_SERVE_ARGS} \
  > "$LOG" 2>&1
RC=$?
echo "[mn] rank=$NODE_RANK exited rc=$RC"
exit $RC
