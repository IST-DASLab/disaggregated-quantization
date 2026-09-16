#!/bin/bash
# Bring up a 1P1D disaggregated vLLM pair + proxy using NixlConnector, inside one
# SLURM allocation.
#
#   prefill server  GPU 0   kv_producer   VLLM_NIXL_SIDE_CHANNEL_PORT=<a>
#   decode  server  GPU 1   kv_consumer   VLLM_NIXL_SIDE_CHANNEL_PORT=<b>
#   proxy           CPU     nixl_proxy.py -> one OpenAI endpoint
#
# FORKED from qad/serving/run_nixl_server.sh. The qad copy stays as it is: its settings
# were established against the nemo container and its published results were measured
# with them. What differs here is documented at KV_BUFFER_DEVICE below.
#
# This is the Nixl counterpart of run_disagg_server.sh (which uses P2pNcclConnector).
# Differences that matter:
#
#   * NixlConnector needs the `nixl` python package, which is NOT in the container.
#     It is installed --no-deps into $NIXL_PREFIX and put on the SERVERS' PYTHONPATH
#     here. Installing it WITHOUT --no-deps pulls a second torch, which shadows the
#     container's and breaks vllm._C with an undefined at::TensorBase symbol.
#     `nixl` is only a shim; the CUDA-matched `nixl-cuNN` wheel must be there too.
#   * Routing is request-level via kv_transfer_params in the HTTP bodies, not via the
#     request id, so the proxy is nixl_proxy.py (see its docstring). The engines find
#     each other over a ZMQ "side channel" whose address is
#     VLLM_NIXL_SIDE_CHANNEL_HOST:VLLM_NIXL_SIDE_CHANNEL_PORT. Both servers share a
#     node, so those PORTS MUST DIFFER; the connector also reserves one port per TP
#     rank starting at that number.
#   * kv_load_failure_policy=fail: without it a failed KV load silently degrades to
#     recomputing the prompt on the decode weights, which is exactly the failure this
#     whole exercise is trying to make impossible to miss.
#   * --enforce-eager on both (per vLLM's disaggregated-serving docs).
#
#   ./serving/run_nixl_server.sh --prefill-model DIR --decode-model DIR \
#        --tokenizer Qwen/Qwen3-0.6B --ready-file /path/ready
#
# Serves until killed. Writes --ready-file (containing the proxy port) once all three
# are answering.

set -uo pipefail

PREFILL_MODEL=""; DECODE_MODEL=""; TOKENIZER=""
SERVED_NAME="${SERVED_NAME:-model}"
PORT_BASE="${PORT_BASE:-8500}"
# TENSOR PARALLEL, PER ENGINE -- so a 1P1D pair on a 4-GPU tray uses TP=2 twice, not
# two TP=1 engines with half the allocation idle. This script used to pin exactly one
# GPU per engine and pass no --tensor-parallel-size at all, so the TP that run_eval.sh
# resolves from models.json was silently dropped on the single-node path (the multi-node
# nixl_engine.sh path always honoured it). That is not a knob that should differ between
# two placements of the same pair.
#
# The GPU lists FOLLOW TP: prefill takes the first TP devices, decode the next TP, so
# the two engines never share a device. At TP=1 this reproduces the old 0 / 1 exactly.
# Both remain overridable for placements this arithmetic does not cover.
# Left EMPTY here and filled in after argument parsing, because --tp may still change TP.
TP="${TP:-1}"
PREFILL_GPU="${PREFILL_GPU:-}"
DECODE_GPU="${DECODE_GPU:-}"
# vLLM allocates KV blocks DYNAMICALLY (PagedAttention) -- max_model_len is a cap,
# not a per-sequence reservation, so lowering it does not raise concurrency. The
# "Maximum concurrency for N tokens per request" line in the engine log is vLLM's
# worst-case estimate, not an allocation.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# 512. Measured over 40k samples: KV cache usage median 5.8%, max 48.7%, ZERO
# preemptions and ZERO OOMs -- the engine was never memory-limited. What WAS observed
# is "Waiting: 72 reqs" queued at the engine while the cache sat ~94% empty, i.e. the
# scheduler cap was binding, not memory.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"

# CUDA graphs ON by default. --enforce-eager was inherited from vLLM's P2pNcclConnector
# docs, but that connector was abandoned; NixlConnector registers the KV cache once at
# startup and graph capture does not relocate it, so the two are compatible. Eager
# costs a lot on a 0.6B model, where decode is launch-bound. DISAGG_EAGER=1 restores it.
EAGER_FLAG=""
[ -n "${DISAGG_EAGER:-}" ] && EAGER_FLAG="--enforce-eager"

# The proxy is a Python process doing JSON work per request; one event loop becomes the
# ceiling well before the GPUs do. Several workers spread that across cores.
PROXY_WORKERS="${PROXY_WORKERS:-8}"
# more of the GPU to KV cache => more concurrent sequences; weights are tiny at 4 bit
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
READY_FILE=""
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"
NIXL_PREFIX="${NIXL_PREFIX:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/nixl_nodeps}"
# Extra flags appended verbatim to BOTH `vllm serve` invocations. Empty by default, so
# every existing caller is unaffected. It exists because some models are unservable
# without model-specific flags -- Muse-Glimmer needs its two channel parsers and
# --generation-config auto -- and forking this script to add them would fork all of the
# hard-won settings above with it. Word-split on purpose: it is a flag list, not a path.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

# Resolve the interpreter instead of spelling `python`. The nemo container this script
# was written in has both names; the vLLM nightly image has ONLY python3, and there the
# bare `python` calls below fail in ways that do not name the cause -- the KV host
# detection silently produced an empty string and the script died with
# "could not determine KV host IP", which reads like a network problem.
PY="$(command -v python3 || command -v python)"
[ -z "$PY" ] && { echo "ERROR: no python interpreter on PATH" >&2; exit 1; }

# Async scheduling runs the scheduler a step ahead of execution while a KV
# connector's metadata is per-step. It cost the P2P track a silent hang; keep it off
# here too unless explicitly re-enabled, so a hang can't be blamed on it.
ASYNC_SCHED_FLAG="--no-async-scheduling"
[ -n "${DISAGG_ASYNC_SCHED:-}" ] && ASYNC_SCHED_FLAG=""

while (($# > 0)); do
  case "$1" in
    --prefill-model) PREFILL_MODEL="$2"; shift 2 ;;
    --decode-model)  DECODE_MODEL="$2";  shift 2 ;;
    --tokenizer)     TOKENIZER="$2";     shift 2 ;;
    --served-name)   SERVED_NAME="$2";   shift 2 ;;
    --extra-serve-args) EXTRA_SERVE_ARGS="$2"; shift 2 ;;
    --port-base)     PORT_BASE="$2";     shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --ready-file)    READY_FILE="$2";    shift 2 ;;
    --tp)            TP="$2";            shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

_gpu_range() {  # start count -> "start,start+1,...,start+count-1"
  local out=() i
  for ((i = 0; i < $2; i++)); do out+=("$(($1 + i))"); done
  local IFS=,; echo "${out[*]}"
}
[ -z "$PREFILL_GPU" ] && PREFILL_GPU="$(_gpu_range 0 "$TP")"
[ -z "$DECODE_GPU" ]  && DECODE_GPU="$(_gpu_range "$TP" "$TP")"
# Fails at submit rather than as a CUDA error 40 minutes into weight loading, which is
# what an over-subscribed CUDA_VISIBLE_DEVICES looks like from the engine log.
_NVIS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$_NVIS" -gt 0 ] && [ "$((TP * 2))" -gt "$_NVIS" ]; then
  echo "ERROR: tp=$TP per engine needs $((TP * 2)) GPUs for a 1P1D pair, node has $_NVIS" >&2
  exit 2
fi
echo "[nixl] tp=$TP  prefill_gpus=$PREFILL_GPU  decode_gpus=$DECODE_GPU"
[ -z "$PREFILL_MODEL" ] && { echo "ERROR: --prefill-model required" >&2; exit 2; }
[ -z "$DECODE_MODEL" ]  && { echo "ERROR: --decode-model required" >&2; exit 2; }
[ -z "$TOKENIZER" ]     && { echo "ERROR: --tokenizer required" >&2; exit 2; }

# Ports are normally supplied by the caller, which binds them first to prove they are
# free (see free_ports() in eval_disagg.py). Deriving them from a base is a fallback
# for manual runs only: on a shared node a collision on a side-channel port surfaces
# deep inside the engine as "Address already in use" and reads like a vLLM bug.
PREFILL_PORT="${PREFILL_PORT:-$((PORT_BASE + 0))}"
DECODE_PORT="${DECODE_PORT:-$((PORT_BASE + 1))}"
PROXY_PORT="${PROXY_PORT:-$((PORT_BASE + 2))}"
PREFILL_SIDE_PORT="${PREFILL_SIDE_PORT:-$((PORT_BASE + 100))}"
DECODE_SIDE_PORT="${DECODE_SIDE_PORT:-$((PORT_BASE + 200))}"

# The decode engine reaches the prefill engine at the host the PREFILL engine
# advertises in kv_transfer_params.remote_host, which is VLLM_NIXL_SIDE_CHANNEL_HOST.
# Use the node's routable address rather than loopback: it is what UCX can actually
# connect to, and it keeps this correct if the two engines are ever split across
# nodes.
KV_HOST="${VLLM_HOST_IP:-$("$PY" - <<'PY'
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

# UCX TRANSPORT SELECTION. Overrides the container/site defaults, and what to override
# them WITH is cluster-specific.
#
# On b200 the container shipped UCX_NET_DEVICES=rocep145s0:1,... -- RDMA devices that do
# not exist inside it -- so UCX had no usable device, fell back to tcp/eth0, could not
# route to the node's own address, and nixl died at createBackend with NIXL_ERR_BACKEND.
# There, "all" was the fix.
#
# On the GB300 NVL72 cluster the site sets UCX_TLS=tcp deliberately and lists it as
# non-tunable, to stop UCX initialising by accident. Cluster support confirmed it MAY be
# overridden per job, and that KV must travel over NVLINK rather than RDMA here:
# NVBug 6261319 -- the VFs expose a 2 MB UAR BAR (PF_LOG_BAR_SIZE=1) and a KV transfer
# overruns it, failing with
#   mlx5dv_devx_alloc_uar(device=mlx5_0, type=WC) failed: Cannot allocate memory
# after a few workers come up. Only racks from the recent scale-up carry the 256 MB
# setting, so any given allocation is likely to hit the small one.
#
# Hence the default below lists cuda_ipc first and OMITS the RDMA verbs transports
# (rc/ud/dc) entirely: the point is that a transfer which cannot use NVLink should fail
# rather than quietly fall back onto the path with the open bug. tcp remains for the
# side-channel handshake. This also requires the two engines to sit in ONE NVL72 rack --
# run_eval.sh passes --segment for that.
export UCX_TLS="${NIXL_UCX_TLS:-cuda_ipc,cuda_copy,sm,self,tcp}"
# MNNVL is what makes cuda_ipc work ACROSS trays rather than only within one. Without it
# cuda_ipc covers the 4 GPUs of a single tray and a cross-tray pair silently has no
# usable transport.
export UCX_CUDA_IPC_ENABLE_MNNVL="${NIXL_UCX_CUDA_IPC_ENABLE_MNNVL:-y}"
# Left at the site value unless told otherwise: these are the eight compute NICs, and
# they are only used for the tcp fallback now that the verbs transports are excluded.
[ -n "${NIXL_UCX_NET_DEVICES:-}" ] && export UCX_NET_DEVICES="$NIXL_UCX_NET_DEVICES"
echo "[nixl] UCX_CUDA_IPC_ENABLE_MNNVL=$UCX_CUDA_IPC_ENABLE_MNNVL"
echo "[nixl] UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES"

# nixl must be importable by the engines. Prepend rather than replace so the caller's
# entries survive; the caller is responsible for having already stripped the
# lm_eval overlay (its huggingface-hub 1.24.0 makes `vllm serve` refuse to start).
export PYTHONPATH="$NIXL_PREFIX${PYTHONPATH:+:$PYTHONPATH}"

# Attention backend. vLLM picks FLASHINFER here and logs "Using HND KV cache layout
# for FLASHINFER backend", i.e. the per-layer KV tensor is laid out with the block
# dimension FIRST. The host-buffer staging path does
#     dst_cache[:, dst_block_indices] = _src_cache.cpu()   (cuda.py swap_out_blocks_to_host)
# which indexes dimension 1 with block ids -- correct only for the (2, num_blocks, ...)
# layout. Under FLASHINFER/HND dimension 1 has size 2, so every block id >= 2 is out
# of range and the prefill engine dies mid-request with
#     IndexKernel.cu:120 ... Assertion `-sizes[i] <= index && index < sizes[i]` failed
#     torch.AcceleratorError: CUDA error: device-side assert triggered
# Pinning FLASH_ATTN keeps the layout the copy helper expects.
export VLLM_ATTENTION_BACKEND="${NIXL_ATTENTION_BACKEND:-FLASH_ATTN}"
echo "[nixl] VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"

# HYBRID SSM / LINEAR-ATTENTION MODELS. A model whose layers carry recurrent state
# rather than a KV cache -- Mamba2, and GatedDeltaNet as in Qwen3.5 -- transfers that
# state through NixlConnector's 3-read conv path, which requires the DS conv layout and
# refuses to start without it:
#     AssertionError: 3-read Mamba conv transfer requires DS conv state layout.
#                     Set VLLM_SSM_CONV_STATE_LAYOUT=DS
# Set unconditionally: a model with no conv state has nothing for this to lay out, so
# it is inert for every pure-attention model already measured through this script.
export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"
echo "[nixl] VLLM_SSM_CONV_STATE_LAYOUT=$VLLM_SSM_CONV_STATE_LAYOUT"

if [ -n "${DISAGG_DEBUG:-}" ]; then
  export VLLM_LOGGING_LEVEL=DEBUG
  echo "[nixl] DEBUG logging enabled"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # qad/serving
QAD_DIR="$(dirname "$SCRIPT_DIR")"                           # qad
# The engines import the noise connector by dotted module path (serving.kv_noise_connector),
# so the qad root has to be importable even when this script is invoked on its own.
case ":$PYTHONPATH:" in
  *":$QAD_DIR:"*) ;;
  *) export PYTHONPATH="$QAD_DIR${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
LOG_DIR="${LOG_DIR:-/tmp}"
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
      echo "[nixl] ${name} ready on ${port}"; return 0
    fi
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "[nixl] ${name} DIED before ready" >&2; return 1; }
    done
    sleep 1
  done
  echo "[nixl] ${name} timed out after ${SERVER_READY_TIMEOUT_S}s" >&2; return 1
}

echo "[nixl] prefill=$PREFILL_MODEL"
echo "[nixl] decode =$DECODE_MODEL"
echo "[nixl] ports: prefill=$PREFILL_PORT decode=$DECODE_PORT proxy=$PROXY_PORT" \
     "side=$PREFILL_SIDE_PORT/$DECODE_SIDE_PORT on $KV_HOST"
echo "[nixl] NIXL_PREFIX=$NIXL_PREFIX"
echo "[nixl] server PYTHONPATH=$PYTHONPATH"
# Fail here rather than 10 minutes into a model load: without nixl the engine logs
# "NIXL is not available" and then dies when the connector is constructed.
"$PY" -c "import nixl._api; print('[nixl] nixl import OK:', nixl.__file__)" || {
  echo "ERROR: nixl is not importable with PYTHONPATH=$PYTHONPATH" >&2
  echo "       install it with: pip install --no-deps --target $NIXL_PREFIX nixl==1.3.1 nixl-cu13==1.3.1" >&2
  exit 1
}

# kv_buffer_device. "cuda" registers the KV blocks with NIXL directly in VRAM, which is
# what you want; "cpu" stages them through a host buffer, which UCX can register without
# CUDA support but which is slower.
#
# The qad copy of this script forces cpu, because the nemo container's UCX was built
# without CUDA:
#   "8 NVIDIA GPU(s) were detected, but UCX CUDA support was not found!"
#   "VRAM memory is detected as host by UCX. VRAM registration cannot proceed."
# and cuda died at init in register_kv_caches with nixlBackendError: NIXL_ERR_BACKEND.
#
# THAT IS A FACT ABOUT THAT CONTAINER, NOT ABOUT NIXL, and this stack runs in a
# different one (vLLM nightly + the nixl_cu13 wheel). It also cannot stay on cpu for a
# hybrid SSM model: host transfer buffers are built for attention layers only, so a
# MambaSpec layer's descriptors still point into VRAM while the agent is registered as
# DRAM, and registration fails with
#   nixl_cu13._bindings.nixlNotFoundError: NIXL_ERR_NOT_FOUND
# from _build_mamba_local. Qwen3.5 has 48 GatedDeltaNet layers, so cpu is simply not an
# option for it.
#
# Default cuda here, overridable per model, and a model measured on cpu should stay on
# cpu rather than be silently re-measured on a different transport.
KV_BUFFER_DEVICE="${NIXL_KV_BUFFER_DEVICE:-cuda}"

# NixlConnector hashes each engine's configuration and refuses the handshake unless
# the two hashes match. A heterogeneous pair (W4A4 prefill -> W4A16 decode) is
# exactly what that check rejects:
#   RuntimeError: NIXL compatibility hash mismatch. Local: dbe0..., Remote: f2f6...
#   Prefill and decode instances have incompatible configurations.
# and the decode engine then dies with a secondary
#   KeyError in block_size_ratio_from_engine_id (the remote was never registered).
# NIXL_ENFORCE_COMPAT=0 sets the connector's own documented escape hatch. Do not use
# it to paper over an accidental mismatch: it is only correct when the difference is
# the WEIGHTS, which do not affect KV layout. If block size, dtype or KV layout
# really differ, disabling the check trades a clear error for corrupt KV.
COMPAT_EXTRA=""
if [ "${NIXL_ENFORCE_COMPAT:-1}" = "0" ]; then
  COMPAT_EXTRA=',"kv_connector_extra_config":{"enforce_handshake_compat":false}'
  echo "[nixl] handshake compatibility check DISABLED (heterogeneous pair)"
fi
# --- optional pseudo KV-cache compression (serving/kv_noise_connector.py) -------------
# KV_BITS_PREFILL / KV_BITS_DECODE are rates: sigma = 2^-bits * RMS(group of 16), so
# 4^-bits of distortion power. >= 32 means OFF, and must be exactly inert.
#
# When on, each engine runs MultiConnector[NoisyKVConnector, NixlConnector] -- noise
# FIRST, so the blocks the decode engine later pulls are already degraded.
#
# kv_role is repeated INSIDE the noise child on purpose: MultiConnector builds each
# child with KVTransferConfig(**child_dict), which does NOT inherit kv_role from the
# parent. Without it the connector cannot tell which phase it is and would silently
# pick the wrong rate.
KV_BITS_PREFILL=${KV_BITS_PREFILL:-32}
KV_BITS_DECODE=${KV_BITS_DECODE:-32}
# Bash [ -lt ] is INTEGER-only: `[ "3.5" -lt 32 ]` raises "integer expression expected"
# AND evaluates false, so a fractional rate would silently fall through to the plain
# NixlConnector and run with no noise at all -- a clean-looking baseline mislabelled as
# a 3.5-bit result. Compare numerically instead.
kv_noise_on() {
  awk -v p="$KV_BITS_PREFILL" -v d="$KV_BITS_DECODE" 'BEGIN{exit !(p+0 < 32 || d+0 < 32)}'
}

kv_cfg() {                                  # $1 = kv_producer | kv_consumer
  local role="$1"
  local conn="NixlConnector" extra="$COMPAT_EXTRA" modpath=""
  if kv_noise_on || [ "${KV_NOISE_WRAP:-0}" = "1" ]; then
    # A SUBCLASS of NixlConnector, not MultiConnector[noise, nixl]. MultiConnector does
    # not forward set_host_xfer_buffer_ops to its children (it is called on the
    # top-level connector, gpu_model_runner.py:6088), so the wrapped NixlConnector never
    # gets its copy operation and the decode engine dies on the first received KV with
    #   nixl_connector.py:1761  assert self.copy_blocks is not None
    # That path is only used when kv_buffer_device=cpu, which is what we run.
    conn="NoisyNixlConnector"
    modpath=',"kv_connector_module_path":"serving.kv_noise_connector"'
    # One flat config: the engine-level fields stay where vLLM reads them, and the rates
    # ride along in extra_config. No wrapper, so nothing can silently revert to defaults.
    extra=",\"kv_connector_extra_config\":{\"kv_bits_prefill\":$KV_BITS_PREFILL,\"kv_bits_decode\":$KV_BITS_DECODE}"
  fi
  printf '%s' "{\"kv_connector\":\"$conn\"$modpath,\"kv_role\":\"$role\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\",\"kv_load_failure_policy\":\"fail\"$extra}"
}
# KV NOISE REQUIRES EAGER. save_kv_layer is a PYTHON hook around the attention op; a
# replayed CUDA graph runs the captured kernels and never re-enters Python, so on the
# decode engine -- where vLLM captures decode graphs by default -- the noise was simply
# not applied for most steps. Measured at kv_bits_decode=1 (50% amplitude): with graphs
# the model still scored gsm8k 0.575 and produced fluent text; with --enforce-eager the
# same config scored 0.0. A partially-applied noise is worse than none: it looks like a
# real, publishable robustness result.
if kv_noise_on && [ -z "$EAGER_FLAG" ]; then
  EAGER_FLAG="--enforce-eager"
  echo "[nixl] KV noise active -> forcing --enforce-eager (CUDA graphs skip the connector hook)"
fi
KV_PRODUCER_CFG="$(kv_cfg kv_producer)"
KV_CONSUMER_CFG="$(kv_cfg kv_consumer)"
echo "[nixl] kv_buffer_device=$KV_BUFFER_DEVICE"
[ -n "$EXTRA_SERVE_ARGS" ] && echo "[nixl] extra serve args: $EXTRA_SERVE_ARGS"
if kv_noise_on; then
  echo "[nixl] KV NOISE ON: prefill=${KV_BITS_PREFILL}bit decode=${KV_BITS_DECODE}bit"
fi

VLLM_NIXL_SIDE_CHANNEL_HOST="$KV_HOST" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT" \
CUDA_VISIBLE_DEVICES="$PREFILL_GPU" \
vllm serve "$PREFILL_MODEL" \
  --port "$PREFILL_PORT" --served-model-name "$SERVED_NAME" \
  --tokenizer "$TOKENIZER" --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${EAGER_FLAG} --enable-request-id-headers ${ASYNC_SCHED_FLAG} \
  --kv-transfer-config "$KV_PRODUCER_CFG" ${EXTRA_SERVE_ARGS} \
  >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=("$!")

VLLM_NIXL_SIDE_CHANNEL_HOST="$KV_HOST" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT" \
CUDA_VISIBLE_DEVICES="$DECODE_GPU" \
vllm serve "$DECODE_MODEL" \
  --port "$DECODE_PORT" --served-model-name "$SERVED_NAME" \
  --tokenizer "$TOKENIZER" --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${EAGER_FLAG} --enable-request-id-headers ${ASYNC_SCHED_FLAG} \
  --kv-transfer-config "$KV_CONSUMER_CFG" ${EXTRA_SERVE_ARGS} \
  >"$LOG_DIR/decode.log" 2>&1 &
PIDS+=("$!")

wait_for prefill "$PREFILL_PORT" || exit 1
wait_for decode  "$DECODE_PORT"  || exit 1

"$PY" "$SCRIPT_DIR/nixl_proxy.py" --workers "$PROXY_WORKERS" \
  --port "$PROXY_PORT" \
  --prefill-host 127.0.0.1 --prefill-port "$PREFILL_PORT" \
  --decode-host  127.0.0.1 --decode-port  "$DECODE_PORT" \
  ${NIXL_ALLOW_MISSING_KV:+--allow-missing-kv-params} \
  >"$LOG_DIR/proxy.log" 2>&1 &
PIDS+=("$!")
wait_for proxy "$PROXY_PORT" || exit 1

echo "[nixl] all up; proxy on $PROXY_PORT"
if [ -n "$READY_FILE" ]; then mkdir -p "$(dirname "$READY_FILE")"; echo "$PROXY_PORT" > "$READY_FILE"; fi
wait
