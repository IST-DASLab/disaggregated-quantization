#!/bin/bash
# Run one benchmark against Muse-Glimmer-30B through a vLLM OpenAI endpoint.
#
#   ./bin/run_muse_eval.sh --bench gpqa                    # 4 repeats, 198 q each
#   ./bin/run_muse_eval.sh --bench ifbench
#   ./bin/run_muse_eval.sh --bench mmmu --long          # vision; the only MMMU subset run
#   ./bin/run_muse_eval.sh --bench gpqa --disagg --tag disagg   # 1P1D, 2 GPUs
#   ./bin/run_muse_eval.sh --bench gpqa --model $MODELS/Muse-Glimmer-30B-NVFP4
#   ./bin/run_muse_eval.sh --bench gpqa --disagg --tag pd \
#       --prefill-model $MODELS/Muse-Glimmer-30B-NVFP4 \
#       --decode-model  $MODELS/Muse-Glimmer-30B-NVFP4A16
#   ./bin/run_muse_eval.sh --bench gpqa --limit 8 --repeats 1 --interactive   # smoke
#
# One job = boot a server, drive one benchmark against it, score, tear down. The server
# is deliberately inside the job rather than long-lived and shared: a shared endpoint
# makes "which weights answered this request" unanswerable after the fact, and the whole
# point of the exercise is to compare serving configurations.
#
# RESTARTS. Every driver appends per item and reloads what it finished, so re-running
# the identical command resumes rather than restarts. --long moves the job to
# batch_long (24h vs the 4h default) for MMMU-Pro vision, which is 5190 items against a
# reasoning model and will not finish in 4h.
#
# Run bin/setup_harnesses.sh once before the first eval: it pins the third-party
# harness checkouts, builds the pip overlay inside the container, and pre-downloads the
# nltk corpora the IFBench scorer needs (compute nodes have no network).
#
# Results:  muse/results/<bench>/<tag>/            (summary.json is versioned)
# Logs:     logs/evals/<timestamp>_<bench>_<tag>/    (gitignored)

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad

set -uo pipefail

# MUSE_ROOT is derived from this script, not hardcoded: the tree lives in the git repo
# and a clone elsewhere must work without editing paths. Only things that are NOT
# versioned -- the container, the 56GB of weights, the HF cache, the pip overlay, the
# third-party harness checkouts -- keep absolute lustre defaults.
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
MODEL=${MODEL:-muse-glimmer}          # key in models.json
# Override the checkpoint while keeping the model's serve flags and served name.
# Empty means "the key's own path".
WEIGHTS=${WEIGHTS:-}
# A HETEROGENEOUS pair -- NVFP4 prefill feeding an NVFP4A16 decode -- is the format this
# exercise is built to serve, so the two engines take independent checkpoints.
PREFILL_WEIGHTS=${PREFILL_WEIGHTS:-}
DECODE_WEIGHTS=${DECODE_WEIGHTS:-}
HF_CACHE=${HF_CACHE:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache}
# Deliberately outside the repo: 108MB of installed wheels. bin/setup_harnesses.sh
# builds it, and it must be built INSIDE the container so the wheels match its
# Python 3.12 / numpy, not the login node's.
PYDEPS=${PYDEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-pydeps}
HARNESSES=${HARNESSES:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses}

BENCH=${BENCH:-gpqa}
TAG=${TAG:-bf16}
LIMIT=${LIMIT:-0}
REPEATS=${REPEATS:-4}
# vision only. standard10 was measured once (72.20 single-engine / 72.37 disaggregated,
# full 1730) and dropped: the published figure is vision's, and a second subset that
# agrees with the first to within 0.2 points buys nothing per GPU-hour. The driver still
# accepts standard4/standard10 if a reason to run them ever comes back.
SETTING=${SETTING:-vision}
MODE=${MODE:-cot}              # mmmu only

# --- ruler only ---------------------------------------------------------------------
# The 13 tasks the RULER paper reports, which is also every key in RULER's synthetic.yaml.
RULER_TASKS=${RULER_TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,vt,cwe,fwe,qa_1,qa_2}
# Powers of two from 4k, RULER's own reporting grid. The top of the sweep must stay
# within BOTH the model's context and --max-model-len, and the length is the point of
# the benchmark, so it is left explicit rather than derived.
RULER_SEQLENS=${RULER_SEQLENS:-4096,8192,16384,32768}
# RULER's own default. 13 tasks x 500 docs x 4 lengths = 26,000 generations, and each
# prompt is as long as the length being tested -- this is a far heavier benchmark than
# its doc count suggests. Lower it for a smoke test.
RULER_SAMPLES=${RULER_SAMPLES:-500}
# Thinking OFF by default. See drivers/ruler_infer.py: RULER's per-task budget is
# 30-128 tokens, which a reasoning model spends entirely inside <think>, scoring 0.
# RULER_THINK_FLAG is derived AFTER argument parsing, since --think sets this.
RULER_THINK=${RULER_THINK:-0}
MAX_TOKENS=${MAX_TOKENS:-}            # from models.json unless overridden
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
# 512, matching MAX_NUM_SEQS. At 32 the server NEVER queued -- measured on
# nemotron-3.5-lightning-30b at Running: 32, Waiting: 0, KV cache usage 1.8% -- so an
# A3B model on a GB300 ran at batch 32 and the GPU idled between requests. The cap that
# actually bound was this one, not MAX_NUM_SEQS, which had already been raised to 512.
#
# Safe because the server is async (vLLM V1 AsyncLLM on uvicorn) and admits sequences up
# to max_num_seqs, dropping to fewer when the KV pool runs short -- it cannot OOM from
# client concurrency alone. The client is threads, not asyncio, so this is 512 real OS
# threads; they sit in socket reads with the GIL released, and common.py now gives each
# one a pooled keep-alive Session instead of a fresh TCP connection per request.
#
# Disaggregated arms are the exception and are NOT helped by this: a decode engine
# saturates near 190 sequences, so 512 oversubscribes it ~2.5x and completions arrive in
# clumps that make a healthy run look hung. Pass --workers explicitly for those.
WORKERS=${WORKERS:-512}
# Per-request read timeout. 1800 was hardcoded in drivers/common.py and it silently cost
# a whole arm: qwen3.8-2.4t nvfp4pd finished INSIDE its wall clock but lost 124 items to
#   ReadTimeout: ... (read timeout=1800)
# and scored 11,908 against every other arm's 12,032. The tail items generate ~16k tokens,
# and at 512 workers against a decode engine that admits ~190 sequences each one crawls at
# ~6 tok/s -- 44 minutes for one request, past a 30 minute timeout. Raise the timeout OR
# lower --workers; both work, and for pd arms both are worth doing.
REQ_TIMEOUT=${REQ_TIMEOUT:-3600}
# SHARDING, for benchmarks too large to finish in one job. Shard K takes the items where
# `index % NUM_SHARDS == K`, writes its own raw.shard{K}of{N}.jsonl and resumes
# independently, so N jobs run the same arm in parallel instead of one job resuming
# across the 4h wall two or three times. Modulo rather than a contiguous slice: MMLU-Pro
# is ordered by category, so slicing would hand one shard all the math.
#
# Only mmlu_pro reads these; every other benchmark fits a single job comfortably. The
# longest job this harness has ever run is 2h52m, measured over 154 completed jobs.
SHARD=${SHARD:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
# SERVER-side concurrency cap, for BOTH the single-engine and the disaggregated path --
# this file pinned each of them at 64 independently, overriding run_nixl_server.sh's own
# default of 512.
#
# It never mattered until now, which is why it went unnoticed: the client's WORKERS=32 is
# below 64, so the server never queued (Waiting: 0 reqs throughout every run) and the GPU
# ran at batch 32 no matter which path served it. Raising this alone changes nothing --
# --workers has to go up with it.
#
# It is a cap, not a reservation: vLLM sizes the KV pool at startup and admits fewer
# sequences when it runs short, so raising it cannot by itself OOM a model whose
# per-sequence state is large.
#
# It is a cap, not a reservation: vLLM sizes the KV pool at startup and admits fewer
# sequences when it runs short, so raising this cannot by itself OOM a model whose
# per-sequence state is large.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-512}
LONG=${LONG:-0}
PORT=${PORT:-8000}
# Smallest GPU count a job may ASK FOR, which is not the same as the number it uses. A
# single engine wants one GPU and a 1P1D pair wants two, but a cluster whose QOSes carry
# MinTRES gres/gpu=N rejects anything smaller with QOSMinGRES -- an accounting error, not
# a capacity one, so it fails at submit with nothing in the log to read. The GB300 tray
# here is 4 GPUs and every QOS on it sets that minimum; b200 has no such floor, so the
# default leaves that cluster's requests exactly as they were.
# 4 on oci-jhb-slurm-1: the GPU nodes are 4-GPU trays and the QOS enforces a whole-node
# floor, so a 1-GPU request is REJECTED AT SUBMIT with
#   sbatch: error: QOSMinGRES
#   Batch job submission failed: Job violates accounting/QOS policy
# (same class as the old cluster's MinTRES gres/gpu=4). Single-engine runs still serve on
# one GPU with TP=1 and leave the other three idle -- that is the floor, not a choice.
# This also raises the ceiling the TP checks below enforce, so --tp up to 4 is legal here.
MIN_GPUS=${MIN_GPUS:-4}
# MULTI-NODE. Both come from models.json (a 2.4T model is unservable on one node, which
# makes them model facts, not run options) and both are overridable for experiments.
# NNODES=1 keeps every existing model on exactly the path it was measured with -- the
# multi-node branch below is not entered at all.
NNODES=${NNODES:-}
TP=${TP:-}
# Port for the torch.distributed rendezvous, distinct from the OpenAI port. Only used
# when NNODES > 1.
MASTER_PORT=${MASTER_PORT:-29501}
# Keep a multi-node allocation inside ONE NVL72 rack. A rack is 18 trays sharing an
# NVLink domain; without this Slurm will happily split a 4-node job across two racks, and
# every TP all-reduce that crosses the boundary drops from NVLink to RoCE. Measured: a
# TP=16 fp8 job landed on nvl72d065 + nvl72d148. It still produces correct scores, so
# nothing fails -- the arms just run under different interconnects, which makes any
# throughput comparison between them meaningless.
#
# It also becomes a correctness requirement for --disagg: KV must cross over NVLink,
# because RDMA on this cluster hits a 2 MB UAR BAR limit (NVBug 6261319) that a KV
# transfer overruns. Cluster guidance is --segment <= 16.
SEGMENT=${SEGMENT:-}
# Job Reaper exemption. The reaper cancels a job whose GPUs sit at
# DCGM_FI_PROF_SM_ACTIVE <= 0.01 for 30 minutes, and a 1.56 TB checkpoint takes ~35
# minutes to stream off Lustre before a single kernel runs -- so a large model is reaped
# mid-load, having done nothing wrong. It killed kimi-k3 mxfp4pd at exactly 30:07.
#
# The exemption is a JSON blob in --comment and is ALL-OR-NOTHING: a missing field, a bad
# reason code or mangled quoting is discarded silently and the job is evaluated as if
# unexempted. There is no confirmation, so check `scontrol show job <id>` for the Comment
# field after submitting.
#
# DEFAULTED, not opt-in. Leaving this empty meant remembering to pass COMMENT on exactly
# the jobs that need it, and the failure mode of forgetting is a job cancelled at ~30
# minutes with no error of its own -- indistinguishable from a user cancel in sacct
# ("CANCELLED b+"). kimi-k3 mxfp4pd lost three separate attempts that way, including one
# after this very comment was written. The exemption is harmless on a job that never
# idles, so the safe default is to always ask for it and size the window to the load.
COMMENT=${COMMENT:-}
# Which --disagg path to take. The legacy one puts both engines on ONE node by GPU index
# (PREFILL_GPU=0 / DECODE_GPU=1) and is what every b200 result was measured with, so it
# stays the default for a model that fits on one node. The multi-node path gives each
# engine its own NNODES nodes and is required for anything that does not.
#
# Forcing it with NNODES=1 is how the cross-tray NVLink transfer gets tested cheaply: the
# legacy path would put both engines on one tray, where UCX uses shared memory and never
# touches MNNVL at all -- so it would pass without exercising the thing under test.
PD_MULTINODE=${PD_MULTINODE:-auto}
# Priority 700 vs normal's 100. For smoke tests the queue wait dominates the work, so
# use it there -- but not for real sweeps, which hold a GPU for hours and should not
# jump the queue.
# How long to wait for a server to answer, in 10s polls. 120 min, which sounds
# absurd until the cluster is busy: with ~30 jobs pulling multi-GB checkpoints off
# Lustre at once, one Gemma run spent 12m25s on safetensors alone and 17m53s in engine
# init, and came up at ~50 minutes -- past the old 40-minute window. It failed with
# SERVER TIMEOUT having done nothing wrong.
#
# A long window costs nothing when the server is actually broken: the poll below breaks
# out the moment the process dies. It only bounds "alive but still starting", which is
# exactly the case that should be waited out rather than thrown away after 40 minutes of
# queue time and weight loading.
READY_POLLS=${READY_POLLS:-720}
QOS=${QOS:-normal}
# --disagg: a prefill engine and a decode engine on separate GPUs exchanging KV through
# vLLM's NixlConnector, fronted by one OpenAI endpoint. The serving stack is qad's --
# run_nixl_server.sh + nixl_proxy.py -- rather than a second copy of it here, because
# every non-default setting in that script (kv_buffer_device=cpu, FLASH_ATTN,
# UCX_TLS=all, kv_load_failure_policy=fail) was established the hard way and a fork
# would drift from it.
DISAGG=${DISAGG:-0}

while (($# > 0)); do
    case "$1" in
        --bench)        BENCH="$2";        shift 2 ;;
        --model)          MODEL="$2";           shift 2 ;;
        --weights)        WEIGHTS="$2";         shift 2 ;;
        --prefill-weights) PREFILL_WEIGHTS="$2"; shift 2 ;;
        --decode-weights)  DECODE_WEIGHTS="$2";  shift 2 ;;
        --tag)          TAG="$2";          shift 2 ;;
        --limit)        LIMIT="$2";        shift 2 ;;
        --repeats)      REPEATS="$2";      shift 2 ;;
        --setting)      SETTING="$2";      shift 2 ;;
        --mode)         MODE="$2";         shift 2 ;;
        --seqlens)      RULER_SEQLENS="$2"; shift 2 ;;
        --ruler-tasks)  RULER_TASKS="$2";  shift 2 ;;
        --ruler-samples) RULER_SAMPLES="$2"; shift 2 ;;
        --think)        RULER_THINK=1;     shift ;;
        --max-tokens)   MAX_TOKENS="$2";   shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --workers)      WORKERS="$2";      shift 2 ;;
        --req-timeout)  REQ_TIMEOUT="$2";  shift 2 ;;
        --shard)        SHARD="$2";        shift 2 ;;
        --num-shards)   NUM_SHARDS="$2";   shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --nnodes)       NNODES="$2";       shift 2 ;;
        --tp)           TP="$2";           shift 2 ;;
        --long)         LONG=1;            shift ;;
        --interactive)  QOS=interactive;   shift ;;
        --disagg)       DISAGG=1;          shift ;;
        # Unknown flags are an error. A silently swallowed --limit turns a smoke test
        # into a full sweep that overwrites the real results directory.
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$BENCH" in
    gpqa|ifbench|mmmu|ocrbench|mmlu_pro|ruler) ;;
    # (RULER_THINK_FLAG is derived just below, once --think has been parsed.)
    *) echo "ERROR: --bench must be gpqa|ifbench|mmmu|ocrbench|mmlu_pro|ruler (got '$BENCH')" >&2
       exit 2 ;;
esac

RULER_THINK_FLAG=""
[ "$RULER_THINK" = 1 ] && RULER_THINK_FLAG="--think"

# Resolve the model spec ONCE, here, so an unknown key fails on the login node in
# milliseconds instead of after a GPU allocation and a 15-minute weight load.
spec() { python3 -c "
import json, sys
m = json.load(open('$MUSE_ROOT/models.json'))
if '$MODEL' not in m:
    keys = [k for k in m if not k.startswith('_')]
    sys.exit('ERROR: unknown --model %r; known: %s' % ('$MODEL', ', '.join(keys)))
v = m['$MODEL'].get('$1')
print(' '.join(v) if isinstance(v, list) else ('' if v is None else v))
"; }
MODEL_PATH_DEFAULT=$(spec path) || exit 2
SERVED_NAME=$(spec served_name)
SERVE_FLAGS=$(spec serve_flags)
# Ad-hoc server flags for one run, without editing models.json. Exists for kernel
# bisection: --moe-backend picks the NvFp4 MoE implementation (flashinfer_trtllm,
# flashinfer_cutlass, vllm_cutlass, marlin, emulation, ...) so a suspect kernel can be
# swapped for a reference one on identical weights.
EXTRA_SERVE=${EXTRA_SERVE:-}
[ -n "$EXTRA_SERVE" ] && SERVE_FLAGS="$SERVE_FLAGS $EXTRA_SERVE"
[ -z "$MAX_MODEL_LEN" ] && MAX_MODEL_LEN=$(spec max_model_len)

# RULER sizes each doc so that input + generation == the requested seqlen exactly
# (niah.py:245 loops until `len(tokens) + tokens_to_generate <= max_seq_length`). It is
# generated with --model_template_type base, so it reserves NOTHING for a chat template
# -- but these requests go through /chat/completions, where the SERVER wraps the prompt
# in this model's template. That wrapper is a few dozen tokens on top of a doc already
# sized to fill the window, so a 65536 sweep against a 65536 window overflows and vLLM
# rejects the request. Invisible at 4k, fatal at the top of the sweep, and it would look
# like a long-context failure rather than a budgeting one.
#
# TEMPLATE_MARGIN is per REQUEST, not per model: 512 is far more than any ChatML wrapper
# needs and costs only KV headroom, which is not the binding constraint here.
if [ "$BENCH" = ruler ]; then
    RULER_MAX_SEQLEN=$(printf '%s\n' ${RULER_SEQLENS//,/ } | sort -n | tail -1)
    RULER_NEED=$((RULER_MAX_SEQLEN + ${RULER_TEMPLATE_MARGIN:-512}))
    if [ "$RULER_NEED" -gt "$MAX_MODEL_LEN" ]; then
        echo "ruler: raising --max-model-len $MAX_MODEL_LEN -> $RULER_NEED" \
             "(max seqlen $RULER_MAX_SEQLEN + chat-template margin)"
        MAX_MODEL_LEN=$RULER_NEED
    fi
fi
[ -z "$MAX_TOKENS" ]    && MAX_TOKENS=$(spec max_tokens)
MODEL_PATH=${WEIGHTS:-$MODEL_PATH_DEFAULT}
[ -z "$NNODES" ] && NNODES=$(spec nnodes)
[ -z "$TP" ]     && TP=$(spec tensor_parallel_size)
[ -z "$NNODES" ] && NNODES=1
[ -z "$TP" ]     && TP=1
# Resolved HERE, not where PD_MULTINODE is declared: NNODES only becomes known once
# models.json has been read, and asking earlier would always see the default of 1.
if [ "$PD_MULTINODE" = auto ]; then
    if [ "$NNODES" -gt 1 ]; then PD_MULTINODE=1; else PD_MULTINODE=0; fi
fi
# world_size must divide evenly across the allocation, and vLLM derives GPUs-per-node as
# TP/NNODES. A mismatch surfaces only after the allocation and the weight load, so it is
# checked here, on the login node, in milliseconds.
if [ $((TP % NNODES)) -ne 0 ]; then
    echo "ERROR: --tp $TP is not divisible by --nnodes $NNODES" >&2; exit 2
fi
if [ "$NNODES" -gt 1 ] && [ $((TP / NNODES)) -ne "$MIN_GPUS" ]; then
    echo "ERROR: tp/nnodes = $((TP / NNODES)) GPUs per node, but this cluster allocates" \
         "$MIN_GPUS per node. Pick a TP that is $MIN_GPUS x nnodes." >&2; exit 2
fi
KV_BUFFER_DEVICE=${KV_BUFFER_DEVICE:-$(spec kv_buffer_device)}

if [ -z "$COMMENT" ]; then
    # Weight load is the idle window: streaming a checkpoint off Lustre runs no kernels.
    # ~15 min per TB per engine here, doubled for a 1P1D pair which loads two.
    _idle=30
    if [ -n "${MODEL_PATH:-}" ] && [ -d "${MODEL_PATH:-}" ]; then
        # -L to FOLLOW SYMLINKS. Several checkpoints here are symlink farms that share
        # one set of shards with a sibling (a W4A4 arm and its weight-only twin differ
        # only in config.json), and without -L du reports the size of the links -- 1 GB
        # for a 1.56 TB model, and an exemption window far too short for its own load.
        _gb=$(du -sBG --apparent-size -L "$MODEL_PATH" 2>/dev/null | cut -dG -f1)
        [ -n "$_gb" ] && _idle=$(( 30 + _gb / 25 ))
    fi
    [ "$DISAGG" = 1 ] && _idle=$(( _idle * 2 ))
    [ "$_idle" -gt 180 ] && _idle=180
    COMMENT="{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"$_idle\",\"reason\":\"model_loading\",\"description\":\"$MODEL $BENCH $TAG: weights stream off Lustre before any kernel runs\"}}"
fi
if [ "$NNODES" -le 1 ] && [ "$TP" -gt "$MIN_GPUS" ]; then
    echo "ERROR: --tp $TP on a single node, but only $MIN_GPUS GPUs are allocated per" \
         "node. Raise --nnodes or lower --tp." >&2; exit 2
fi
# The sampling guard asserts that the server applied the model's generation_config. That
# is only meaningful for a model that HAS vendor sampling overrides; gpt-oss ships a
# generation_config.json carrying token ids and nothing else, so vLLM never logs the line
# and the guard aborts a server that is serving correctly. Opt out per model, in
# models.json, rather than weakening the check for the models it protects.
SAMPLING_GUARD=$(spec sampling_guard)
# SERVER-SIDE overlay, opt-in per model. The client's $PYDEPS is deliberately kept off
# the server (see the note above the server block): it carries its own huggingface-hub
# and numpy, which shadow vLLM's and stop it booting. But some models need a package the
# container does not ship -- nemotron-3-super-120b's modeling file hard-raises
#   'mamba-ssm is required by the Mamba model but cannot be imported'
# and dies before serving a single token. muse-mambadeps was built with --no-deps and
# holds ONLY causal_conv1d and mamba_ssm, so it adds the missing import without shadowing
# anything vLLM needs. Exported here, before every launch branch, because the multi-node
# server starts as its own srun step further up and would not see a later export.
SERVER_PYTHONPATH=$(spec server_pythonpath)
if [ -n "$SERVER_PYTHONPATH" ]; then
    export PYTHONPATH="$SERVER_PYTHONPATH"
    echo "server PYTHONPATH overlay: $SERVER_PYTHONPATH"
fi
# Nested JSON, so it cannot go through spec(), which flattens lists and stringifies
# anything else. Emitted as compact JSON and parsed back by drivers/common.py.
EXTRA_BODY=$(python3 -c "
import json
m = json.load(open('$MUSE_ROOT/models.json'))
print(json.dumps(m['$MODEL'].get('request_extra_body') or {}, separators=(',', ':')))
") || exit 2
[ -z "$KV_BUFFER_DEVICE" ] && KV_BUFFER_DEVICE=cuda

# ---------------------------------------------------------------------------
# LOGIN mode: submit
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    LABEL="${MODEL}_${BENCH}_${TAG}"
    # Per NODE. A single engine is TP GPUs; a single-node 1P1D pair is two engines side
    # by side on one node, so twice that. (Multi-node disagg gives each engine its own
    # nodes, so NGPU stays per-engine and ALLOC_NODES doubles instead -- see below.)
    NGPU=$TP
    [ "$DISAGG" = 1 ] && [ "$PD_MULTINODE" = 0 ] && NGPU=$((TP * 2))
    [ "$NGPU" -lt "$MIN_GPUS" ] && NGPU=$MIN_GPUS
    # NNODES is per ENGINE. A disaggregated pair is two of them, so the ALLOCATION is
    # twice that -- and both engines must land in one NVL72 rack for the KV to cross over
    # NVLink, which is what --segment below enforces.
    ALLOC_NODES=$NNODES
    if [ "$DISAGG" = 1 ] && [ "$PD_MULTINODE" = 1 ]; then ALLOC_NODES=$((NNODES * 2)); fi
    [ "$BENCH" = mmmu ] && LABEL="${MODEL}_${BENCH}_${SETTING}_${TAG}"
    [ "$NUM_SHARDS" != 1 ] && LABEL="${LABEL}_s${SHARD}of${NUM_SHARDS}"
    LOGS="$REPO_ROOT/logs/evals/$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "$LOGS"
    if [ "$LONG" = 1 ]; then PART=batch_long; TIME=${TIME:-24:00:00}
    else                     PART=batch;      TIME=${TIME:-04:00:00}; fi
    MUSE_SELF="$SELF"
    export MUSE_SELF MUSE_ROOT CONTAINER MODEL MODEL_PATH WEIGHTS PREFILL_WEIGHTS DECODE_WEIGHTS \
           SERVED_NAME SERVE_FLAGS EXTRA_SERVE KV_BUFFER_DEVICE EXTRA_BODY MAX_NUM_SEQS SERVER_PYTHONPATH REQ_TIMEOUT \
           SAMPLING_GUARD \
           SHARD NUM_SHARDS NNODES TP MASTER_PORT \
           HF_CACHE PYDEPS HARNESSES \
           BENCH TAG LIMIT REPEATS SETTING MODE MAX_TOKENS MAX_MODEL_LEN WORKERS \
           RULER_TASKS RULER_SEQLENS RULER_SAMPLES RULER_THINK RULER_DATA \
           LONG PORT LOGS REPO_ROOT QOS DISAGG READY_POLLS
    echo "logs -> $LOGS   ($PART, $TIME, qos=$QOS, nodes=$ALLOC_NODES, gpus/node=$NGPU, tp=$TP)"
    SEG=()
    if [ "$ALLOC_NODES" -gt 1 ]; then
        S=${SEGMENT:-$ALLOC_NODES}
        [ "$S" -gt 16 ] && S=16
        SEG=(--segment="$S")
    fi
    CMT=(); [ -n "$COMMENT" ] && CMT=(--comment="$COMMENT")
    exec sbatch --export=ALL --partition="$PART" --time="$TIME" --qos="$QOS" "${CMT[@]}" \
        --nodes="$ALLOC_NODES" --gpus-per-node="$NGPU" "${SEG[@]}" --job-name="eval-$LABEL" \
        --output="$LOGS/%j.out" --error="$LOGS/%j.err" "$SELF"
fi

# ---------------------------------------------------------------------------
# HOST mode: re-enter inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null && [ -z "${MUSE_IN_CONTAINER:-}" ]; then
    # `exit` after the first match is required: scontrol prints every array record for
    # the last element, and a multi-line path kills the task with exit 127.
    # MUSE_SELF first, scontrol only as a fallback. scontrol talks to slurmctld, and
    # under a burst of submissions it TIMES OUT -- the same "Socket timed out on
    # send/recv operation" seen at submit time. The failure was silent and lethal: an
    # empty SCRIPT_PATH ran `bash ""`, which exits 127 after the engines were already
    # up, so five multi-node jobs died having loaded their weights. Retried, then
    # asserted, so it can never again reach the srun as an empty string.
    SCRIPT_PATH="${MUSE_SELF:-}"
    for _try in 1 2 3; do
        [ -n "$SCRIPT_PATH" ] && break
        SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                      | awk -F= '/Command=/{print $2; exit}')
        [ -n "$SCRIPT_PATH" ] || sleep 5
    done
    if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then
        echo "ERROR: cannot resolve this script's path (MUSE_SELF='${MUSE_SELF:-}')" >&2
        exit 1
    fi
    export MUSE_IN_CONTAINER=1

    # MULTI-NODE DISAGGREGATED, 1P1D. Each engine is itself a multi-node vLLM instance,
    # so the pair is four srun steps in one allocation: prefill across the first NNODES
    # nodes, decode across the last NNODES, the proxy on the head node, and the client
    # beside it. run_nixl_server.sh cannot express this -- it places both engines on one
    # host by GPU index.
    #
    # --container-env is how the UCX overrides reach the container. The site injects
    # UCX_TLS=tcp / UCX_NET_DEVICES into every enroot job, and without naming them here
    # pyxis applies those defaults over whatever we exported, which puts the KV back on
    # the RDMA path that NVBug 6261319 breaks.
    if [ "$DISAGG" = 1 ] && [ "$PD_MULTINODE" = 1 ]; then
        mapfile -t ALLNODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
        if [ "${#ALLNODES[@]}" -lt "$((NNODES * 2))" ]; then
            echo "ERROR: need $((NNODES * 2)) nodes for a 1P1D pair, got ${#ALLNODES[@]}" >&2
            exit 1
        fi
        PN=$(IFS=,; echo "${ALLNODES[*]:0:$NNODES}")
        DN=$(IFS=,; echo "${ALLNODES[*]:$NNODES:$NNODES}")
        PHEAD=${ALLNODES[0]}; DHEAD=${ALLNODES[$NNODES]}
        PIP=$(getent hosts "$PHEAD" | awk '{print $1; exit}')
        DIP=$(getent hosts "$DHEAD" | awk '{print $1; exit}')
        [ -z "$PIP" ] || [ -z "$DIP" ] && { echo "ERROR: cannot resolve engine heads" >&2; exit 1; }
        echo "=== 1P1D: prefill=[$PN] decode=[$DN] ==="

        PM=${PREFILL_WEIGHTS:-$MODEL_PATH}
        DM=${DECODE_WEIGHTS:-$MODEL_PATH}
        # The connector hashes each engine's config and rejects a mismatch; the model PATH
        # is in that hash, so a heterogeneous pair is rejected for differing in the one
        # field meant to differ. Only opened when the paths actually differ.
        if [ "$PM" != "$DM" ]; then
            export NIXL_ENFORCE_COMPAT=0
            echo "heterogeneous pair: prefill=$(basename "$PM") decode=$(basename "$DM")"
        fi

        export UCX_TLS="${NIXL_UCX_TLS:-cuda_ipc,cuda_copy,sm,self,tcp}"
        export UCX_CUDA_IPC_ENABLE_MNNVL="${NIXL_UCX_CUDA_IPC_ENABLE_MNNVL:-y}"
        CENV="--container-env=UCX_TLS,UCX_CUDA_IPC_ENABLE_MNNVL"
        CMNT="--container-mounts=/scratch:/scratch,/lustre:/lustre"
        export SERVED_NAME TP NNODES MAX_MODEL_LEN MAX_NUM_SEQS KV_BUFFER_DEVICE
        export EXTRA_SERVE_ARGS="$SERVE_FLAGS --generation-config auto"
        export LOG_DIR="$LOGS"
        # A 500-port block derived from the JOB ID, not a fixed 8500. The first attempt
        # used 8800/8900 for the two rendezvous stores and the decode engine died with
        #   DistNetworkError: server socket has failed to listen ... port: 8900 EADDRINUSE
        # while the prefill engine's 8800 on another node came up fine -- something
        # outside the job already held that port on that node. Ports in the 8000s are
        # common service ports, and two of these jobs on one node would collide anyway.
        # The block is 500 wide because the NIXL side channel reserves one port per TP
        # rank from its base, so the engines need room, not just single ports.
        PBASE=${PORT_BASE:-$((20000 + (SLURM_JOB_ID % 80) * 500))}
        echo "port block: $PBASE-$((PBASE + 499))"

        ROLE=kv_producer ENGINE=prefill MODEL_PATH="$PM" KV_HOST="$PIP" \
        MASTER_ADDR="$PIP" MASTER_PORT=$((PBASE + 300)) \
        PORT=$((PBASE + 0)) SIDE_PORT=$((PBASE + 100)) \
        srun --overlap --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
            --nodelist="$PN" --container-image="$CONTAINER" --no-container-mount-home \
            $CMNT $CENV --export=ALL "$MUSE_ROOT/serving/nixl_engine.sh" &
        PSRV=$!

        ROLE=kv_consumer ENGINE=decode MODEL_PATH="$DM" KV_HOST="$DIP" \
        MASTER_ADDR="$DIP" MASTER_PORT=$((PBASE + 400)) \
        PORT=$((PBASE + 1)) SIDE_PORT=$((PBASE + 200)) \
        srun --overlap --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
            --nodelist="$DN" --container-image="$CONTAINER" --no-container-mount-home \
            $CMNT $CENV --export=ALL "$MUSE_ROOT/serving/nixl_engine.sh" &
        DSRV=$!
        trap 'kill $PSRV $DSRV ${PXY:-} 2>/dev/null' EXIT

        # Wait for both engines to answer before starting the proxy: nixl_proxy.py probes
        # them at startup and exits if either is absent.
        for ep in "$PIP:$((PBASE + 0))" "$DIP:$((PBASE + 1))"; do
            for i in $(seq 1 "$READY_POLLS"); do
                curl -sf "http://$ep/v1/models" >/dev/null 2>&1 && break
                kill -0 $PSRV 2>/dev/null && kill -0 $DSRV 2>/dev/null || {
                    echo "ENGINE DIED before $ep answered"
                    tail -40 "$LOGS"/prefill_rank0.log "$LOGS"/decode_rank0.log 2>/dev/null
                    exit 1; }
                sleep 10
            done
            curl -sf "http://$ep/v1/models" >/dev/null 2>&1 || {
                echo "ENGINE TIMEOUT waiting for $ep"
                tail -40 "$LOGS"/prefill_rank0.log "$LOGS"/decode_rank0.log 2>/dev/null
                exit 1; }
            echo "  engine up: $ep"
        done

        srun --overlap --nodes=1 --ntasks=1 --nodelist="$PHEAD" \
            --container-image="$CONTAINER" --no-container-mount-home $CMNT --export=ALL \
            python3 "$MUSE_ROOT/serving/nixl_proxy.py" --workers 8 \
                --port $((PBASE + 2)) \
                --prefill-host "$PIP" --prefill-port $((PBASE + 0)) \
                --decode-host "$DIP" --decode-port $((PBASE + 1)) \
            > "$LOGS/proxy.log" 2>&1 &
        PXY=$!
        for i in $(seq 1 60); do
            curl -sf "http://$PIP:$((PBASE + 2))/v1/models" >/dev/null 2>&1 && break
            sleep 5
        done
        curl -sf "http://$PIP:$((PBASE + 2))/v1/models" >/dev/null 2>&1 || {
            echo "PROXY TIMEOUT"; tail -30 "$LOGS/proxy.log"; exit 1; }
        echo "=== 1P1D up on http://$PIP:$((PBASE + 2))/v1 ==="

        export MUSE_EXTERNAL_BASE="http://$PIP:$((PBASE + 2))/v1"
        export MUSE_GUARD_LOGS="$LOGS/prefill_rank0.log $LOGS/decode_rank0.log"
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$PHEAD" \
            --container-image="$CONTAINER" --no-container-mount-home $CMNT \
            --export=ALL bash "$SCRIPT_PATH"
        RC=$?
        kill $PSRV $DSRV $PXY 2>/dev/null
        exit $RC
    fi

    # MULTI-NODE, single engine. The server cannot live inside the client's task the way
    # it does on one node: it needs one task PER NODE, and the client needs exactly one.
    # So they become two srun steps in the same allocation -- the server across all
    # nodes, the client on rank 0 with --overlap, since the server step already holds
    # every GPU and without it Slurm queues the client behind its own job forever.
    if [ "$NNODES" -gt 1 ] && [ "$DISAGG" = 0 ]; then
        HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
        # The routable address, not the hostname: it is what the other ranks hand to
        # torch.distributed, and a name that resolves differently inside the container
        # than out of it hangs the rendezvous with no error.
        MASTER_ADDR=$(getent hosts "$HEAD" | awk '{print $1; exit}')
        [ -z "$MASTER_ADDR" ] && { echo "ERROR: cannot resolve head node $HEAD" >&2; exit 1; }
        # --generation-config auto is appended HERE for the same reason it is appended in
        # container mode below: it is what makes the server, not the harness, decide
        # sampling. This path never reaches that line, so omitting it here would quietly
        # measure vLLM's default sampling instead of the vendor's.
        export MODEL_PATH SERVED_NAME TP NNODES PORT MAX_MODEL_LEN MAX_NUM_SEQS \
               MASTER_ADDR MASTER_PORT
        export EXTRA_SERVE_ARGS="$SERVE_FLAGS --generation-config auto"
        export LOG_DIR="$LOGS"
        echo "=== multi-node server: $NNODES nodes, tp=$TP, master=$MASTER_ADDR:$MASTER_PORT ==="
        srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
            --container-image="$CONTAINER" --no-container-mount-home \
            --container-mounts="/scratch:/scratch,/lustre:/lustre" --export=ALL \
            "$MUSE_ROOT/serving/run_multinode_server.sh" &
        SRV=$!
        trap 'kill $SRV 2>/dev/null' EXIT
        # Rank 0 serves on the head node, and the client runs there too, so loopback
        # reaches it. MUSE_EXTERNAL_BASE is what tells container mode not to start a
        # second server of its own.
        export MUSE_EXTERNAL_BASE="http://127.0.0.1:$PORT/v1"
        srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD" \
            --container-image="$CONTAINER" --no-container-mount-home \
            --container-mounts="/scratch:/scratch,/lustre:/lustre" --export=ALL bash "$SCRIPT_PATH"
        RC=$?
        kill $SRV 2>/dev/null
        exit $RC
    fi

    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/scratch:/scratch,/lustre:/lustre" --export=ALL bash "$SCRIPT_PATH"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode
# ---------------------------------------------------------------------------
export HF_HOME=$HF_CACHE
export HF_HUB_OFFLINE=1        # everything needed is cached; a miss should fail loudly
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Results are keyed by MODEL first. Without that level two models writing the same
# benchmark and tag land in one directory and silently average together.
RESULTS="$MUSE_ROOT/results/$MODEL/$BENCH/$TAG"
[ "$BENCH" = mmmu ] && RESULTS="$MUSE_ROOT/results/$MODEL/$BENCH/${SETTING}_${MODE}/$TAG"
mkdir -p "$RESULTS"
SRVLOG="${LOGS:-$REPO_ROOT/logs/muse}/server_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$SRVLOG")"

echo "node=$(hostname) model=$MODEL bench=$BENCH tag=$TAG weights=$MODEL_PATH"
echo "serve flags: $SERVE_FLAGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- server ---------------------------------------------------------------
# PYTHONPATH is NOT set for the servers on purpose. $PYDEPS carries its own
# huggingface-hub and numpy; in front of the container's they shadow vLLM's copies and
# it refuses to start. Only the client below gets the overlay.
#
# SERVE_FLAGS comes from models.json and is not optional: it carries the parsers each
# model needs to be servable at all. --generation-config auto is appended for every
# model -- it is what makes the server, not the harness, decide sampling.
SERVE_FLAGS="$SERVE_FLAGS --generation-config auto"

if [ -n "${MUSE_EXTERNAL_BASE:-}" ]; then
    # The multi-node branch above already started the server as its own srun step across
    # every node. Nothing to launch here, and nothing to kill on the way out -- the
    # launching shell owns that process.
    BASE="$MUSE_EXTERNAL_BASE"
    for i in $(seq 1 "$READY_POLLS"); do
        curl -sf "$BASE/models" >/dev/null 2>&1 && break
        sleep 10
    done
    curl -sf "$BASE/models" >/dev/null || {
        echo "SERVER TIMEOUT (multi-node)"; tail -60 "$LOGS/server_rank0.log" 2>/dev/null; exit 1; }
    # Rank 0 is the only rank that runs an API server, so it is the only log that reports
    # the sampling defaults the guard below checks. On the 1P1D path the launcher names
    # both engines' rank-0 logs instead, so a half-configured pair cannot pass.
    if [ -n "${MUSE_GUARD_LOGS:-}" ]; then
        read -r -a GUARD_LOGS <<< "$MUSE_GUARD_LOGS"
    else
        GUARD_LOGS=("$LOGS/server_rank0.log")
    fi
elif [ "$DISAGG" = 1 ]; then
    READY="$LOGS/nixl_ready_${SLURM_JOB_ID}"
    rm -f "$READY"
    PM=${PREFILL_WEIGHTS:-$MODEL_PATH}
    DM=${DECODE_WEIGHTS:-$MODEL_PATH}
    # NixlConnector hashes each engine's configuration and refuses the handshake unless
    # the hashes match. That hash covers the MODEL PATH STRING, so an NVFP4 prefill and
    # an NVFP4A16 decode are rejected for differing in the one field that is supposed to
    # differ. Everything the KV transfer actually depends on -- dtype, num_kv_heads,
    # head_size, num_hidden_layers, cache_dtype -- is identical: both checkpoints
    # quantize weights only and leave the KV cache bf16 (kv_cache_scheme: null).
    #
    # Opened ONLY when the two paths differ, so a homogeneous run that somehow becomes
    # mismatched still fails loudly instead of transferring corrupt KV.
    # export, NOT a `VAR=0 cmd` prefix built in an array. Bash decides whether a word is
    # an assignment while PARSING, before expansion, so a quoted "${arr[@]}" holding
    # `NIXL_ENFORCE_COMPAT=0` is passed as a positional ARGUMENT -- which
    # run_nixl_server.sh rejects as an unknown flag, and which would have looked like a
    # flag-parsing bug rather than a quoting one.
    if [ "$PM" != "$DM" ]; then
        export NIXL_ENFORCE_COMPAT=0
        echo "heterogeneous pair: prefill=$(basename "$PM") decode=$(basename "$DM")"
    fi
    # evals/serving, not qad/serving: the qad copy forces a cpu KV buffer, which its
    # results were measured with and which a hybrid SSM model cannot use at all.
    LOG_DIR="$LOGS" SERVED_NAME="$SERVED_NAME" MAX_NUM_SEQS="$MAX_NUM_SEQS" GPU_MEM_UTIL=0.90 \
    NIXL_KV_BUFFER_DEVICE="$KV_BUFFER_DEVICE" \
    "$MUSE_ROOT/serving/run_nixl_server.sh" \
        --prefill-model "$PM" --decode-model "$DM" \
        --tokenizer "$MODEL_PATH" --served-name "$SERVED_NAME" \
        --max-model-len "$MAX_MODEL_LEN" --ready-file "$READY" --tp "$TP" \
        --extra-serve-args "$SERVE_FLAGS" > "$SRVLOG" 2>&1 &
    SRV=$!
    trap 'kill $SRV 2>/dev/null' EXIT
    for i in $(seq 1 "$READY_POLLS"); do
        [ -s "$READY" ] && break
        kill -0 $SRV 2>/dev/null || { echo "STACK DIED"; tail -60 "$SRVLOG"; \
            tail -40 "$LOGS"/prefill.log "$LOGS"/decode.log 2>/dev/null; exit 1; }
        sleep 10
    done
    [ -s "$READY" ] || { echo "STACK TIMEOUT"; tail -60 "$SRVLOG"; exit 1; }
    BASE="http://127.0.0.1:$(cat "$READY")/v1"
    # The sampling guard reads the PREFILL engine's log. Checking only one engine would
    # be enough for sampling (decode does the sampling), but both are grepped below so a
    # half-configured pair cannot pass.
    GUARD_LOGS=("$LOGS/prefill.log" "$LOGS/decode.log")
else
    # $TP, not a hardcoded 1. The submit line has always PRINTED tp=$TP, but this path
    # pinned the server to one GPU, so any single-node model asking for tp>1 silently ran
    # on 1 of the 4 GPUs the QOS floor allocates anyway. Harmless for the models that
    # leave tensor_parallel_size unset (TP resolves to 1), fatal for
    # nemotron-3-super-120b: 120B on one GPU left only 458 Mamba cache blocks against
    # max_num_seqs 512, and the engine refused to capture CUDA graphs.
    vllm serve "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.90 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        $SERVE_FLAGS \
        --host 127.0.0.1 --port "$PORT" > "$SRVLOG" 2>&1 &
    SRV=$!
    trap 'kill $SRV 2>/dev/null' EXIT
    BASE="http://127.0.0.1:$PORT/v1"
    for i in $(seq 1 "$READY_POLLS"); do
        curl -sf "$BASE/models" >/dev/null 2>&1 && break
        kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -50 "$SRVLOG"; exit 1; }
        sleep 10
    done
    curl -sf "$BASE/models" >/dev/null || { echo "SERVER TIMEOUT"; tail -50 "$SRVLOG"; exit 1; }
    GUARD_LOGS=("$SRVLOG")
fi
echo "=== stack up on $BASE ==="

# GUARD, not decoration. The whole point of omitting sampling params client-side is that
# the server supplies the vendor's. If --generation-config auto ever stops taking
# effect, every request silently falls back to vLLM's defaults and the numbers are
# quietly not comparable to the published ones. Fail here instead of discovering it from
# a score that is merely a bit low.
#
# The pattern stops at "overridden by the model" on purpose. vLLM's actual line is
#   Default vLLM sampling parameters have been overridden by the model's
#   `generation_config.json`: {'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}
# and matching further -- "...by the model's generation_config" -- fails on the
# BACKTICK, which is how this guard first fired against a correctly configured server.
# A guard that cries wolf gets deleted, so it matches the stable part of the sentence.
#
# The wait is separate from the readiness poll: /v1/models starts answering before the
# API server has finished logging its sampling defaults, so grepping the instant the
# endpoint comes up is a race the guard loses.
GUARD_RE="overridden by the model"
if [ "${SAMPLING_GUARD:-}" = "False" ]; then
    echo "  sampling guard DISABLED for $MODEL (models.json: sampling_guard=false)."
    echo "  This model ships no vendor sampling overrides, so there is nothing to assert;"
    echo "  the server's own defaults are what is measured."
    GUARD_LOGS=()
fi
for gl in "${GUARD_LOGS[@]}"; do
    for _ in $(seq 1 30); do
        grep -q "$GUARD_RE" "$gl" 2>/dev/null && break
        sleep 2
    done
    if ! grep -q "$GUARD_RE" "$gl" 2>/dev/null; then
        echo "ERROR: $gl never reported applying the model's generation_config." >&2
        grep -iE "generation.config|sampling" "$gl" | tail -10 >&2
        exit 1
    fi
    echo "  $(basename "$gl"): $(grep "$GUARD_RE" "$gl" | tail -1)"
done

# --- client ---------------------------------------------------------------
export PYTHONPATH="$PYDEPS"
# The IFBench scorer calls nltk.download(quiet=True) at import. Compute nodes have no
# network, so that call fails SILENTLY and the failure only surfaces later as a
# LookupError from sent_tokenize. setup_harnesses.sh pre-populates this directory.
export NLTK_DATA="$HARNESSES/IFBench/ifbench/.nltk_data"
export EVAL_BASE_URL="$BASE" EVAL_WORKERS="$WORKERS" EVAL_MODEL="$SERVED_NAME"
export EVAL_TIMEOUT="$REQ_TIMEOUT"
# Model-specific NON-SAMPLING request fields (models.json request_extra_body).
# Printed, not silent: a stop condition that only exists in the request body is
# invisible in the server log, so the one place it can be seen after the fact is
# this line in the job output.
export EVAL_EXTRA_BODY="$EXTRA_BODY"
[ "$EXTRA_BODY" != "{}" ] && echo "request extra_body: $EXTRA_BODY"
LIM=(); [ "$LIMIT" != 0 ] && LIM=(--limit "$LIMIT")
RC=0

case "$BENCH" in
  gpqa)
    python3 "$MUSE_ROOT/drivers/gpqa_diamond.py" --out "$RESULTS" \
        --repeats "$REPEATS" --max-tokens "$MAX_TOKENS" --workers "$WORKERS" \
        --base-url "$BASE" "${LIM[@]}" ; RC=$?
    ;;
  ifbench)
    python3 "$MUSE_ROOT/drivers/ifbench_gen.py" --out "$RESULTS" \
        --max-tokens "$MAX_TOKENS" --workers "$WORKERS" \
        --base-url "$BASE" "${LIM[@]}" ; RC=$?
    if [ $RC -eq 0 ]; then
        # Upstream scorer, unmodified. It must run from the harness root: run_eval
        # imports evaluation_lib and the ifbench package by bare name.
        ( cd "$HARNESSES/IFBench" && \
          python3 -m run_eval --input_data=data/IFBench_test.jsonl \
              --input_response_data="$RESULTS/responses.jsonl" \
              --output_dir="$RESULTS" ) ; RC=$?
        python3 "$MUSE_ROOT/drivers/ifbench_report.py" "$RESULTS"
    fi
    ;;
  ocrbench)
    # 8192, not the model's default 32768. OCRBench answers are transcriptions, not
    # reasoning chains -- a full page is a few thousand tokens -- and 7400 items at a
    # 32k ceiling would spend most of the wall clock on a tail that never produces a
    # better score. Truncation is recorded, so raise this if the diagnostics show any.
    export OCRBENCH_HARNESS="$HARNESSES/MultimodalOCR/OCRBench_v2"
    python3 "$MUSE_ROOT/drivers/ocrbench_infer.py" --out "$RESULTS" \
        --lang "${OCRBENCH_LANG:-en}" --max-tokens "${OCRBENCH_MAX_TOKENS:-8192}" \
        --workers "$WORKERS" --base-url "$BASE" --model "$SERVED_NAME" \
        "${LIM[@]}" ; RC=$?
    # Scored inline, like every other benchmark, so one job produces one finished
    # result. This costs ~15-20 min of GPU-node wall clock on CPU-only work (uncapped
    # O(n^2) edit distance over full-page transcriptions, plus TEDS on tables) -- a
    # deliberate trade of GPU-hours for a single-step workflow. bin/score_ocrbench.sh
    # remains for re-scoring existing generations without regenerating them.
    #
    # Runs even on a non-zero rc: a partial file still yields a partial per-category
    # table plus the count it rests on, which is what decides whether resuming is worth
    # it. The scorer never mutates the generations.
    python3 "$MUSE_ROOT/drivers/ocrbench_score.py" "$RESULTS"
    ;;
  mmlu_pro)
    # max_tokens is the MODEL's, not the official script's 4096. These are reasoning
    # models: muse's GPQA p50 is 3907 tokens and its p99 reaches the 32768 ceiling, so a
    # 4096 cap would truncate roughly half of every reasoning arm -- and a truncated
    # answer comes back with EMPTY content that scores as wrong, which is
    # indistinguishable from a model that answered badly.
    #
    # Scoring is a regex per row, so it runs inline in the same process rather than as a
    # separate scorer. It globs every shard in the directory, so whichever shard finishes
    # last writes a summary over the complete set and the earlier ones write partial
    # summaries that are simply superseded.
    python3 "$MUSE_ROOT/drivers/mmlu_pro_infer.py" --out "$RESULTS" \
        --max-tokens "$MAX_TOKENS" --workers "$WORKERS" \
        --base-url "$BASE" --model "$SERVED_NAME" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS" "${LIM[@]}" ; RC=$?
    ;;
  mmmu)
    export MMMU_HARNESS="$HARNESSES/MMMU/mmmu-pro"
    python3 "$MUSE_ROOT/drivers/mmmu_pro_infer.py" --mode "$MODE" \
        --setting "$SETTING" --out "$RESULTS" --max-tokens "$MAX_TOKENS" \
        --workers "$WORKERS" --base-url "$BASE" "${LIM[@]}" ; RC=$?
    # Scored even when generation returned non-zero: a partial file is still worth a
    # partial score plus its coverage count, and the alternative is finding out how far
    # a 24h job got only by counting lines by hand.
    python3 "$MUSE_ROOT/drivers/mmmu_pro_score.py" "$RESULTS"
    ;;
  ruler)
    export RULER_HARNESS="$HARNESSES/RULER"
    # DATA IS GENERATED, NOT DOWNLOADED, and it is generated PER TOKENIZER: RULER sizes
    # every haystack to hit the target length in the model's own tokens. It lives
    # outside results/ because it is an input, not an output.
    #
    # KEYED BY THE TOKENIZER'S CONTENT, not by the weights directory. The corpus is a
    # pure function of the tokenizer, and every arm of a quantization sweep shares one:
    # the base model and all four GSQ-RCO dequantized checkpoints here have byte-
    # identical tokenizer.json. Keying on the weights path made each arm regenerate the
    # same multi-GB corpus from scratch -- hours of prep per arm, with the GPUs idle,
    # producing files that compare equal.
    #
    # Hashing rather than assuming, because tokenizers do differ in practice and a
    # wrongly shared corpus would mis-size every context silently: the NVFP4 checkpoint
    # in this tree has a tokenizer llm-compressor rewrote on save, with a different
    # pre_tokenizer regex (drops \p{M}) and different ByteLevel decoder flags. A content
    # hash shares what is genuinely identical and separates what is not, with no list to
    # maintain.
    if [ -z "${RULER_DATA:-}" ]; then
        RULER_TOK_KEY=$(cat "$MODEL_PATH/tokenizer.json" \
                            "$MODEL_PATH/tokenizer_config.json" 2>/dev/null \
                        | md5sum | cut -c1-12)
        [ -z "$RULER_TOK_KEY" ] && { echo "ERROR: no tokenizer at $MODEL_PATH" >&2; exit 2; }
        RULER_DATA=$MUSE_ROOT/data/ruler/tok-$RULER_TOK_KEY
        mkdir -p "$RULER_DATA"
        # Which checkpoints map here, so an opaque hash directory stays legible.
        grep -qxF "$MODEL_PATH" "$RULER_DATA/TOKENIZERS.txt" 2>/dev/null || \
            echo "$MODEL_PATH" >> "$RULER_DATA/TOKENIZERS.txt"
        echo "ruler data: $RULER_DATA (tokenizer $RULER_TOK_KEY)"
    fi
    # prepare.py's generators import their siblings by bare name (niah.py does
    # `from manifest_utils import write_manifest`), so scripts/data must be on the path
    # and prepare.py must run from there -- it resolves ../synthetic.yaml relative to
    # its own file, not to $PWD.
    # RULER's generators need nltk/wonderwords/tenacity, which the container does not
    # carry; RULERDEPS is --no-deps so it adds exactly those and shadows nothing. It
    # deliberately does NOT carry `regex`: nltk imports it, but the container's own copy
    # is the one built for this interpreter, and an overlaid wheel shadows it with a
    # _regex extension that cannot load.
    RULERDEPS=${RULERDEPS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/muse-rulerdeps}
    export NLTK_DATA=${NLTK_DATA:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/nltk_data}
    # RULERDEPS/bin holds a `python` -> python3 shim, and it is load-bearing:
    # prepare.py builds its generator command as a literal `python {script}` string
    # (prepare.py:124) and runs it through a shell that only has python3. Worse, it does
    # not check the return code -- the failure surfaces as an indented "Error output:
    # /bin/sh: 1: python: not found" line and is then followed by
    # "Prepare <task> with lines: N", and prepare.py exits 0. Measured: six tasks all
    # reported success and generated ZERO files. Without the shim this bench silently
    # evaluates an empty dataset.
    export PATH="$RULERDEPS/bin:$PATH"
    for SL in ${RULER_SEQLENS//,/ }; do
      for T in ${RULER_TASKS//,/ }; do
        # The DOC COUNT is the cache key, not mere existence. prepare.py's own skip is
        # `len(data) == num_samples`, so it would regenerate a short file -- but this
        # check runs FIRST and would short-circuit it. A 3-doc file left by a smoke test
        # would then be silently evaluated as if it were the full 500, and the only
        # visible trace is an n= in the summary that nobody reads. Not hypothetical: the
        # smoke run in this tree left exactly such a directory.
        HAVE=0
        [ -s "$RULER_DATA/$SL/$T/validation.jsonl" ] && \
            HAVE=$(wc -l < "$RULER_DATA/$SL/$T/validation.jsonl")
        if [ "$HAVE" = "$RULER_SAMPLES" ]; then continue; fi
        [ "$HAVE" != 0 ] && echo "  ruler prepare: $T @ $SL (have $HAVE, want $RULER_SAMPLES)"
        echo "  ruler prepare: $T @ $SL"
        # timeout, because qa.py CAN HANG FOREVER rather than fail. Its per-sample retry
        # only shrinks used_docs while `used_docs > incremental` (10), so once the length
        # cannot hold the task's mandatory documents the handler changes nothing and the
        # loop spins at 100% CPU with no output. HotpotQA's context is all 10 distractor
        # paragraphs, so qa_2 at 4096 fails for every sample with this tokenizer:
        # measured at 2h05m and zero bytes, while the same task finished at 8192 in 9.3
        # minutes. Untimed, that consumes the GPU job's entire walltime before a single
        # token is generated. See RULER_TASKS if a task is unconstructible at a length.
        ( cd "$RULER_HARNESS/scripts/data" && \
          PYTHONPATH="$RULER_HARNESS/scripts/data:$RULERDEPS:$PYTHONPATH" \
          timeout "${RULER_PREP_TIMEOUT:-5400}" \
          python3 prepare.py --save_dir "$RULER_DATA/$SL" --benchmark synthetic \
              --task "$T" --tokenizer_path "$MODEL_PATH" --tokenizer_type hf \
              --max_seq_length "$SL" --num_samples "$RULER_SAMPLES" \
              --model_template_type base )
        [ $? = 124 ] && echo "  ruler prepare TIMED OUT: $T @ $SL" >&2
        # Checked explicitly BECAUSE prepare.py's exit code cannot be trusted (above).
        if [ ! -s "$RULER_DATA/$SL/$T/validation.jsonl" ]; then
            echo "ERROR: ruler prepare produced nothing for $T @ $SL" >&2
            RC=1; break 2
        fi
      done
    done
    # --model_template_type base, NOT one of RULER's baked-in chat templates: the
    # request goes through /chat/completions, so the SERVER applies this model's own
    # template. Picking a template here would apply a second, wrong one inside the user
    # turn -- RULER's list predates this model and has no entry for it.
    [ "${RC:-0}" = 0 ] && { python3 "$MUSE_ROOT/drivers/ruler_infer.py" --out "$RESULTS" \
        --data-dir "$RULER_DATA" --seqlens "$RULER_SEQLENS" --tasks "$RULER_TASKS" \
        --workers "$WORKERS" --base-url "$BASE" --model "$SERVED_NAME" \
        $RULER_THINK_FLAG "${LIM[@]}" ; RC=$?; }
    python3 "$MUSE_ROOT/drivers/ruler_score.py" "$RESULTS"
    ;;
esac

# EXPORT THE PER-ITEM SCORES AS PART OF THE RUN, not as a step someone remembers.
#
# evals/scores is what a CLONE reads: the generations are gitignored, so off-cluster
# every paired test and every table falls back to the export. Left as a manual step it
# silently drifts -- it held the pre-archive b200 tags for four models and had never
# heard of the other four, so a table built from it printed stale numbers for half the
# rows and "-" for the rest, both of which look like legitimate output.
#
# Scoped to this model and bench so it costs seconds, and run even when RC is non-zero:
# a partial arm is exactly the one a reader must be able to see is partial.
if [ -x "$MUSE_ROOT/bin/export_scores.py" ] || [ -f "$MUSE_ROOT/bin/export_scores.py" ]; then
    python3 "$MUSE_ROOT/bin/export_scores.py" --model "$MODEL" --benches "$BENCH" \
        || echo "WARNING: score export failed for $MODEL/$BENCH; evals/scores is now stale" >&2
fi

# Verify the disaggregation AFTER generating, not before: the evidence is the engines'
# own throughput and external-prefix-hit counters accumulated under load, and there is
# nothing to read until traffic has flowed. Runs even when RC is non-zero -- a partial
# run still answers "was this really disaggregated", and that answer is what decides
# whether the generations already on disk are worth resuming.
if [ "$DISAGG" = 1 ]; then
    # verify_disagg.py opens <log_dir>/prefill.log and decode.log by name. The multi-node
    # launcher writes one log per rank, and rank 0 is the only one that serves and so the
    # only one carrying the throughput and external-prefix-hit counters the check reads.
    for e in prefill decode; do
        [ -f "$LOGS/$e.log" ] || [ ! -f "$LOGS/${e}_rank0.log" ] || ln -sf "${e}_rank0.log" "$LOGS/$e.log"
    done
    python3 "$MUSE_ROOT/drivers/verify_disagg.py" "$LOGS" --out "$RESULTS" || {
        echo "ERROR: this run is labelled disaggregated but the engines do not show a" >&2
        echo "       KV transfer. The scores in $RESULTS describe something else." >&2
        RC=1
    }
fi

echo "=== rc=$RC  results in $RESULTS ==="
# ${SRV:-}: on the multi-node path the server is a separate srun step owned by the
# launching shell, so this branch never set SRV. Under `set -u` a bare $SRV would abort
# here -- AFTER the scores are written and the job is otherwise a success, which is the
# worst place to fail.
kill ${SRV:-} 2>/dev/null
exit $RC
