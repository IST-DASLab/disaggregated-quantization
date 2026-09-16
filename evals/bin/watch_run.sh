#!/bin/bash
# Block until one PHASE of a running eval finishes, then print what it cost and exit.
#
#   ./bin/watch_run.sh --logs <logdir> --phase loading
#   ./bin/watch_run.sh --logs <logdir> --phase compile
#   ./bin/watch_run.sh --logs <logdir> --phase requests
#   ./bin/watch_run.sh --logs <logdir> --phase all --interval 60
#
# WHY PHASES AND NOT ONE TAIL. On a 2.4T model the three phases fail differently and on
# wildly different timescales: weights stream off Lustre for tens of minutes, compilation
# is a few minutes of CPU with the GPUs idle, and generation is hours. A single "is it
# done yet" gives no way to tell a slow load from a hung one, and the interesting question
# is always "which phase is it in, and is that phase still making progress".
#
# Each phase exits 0 the moment it completes, so a caller can wait on exactly one of them
# and be told, rather than polling. It exits 2 if the job disappears first -- a phase that
# will never complete must not block forever.
#
# The markers are the ones vLLM and the drivers actually print, verified against a
# completed 2-node run; each is counted rather than merely matched, because the multi-node
# case has to know that EVERY rank finished, not just the one that logs first.
set -uo pipefail

LOGS=""; PHASE="all"; INTERVAL=30; JOB=""; TP=""
while (($# > 0)); do
  case "$1" in
    --logs)     LOGS="$2";     shift 2 ;;
    --phase)    PHASE="$2";    shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --job)      JOB="$2";      shift 2 ;;
    --tp)       TP="$2";       shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$LOGS" ] || { echo "ERROR: --logs is required" >&2; exit 2; }
case "$PHASE" in loading|compile|requests|all) ;;
  *) echo "ERROR: --phase must be loading|compile|requests|all" >&2; exit 2 ;; esac

# The job id is only used to answer "is this still alive". Derived from the log directory
# so the caller does not have to thread it through, since run_eval.sh names the files
# after it.
if [ -z "$JOB" ]; then
    JOB=$(ls "$LOGS"/*.out 2>/dev/null | head -1 | xargs -r basename | sed 's/\.out$//')
fi

alive () {
    [ -z "$JOB" ] && return 0
    squeue -h -j "$JOB" -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'
}
# Counted across EVERY rank log: on the multi-node path each rank logs only its own
# workers, so rank 0 reporting four loaded workers says nothing about rank 1.
#
# ONE GLOB, not two. `server_*.log` already matches both the multi-node `server_rank0.log`
# and the single-node `server_<jobid>.log`; adding `server_rank*.log` beside it counts
# every multi-node line twice, and a doubled count reaches --tp when only half the workers
# have actually loaded -- which is the exact failure this counting exists to catch.
#
# No `|| echo 0` on the grep. `grep -c` ALREADY prints 0 when it matches nothing, and it
# also exits 1 -- so the fallback fires too and the function returns "0\n0", which every
# numeric test then rejects with "integer expression expected". The count is the output,
# never the exit status.
count_all () { cat "$LOGS"/server_*.log 2>/dev/null | grep -c "$1"; }
elapsed () { printf '%dm%02ds' $(( ($(date +%s) - T0) / 60 )) $(( ($(date +%s) - T0) % 60 )); }

T0=$(date +%s)

# --- loading ---------------------------------------------------------------
# "Model loading took N GiB memory and N seconds" is printed once per WORKER. With --tp
# given, the phase is complete only when all of them have; without it, the first one is
# taken as the signal, which is right for a single-GPU run and optimistic otherwise.
wait_loading () {
    local want=${TP:-1} n=0 last=-1
    while :; do
        n=$(count_all "Model loading took")
        [ "$n" != "$last" ] && { echo "[$(elapsed)] loading: $n/$want workers"; last=$n; }
        [ "$n" -ge "$want" ] && { echo "[$(elapsed)] LOADING DONE ($n workers)"
            cat "$LOGS"/server_rank0.log 2>/dev/null | grep -m1 "Loading weights took"; return 0; }
        alive || { echo "[$(elapsed)] job $JOB gone during loading ($n/$want)"; return 2; }
        sleep "$INTERVAL"
    done
}

# --- compile ---------------------------------------------------------------
# torch.compile then CUDA-graph capture. "init engine (profile, create kv cache, warmup
# model) took" is the line that closes the whole startup, and "Application startup
# complete" is uvicorn accepting connections -- the first is the honest end of
# compilation, the second is when the endpoint answers.
wait_compile () {
    local last=""
    while :; do
        local dyn cap init
        dyn=$(count_all "Dynamo bytecode transform time")
        cap=$(count_all "Capturing CUDA graphs")
        init=$(count_all "init engine")
        local now="dynamo=$dyn cudagraph=$cap init=$init"
        [ "$now" != "$last" ] && { echo "[$(elapsed)] compile: $now"; last="$now"; }
        if [ "$init" -gt 0 ]; then
            echo "[$(elapsed)] COMPILE DONE"
            grep -h -m1 "init engine" "$LOGS"/server_rank0.log 2>/dev/null
            grep -h -m1 "Dynamo bytecode transform time" "$LOGS"/server_rank0.log 2>/dev/null
            return 0
        fi
        alive || { echo "[$(elapsed)] job $JOB gone during compile ($now)"; return 2; }
        sleep "$INTERVAL"
    done
}

# --- requests --------------------------------------------------------------
# Progress is the RAW FILE, not the log: every driver appends and fsyncs per item, so its
# line count is the number of items actually finished and durable. The results directory
# is read out of the job's own stdout rather than reconstructed from the model/bench/tag,
# which would have to duplicate run_eval.sh's naming rules and drift from them.
wait_requests () {
    local last=-1 rd=""
    while :; do
        [ -z "$rd" ] && rd=$(grep -ho '/[^ ]*/raw[^ ]*\.jsonl' "$LOGS"/*.out 2>/dev/null \
                             | head -1 | xargs -r dirname)
        local n=0
        [ -n "$rd" ] && n=$(cat "$rd"/raw*.jsonl 2>/dev/null | wc -l)
        [ "$n" != "$last" ] && { echo "[$(elapsed)] requests: $n done"; last=$n; }
        if grep -qh "generation diagnostics" "$LOGS"/*.out 2>/dev/null; then
            echo "[$(elapsed)] REQUESTS DONE"
            grep -h "generation diagnostics" "$LOGS"/*.out | tail -1
            grep -hE "acc=|unparsed" "$LOGS"/*.out | tail -1
            return 0
        fi
        alive || { echo "[$(elapsed)] job $JOB gone during generation ($n done)"; return 2; }
        sleep "$INTERVAL"
    done
}

echo "watching $LOGS (job=${JOB:-?}, phase=$PHASE, every ${INTERVAL}s)"
RC=0
case "$PHASE" in
  loading)  wait_loading;  RC=$? ;;
  compile)  wait_compile;  RC=$? ;;
  requests) wait_requests; RC=$? ;;
  all)      wait_loading && wait_compile && wait_requests; RC=$? ;;
esac
exit $RC
