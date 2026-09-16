#!/bin/bash
# Kill eval jobs that are burning GPUs without producing anything, and say why.
#
#   ./bin/watch_evals.sh                      # one pass, report only
#   ./bin/watch_evals.sh --kill               # one pass, cancel the dead ones
#   ./bin/watch_evals.sh --kill --daemon      # keep doing that until no jobs remain
#
# WHY THIS EXISTS. `squeue` says RUNNING and a stalled job looks exactly like a healthy
# one from outside, so failures here are silent and expensive. Three have happened in
# this tree, each wasting 4 GPUs for 30+ minutes before anyone noticed:
#
#   * a decode engine segfaulted inside UCX after 82 KV transfers; the prefill engine
#     stayed alive computing KV nobody consumed, for 45 minutes;
#   * a trained prefill failed weight loading ("no module or parameter named
#     ...input_layernorm.inner"), decode came up fine, job sat at 0 rows;
#   * the same checkpoint after a fix, failing differently ("'MergedColumnParallelLinear'
#     object has no attribute 'data'") -- again 0 rows, again nothing in squeue.
#
# WHAT COUNTS AS DEAD, and why these two rules rather than a generic timeout:
#
#   DEAD-ENGINE  an engine log contains a fatal marker (EngineDeadError, Segfault,
#                "Engine core initialization failed", "prefill DIED", "STACK DIED").
#                This is unambiguous and fires within seconds of the failure.
#   NO-PROGRESS  the job is RUNNING, has been for longer than --grace, and raw.jsonl has
#                not grown in --stall seconds. --grace must exceed real startup: two
#                engines loading 25-51 GB plus CUDA-graph capture is 10-15 minutes here,
#                so the default is deliberately generous. A job still legitimately
#                loading is NOT dead.
#
# It never cancels a job that is writing rows, however slowly -- 64k RULER items take
# half a second each, and "slow" is not "stuck".
set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
LOGS="$REPO_ROOT/logs/evals"

KILL=0; DAEMON=0
GRACE=${GRACE:-1500}      # 25 min: startup is 10-15 min for a disaggregated pair
STALL=${STALL:-900}       # 15 min with no new row
EVERY=${EVERY:-300}

while (($# > 0)); do
    case "$1" in
        --kill)   KILL=1;      shift ;;
        --daemon) DAEMON=1;    shift ;;
        --grace)  GRACE="$2";  shift 2 ;;
        --stall)  STALL="$2";  shift 2 ;;
        --every)  EVERY="$2";  shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

FATAL='EngineDeadError|Segfault encountered|Engine core initialization failed|prefill DIED|decode DIED|STACK DIED|ENGINE DIED|ENGINE TIMEOUT'

pass() {
    local now killed=0 checked=0
    now=$(date +%s)
    while read -r id elapsed name; do
        [ -z "${name:-}" ] && continue
        case "$name" in eval-*) ;; *) continue ;; esac
        checked=$((checked + 1))
        # Elapsed as seconds: squeue prints [[D-]HH:]MM:SS
        local e=0 p
        IFS='-' read -r d rest <<< "${elapsed}"
        if [ "$d" = "$elapsed" ]; then rest="$elapsed"; d=0; fi
        IFS=':' read -ra p <<< "$rest"
        case "${#p[@]}" in
            3) e=$((10#${p[0]}*3600 + 10#${p[1]}*60 + 10#${p[2]})) ;;
            2) e=$((10#${p[0]}*60 + 10#${p[1]})) ;;
            *) e=0 ;;
        esac
        e=$((e + 10#$d * 86400))

        local dir reason=""
        dir=$(ls -1dt "$LOGS"/*"${name#eval-qwen3.8-27b_}" 2>/dev/null | head -1)
        [ -z "$dir" ] && dir=$(grep -l . "$LOGS"/*/"$id".out 2>/dev/null | head -1 | xargs -r dirname)
        [ -z "$dir" ] && continue

        if grep -qhE "$FATAL" "$dir"/*.log "$dir"/*.out 2>/dev/null; then
            reason="dead-engine: $(grep -hoE "$FATAL" "$dir"/*.log "$dir"/*.out 2>/dev/null | head -1)"
        elif [ "$e" -gt "$GRACE" ]; then
            local raw age=999999
            raw=$(ls -1t "$dir"/../../evals/results 2>/dev/null >/dev/null; echo "")
            # The results path is recorded in the job's own header line, so it is read
            # rather than reconstructed from the tag (mmmu nests under <setting>_<mode>).
            raw=$(grep -hoE '/[^ ]*/results/[^ ]*/raw[^ ]*\.jsonl' "$dir"/*.out 2>/dev/null | head -1)
            if [ -n "$raw" ] && [ -e "$raw" ]; then
                age=$(( now - $(stat -c %Y "$raw") ))
                [ "$age" -gt "$STALL" ] && reason="no-progress: raw.jsonl unchanged for ${age}s"
            else
                reason="no-progress: no raw.jsonl after ${e}s"
            fi
        fi

        if [ -n "$reason" ]; then
            echo "  DEAD $id  $name"
            echo "       $reason"
            if [ "$KILL" = 1 ]; then scancel "$id" && killed=$((killed + 1)); fi
        fi
    done < <(squeue -u "$USER" -h -o "%i %M %j" 2>/dev/null)
    echo "checked $checked running eval job(s); $( [ "$KILL" = 1 ] && echo "cancelled $killed" || echo "report only")"
}

if [ "$DAEMON" = 1 ]; then
    while squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q '^eval-'; do
        echo "--- $(date +%H:%M:%S) ---"; pass; sleep "$EVERY"
    done
    echo "no eval jobs left; watcher exiting"
else
    pass
fi
