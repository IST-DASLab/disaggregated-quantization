#!/bin/bash
# Submit prefill/decode evals for QAD checkpoints as soon as a given step is written.
#
#   ./bin/autosubmit_qad.sh --tag reason8k --every 250
#
# Training writes step_NNNNNNN/prefill at its own pace across eight runs, and waiting for
# all of them before submitting wastes hours of idle cluster. This polls, and for every
# (run, step) whose step is a positive multiple of --every, it prepares the serving view
# (bin/prep_qad_prefill.sh) and submits MMLU-Pro + MMMU-Pro against the matching unsloth
# decode checkpoint.
#
# MIND THE VOLUME. This is 2 jobs per (arm, step): eight arms at every 250 steps is 16
# jobs per tier, so a run reaching step 2000 submits 128 four-GPU jobs. --every is the
# only throttle; raise it if the cluster is busy.
#
# EVERY GUARD BELOW EXISTS TO STOP IT BURNING GPUS. A poll loop that submits is far more
# dangerous than one that reports, so the invariants are: submit at most once per arm,
# never submit onto a half-written checkpoint, and stop retrying anything that fails.
#
#   ONE SHOT PER ARM, VIA A MARKER WRITTEN BEFORE sbatch. The obvious readiness test --
#   "does the results directory exist yet" -- IS WRONG and was the first version of this
#   script: run_eval.sh creates RESULTS at line 622, in CONTAINER mode, which is minutes
#   after `exec sbatch`. During that window the check reads false and a 5-minute poll
#   resubmits the same arm again and again, each taking four GPUs. The marker is written
#   BEFORE submitting, not after, so a crash between the two costs one missed eval rather
#   than an unbounded resubmit loop.
#
#   FAILURES ARE TERMINAL, NOT RETRIED. A prep that fails twice, a missing decode
#   checkpoint, or an unknown hash writes a .failed marker and is never attempted again.
#   Retrying a broken setup on a schedule is precisely how a watchdog becomes the thing
#   wasting the cluster.
#
#   THE WRITE MUST SETTLE. A 25 GB model.safetensors is visible long before it is
#   complete, and a truncated one dies deep inside vLLM's loader after the job already
#   holds four GPUs. A checkpoint is ready only once config.json exists and the
#   safetensors size is unchanged across two consecutive polls.
#
#   SINGLE INSTANCE. An atomic mkdir lock stops a second copy (an accidental restart)
#   from racing the first and double-submitting.
#
# HASH -> DECODE ARM is fixed by the training config and not derivable from the
# checkpoint, so it is tabulated. An unknown hash is skipped loudly rather than guessed:
# pairing a prefill with the wrong decode yields a plausible number that is meaningless.

set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
P=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode
CKPTS=${CKPTS:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_nextgen/users/apanferov/prefill_decode/checkpoints}

TAG=""; EVERY=250; POLL=${POLL:-300}; MAXPOLL=${MAXPOLL:-288}; MAX_PREP_TRIES=${MAX_PREP_TRIES:-2}
while (($# > 0)); do
    case "$1" in
        --tag)  TAG="$2";  shift 2 ;;
        --every) EVERY="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$TAG" ] || { echo "ERROR: --tag is required" >&2; exit 2; }

STATE="$REPO_ROOT/logs/evals/.autosubmit"
mkdir -p "$STATE"
LOCK="$STATE/$TAG.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ERROR: another autosubmit for $TAG holds $LOCK; refusing to double-submit" >&2
    exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

declare -A ARM=( [54d4e3e5]=IQ1_S [a069b041]=IQ1_M [60f2e0a9]=IQ2_XXS [e2694575]=IQ2_S
                 [d27e7f43]=Q2_K_XL [4774f820]=IQ3_XXS [48aa1bba]=IQ3_S [7f23c80a]=Q3_K_XL )
declare -A SIZE=() TRIES=()

# Anything already evaluated before this script existed counts as done, so a first run
# does not resubmit work that is already on disk. Tag and step are parsed back out of the
# results directory name (<tag><step>-<arm>-d).
for d in "$MUSE_ROOT"/results/qwen3.8-27b/mmlu_pro/"$TAG"*-*-d; do
    [ -d "$d" ] || continue
    b=$(basename "$d"); b=${b#"$TAG"}; b=${b%-d}
    st=${b%%-*}; ar=${b#*-}
    case "$st" in ''|*[!0-9]*) continue ;; esac
    : > "$STATE/$TAG$st-$ar.submitted"
done

for ((i = 0; i < MAXPOLL; i++)); do
    pending=0
    for d in "$CKPTS"/${TAG}-*; do
        [ -d "$d" ] || continue
        h=${d##*-}
        A=${ARM[$h]:-}
        if [ -z "$A" ]; then
            if [ ! -e "$STATE/$TAG-unknown-$h.failed" ]; then
                echo "  SKIP unknown hash $h -- not in the arm table, refusing to guess the decode pair"
                : > "$STATE/$TAG-unknown-$h.failed"
            fi
            continue
        fi
        low=$(echo "$A" | tr 'A-Z' 'a-z' | tr -d '_')

        # Every step this run has written that is a positive multiple of --every.
        for sd in "$d"/weights/step_*; do
            [ -d "$sd" ] || continue
            N=$(basename "$sd"); N=${N#step_}; N=$((10#$N))
            [ "$N" -gt 0 ] || continue
            [ $((N % EVERY)) -eq 0 ] || continue

            M="$STATE/$TAG$N-$low"
            if [ -e "$M.submitted" ] || [ -e "$M.failed" ]; then continue; fi
            pending=$((pending + 1))

            C="$sd/prefill"
            [ -s "$C/config.json" ] && [ -s "$C/model.safetensors" ] || continue
            key="$h:$N"
            sz=$(stat -c %s "$C/model.safetensors" 2>/dev/null || echo 0)
            if [ "${SIZE[$key]:-}" != "$sz" ]; then
                SIZE[$key]=$sz
                echo "  $A step $N: appearing ($((sz / 1000000)) MB) -- waiting for the write to settle"
                continue
            fi

            D="$P/models/Qwen3.8-27B-unsloth-UD-$A-bf16"
            if [ ! -s "$D/model.safetensors.index.json" ]; then
                echo "  $A: no decode checkpoint at $D -- marking failed, will not retry" >&2
                : > "$M.failed"; continue
            fi

            echo "=== $A step $N ready -- preparing ==="
            if ! "$_SELF_DIR/prep_qad_prefill.sh" --ckpt "$C" --out-name "$TAG$N-$A" 2>&1 | sed 's/^/    /'; then
                TRIES[$key]=$(( ${TRIES[$key]:-0} + 1 ))
                if [ "${TRIES[$key]}" -ge "$MAX_PREP_TRIES" ]; then
                    echo "  $A step $N: prep failed ${TRIES[$key]}x -- marking failed, will not retry" >&2
                    : > "$M.failed"
                else
                    echo "  $A step $N: prep failed (attempt ${TRIES[$key]}), will retry once" >&2
                fi
                continue
            fi

            # Marker BEFORE sbatch: see the header. One missed eval beats a resubmit loop.
            : > "$M.submitted"
            V="$P/models/$TAG$N-$A-prefill"
            for bench in mmlu_pro mmmu; do
                EXTRA=(--long); [ "$bench" = mmmu ] && EXTRA=()
                if ! "$_SELF_DIR/run_eval.sh" --bench "$bench" --model qwen3.8-27b --disagg --tp 2 \
                        "${EXTRA[@]}" --prefill-weights "$V" --decode-weights "$D" \
                        --tag "$TAG$N-$low-d" 2>&1 | tail -1; then
                    echo "  $A step $N/$bench: sbatch FAILED -- not retried (marker already set)" >&2
                fi
            done
        done
    done
    # Training still running means more steps are coming, so an empty pass is not done.
    if [ "$pending" = 0 ] && ! squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "^$TAG-"; then
        echo "all $TAG steps resolved and no $TAG training jobs remain"; exit 0
    fi
    sleep "$POLL"
done
echo "autosubmit: stopped after $((MAXPOLL * POLL))s"
