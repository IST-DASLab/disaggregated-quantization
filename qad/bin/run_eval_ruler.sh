#!/bin/bash
# Evaluate a checkpoint on RULER (long-context) through REAL disaggregated vLLM
# serving. Separate entrypoint from run_eval_disagg.sh, not a --tasks addition to it:
# RULER needs a per-model context-length sweep and a completely different
# concurrency budget (long prompts eat the KV cache fast), neither of which fits the
# gsm8k/mmlu_pro-shaped flag surface there.
#
#   ./bin/run_eval_ruler.sh --model Qwen/Qwen3-4B --quantizer nvfp4 \
#        --run-name qad3x-Qwen-Qwen3-4B --iter 2450
#     -> seqlens auto: powers of 2 from 8192 up to the model's OWN max_position_embeddings
#     -> mode auto (one job): no-think for Qwen, think for everyone else. RULER's
#        per-task generation budget is a fixed 128 tokens (the spec -- these are
#        short retrieval/extraction tasks, not reasoning ones), nowhere near enough
#        room for a <think> trace plus an answer, so a "think" run would measure
#        truncation rather than reasoning. --think/--no-think still overrides this
#        if you want the single mode explicitly. Gemma has no thinking axis at all;
#        --no-think hard-fails there the same way it does in eval_disagg.py.
#
#   ./bin/run_eval_ruler.sh ... --seqlens 8192,16384          # override the sweep
#   ./bin/run_eval_ruler.sh ... --max-context 65536           # override the model's own max
#   ./bin/run_eval_ruler.sh ... --limit 20                    # SMOKE TEST ONLY, see eval_ruler.py
#
# Results:  qad/results/ruler/think/<tag>/step_<N>.json   (or nothink/)
# Logs:     logs/eval_ruler/<timestamp>_<label>/
#
# COST. lm-eval's RULER tasks generate 500 fresh docs PER requested length, fixed,
# not a flag here (see eval_ruler.py's docstring) -- 13 tasks x 500 docs x however
# many lengths this submits is the real unit of work, and it grows fast at the top of
# the sweep: one NVIDIA/RULER GitHub report measured ~15 min for 5 requests at 128K
# context on a single A100 for an 8B model. Measure with --limit before committing to
# the full sweep on a model you have not run this against yet.

#SBATCH --job-name=qad-ruler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad

CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
HF_CACHE=${HF_CACHE:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache}
LM_EVAL_OVERLAY=${LM_EVAL_OVERLAY:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/lm_eval_overlay_vllm}
# normal, not the #SBATCH header's default directly -- QOS=interactive lets a one-off
# verification/test run (e.g. re-checking a bugfix) jump a saturated normal-QOS queue
# instead of waiting behind a large sweep for a GPU slot.
QOS=${QOS:-normal}
# Partition and wall clock are chosen from the SEQLEN SWEEP further down, not fixed
# here. Set either to override that choice.
PARTITION=${PARTITION:-}
TIME=${TIME:-}
# The DEFAULT sweep is 8192..32768. 65536 was dropped 2026-09-12.
#
# 64k cost far more than it told us: it is where gemma-3-12b's jobs ran 3:50-4:00
# against a 4h wall and one TIMED OUT outright, and on the BF16 12b run it was the
# segment that turned a ~2h job into a ~7h one.
#
# Dropping the TOP invalidates nothing. ruler_landed_steps() tests want_seqlens <=
# have_seqlens, so a gemma-4b/12b result holding [8192,16384,32768,65536] still covers
# the narrower want set and stays complete. Nothing is resubmitted.
#
# 4096 was added 2026-09-12, once both of its preconditions held. Lowering the floor is
# the opposite case to raising the top: no result on disk had 4096, so it marks every
# existing point partial at once. That is only safe because
#   (a) the previous sweep had fully drained -- jobs still running would have carried
#       the OLD eval_ruler.py, whose task-level merge overwrote a freshly written score
#       with its own -1 placeholder (verified: 0 jobs in flight at the time), and
#   (b) submit_missing_evals now requests ONLY the lengths a step lacks
#       (ruler_step_coverage / ruler_missing_seqlens), so a point holding 8k/16k/32k
#       gets a 4096-only job rather than a full re-sweep. That matters beyond cost:
#       RULER regenerates its documents every run, so recomputing a length it already
#       has would move the number under figures already drawn from it.
RULER_MIN_SEQLEN=${RULER_MIN_SEQLEN:-4096}
RULER_TOP_SEQLEN=${RULER_TOP_SEQLEN:-32768}

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
QUANTIZER=${QUANTIZER:-nvfp4}
QUANT_PARAMS=${QUANT_PARAMS:-}
RUN_NAME=${RUN_NAME:-}
ITER=${ITER:-}
SEQLENS=${SEQLENS:-}            # empty -> auto (powers of 2, 8192..max-context)
MAX_CONTEXT=${MAX_CONTEXT:-}    # empty -> read from the model's own config
# unset, not 0/1: THINK must distinguish "the caller asked for a specific mode" from
# "let this script decide (both for Qwen, think-only for everyone else)". A THINK=1
# default would silently drop the no-think arm for every Qwen job.
THINK=${THINK:-}
LOG_SAMPLES=${LOG_SAMPLES:-1}
CKPT_DIR=${CKPT_DIR:-}
LIMIT=${LIMIT:-}
CONCURRENCY=${CONCURRENCY:-}    # empty -> sized from max-model-len, see below
UNQUANT=${UNQUANT:-0}
FULL_DISAG=${FULL_DISAG:-0}
TAG=${TAG:-}
TASKS=${TASKS:-}                # empty -> eval_ruler.py's own RULER_TASKS default

while (($# > 0)); do
    case "$1" in
        --model)             MODEL="$2";        shift 2 ;;
        --quantizer)         QUANTIZER="$2";    shift 2 ;;
        --quantizer-params)  QUANT_PARAMS="$2"; shift 2 ;;
        --run-name)          RUN_NAME="$2";     shift 2 ;;
        --iter)              ITER="$2";         shift 2 ;;
        --seqlens)           SEQLENS="$2";      shift 2 ;;
        --max-context)       MAX_CONTEXT="$2";  shift 2 ;;
        --tasks)             TASKS="$2";        shift 2 ;;
        --limit)             LIMIT="$2";        shift 2 ;;
        --ckpt-dir)          CKPT_DIR="$2";     shift 2 ;;
        --concurrency)       CONCURRENCY="$2";  shift 2 ;;
        --think)             THINK=1;           shift ;;
        --no-think)          THINK=0;           shift ;;
        --log-samples)       LOG_SAMPLES=1;     shift ;;
        --no-log-samples)    LOG_SAMPLES=0;     shift ;;
        --unquantized)       UNQUANT=1;         shift ;;
        --full-disag)        FULL_DISAG=1;      shift ;;
        --tag)                TAG="$2";          shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# LOGIN mode: resolve the model's max context (needs the container for
# transformers/HF cache access), compute the seqlen sweep, then submit --
# ONE job for the caller's --think/--no-think, or TWO (think + no-think) if
# neither was given and the model is a Qwen (Gemma has no thinking axis).
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELF="$SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
    ROOT="$(dirname "$SELF_DIR")/.."

    if [ -z "$MAX_CONTEXT" ]; then
        echo "resolving max context for $MODEL ..."
        MAX_CONTEXT=$(srun --account=coreai_psx_qad --partition=cpu --time=00:05:00 \
            --ntasks=1 --cpus-per-task=2 --container-image="$CONTAINER" \
            --no-container-mount-home --container-mounts=/scratch:/scratch,/lustre:/lustre \
            bash -c "HF_HOME=$HF_CACHE HF_HUB_OFFLINE=1 python3 -c \"
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('$MODEL')
tc = getattr(cfg, 'text_config', cfg)
# NOT getattr(tc, 'x', getattr(cfg, 'x')) -- getattr's default is an ordinary
# argument, evaluated eagerly regardless of whether tc already has 'x', so that
# form crashes on any wrapper whose OWN config lacks the attribute (Gemma-3
# 4b/12b's Gemma3Config: max_position_embeddings lives only on .text_config).
mpe = getattr(tc, 'max_position_embeddings', None)
if mpe is None:
    mpe = getattr(cfg, 'max_position_embeddings', None)
print(mpe)
\"" 2>/dev/null | tail -1)
        if ! [[ "$MAX_CONTEXT" =~ ^[0-9]+$ ]]; then
            echo "ERROR: could not resolve max context for $MODEL (got '$MAX_CONTEXT')," >&2
            echo "       pass --max-context N explicitly." >&2
            exit 1
        fi
        echo "  $MODEL max_position_embeddings=$MAX_CONTEXT"
    fi

    if [ -z "$SEQLENS" ]; then
        # Powers of 2 from RULER_MIN_SEQLEN up to min(MAX_CONTEXT, RULER_TOP_SEQLEN),
        # EXCEPT a top value landing exactly ON the ceiling is dropped. eval_ruler.py's
        # served context is max(seqlens) + generation budget + margin, so a top
        # seqlen == MAX_CONTEXT leaves that margin nowhere to go -- vLLM then
        # hard-refuses to serve (confirmed live: gemma-3-270m/1b/4b/12b-it, whose
        # ceilings are exact powers of 2, all crashed here; Qwen3 never has since
        # its ceiling, 40960, isn't one -- the sweep tops out at 32768 with 8192 to
        # spare). eval_ruler.py still clamps as a last resort if this is bypassed
        # (e.g. a manual --seqlens landing on the ceiling), but that clamp eats the
        # margin instead of avoiding the situation, so prefer not needing it here.
        SEQLENS=$(python3 -c "
n = $RULER_MIN_SEQLEN
cap = min($MAX_CONTEXT, $RULER_TOP_SEQLEN)
out = []
while n <= cap:
    out.append(n)
    n *= 2
if out and out[-1] == $MAX_CONTEXT and len(out) > 1:
    out.pop()
print(','.join(str(x) for x in out))
")
        [ -z "$SEQLENS" ] && { echo "ERROR: $MODEL's max_context=$MAX_CONTEXT is below $RULER_MIN_SEQLEN" >&2; exit 1; }
        echo "  seqlens: $SEQLENS"
    fi

    # Qwen family only: "with and without reasoning" per the ask that created this
    # script. Gemma-3 is think-only everywhere else in this repo; a Gemma --no-think request is passed through so
    # eval_ruler.py's own hard-fail still catches a mistaken override rather than
    # this script silently swallowing it.
    # NOT both modes for Qwen: RULER's per-task generation budget is a fixed 128
    # tokens (the RULER spec -- these are short extraction/retrieval tasks, not
    # reasoning ones), nowhere near enough room for a <think> trace plus an answer,
    # so a "think" run would just measure truncation, not reasoning. Default to
    # no-think for Qwen and think for everyone else (Gemma has no thinking axis;
    # --no-think hard-fails there rather than being a no-op, same as eval_disagg.py).
    MODES=()
    if [ -n "$THINK" ]; then
        MODES=("$THINK")
    elif [[ "$MODEL" == Qwen/* ]]; then
        MODES=(0)
    else
        MODES=(1)
    fi

    LABEL_BASE="$(echo "${RUN_NAME:-$MODEL}" | tr '/' '-')-$([ "$UNQUANT" = 1 ] && echo unquantized || echo "$QUANTIZER")"
    [ -n "$TAG" ] && LABEL_BASE="$TAG"

    if [ "$UNQUANT" = 1 ]; then ITER=0; fi
    if [ -z "$ITER" ]; then echo "ERROR: pass --iter N" >&2; exit 1; fi

    # A 64k sweep does not fit the `batch` partition's 4h cap. Measured on
    # gemma-3-12b-it, whose sweep is 8192,16384,32768,65536: five jobs finished at
    # 03:50:49, 03:55:34, 03:56:13, 03:58:06 -- and job 311508 (12b nvfp4prefill)
    # hit TIMEOUT at 04:00:29. A timed-out point is strictly worse than a slow one:
    # it writes NO result file, so the gap scan re-offers it, autoeval_watch
    # resubmits, and the next attempt burns another 4h against the same wall. That
    # is an unbounded retry loop, not a delay.
    # Since RULER_TOP_SEQLEN dropped to 32768 no DEFAULT sweep reaches 64k any more, so
    # this branch is now only reachable through an explicit --seqlens/--max-context
    # override. Kept precisely for that case -- it is the override that reintroduces the
    # cost, so it should also reintroduce the wall clock.
    # So anything sweeping to 64k goes to batch_long (7d cap) with 8h of wall. Models
    # topping out at 32768 (every Qwen3, whose 40960 ceiling is not a power of two)
    # keep the 4h `batch` slot, which they clear comfortably -- the median RULER job
    # is 52 min -- and which schedules sooner.
    TOP_SEQLEN=$(echo "$SEQLENS" | tr ',' '\n' | sort -n | tail -1)
    if [ -z "$PARTITION" ] && [ "${TOP_SEQLEN:-0}" -ge 65536 ]; then
        PARTITION=batch_long
        [ -z "$TIME" ] && TIME=08:00:00
    fi
    PARTITION=${PARTITION:-batch}
    TIME=${TIME:-04:00:00}
    echo "  partition: $PARTITION  time: $TIME  (top seqlen $TOP_SEQLEN)"

    for mode in "${MODES[@]}"; do
        THINK_MODE="$mode"
        LABEL="${LABEL_BASE}_$([ "$mode" = 1 ] && echo think || echo nothink)"
        LOGS="$ROOT/logs/eval_ruler/$(date +%Y%m%d_%H%M%S)_${LABEL}"
        mkdir -p "$LOGS"
        export MODEL QUANTIZER QUANT_PARAMS RUN_NAME ITER SEQLENS MAX_CONTEXT TASKS \
               LIMIT LOG_SAMPLES CKPT_DIR CONCURRENCY UNQUANT FULL_DISAG TAG \
               CONTAINER HF_CACHE LM_EVAL_OVERLAY
        sbatch --export=ALL,THINK="$THINK_MODE" --qos="$QOS" \
            --partition="$PARTITION" --time="$TIME" \
            --output="$LOGS/%j.out" --error="$LOGS/%j.err" "$SELF"
        echo "logs → $LOGS"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# HOST mode: re-enter inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null && [ -z "${RULER_IN_CONTAINER:-}" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
    export QAD_DIR=$(dirname "$SCRIPT_DIR")
    export RULER_IN_CONTAINER=1
    srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
        --container-mounts="/scratch:/scratch,/lustre:/lustre,$HOME/.netrc:/root/.netrc" --export=ALL \
        bash "$SCRIPT_PATH"
    exit $?
fi

# ---------------------------------------------------------------------------
# CONTAINER mode
# ---------------------------------------------------------------------------
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
# RULER's niah tasks auto-fetch nltk's punkt_tab on first use. Without this it lands in a
# container-internal path that --no-container-mount-home throws away, so EVERY job
# re-downloads it. Point it at a persistent shared dir instead; it is
# already warm, so RULER jobs no longer depend on that fetch succeeding at all.
export NLTK_DATA=${NLTK_DATA:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/nltk_data}
export HF_DATASETS_OFFLINE=1
# ruler_qa_squad/ruler_qa_hotpot fetch their raw JSON over plain HTTP (not through
# huggingface_hub), so they are NOT covered by HF_HUB_OFFLINE either way -- this
# only controls the NIAH haystack essays and the tokenizer/model config, which ARE
# regular HF datasets/hub lookups.
export HF_HUB_OFFLINE=1
export PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:${PYTHONPATH:-}

if [ -z "$CONCURRENCY" ]; then
    # Sized off the one real calibration point this repo has (run_eval_disagg.sh):
    # 128 concurrent requests fit an 8192-token server comfortably. KV budget is
    # roughly concurrency x max_model_len, so hold that product ~constant instead
    # of reusing 128 (or 512) unchanged at 131072 tokens, which would ask for a KV
    # cache an order of magnitude past what fits.
    MAX_SEQLEN=$(echo "$SEQLENS" | tr ',' '\n' | sort -n | tail -1)
    CONCURRENCY=$(( 1000000 / MAX_SEQLEN ))
    [ "$CONCURRENCY" -lt 1 ] && CONCURRENCY=1
    echo "concurrency (auto, from max seqlen $MAX_SEQLEN): $CONCURRENCY"
fi

ARGS=(
    --tokenizer "$MODEL"
    --model "$MODEL"
    --seqlens "$SEQLENS"
    --concurrency "$CONCURRENCY"
    --log-dir "$(dirname "$QAD_DIR")/logs/eval_ruler/stack_${SLURM_JOB_ID}"
)
[ -n "$TASKS" ] && ARGS+=(--tasks $TASKS)
if [ "$UNQUANT" = 1 ]; then
    ARGS+=(--unquantized)
else
    ARGS+=(--quantizer "$QUANTIZER" --iter "$ITER")
fi
[ -n "$QUANT_PARAMS" ] && [ "$UNQUANT" = 0 ] && ARGS+=(--quantizer-params "$QUANT_PARAMS")
[ -n "$RUN_NAME" ]     && [ "$UNQUANT" = 0 ] && ARGS+=(--run-name "$RUN_NAME")
[ -n "$CKPT_DIR" ]     && ARGS+=(--ckpt-dir "$CKPT_DIR")
[ -n "$TAG" ]          && ARGS+=(--tag "$TAG")
[ "$FULL_DISAG" = 1 ]  && ARGS+=(--full-disag)
[ -n "$LIMIT" ]        && ARGS+=(--limit "$LIMIT")
[ "$LOG_SAMPLES" = 1 ] && ARGS+=(--log-samples)
[ "$THINK" = 0 ]       && ARGS+=(--no-think)

if [ -z "$ITER" ] && [ "$UNQUANT" = 0 ]; then
    echo "ERROR: ITER is empty inside the container -- a value was lost crossing the" >&2
    echo "       submit/container boundary. Check that every default uses \${VAR:-...}." >&2
    exit 1
fi
if [ "$UNQUANT" = 1 ]; then
    echo "[ruler-eval] model=$MODEL UNQUANTIZED (BF16) think=$THINK seqlens=$SEQLENS"
else
    echo "[ruler-eval] model=$MODEL quantizer=$QUANTIZER iter=$ITER think=$THINK seqlens=$SEQLENS"
fi
python3 "$QAD_DIR/eval/eval_ruler.py" "${ARGS[@]}"; rc=$?
echo "JOB_COMPLETE rc=$rc"
exit $rc
