#!/bin/bash
# Run the eval suite through REAL disaggregated vLLM serving (Nixl 1P1D).
#
# This is the deployment-accurate counterpart to run_eval_vllm.sh: a prefill engine and
# a decode engine on separate GPUs exchanging KV, driven by lm-eval through one OpenAI
# endpoint. It takes the SAME arguments, resolves checkpoints with the SAME tag
# convention, and writes the SAME on-disk layout, so results drop straight into the
# plotting notebook next to the single-server ones.
#
#   ./bin/run_eval_disagg.sh --model Qwen/Qwen3-0.6B --quantizer nvfp4pdshared \
#        --run-name qad3x-Qwen-Qwen3-0.6B --iter 2450
#   ./bin/run_eval_disagg.sh ... --steps 250,750,1250,2450        # job array over steps
#   ./bin/run_eval_disagg.sh ... --no-think                       # thinking suppressed
#   ./bin/run_eval_disagg.sh ... --full-disag                     # a --full-disag run
#
# A dual checkpoint exports weights/step_N/{prefill,decode}; every other method exports
# weights/step_N and both engines serve it. Both are handled -- a homogeneous model is
# just the case where the two halves are identical -- so the whole sweep goes through
# one path.
#
# Results:  qad/results/disagg/think/<tag>/step_<N>.json   (or nothink/)
# Logs:     logs/eval_disagg/<timestamp>_<label>/
#
# Requires TWO GPUs per task (prefill on 0, decode on 1). See DISAGG.md for the two
# non-default connector settings this cluster needs and why.

#SBATCH --job-name=qad-disagg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Only 2 GPUs are used (prefill=0, decode=1), but every QOS on this cluster carries
# MinTRES gres/gpu=4, so a --gpus-per-node=2 request is refused at submit with
# QOSMinGRES. Ask for the floor; the extra 2 sit idle. See evals/bin/run_quantize.sh
# for the same MIN_GPUS convention.
#SBATCH --gpus-per-node=4
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad

CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/vllm-nightly.sqsh}
HF_CACHE=${HF_CACHE:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache}
LM_EVAL_OVERLAY=${LM_EVAL_OVERLAY:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/lm_eval_overlay_vllm}

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
QUANTIZER=${QUANTIZER:-nvfp4pdshared}
# EVERY default must be ${VAR:-...}, never a bare assignment. This script re-executes
# itself inside the container with the settings passed through the environment, so a
# bare `ITER=""` here silently wipes the value the caller gave on the command line --
# which is exactly what a smoke test caught: `--iter 2450` arrived as `iter=`.
QUANT_PARAMS=${QUANT_PARAMS:-}
RUN_NAME=${RUN_NAME:-}
ITER=${ITER:-}
STEPS=${STEPS:-}
# DEPRECATED, do not re-add without a reason: aime25 (30 docs -- one item is 3.3
# points, so step-to-step motion is almost entirely noise, and its 32768-token
# max_gen_toks once drove max_context_len negative and silently emptied a whole
# sweep) and mmlu_flan_cot_zeroshot (1531 docs spread over 57 subjects, ~27 each).
# mmlu_pro is the MMLU that stays: 12032 docs, and it discriminates.
TASKS=${TASKS:-"gsm8k minerva_math500"}
THINK=${THINK:-1}        # thinking ON by default, matching results/vllm/think/
LIMIT=${LIMIT:-}
# Generations are logged BY DEFAULT: the scores alone cannot answer questions that
# come up later (length, refusals, format failures, repetition loops), and re-running
# a sweep to recover them costs far more than the disk. They land beside the results
# as step_<N>_samples_<task>.jsonl and are gitignored -- ~1 MB per file, which would
# add gigabytes to the repo. Pass --no-log-samples to opt out.
LOG_SAMPLES=${LOG_SAMPLES:-1}
CKPT_DIR=${CKPT_DIR:-}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
# 128, not 32. The API path caps in-flight requests, and at 32 the engines idled at
# ~31 running seqs while in-process vLLM queues hundreds -- ~25x less throughput
# (926 vs 23,435 output tok/s). It does NOT change accuracy (measured: 16.00 at 32 vs
# 19.00 at 128, both far below in-process), so this is purely a speed setting.
# 128 x 8192 tokens fits the ~1.07M-token KV cache; 256 would overflow into preemption.
# 512, matching the engine's max_num_seqs. Never OOMed at any setting so far; KV
# cache peaked at 48.7% with the client at 128.
CONCURRENCY=${CONCURRENCY:-512}
UNQUANT=${UNQUANT:-0}   # --unquantized: BF16 base model on both engines
FULL_DISAG=${FULL_DISAG:-0}   # --full-disag: select the full-disag run's checkpoint tag
# Results subdirectory override. A kv-noise sweep reuses ONE checkpoint at several
# rates, so without this every rate would overwrite the same step_0000000.json.
TAG=${TAG:-}

while (($# > 0)); do
    case "$1" in
        --model)             MODEL="$2";        shift 2 ;;
        --quantizer)         QUANTIZER="$2";    shift 2 ;;
        --quantizer-params)  QUANT_PARAMS="$2"; shift 2 ;;
        --run-name)          RUN_NAME="$2";     shift 2 ;;
        --iter)              ITER="$2";         shift 2 ;;
        --steps)             STEPS="$2";        shift 2 ;;
        --tasks)             TASKS="$2";        shift 2 ;;
        --limit)             LIMIT="$2";        shift 2 ;;
        --ckpt-dir)          CKPT_DIR="$2";     shift 2 ;;
        --max-gen-toks)      MAX_GEN_TOKS="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2";shift 2 ;;
        --concurrency)       CONCURRENCY="$2";  shift 2 ;;
        --think)             THINK=1;           shift ;;
        --no-think)          THINK=0;           shift ;;
        --log-samples)       LOG_SAMPLES=1;     shift ;;
        --no-log-samples)    LOG_SAMPLES=0;     shift ;;
        --unquantized)       UNQUANT=1;         shift ;;
        --full-disag)        FULL_DISAG=1;      shift ;;
        --tag)               TAG="$2";          shift 2 ;;
        # Unknown flags are an ERROR, never silently dropped: run_eval_vllm.sh used to
        # swallow --limit, which quietly turned a 100-doc smoke test into a full sweep.
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# LOGIN mode: submit (single job, or an array over --steps)
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    # $QUANTIZER keeps its default even for --unquantized runs (it is never passed),
    # so naming the log dir after it filed every BF16 baseline under "nvfp4pdshared".
    LABEL="$(echo "${RUN_NAME:-$MODEL}" | tr '/' '-')-$([ "$UNQUANT" = 1 ] && echo unquantized || echo "$QUANTIZER")"
    # When --tag is given it IS the identity of the run (a KV-noise sweep reuses one
    # checkpoint at several rates and is distinguished by nothing else), so name the log
    # dir after it. Otherwise the rate appears only in the results tree, and a progress
    # view built from live jobs shows every arm as the same plain baseline.
    [ -n "$TAG" ] && LABEL="$TAG"
    LOGS="$(dirname "$(dirname "$(dirname "$SELF")")")/logs/eval_disagg/$(date +%Y%m%d_%H%M%S)_${LABEL}"
    mkdir -p "$LOGS"
    if [ "$UNQUANT" = 1 ]; then ITER=0; fi
    if [ -z "$STEPS" ] && [ -z "$ITER" ]; then
        echo "ERROR: pass --iter N or --steps a,b,c" >&2; exit 1
    fi
    ARRAY=()
    if [ -n "$STEPS" ]; then
        N=$(echo "$STEPS" | tr ',' '\n' | grep -c .)
        ARRAY=(--array="0-$((N - 1))")
    fi
    # Values travel through the environment, never --export=ALL,VAR=...: sbatch splits
    # that list on commas and silently corrupts anything containing one.
    # TAG belongs here, not just in the arg parser: stage 0 re-execs via sbatch and
    # passes NOTHING on the command line, so a variable that is parsed but not exported
    # is silently dropped. That is not a cosmetic loss -- TAG is what keeps a sweep from
    # writing over the shared baseline directory, and omitting it let a --limit 100
    # canary overwrite the real 0.6B BF16 gsm8k baseline (0.664 -> 0.10).
    export MODEL QUANTIZER QUANT_PARAMS RUN_NAME ITER STEPS TASKS THINK LIMIT \
           LOG_SAMPLES CKPT_DIR MAX_GEN_TOKS MAX_MODEL_LEN CONCURRENCY UNQUANT \
           FULL_DISAG \
           CONTAINER HF_CACHE LM_EVAL_OVERLAY TAG
    echo "logs → $LOGS"
    exec sbatch --export=ALL "${ARRAY[@]}" \
        --output="$LOGS/%A_%a.out" --error="$LOGS/%A_%a.err" "$SELF"
fi

# ---------------------------------------------------------------------------
# HOST mode: re-enter inside the container
# ---------------------------------------------------------------------------
if command -v scontrol &>/dev/null && [ -z "${DISAGG_IN_CONTAINER:-}" ]; then
    # `exit` after the first match is REQUIRED: for the last array element scontrol
    # prints every array record, and a multi-line path kills the task with exit 127.
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
    export SCRIPT_DIR=$(dirname "$SCRIPT_PATH")   # qad/bin
    export QAD_DIR=$(dirname "$SCRIPT_DIR")       # qad -- anchors checkpoints/ and imports
    export DISAGG_IN_CONTAINER=1
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
export HF_DATASETS_OFFLINE=1
# OFFLINE. A sweep launches ~100 jobs that all resolve the same tokenizer at once;
# the Hub rate-limits, and the failure is not a clean 429 -- it is a TRUNCATED
# download surfacing as "Unable to load vocabulary from file ... not corrupted",
# which reads like a bad checkpoint. Everything needed is in HF_HOME already, so
# serve it from cache; a genuine cache miss then fails immediately and legibly.
export HF_HUB_OFFLINE=1
# The overlay is needed by THIS process (lm-eval) but eval_disagg.py strips it from the
# servers' PYTHONPATH: its huggingface-hub 1.24.0 shadows the container's and vLLM then
# refuses to start.
export PYTHONPATH=$LM_EVAL_OVERLAY:$QAD_DIR:${PYTHONPATH:-}

if [ -n "$STEPS" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ITER=$(echo "$STEPS" | cut -d, -f$((SLURM_ARRAY_TASK_ID + 1)))
    echo "array index $SLURM_ARRAY_TASK_ID -> step $ITER"
fi

ARGS=(
    --tokenizer "$MODEL"
    --model "$MODEL"
    --tasks $TASKS
    --max-gen-toks "$MAX_GEN_TOKS"
    --max-model-len "$MAX_MODEL_LEN"
    --concurrency "$CONCURRENCY"
    # SLURM_ARRAY_JOB_ID, not SLURM_JOB_ID: for an array element those differ, and the
    # .out file is named after the ARRAY id. Using the element id here meant nothing
    # could join a queued job to its proxy log, so live generation progress was
    # unreadable. This makes the stack dir match the .out basename exactly.
    --log-dir "$(dirname "$QAD_DIR")/logs/eval_disagg/stack_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
)
if [ "$UNQUANT" = 1 ]; then
    ARGS+=(--unquantized)
else
    ARGS+=(--quantizer "$QUANTIZER" --iter "$ITER")
fi
[ -n "$QUANT_PARAMS" ] && [ "$UNQUANT" = 0 ] && ARGS+=(--quantizer-params "$QUANT_PARAMS")
[ -n "$RUN_NAME" ] && [ "$UNQUANT" = 0 ] && ARGS+=(--run-name "$RUN_NAME")
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
    echo "[disagg-eval] model=$MODEL UNQUANTIZED (BF16) think=$THINK tasks=$TASKS"
else
    echo "[disagg-eval] model=$MODEL quantizer=$QUANTIZER iter=$ITER think=$THINK tasks=$TASKS"
fi
python3 "$QAD_DIR/eval/eval_disagg.py" "${ARGS[@]}"; rc=$?
# Sentinel for reap_stalled.sh: printed ONLY after all work is done, so a job hung in
# vLLM teardown is distinguishable from one still generating.
echo "JOB_COMPLETE rc=$rc"
exit $rc
