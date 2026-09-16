#!/bin/bash
# Generate RULER's corpora on the CPU partition, one array task per (task, length).
#
#   ./bin/prep_ruler.sh --model qwen3.8-27b --seqlens 4096,8192,16384,32768,65536
#
# WHY THIS EXISTS SEPARATELY FROM run_eval.sh. --bench ruler prepares its own data, so
# this is never REQUIRED -- but that preparation is serialized ahead of every token and
# it is pure single-core CPU work, so a GPU job spends hours with four idle GPUs and a
# loaded vLLM server waiting on it. Measured on this model: the 12 non-QA tasks take
# ~8 min at 4k and qa_2 alone takes 15+, and the join/tokenize cost scales with length,
# putting a 5-length sweep in the 6-12 hour range before the first generation.
#
# The work is embarrassingly parallel -- every (task, length) is independent -- so an
# array on the `cpu` partition collapses that to the slowest single cell, which is the
# only thing that cannot be split further (probably qa_2 at the top length).
#
# THE CACHE IS SHARED WITH run_eval.sh BY CONSTRUCTION: both derive the directory from a
# hash of the tokenizer's own bytes, so a GPU job submitted afterwards finds every file
# present and goes straight to generating. Nothing here is ruler-run specific; it is the
# same prepare.py with the same arguments.
#
# DO NOT RUN THIS WHILE A --bench ruler JOB IS PREPARING the same tokenizer. Both write
# the same paths, and a half-written file trips the other side's doc-count check into
# regenerating on top of an in-progress write.
#
# qa.py is slow for a reason worth knowing before tuning anything: for EVERY sample it
# rebuilds `[i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]` over
# HotpotQA's 73,754 documents, allocating a fresh concatenation inside the comprehension
# -- ~37M list allocations per task. That cost is per sample and independent of length.

#SBATCH --job-name=prep-ruler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=coreai_psx_qad

set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
REPO_ROOT=${REPO_ROOT:-$(dirname "$MUSE_ROOT")}
P=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode
CONTAINER=${CONTAINER:-$P/containers/vllm-nightly.sqsh}
RULER_HARNESS=${RULER_HARNESS:-$P/harnesses/RULER}
RULERDEPS=${RULERDEPS:-$P/muse-rulerdeps}
NLTK_DATA=${NLTK_DATA:-$P/nltk_data}

MODEL=${MODEL:-qwen3.8-27b}
SEQLENS=${SEQLENS:-4096,8192,16384,32768,65536}
TASKS=${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,vt,cwe,fwe,qa_1,qa_2}
SAMPLES=${SAMPLES:-500}
WEIGHTS=${WEIGHTS:-}
# Per TASK, not per job: see the qa.py hang documented at the srun step below.
PREP_TIMEOUT=${PREP_TIMEOUT:-5400}
TIME=${TIME:-08:00:00}
PART=${PART:-cpu}

while (($# > 0)); do
    case "$1" in
        --model)    MODEL="$2";   shift 2 ;;
        --weights)  WEIGHTS="$2"; shift 2 ;;
        --seqlens)  SEQLENS="$2"; shift 2 ;;
        --tasks)    TASKS="$2";   shift 2 ;;
        --samples)  SAMPLES="$2"; shift 2 ;;
        --time)     TIME="$2";    shift 2 ;;
        --partition) PART="$2";   shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The tokenizer the corpus is sized by -- the same resolution run_eval.sh does
# (MODEL_PATH=${WEIGHTS:-<models.json path>}), so the two agree on the cache directory.
if [ -z "$WEIGHTS" ]; then
    MODEL_PATH=$(python3 -c "
import json,sys
m=json.load(open('$MUSE_ROOT/models.json'))
if '$MODEL' not in m: sys.exit('unknown --model $MODEL')
print(m['$MODEL']['path'])") || exit 2
else
    MODEL_PATH="$WEIGHTS"
fi
[ -d "$MODEL_PATH" ] || { echo "ERROR: no such model dir: $MODEL_PATH" >&2; exit 2; }

TOK_KEY=$(cat "$MODEL_PATH/tokenizer.json" "$MODEL_PATH/tokenizer_config.json" \
          2>/dev/null | md5sum | cut -c1-12)
[ -z "$TOK_KEY" ] && { echo "ERROR: no tokenizer at $MODEL_PATH" >&2; exit 2; }
RULER_DATA=${RULER_DATA:-$MUSE_ROOT/data/ruler/tok-$TOK_KEY}

# ONE ARRAY ELEMENT PER LENGTH, fanning that length's tasks across cores of a single
# node -- NOT one element per (task, length). Slurm allocates a whole node either way,
# and these nodes have 76 cores and 350 GB; a 1-core-per-node array asks for 53 nodes to
# run 53 single-threaded processes and is immediately throttled to ~2 concurrent by
# QOSMaxNodePerUserLimit, since a busy eval sweep is already holding most of the node
# budget. Grouping by length needs 5 nodes, runs 13 tasks at once on each, and finishes
# in about the time of the slowest single task.
#
# The missing work is computed on the LOGIN node so the array is exactly as large as what
# is left; rerunning after a partial pass queues only the gaps.
LENGTHS=()
for SL in ${SEQLENS//,/ }; do
    MISSING=""
    for T in ${TASKS//,/ }; do
        F="$RULER_DATA/$SL/$T/validation.jsonl"
        HAVE=0; [ -s "$F" ] && HAVE=$(wc -l < "$F")
        [ "$HAVE" = "$SAMPLES" ] || MISSING="$MISSING $T"
    done
    [ -n "$MISSING" ] && LENGTHS+=("$SL$MISSING")
done

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ "${#LENGTHS[@]}" = 0 ]; then
        echo "nothing to do: every (task, length) already has $SAMPLES docs in $RULER_DATA"
        exit 0
    fi
    mkdir -p "$RULER_DATA"
    printf '%s\n' "${LENGTHS[@]}" > "$RULER_DATA/.pairs.$$"
    LOGS="$REPO_ROOT/logs/evals/$(date +%Y%m%d_%H%M%S)_prepruler_${MODEL}"
    mkdir -p "$LOGS"
    export MUSE_ROOT REPO_ROOT CONTAINER RULER_HARNESS RULERDEPS NLTK_DATA \
           MODEL_PATH RULER_DATA SAMPLES LOGS PREP_TIMEOUT
    export PAIRS_FILE="$RULER_DATA/.pairs.$$"
    echo "data -> $RULER_DATA (tokenizer $TOK_KEY)"
    echo "${#LENGTHS[@]} lengths to generate; logs -> $LOGS"
    printf '  %s\n' "${LENGTHS[@]}" | awk '{printf "  %-7s %d tasks\n", $1, NF-1}'
    exec sbatch --export=ALL --partition="$PART" --time="$TIME" \
        --array=0-$(( ${#LENGTHS[@]} - 1 )) --job-name="prepruler-$MODEL" \
        --output="$LOGS/%A_%a.out" --error="$LOGS/%A_%a.err" \
        "$_SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
fi

# --- array task: one LENGTH, all of its missing tasks in parallel ---------------------
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PAIRS_FILE")
SL=${LINE%% *}; MY_TASKS=${LINE#* }
[ -z "${SL:-}" ] && { echo "ERROR: no entry at index $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
echo "=== prepare @ $SL on $(hostname): $MY_TASKS ==="

# One srun step, one container, N background processes. prepare.py is single-threaded
# and holds ~1 GB, so 13 of them fit comfortably in a 76-core / 350 GB node.
# TOKENIZERS_PARALLELISM=false matters here: 13 fast tokenizers each spawning a thread
# pool would oversubscribe the node and make every one of them slower.
srun --ntasks=1 --container-image="$CONTAINER" --no-container-mount-home \
     --container-mounts=/scratch:/scratch,/lustre:/lustre --export=ALL bash -c "
set -uo pipefail
# RULERDEPS/bin holds a python -> python3 shim. prepare.py builds its generator command
# as a literal 'python {script}' and does NOT check the return code: without the shim it
# prints 'Prepare <task> with lines: N' and exits 0 having written nothing.
export PATH='$RULERDEPS/bin':\$PATH
export PYTHONPATH='$RULER_HARNESS/scripts/data':'$RULERDEPS'
export NLTK_DATA='$NLTK_DATA'
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd '$RULER_HARNESS/scripts/data' || exit 1
for T in $MY_TASKS; do
  # timeout, because qa.py CAN HANG FOREVER rather than fail. Its per-sample retry is
  #     while True:
  #         try: ... assert length <= max_seq_length ...
  #         except:
  #             if used_docs > incremental: used_docs -= incremental
  # so once used_docs <= incremental (10) the handler changes nothing and the loop spins
  # at 100% CPU with no output. Reachable whenever the length cannot hold the task's
  # mandatory documents: HotpotQA's context is all 10 distractor paragraphs, so qa_2 at
  # 4096 fails for every sample with this tokenizer. Measured: 2h05m, zero bytes written,
  # while the same task finished at 8192 in 9.3 minutes. Without this the hang is
  # indistinguishable from slow generation and silently consumes the whole job.
  ( timeout '$PREP_TIMEOUT' python3 prepare.py --save_dir '$RULER_DATA/$SL' \
        --benchmark synthetic --task \$T \
        --tokenizer_path '$MODEL_PATH' --tokenizer_type hf --max_seq_length '$SL' \
        --num_samples '$SAMPLES' --model_template_type base \
    > '$LOGS'/prep_${SLURM_ARRAY_JOB_ID}_${SL}_\$T.log 2>&1
    [ \$? = 124 ] && echo \"TIMEOUT after ${PREP_TIMEOUT}s\" \
        >> '$LOGS'/prep_${SLURM_ARRAY_JOB_ID}_${SL}_\$T.log ) &
done
wait
"
# Verified per task, because prepare.py's exit code cannot be trusted (see above) and a
# backgrounded failure would otherwise be invisible behind a successful \`wait\`.
RC=0
for T in $MY_TASKS; do
    F="$RULER_DATA/$SL/$T/validation.jsonl"
    HAVE=0; [ -s "$F" ] && HAVE=$(wc -l < "$F")
    if [ "$HAVE" != "$SAMPLES" ]; then
        echo "ERROR: $T @ $SL produced $HAVE docs, wanted $SAMPLES" >&2; RC=1
    else
        echo "=== ok: $T @ $SL -> $HAVE docs ==="
    fi
done
exit $RC
