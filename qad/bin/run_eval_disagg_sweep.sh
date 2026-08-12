#!/bin/bash
# Launch the full disaggregated eval sweep: every format x both thinking modes x steps.
#
#   ./bin/run_eval_disagg_sweep.sh                      # 0.6B, both modes, all formats
#   ./bin/run_eval_disagg_sweep.sh --model Qwen/Qwen3-1.7B
#   ./bin/run_eval_disagg_sweep.sh --modes think        # one mode only
#   ./bin/run_eval_disagg_sweep.sh --dry-run            # print the submissions, submit none
#
# One sbatch ARRAY per (format, thinking mode); each array element is one step. Every
# job holds TWO GPUs (prefill + decode), so check the queue before launching all sizes.
#
# Thinking mode is a separate benchmark, not a variant: the same checkpoint differs by
# ~20 points on GSM8K between modes, so both are swept and they land in separate trees
# (results/disagg/think/ and nothink/).

set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAD_DIR="$(dirname "$SELF_DIR")"   # checkpoints/ and results/ live here

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
RUN=${RUN:-qad3x}
# every 250 plus the final step. NOTE: step 0 is NOT exported for quantized runs (the
# earliest checkpoint is step 25); the step-0 column in the plots is the BF16 point,
# which is covered by the `unquantized` format below.
STEPS=${STEPS:-250,500,750,1000,1250,1500,1750,2000,2250,2450}
MODES=${MODES:-"think nothink"}
# DEPRECATED, do not re-add without a reason: aime25 (30 docs -- one item is 3.3
# points, so step-to-step motion is almost entirely noise, and its 32768-token
# max_gen_toks once drove max_context_len negative and silently emptied a whole
# sweep) and mmlu_flan_cot_zeroshot (1531 docs spread over 57 subjects, ~27 each).
# mmlu_pro is the MMLU that stays: 12032 docs, and it discriminates.
TASKS=${TASKS:-"gsm8k minerva_math500"}
# label:quantizer   ("unquantized" is special-cased: BF16 base model, single step)
FORMATS=${FORMATS:-"BF16:unquantized NVFP4:nvfp4 NVFP4A16:nvfp4a16 W3A16-Lloyd:lloyd3bit NVFP4-pd-shared:nvfp4pdshared NVFP4-pd-split:nvfp4pdsplit"}
DRY=0
MISSING_ONLY=0
LIMIT=${LIMIT:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}

while (($# > 0)); do
  case "$1" in
    --model)   MODEL="$2";   shift 2 ;;
    --run)     RUN="$2";     shift 2 ;;
    --steps)   STEPS="$2";   shift 2 ;;
    --modes)   MODES="$2";   shift 2 ;;
    --tasks)   TASKS="$2";   shift 2 ;;
    --formats) FORMATS="$2"; shift 2 ;;
    --limit)   LIMIT="$2";   shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --dry-run) DRY=1;        shift ;;
    --missing-only) MISSING_ONLY=1; shift ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done


# Print the subset of $2 (comma list of steps) that has no result yet for tag $1 in
# tree $3, i.e. no step_%07d.json containing every task in $4. Re-running a sweep with
# --missing-only then fills exactly the gaps, whatever caused them.
missing_steps() {
  local tag_glob="$1" steps="$2" tree="$3" tasks="$4"
  python - "$tag_glob" "$steps" "$tree" "$tasks" <<'PYEOF'
import glob, json, os, sys
tag_glob, steps, tree, tasks = sys.argv[1:5]
want = tasks.split()
dirs = glob.glob(tag_glob)
out = []
for st in steps.split(","):
    if not st:
        continue
    have = False
    for d in dirs:
        f = os.path.join(tree, os.path.basename(d), "step_%07d.json" % int(st))
        if not os.path.exists(f):
            continue
        try:
            res = json.load(open(f)).get("results", {})
        except Exception:
            continue
        # a task counts as present only if it (or one of its subject leaves) reported
        if all(any(k == t or k.startswith(t + "_") for k in res) for t in want):
            have = True
            break
    if not have:
        out.append(st)
print(",".join(out))
PYEOF
}

MODEL_TAG="$(echo "$MODEL" | tr '/' '-')"
RUN_NAME="${RUN}-${MODEL_TAG}"
n_sub=0

for fmt in $FORMATS; do
  label="${fmt%%:*}"; quant="${fmt##*:}"
  for mode in $MODES; do
    think_flag="--think"; [ "$mode" = "nothink" ] && think_flag="--no-think"

    if [ "$quant" = "unquantized" ]; then
      # BF16 has no checkpoint tag and no training steps: one point, tagged at step 0.
      # --missing-only must skip it too, otherwise every gap-fill pass needlessly
      # re-runs the baseline it already has.
      if [ "$MISSING_ONLY" = 1 ]; then
        tree="$QAD_DIR/results/disagg/think"
        [ "$mode" = "nothink" ] && tree="$QAD_DIR/results/disagg/nothink"
        bf16_tag="$(echo "$MODEL" | tr '/' '-')-unquantized"
        # Existence is NOT enough: step_0000000.json is shared by every benchmark, so a
        # file holding only gsm8k made --missing-only skip the BF16 mmlu_pro baseline.
        # Reuse missing_steps(), which checks that each requested TASK actually reported.
        if [ -z "$(missing_steps "$tree/$bf16_tag" 0 "$tree" "$TASKS")" ] \
           && [ -f "$tree/$bf16_tag/step_0000000.json" ]; then
          printf 'done    %-18s %-8s  baseline already has results\n' "$label" "$mode"
          continue
        fi
      fi
      cmd=("$SELF_DIR/run_eval_disagg.sh" --model "$MODEL" --unquantized
           --tasks "$TASKS" $think_flag)
      [ -n "$LIMIT" ] && cmd+=(--limit "$LIMIT")
      [ -n "$MAX_MODEL_LEN" ] && cmd+=(--max-model-len "$MAX_MODEL_LEN")
    else
      # Skip a format with no checkpoints rather than submitting jobs that will fail
      # one by one inside the container.
      if ! ls -d "$QAD_DIR/checkpoints/${RUN_NAME}-${quant}-"* >/dev/null 2>&1; then
        echo "skip  $label ${mode}: no checkpoints for ${RUN_NAME}-${quant}-*"
        continue
      fi
      use_steps="$STEPS"
      if [ "$MISSING_ONLY" = 1 ]; then
        tree="$QAD_DIR/results/disagg/${mode}"
        [ "$mode" = "think" ] && tree="$QAD_DIR/results/disagg/think"
        use_steps=$(missing_steps "$QAD_DIR/checkpoints/${RUN_NAME}-${quant}-*" \
                                  "$STEPS" "$tree" "$TASKS")
        if [ -z "$use_steps" ]; then
          printf 'done    %-18s %-8s  all steps already have results\n' "$label" "$mode"
          continue
        fi
        printf 'gaps    %-18s %-8s  %s\n' "$label" "$mode" "$use_steps"
      fi
      cmd=("$SELF_DIR/run_eval_disagg.sh" --model "$MODEL" --quantizer "$quant"
           --run-name "$RUN_NAME" --steps "$use_steps" --tasks "$TASKS" $think_flag)
      [ -n "$LIMIT" ] && cmd+=(--limit "$LIMIT")
      [ -n "$MAX_MODEL_LEN" ] && cmd+=(--max-model-len "$MAX_MODEL_LEN")
    fi

    if [ "$DRY" = 1 ]; then
      printf 'would submit  %-18s %-8s  %s\n' "$label" "$mode" "${cmd[*]}"
    else
      printf 'submit  %-18s %-8s ' "$label" "$mode"
      "${cmd[@]}" 2>&1 | tail -1
    fi
    n_sub=$((n_sub + 1))
  done
done
echo "---"
echo "$([ "$DRY" = 1 ] && echo 'would submit' || echo 'submitted') $n_sub array(s) for $MODEL ($RUN_NAME)"
echo "steps: $STEPS"
echo "modes: $MODES"
echo "tasks: $TASKS"
