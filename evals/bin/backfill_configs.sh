#!/bin/bash
# Copy configs a derived checkpoint is missing from the model it was derived from.
#
#   ./bin/backfill_configs.sh                    # every pair in models.json
#   ./bin/backfill_configs.sh <derived> <parent> # one pair
#
# A quantization pipeline saves weights and whatever the library it used thought to
# write. What it does NOT write is everything else the serving stack needs, and the
# omission is silent until vLLM tries to build a processor and fails on a bare OSError.
# Seen twice now:
#
#   * llm-compressor writes processor_config.json but not preprocessor_config.json.
#     Muse-Glimmer needs only the former, so this went unnoticed until Qwen -- whose
#     image processor needs the latter -- quantized perfectly and then could not take a
#     picture.
#   * the daslab-testing lloyd43 checkpoints carry chat_template/processor_config but
#     not Qwen's preprocessor_config.json, video_preprocessor_config.json, merges.txt
#     or vocab.json.
#
# Copying by DIFFERENCE rather than by a fixed list is the point: the next checkpoint
# will omit some other file, and a list would have to be updated after the failure
# instead of before it. Weights are never touched -- only the derived checkpoint's own
# config.json, which describes its quantization, is protected by never overwriting.
set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}

backfill() {   # $1 = derived dir, $2 = parent dir
    local derived="$1" parent="$2" copied=()
    [ -d "$derived" ] || { echo "  skip (absent): $derived"; return; }
    [ -d "$parent" ]  || { echo "  skip (no parent): $parent"; return; }
    for f in "$parent"/*; do
        local b; b="$(basename "$f")"
        [ -f "$f" ] || continue
        case "$b" in
            *.safetensors|*.bin|*.pt|*.index.json) continue ;;   # weights and their map
            config.json) continue ;;   # describes the DERIVED model; never overwrite
        esac
        [ -e "$derived/$b" ] && continue
        cp "$f" "$derived/$b" && copied+=("$b")
    done
    if ((${#copied[@]})); then
        echo "  $(basename "$derived") <- ${copied[*]}"
    else
        echo "  $(basename "$derived") already complete"
    fi
}

if (($# == 2)); then
    backfill "$1" "$2"; exit 0
fi

# Every derived checkpoint models.json names, against its model's own base path.
while IFS='|' read -r derived parent; do
    [ -n "$derived" ] && backfill "$derived" "$parent"
done < <(python3 -c "
import json
m = json.load(open('$MUSE_ROOT/models.json'))
for k, v in m.items():
    if k.startswith('_'):
        continue
    for _, path in (v.get('weights') or {}).items():
        print(f\"{path}|{v['path']}\")
")
