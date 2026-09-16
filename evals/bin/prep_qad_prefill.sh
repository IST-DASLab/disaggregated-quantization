#!/bin/bash
# Turn a QAD training checkpoint into something vLLM can serve, without touching it.
#
#   ./bin/prep_qad_prefill.sh --ckpt <...>/step_0000250/prefill --out-name fixpaths250-IQ1_S
#
# Builds a serving VIEW under models/: weights symlinked (no copy), the checkpoint's own
# config.json, and the three repairs the exporter currently needs. Every repair is
# idempotent and is skipped when the exporter has already done it, so this keeps working
# unchanged as the saving side is fixed.
#
# REPAIR 1 -- tokenizer/processor. The export ships only config/generation_config/weights.
# vLLM resolves the IMAGE PROCESSOR from the model directory (not from --tokenizer), so a
# multimodal model dies with "OSError: Can't load image processor". Copied from --base.
#
# REPAIR 2 -- `.inner` fp32 masters. The norm wrapper exposes its parameter one level
# down, so norms were exported twice: `...input_layernorm.weight` (BF16) and
# `...input_layernorm.inner.weight` (F32). vLLM has no `.inner` submodule and rejects the
# whole load. Delegated to bin/strip_inner_weights.py, which refuses to drop any `.inner`
# tensor that has no plain counterpart.
#
# REPAIR 3 -- the `ignore` list. compressed-tensors builds a quantized layer for every
# module matching `targets` that is not in `ignore`; if the weights store that module as
# plain `.weight`, loading dies with
#     AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
# Seen twice with different shapes: `ignore: ['lm_head']` (nothing declared), and a
# 207-entry list covering only layers 0-30 of 64 -- the second half of the Gated-DeltaNet
# stack silently omitted.
#
# THE LIST IS DERIVED FROM THE WEIGHTS, not copied from a reference checkpoint. A module
# is quantized iff it has `weight_packed`; anything else with a 2-D `.weight` is a Linear
# that was left alone and must be declared. That is the ground truth, it cannot drift
# from the file it describes, and it stays correct for a checkpoint quantizing a
# different module set.
set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSE_ROOT=${MUSE_ROOT:-$(dirname "$_SELF_DIR")}
P=/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode
BASE=${BASE:-$P/models/Qwen3.8-27B}
MODELS=${MODELS:-$P/models}

CKPT=""; OUT_NAME=""
while (($# > 0)); do
    case "$1" in
        --ckpt)     CKPT="$2";     shift 2 ;;
        --out-name) OUT_NAME="$2"; shift 2 ;;
        --base)     BASE="$2";     shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -s "$CKPT/config.json" ] || { echo "ERROR: no config.json in $CKPT" >&2; exit 2; }
[ -n "$OUT_NAME" ] || { echo "ERROR: --out-name is required" >&2; exit 2; }
V="$MODELS/$OUT_NAME-prefill"
mkdir -p "$V"

cp -f "$CKPT/config.json" "$V/config.json"
[ -e "$CKPT/generation_config.json" ] && cp -f "$CKPT/generation_config.json" "$V/"
for f in tokenizer.json tokenizer_config.json merges.txt vocab.json chat_template.jinja \
         preprocessor_config.json video_preprocessor_config.json; do
    [ -e "$BASE/$f" ] && cp -f "$BASE/$f" "$V/"
done

# Strip only when needed; otherwise symlink the single-file checkpoint in place.
if python3 -c "
import json,struct,sys,glob
ks=[]
for p in sorted(glob.glob('$CKPT/*.safetensors')):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; ks+=list(json.loads(f.read(n)))
sys.exit(0 if any('.inner.' in k for k in ks) else 1)"; then
    echo "  repair 2: stripping .inner masters"
    rm -f "$V"/model*.safetensors "$V"/model.safetensors.index.json
    python3 "$_SELF_DIR/strip_inner_weights.py" --src "$CKPT/model.safetensors" --out "$V" || exit 1
else
    echo "  repair 2: no .inner tensors (exporter already fixed)"
    rm -f "$V"/model*.safetensors "$V"/model.safetensors.index.json
    for s in "$CKPT"/*.safetensors; do ln -sf "$s" "$V/$(basename "$s")"; done
    [ -e "$CKPT/model.safetensors.index.json" ] && cp -f "$CKPT/model.safetensors.index.json" "$V/"
fi

python3 - "$V" <<'PY'
import glob, json, struct, sys

view = sys.argv[1]
names = []
for p in sorted(glob.glob(f"{view}/*.safetensors")):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    names += [(k, v.get("shape")) for k, v in h.items() if k != "__metadata__"]

packed = {k.rsplit(".weight_packed", 1)[0] for k, _ in names if k.endswith("weight_packed")}
# 2-D `.weight` == a Linear. Conv patch-embeds are 4/5-D and are not `targets: ['Linear']`,
# so listing them would be noise; norms are 1-D for the same reason.
need = sorted({k.rsplit(".weight", 1)[0] for k, s in names
               if k.endswith(".weight") and s and len(s) == 2
               and k.rsplit(".weight", 1)[0] not in packed})

cfg_path = f"{view}/config.json"
cfg = json.load(open(cfg_path))
q = cfg.get("quantization_config")
if not q:
    print("  repair 3: no quantization_config; nothing to do")
else:
    old = q.get("ignore") or []
    if set(old) == set(need):
        print(f"  repair 3: ignore already matches the weights ({len(old)} entries)")
    else:
        q["ignore"] = need
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        missed = sorted(set(need) - set(old))
        print(f"  repair 3: ignore {len(old)} -> {len(need)} "
              f"({len(missed)} undeclared unquantized Linears added)")
        for m in missed[:3]:
            print(f"             e.g. {m}")
print(f"  quantized Linears: {len(packed)}   declared ignore: {len(json.load(open(cfg_path)).get('quantization_config',{}).get('ignore') or [])}")
PY

echo "  serving view ready: $V"
