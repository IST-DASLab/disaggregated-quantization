#!/bin/bash
# Make every run under CKPT_DIR discoverable through qad/checkpoints/<tag>.
#
#   ./cluster_scripts/link_checkpoints.sh          # create missing links
#   ./cluster_scripts/link_checkpoints.sh --dry    # list what it would create
#
# WHY THIS EXISTS
# ---------------
# Checkpoints live on the coreai_psx_nextgen project quota (see run_qad.sh's CKPT_DIR),
# but submit_missing_evals.py and plots.ipynb discover runs by scanning qad/checkpoints.
# run_qad.sh writes to CKPT_DIR and does NOT create the back-link, so a freshly launched
# run is invisible to the eval sweep: it trains and exports perfectly while the gap scan
# reports nothing to do. That is exactly what happened to the eight nvfp4a16 runs --
# 270m reached step 1500 with no RULER job ever submitted, and the only symptom was a
# format silently absent from the gap scan.
#
# The links are made from the DESTINATION side rather than in run_qad.sh because the tag
# ends in a hash of the quantizer params, computed in qad.py (ckpt_tag, qad.py:379) and
# not reconstructible in shell. Globbing what actually landed needs no hash logic and
# covers every run at once, including ones launched before this script existed.
#
# Safe to run repeatedly and while jobs are writing: it only ever ADDS symlinks for tags
# that have no entry at all. An existing directory or link is left untouched, so it can
# never replace a real checkpoint dir with a link to somewhere else.
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAD_DIR="$(dirname "$SELF_DIR")"

CKPT_DIR=${CKPT_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_nextgen/users/apanferov/prefill_decode/checkpoints}
LINK_DIR=${LINK_DIR:-$QAD_DIR/checkpoints}

DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

[ -d "$CKPT_DIR" ] || { echo "no CKPT_DIR: $CKPT_DIR" >&2; exit 1; }
mkdir -p "$LINK_DIR"

made=0; have=0
for d in "$CKPT_DIR"/*/; do
    [ -d "$d" ] || continue
    tag=$(basename "$d")
    # -e follows symlinks and would call a link to a deleted target "missing", then fail
    # to create it because the dangling link still occupies the name. -e OR -L.
    if [ -e "$LINK_DIR/$tag" ] || [ -L "$LINK_DIR/$tag" ]; then
        have=$((have + 1)); continue
    fi
    if [ "$DRY" = 1 ]; then
        echo "would link $tag"
    else
        ln -s "${d%/}" "$LINK_DIR/$tag" && echo "linked $tag"
    fi
    made=$((made + 1))
done
echo "link_checkpoints: $made new, $have already present"
