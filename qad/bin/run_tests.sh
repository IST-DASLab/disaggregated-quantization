#!/bin/bash
# Run the QAD tests inside the container on an interactive allocation.
#
#   ./bin/run_tests.sh                      # every test (8 GPUs: test_checkpoint needs them)
#   ./bin/run_tests.sh gsq_lloyd            # one file (tests/test_gsq_lloyd.py), 1 GPU
#   ./bin/run_tests.sh gsq_lloyd quantizers # several
#   ./bin/run_tests.sh -v gsq_lloyd         # full output instead of just PASS/FAIL lines
#
# test_checkpoint.py is distributed (it verifies the ZeRO-2 optimizer shards restore
# exactly), so it is launched under torchrun with as many ranks as there are GPUs;
# everything else runs single-process.
set -uo pipefail

CONTAINER=${CONTAINER:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/containers/nemo-26.02.sqsh}
ACCOUNT=coreai_psx_qad
TIME=${TIME:-00:30:00}

# tests/ and the importable packages live in the qad root, one level up from bin/.
# cd+pwd (bash builtins, no -P) rather than realpath: realpath calls getcwd(), which
# resolves the /lustre->/scratch symlink.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SELF_DIR")" || exit 1

VERBOSE=0
TESTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -v|--verbose) VERBOSE=1; shift ;;
        *)            TESTS+=("${1%.py}"); shift ;;
    esac
done

if [ ${#TESTS[@]} -eq 0 ]; then
    for f in tests/test_*.py; do TESTS+=("$(basename "${f%.py}")"); done
fi

# Only the distributed test needs the full node. MIN_GPUS is what the scheduler must be
# ASKED for -- every QOS on this cluster carries MinTRES gres/gpu=4, so a 1-GPU request
# is refused at submit with QOSMinGRES (see evals/bin/run_quantize.sh for the same knob).
GPUS=${MIN_GPUS:-1}
for t in "${TESTS[@]}"; do
    case "$t" in
        *checkpoint*) GPUS=8 ;;
        # test_pipeline needs the FULL node, not merely an even world. At world=2 there
        # is a single pipeline pair, and three real bugs were invisible there because
        # they only appear with SEVERAL pairs or a wider mesh:
        #   * the export handoff filename raced between pairs (FileNotFoundError);
        #   * the "not the cross-node block layout" assertion is vacuous at world=2,
        #     where the adjacent peer and the block peer are the same rank;
        #   * dp_size is 1, so nothing exercises the DP/PP group interaction.
        # 8 gives 4 pairs and dp_size=4, which is what the real runs use.
        *pipeline*) GPUS=8 ;;
    esac
done

# Build the in-container command: each test is a line, so one failure does not hide
# the rest, and a trailing summary reports which files failed.
# pipefail is REQUIRED: each test is piped through grep, and without it the
# pipeline reports GREP's status. A python traceback still makes grep print
# lines and exit 0, so every failing test was reported as passing -- which is
# how tests/test_dual.py sat broken at import time (export_variants does not
# exist in export.save) while the runner said ALL TEST FILES PASSED.
CMD='cd "$PWD"; export PYTHONPATH="$PWD"; set -o pipefail; failed=""'
for t in "${TESTS[@]}"; do
    f="tests/${t#test_}"; f="tests/test_${t#test_}.py"
    case "$t" in
        *checkpoint*|*pipeline*|*flat_shard*) run="torchrun --nproc_per_node=$GPUS $f" ;;
        *)                   run="python3 $f" ;;
    esac
    CMD="$CMD; echo; echo \"=== $f ===\"; $run 2>&1 | grep -viE 'futurewarning|pynvml|^  import' || failed=\"\$failed $f\""
done
CMD="$CMD; echo; if [ -n \"\$failed\" ]; then echo \"FAILED:\$failed\"; exit 1; else echo 'ALL TEST FILES PASSED'; fi"

echo "tests: ${TESTS[*]}  (gpus=$GPUS)"
OUT=$(srun --account="$ACCOUNT" --partition=batch --qos=interactive --time="$TIME" \
    --nodes=1 --ntasks=1 --gpus-per-node="$GPUS" \
    --container-image="$CONTAINER" --no-container-mount-home \
    --container-mounts=/scratch:/scratch,/lustre:/lustre \
    bash -c "PWD=$PWD; $CMD" 2>&1)
rc=$?

if [ "$VERBOSE" = 1 ]; then
    echo "$OUT"
else
    echo "$OUT" | grep -E '^===|PASS|FAIL|OK$|Error|error:|ALL |Traceback|^  [a-zA-Z].*='
fi
exit $rc
