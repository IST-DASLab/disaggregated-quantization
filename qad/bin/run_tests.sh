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

CONTAINER=/lustre/fsw/portfolios/adlr/users/apanferov/containers/nemo:26.02.nemotron_3_super_luts_v2.sqsh
ACCOUNT=adlr_psx_numerics
TIME=${TIME:-00:30:00}

# tests/ and the importable packages live in the qad root, one level up from bin/.
cd "$(dirname "$(dirname "$(realpath "$0")")")" || exit 1

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

# Only the distributed test needs the full node.
GPUS=1
for t in "${TESTS[@]}"; do
    case "$t" in *checkpoint*) GPUS=8 ;; esac
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
        *checkpoint*) run="torchrun --nproc_per_node=$GPUS $f" ;;
        *)            run="python $f" ;;
    esac
    CMD="$CMD; echo; echo \"=== $f ===\"; $run 2>&1 | grep -viE 'futurewarning|pynvml|^  import' || failed=\"\$failed $f\""
done
CMD="$CMD; echo; if [ -n \"\$failed\" ]; then echo \"FAILED:\$failed\"; exit 1; else echo 'ALL TEST FILES PASSED'; fi"

echo "tests: ${TESTS[*]}  (gpus=$GPUS)"
OUT=$(srun --account="$ACCOUNT" --partition=batch --qos=interactive --time="$TIME" \
    --nodes=1 --ntasks=1 --gpus-per-node="$GPUS" \
    --container-image="$CONTAINER" --no-container-mount-home \
    --container-mounts=/lustre:/lustre \
    bash -c "PWD=$PWD; $CMD" 2>&1)
rc=$?

if [ "$VERBOSE" = 1 ]; then
    echo "$OUT"
else
    echo "$OUT" | grep -E '^===|PASS|FAIL|OK$|Error|error:|ALL |Traceback|^  [a-zA-Z].*='
fi
exit $rc
