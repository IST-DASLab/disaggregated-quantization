#!/bin/bash
#SBATCH --job-name=qad
# ^ fallback only: stage 0 always passes --job-name=qad-<model>-<quantizer>, which is
#   what --dependency=singleton keys on. Do not point tooling at this literal.
# This cluster's nodes are 4-GPU trays (Gres=gpu:4), not the old cluster's 8-GPU boxes --
# 2 nodes x 4 GPUs = the world-8 default the README's scaling table assumes. --nodes here
# means "4-GPU nodes", so what used to be "--nodes 2" (16 GPUs, e.g. 12b) is now "--nodes 4".
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=coreai_psx_qad
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# STAGE 0: pre-submit (login node). Create nested log dir logs/train/<stamp>_<tag>/
# and re-submit into it (SLURM can't create --output dirs; mkdir-in-job is too
# late). Invoke directly, e.g.:
#   ./bin/run_qad.sh --quantizer ste4bit            # normal
#   ./bin/run_qad.sh --quantizer ste4bit --debug    # interactive QoS
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    # cd+pwd (bash builtins, no -P) rather than realpath: realpath calls getcwd(),
    # which resolves the /lustre->/scratch symlink and would hand sbatch a /scratch
    # path -- invisible to the container, which only mounts /lustre. See MIGRATION.md.
    SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELF="$SELF_DIR/$(basename "${BASH_SOURCE[0]}")"
    # This script lives in qad/bin/, so the repo root is two levels up.
    ROOT="$(dirname "$SELF")/../.."
    STAMP="$(date +%Y%m%d_%H%M%S)"
    TAG="run"; QOS_ARGS=(); NODE_ARGS=(); PASS=(); CHAIN=1
    while [ $# -gt 0 ]; do
        case "$1" in
            --debug)       QOS_ARGS=(--qos=interactive --time=1:00:00);   shift ;;
            # --short: qos=short is capped at 2h wall and 4 nodes, but runs at priority
            # 200 against normal's 100, so it starts sooner. Fits the small models --
            # 270m is ~1h51m and 1b ~1h40m measured -- but NOT 4b (>4h) or 12b, and a
            # 2h cap leaves 1b only ~20 min of margin, so a slow node can still time out.
            --short)       QOS_ARGS=(--qos=short --time=2:00:00);         shift ;;
            # --long: the `batch_long` partition's cap is 7 DAYS, not 4 hours, so a run
            # that cannot fit the wall finishes in ONE job instead of being chained.
            # That matters at 27B: measured 92.5 s/step at mbs=1, i.e. ~64h for the
            # 2485-step recipe -- 16 chained 4h jobs, where each handover reloads a 52 GB
            # model and repeats the teacher baseline. Chaining is the workaround for a
            # wall; this removes the wall. Takes an optional duration, default 48h.
            # The duration is decided BEFORE it is used: "${2:-48h}" substitutes the
            # default only when $2 is EMPTY, so a bare `--long --train-tokens N` would
            # take "--train-tokens" as the wall clock and swallow the real flag.
            --long)        case "${2:-}" in
                               [0-9]*) _LONG_T="$2"; shift 2 ;;
                               *)      _LONG_T=48:00:00; shift ;;
                           esac
                           QOS_ARGS=(--partition=batch_long --time="$_LONG_T") ;;
            # --time is capped at 04:00:00 by the cluster, and that is NOT enough past
            # ~4B: a 4B run measures 5.81 s/step, i.e. 4.01h for 2485 steps, so it
            # TIMEOUTs a few steps from the end. 8B is roughly double. --chain N submits
            # N jobs sharing one --job-name and holding --dependency=singleton, so SLURM
            # runs them strictly one at a time and each continues where the last stopped
            # via --resume auto (state/ is written every --save-every=100 steps, so a
            # timeout costs at most 100 steps). Over-provision freely: a chain job that
            # finds the run already finished exits at once with "Nothing to do: state at
            # step N >= total" rather than retraining anything.
            --chain)       CHAIN="$2";                                    shift 2 ;;
            --chain=*)     CHAIN="${1#--chain=}";                         shift ;;
            # NODE_ARGS, not QOS_ARGS: --debug and --short ASSIGN QOS_ARGS, so a
            # --nodes appended before either of them was silently discarded and the
            # job ran on ONE node. Invisible at submit time -- it showed up only as
            # a smaller world_size in the training log.
            --nodes)       NODE_ARGS=(--nodes="$2");                      shift 2 ;;
            --quantizer=*) TAG="${1#--quantizer=}"; PASS+=("$1");      shift ;;
            --quantizer)   TAG="$2";                PASS+=("$1" "$2"); shift 2 ;;
            *)             PASS+=("$1");                                 shift ;;
        esac
    done
    # Singleton is keyed on (user, job-name), so the NAME must identify this run. The
    # old hardcoded qad-qwen3-4b was shared by every training job, which under singleton
    # would have serialised the entire cluster's worth of runs against each other.
    # Naming it per (model, quantizer) also stops two jobs from ever writing one state/
    # directory concurrently, which would corrupt the resume point.
    # The repo name without the vendor: Qwen/Qwen3-4B -> Qwen3-4B, and
    # google/gemma-3-4b-it -> gemma-3-4b-it. The old form stripped a literal "Qwen-"
    # prefix, which left Gemma job names as "google-gemma-3-4b-it".
    # basename, not `cut -d/ -f2`: field 2 of a LOCAL PATH is a leading directory, so
    # /scratch/.../models/Qwen3.8-27B became "scratch" and every run of one quantizer
    # shared a job name. With --dependency=singleton that silently SERIALISES them --
    # four 13.5h 27B runs queued behind each other as "Reason: Dependency" instead of
    # running in parallel.
    MODEL_TAG=$(basename "${MODEL:-Qwen/Qwen3-4B}")
    # JOB_NAME override: singleton is keyed on (user, job-name), and its purpose is to
    # stop two jobs writing one state/ directory. Runs that differ only in
    # --quantizer-params (e.g. a different frozen decode checkpoint) get DIFFERENT
    # checkpoint tags, so they are safe in parallel -- but the derived name cannot tell
    # them apart. Set JOB_NAME to give each one its own singleton domain.
    JOB_NAME="${JOB_NAME:-qad-${MODEL_TAG}-${TAG}}"
    LOGDIR="$ROOT/logs/train/${STAMP}_${TAG}"
    mkdir -p "$LOGDIR"
    echo "logs → $LOGDIR"
    echo "job  → $JOB_NAME   (chain of $CHAIN, --dependency=singleton)"
    for _i in $(seq 1 "$CHAIN"); do
        sbatch --job-name="$JOB_NAME" --dependency=singleton \
            "${QOS_ARGS[@]}" "${NODE_ARGS[@]}" \
            --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
            "$SELF" "${PASS[@]}"
    done
    exit 0
fi

# ---- under SLURM allocation ----
EXTRA_ARGS=("$@")   # --debug already consumed in STAGE 0

# `exit` after the first match: `scontrol show job` can print more than one record
# (always the case for the last element of a job array), which would otherwise make
# SCRIPT_PATH a multi-line string and fail with bash "No such file" (exit 127).
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}')
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")     # qad/bin
QAD_DIR=$(dirname "$SCRIPT_DIR")         # qad -- anchors checkpoints/, wandb/, imports

# oci-jhb-slurm-1: /lustre is a root-level symlink to /scratch, AND
# /scratch/fsw/portfolios/coreai/users/apanferov is itself a symlink to
# ../projects/coreai_psx_qad/users/apanferov -- so the same directory is reachable by
# three path spellings. Rather than replay the old cluster's logical-vs-physical dance,
# everything here uses the PHYSICAL /scratch path and the srun below mounts
# /scratch:/scratch. getcwd()/realpath then agree with the mount, so the whole class of
# "file is right there but invisible inside the container" bugs cannot occur. See
# MIGRATION.md §1 for what this replaces.
PD_ROOT=${PD_ROOT:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode}
CONTAINER=${CONTAINER:-$PD_ROOT/containers/nemo-26.02.sqsh}
HF_CACHE=${HF_CACHE:-$PD_ROOT/hf_cache}
MODEL=${MODEL:-Qwen/Qwen3-4B}
# RUN_PREFIX namespaces the checkpoint tag (<prefix>-<model>-<quant>-<hash>).
# Use a fresh prefix for a new training recipe so earlier runs are never overwritten.
RUN_PREFIX=${RUN_PREFIX:-qad3x}
# Checkpoints live on the coreai_psx_NEXTGEN project quota, not coreai_psx_qad.
# Same filesystem, separate 100 TiB quota -- qad was FULL (writes failing with
# "Disk quota exceeded" inside safetensors save_file, killing jobs at their first
# export), while nextgen was ~0.5 TiB used. A qad-side symlink per run keeps
# $QAD_DIR/checkpoints/<tag> resolving, so submit_missing_evals.py and plots.ipynb
# discover runs through the unchanged path.
CKPT_DIR=${CKPT_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_psx_nextgen/users/apanferov/prefill_decode/checkpoints}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

srun \
    --ntasks-per-node=1 \
    --container-image="$CONTAINER" \
    --no-container-mount-home \
    --container-mounts="/scratch:/scratch,/lustre:/lustre,$HOME/.netrc:/root/.netrc" \
    bash -c "
        export HF_HOME=$HF_CACHE
        export TOKENIZERS_PARALLELISM=false
        # SHARDED checkpoints + 8 ranks = a cache race that kills the run in 90 seconds.
        # 4b/12b ship as model-0000N-of-00002.safetensors, so from_pretrained goes through
        # get_checkpoint_shard_files(); all 8 ranks then hit the hub concurrently and one
        # loses, with
        #   OSError: google/gemma-3-4b-it does not appear to have a file named
        #            model-00001-of-00002.safetensors
        # even though the cache is complete and every shard is present. 270m/1b/Qwen never
        # showed it because they are single-file and skip that code path entirely.
        # Offline mode removes the fetch, so every rank reads the warm snapshot directly.
        # Same setting bin/run_eval_disagg.sh already uses. Override only to warm a cache.
        export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
        # psx-luts carries the luts extension nvr2bit imports lazily. NO BACKTICKS:
        # this whole block is a double-quoted bash -c string, so backticks are
        # command substitution and even a COMMENT gets executed.
        # venv_overlay: this container's extras (wandb, datasets, ...) live only in
        # /opt/venv, which the NVIDIA GPU container-runtime hooks silently hide/replace
        # on a GPU-allocated job (confirmed: PATH loses /opt/venv/bin and the
        # site-packages dir itself 404s, though it's readable fine on a GPU-less
        # container). A verbatim copy of /opt/venv's site-packages onto Lustre,
        # picked up by qad/sitecustomize.py via site.addsitedir() -- APPENDED, not
        # prepended on PYTHONPATH, so the base container's own packages still win
        # where both exist (a plain prepend let the overlay's stale bundled botocore
        # shadow the base's boto3-compatible one).
        # NOT NEEDED ON oci-jhb-slurm-1, and the directory deliberately does not exist:
        # sitecustomize.py checks os.path.isdir first, so this is a no-op. Verified on
        # this cluster + nemo:26.02 that a GPU-allocated job still sees /opt/venv
        # (torch/transformers/datasets/wandb all import from
        # /opt/venv/lib/python3.12/site-packages under a 4-GPU allocation), i.e. the old
        # cluster's hide-on-GPU-allocation quirk does not reproduce here. Kept wired up
        # so that if a future image DOES hide it, populating this path is the only fix
        # needed. This is the TRAINING overlay and stays a SEPARATE tree from evals'
        # lm_eval_overlay: this one is APPENDED to sys.path (container packages win),
        # the eval one is PREPENDED (lm_eval must win). See MIGRATION.md §3/§4.
        export VENV_OVERLAY=${VENV_OVERLAY:-$PD_ROOT/venv_overlay}
        export PYTHONPATH=$QAD_DIR:${PSX_LUTS_PATH:-$PD_ROOT/psx-luts}:\$PYTHONPATH
        export WANDB_MODE=${WANDB_MODE:-online}
        # wandb creates its run directory under \$WANDB_DIR, which defaults to the
        # CWD — and inside the container that is not a writable path, so wandb.init()
        # blocks until it times out. This hits offline mode too (it needs the same
        # local dir), which is how three jobs ended up training with logging disabled
        # after burning 6 minutes each on two 180s timeouts. Point it somewhere real.
        export WANDB_DIR=$QAD_DIR
        export RUN_PREFIX=$RUN_PREFIX
        cd $QAD_DIR
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # Flight recorder only -- silent unless a collective actually times out, unlike
        # NCCL_DEBUG=INFO/TORCH_DISTRIBUTED_DEBUG=DETAIL (used to diagnose the
        # total_steps rank-desync bug above; verbose enough to bloat every job's log,
        # so not left on by default now that it's fixed).
        export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
        export TORCH_NCCL_DUMP_ON_TIMEOUT=1

        # --master_port below is PER JOB, derived from the job id: every rank of one job
        # computes the same value, while two jobs sharing a node cannot collide. A
        # hardcoded 29500 killed a gemma-12b run and then the IQ3_XXS 27B run, both with
        # 'EADDRINUSE ... port: 29500'. It only ever worked because concurrent jobs
        # rarely shared a node -- which stops being rare the moment four 16-node jobs
        # start together. The range is kept BELOW ip_local_port_range (9000-65000 here):
        # inside it the kernel can hand the same port to an outgoing connection between
        # rendezvous and bind, and 29500 sat inside it too.
        # The forwarded arguments below are \"\$@\", QUOTED. Unquoted, every one of them
        # word-splits, so any argument containing a space silently arrives as several --
        # which is how --quantizer-params '{\"decode_model\": \"/path\"}' reached qad.py as
        # '{\"decode_model\":' plus an unrecognized '\"/path\"}' and died at argparse.
        # Nothing caught it earlier because every other caller passes space-free values.
        python3 -m torch.distributed.run \
            --nproc_per_node=4 \
            --nnodes=$SLURM_NNODES \
            --node_rank=\$SLURM_PROCID \
            --master_addr=$MASTER_ADDR \
            --master_port=\$((2000 + SLURM_JOB_ID % 6000)) \
            $QAD_DIR/training/qad.py \
                --model $MODEL \
                --run-name $RUN_PREFIX-\$(echo $MODEL | tr '/' '-') \
                --ckpt-dir $CKPT_DIR \
                --global-batch-size 64 \
                \"\$@\"
    " -- "${EXTRA_ARGS[@]}"
