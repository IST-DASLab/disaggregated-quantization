"""Persistent training state for restartable QAD runs.

A "training state" is everything needed to resume bit-faithfully: the FP32 master
weights, the optimizer moments, the step counter and the RNG states. It is written
to `<ckpt_dir>/<ckpt_tag>/state/step_XXXXXXX/` and is separate from the eval-ready
weight exports under `.../weights/` (which are lossy — bf16 or packed 4-bit).

The optimizer needs care. DistAdamW is ZeRO-2: for a parameter with numel >= 1024
each rank keeps only the shard `p[rank*rsize:(rank+1)*rsize]` of exp_avg /
exp_avg_sq, so rank 0's state_dict() alone would silently lose 7/8 of the moments.
On save we therefore all-gather each moment into a full tensor and immediately move
it to CPU (one parameter at a time, so the extra GPU footprint is a single
parameter, not a second copy of the whole optimizer). On load every rank reads the
same file from the shared filesystem and slices its own shard back out — no scatter
collective needed, and no rank-0 memory spike.

Small parameters (numel < 1024) are not sharded: their grads are all-reduced, so
every rank holds identical full-size moments and rank 0's copy is authoritative.

Data position needs no bookkeeping: the training loop indexes chunks as
`step * chunks_per_step + acc * mbs`, a pure function of the step, so resuming at
`start_step` fast-forwards the dataset exactly.
"""

import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist


def _params(optimizer) -> list:
    """Optimizer parameters in a stable order (identical on every rank)."""
    return [p for g in optimizer.param_groups for p in g["params"]]


@torch.no_grad()
def gather_optimizer_state(optimizer) -> list | None:
    """All-gather the sharded optimizer moments; CPU list on rank 0, else None.

    Every rank must call this (all_gather is collective). Peak extra GPU memory is
    one full-size parameter, since each gathered moment is moved to CPU immediately.

    Which moments exist depends on the group's algorithm — AdamW keeps exp_avg and
    exp_avg_sq, Lion only exp_avg — so the tensor keys are discovered rather than
    hardcoded. Every rank walks the same parameters in the same order and each
    parameter has the same keys on every rank, so the collectives stay in lockstep.

    SCOPED TO THE OPTIMIZER'S OWN GROUP. DistOptimizer shards the moments across the
    group it reduces on, which under pipeline parallelism is the DP group, not the
    world. Gathering on the default group then asks for world_size x a 1/dp_size shard
    and torch rejects it outright:
        ValueError: output tensor size must be equal to world_size times input size
    -- which is the good outcome; a group that happened to divide evenly would have
    reassembled garbage silently. The group is read off the optimizer so the two can
    never disagree.
    """
    pg = getattr(optimizer, "_pg", None)
    rank, world_size = dist.get_rank(group=pg), dist.get_world_size(group=pg)
    out: list = []
    for p in _params(optimizer):
        st = optimizer.state.get(p, None)
        if not st:                       # parameter never stepped
            out.append(None)
            continue
        small = p.numel() < 1024         # not sharded -> replicated on every rank
        entry = {"step": int(st["step"])}
        for key in sorted(k for k, v in st.items() if torch.is_tensor(v)):
            shard = st[key].contiguous()
            if small:
                entry[key] = shard.to("cpu", copy=True) if rank == 0 else None
            else:
                full = torch.empty(p.shape, dtype=shard.dtype, device=shard.device)
                dist.all_gather_into_tensor(full, shard, group=pg)
                entry[key] = full.to("cpu") if rank == 0 else None
                del full                 # free the GPU buffer before the next moment
        out.append(entry if rank == 0 else None)
    return out if rank == 0 else None


@torch.no_grad()
def load_optimizer_state(optimizer, saved: list) -> None:
    """Restore moments, slicing each rank's shard out of the full saved tensors.

    Same group as gather_optimizer_state: the slice index must be this rank's position
    within the group the moments were sharded across, which under PP is the DP group.
    """
    pg = getattr(optimizer, "_pg", None)
    rank, world_size = dist.get_rank(group=pg), dist.get_world_size(group=pg)
    for p, entry in zip(_params(optimizer), saved):
        if entry is None:
            continue
        st = optimizer.state[p]
        st["step"] = entry["step"]
        small = p.numel() < 1024
        rsize = p.shape[0] // world_size
        for key in sorted(k for k in entry if k != "step"):
            full = entry[key]
            shard = full if small else full[rank * rsize : (rank + 1) * rsize]
            st[key] = shard.to(device=p.device, dtype=p.dtype).contiguous()


def state_root(ckpt_dir: str, ckpt_tag: str) -> Path:
    return Path(ckpt_dir) / ckpt_tag / "state"


def find_latest(ckpt_dir: str, ckpt_tag: str) -> tuple[Path, int] | None:
    """Newest complete training state, or None. Incomplete (*.tmp) dirs are ignored.

    A PIPELINE state is complete when EVERY stage has landed: the meta sits at
    step_N/stage<k>/meta.json, not step_N/meta.json. Requiring only the top-level file
    made find_latest skip every PP state and return None -- so a chained follow-up would
    not resume, it would silently restart the run from step 0 and overwrite the
    checkpoints as it went. Both stages are required, so a state written by a job that
    died between the two renames is correctly treated as incomplete.
    """
    root = state_root(ckpt_dir, ckpt_tag)
    if not root.is_dir():
        return None
    best = None
    for d in root.glob("step_*"):
        if d.name.endswith(".tmp"):
            continue
        stages = sorted(d.glob("stage*/meta.json"))
        if not (d / "meta.json").exists() and not stages:
            continue
        if stages and len(stages) < 2:          # a half-written pipeline state
            continue
        try:
            step = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if best is None or step > best[1]:
            best = (d, step)
    return best


@torch.no_grad()
def save_training_state(student, optimizer, step: int, args, keep_last: int = 1) -> Path | None:
    """Write a resumable training state. Collective: call on every rank.

    Written to a .tmp directory and renamed, so an interrupted save can never be
    mistaken for a complete one. Older states are pruned to `keep_last` (these are
    large: FP32 weights + two FP32 moments ~ 12 bytes/param).
    """
    groups = getattr(args, "_pp_groups", None)
    # UNDER PIPELINE PARALLELISM each stage owns a DIFFERENT half of the model, so one
    # writer cannot produce a complete state. Each stage writes its OWN half into
    # stage<k>/, and the writer for a stage is its dp_rank 0 -- the rank that
    # gather_optimizer_state() collects the ZeRO-2 shards onto within that stage's DP
    # group. Resume therefore requires the SAME pp and dp, which load_training_state
    # enforces; that is a deliberate trade for not having to merge halves here.
    opt_state = gather_optimizer_state(optimizer)      # collective — all ranks
    if groups is not None:
        writer = groups.dp_rank == 0
        subdir = f"stage{groups.pp_rank}"
    else:
        writer = dist.get_rank() == 0
        subdir = None
    model_state = {k: v.detach().cpu() for k, v in student.state_dict().items()} \
        if writer else None
    if not writer:
        dist.barrier()
        return None

    root = state_root(args.ckpt_dir, args.ckpt_tag)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{step:07d}"
    tmp = root / f"step_{step:07d}.tmp"
    if subdir is not None:
        final, tmp = final / subdir, tmp / subdir
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    torch.save(model_state, tmp / "model.pt")
    torch.save(opt_state, tmp / "optim.pt")
    (tmp / "meta.json").write_text(json.dumps({
        "step": step,
        "world_size": dist.get_world_size(),
        "pp_size": (2 if groups is not None else 1),
        "dp_size": (groups.dp_size if groups is not None else dist.get_world_size()),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "torch_rng": None,   # RNG lives in rng.pt (tensors are not JSON-serializable)
    }, indent=2))
    torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
               tmp / "rng.pt")

    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    if subdir is not None:
        # os.replace moved the stage subdir OUT of step_N.tmp/, leaving that parent
        # behind empty. Left alone it accumulates one dead directory per save.
        try:
            tmp.parent.rmdir()
        except OSError:
            pass                                # the other stage still has its subdir


    # prune old states (they are huge). Only one writer prunes, or two stages would
    # race to rmtree the same directory.
    if subdir is not None and dist.get_rank() != 0:
        dist.barrier()
        return final
    states = sorted(root.glob("step_*"), key=lambda d: d.name)
    states = [d for d in states if not d.name.endswith(".tmp")]
    for old in states[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)

    print(f"[rank0] training state → {final}", flush=True)
    dist.barrier()
    return final


def load_training_state(path: Path, student, optimizer, device, groups=None) -> int:
    """Restore weights + optimizer + RNG from `path`. Every rank reads the same files
    (shared filesystem) and keeps only its own optimizer shard. Returns the step to
    resume AT (i.e. the next step to run)."""
    # Under PP each stage restores its OWN half from stage<k>/. The layout is only
    # meaningful for the exact (pp, dp) it was written with -- the model half is stage-
    # specific and the optimizer payload is ZeRO-2-sharded across that stage's DP group
    # -- so both are checked below rather than being silently reinterpreted.
    if groups is not None:
        path = path / f"stage{groups.pp_rank}"
    meta = json.loads((path / "meta.json").read_text())
    step = int(meta["step"])
    ws = dist.get_world_size()
    want_pp = 2 if groups is not None else 1
    want_dp = groups.dp_size if groups is not None else ws
    if meta.get("pp_size", 1) != want_pp:
        raise RuntimeError(
            f"state was saved with pp_size={meta.get('pp_size', 1)}, now {want_pp}; the "
            "halves of the model do not correspond — resume with the same --pp")
    if meta.get("dp_size", meta.get("world_size")) not in (None, want_dp):
        raise RuntimeError(
            f"state was saved with dp_size={meta.get('dp_size')}, now {want_dp}; the "
            "ZeRO-2 optimizer shards would not line up — rerun with the same data-"
            "parallel width (nodes x gpus / pp)")
    if groups is None and meta.get("world_size") not in (None, ws):
        raise RuntimeError(
            f"checkpoint was saved with world_size={meta['world_size']}, now {ws}; "
            "optimizer shards would not line up — rerun with the same world size"
        )

    # EVERY rank reads the same file off the shared filesystem, as this module has
    # always done.
    model_state = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    student.load_state_dict(model_state)
    student.to(device)
    del model_state

    opt_state = torch.load(path / "optim.pt", map_location="cpu", weights_only=False)
    load_optimizer_state(optimizer, opt_state)
    del opt_state

    # Only rank 0 saved its RNG, so only rank 0 restores it — pushing one rank's
    # stream onto every rank would correlate them, which is worse than letting the
    # others carry on. Nothing in the step depends on RNG anyway (no dropout; the
    # chunk shuffle is seeded per rank at build time and cached).
    if dist.get_rank() == 0:
        rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])

    if dist.get_rank() == 0:
        print(f"[rank0] resumed from {path} — continuing at step {step + 1}", flush=True)
    return step + 1
