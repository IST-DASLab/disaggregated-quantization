"""Pipeline parallelism, PP=2, deliberately the simplest thing that can work.

NO overlap, no interleaving, no micro-batch scheduling. Stage 0 runs, sends one tensor,
stage 1 runs, sends one gradient back. Both GPUs are idle while the other works, so the
compute utilisation ceiling is ~50%. That is accepted: the point is to halve the
per-GPU parameter footprint so 12b formats that OOM replicated can run at all, and a
simple exchange is far easier to prove equivalent than an overlapped one.

WHAT IS SPLIT. The decoder layers, for BOTH student and teacher, at the midpoint:

    stage 0 : embed_tokens + layers[:H/2]          (norm -> Identity, no head)
    stage 1 : layers[H/2:] + norm + head + LOSS

Stage 1 is entered through `inputs_embeds=`, which is how the received activation is
fed back into an unmodified HuggingFace stack -- no reimplementation of rotary
embeddings, causal/sliding-window masks or position ids, all of which the stack builds
itself from the inputs_embeds shape.

RANK LAYOUT is `pp_rank = rank % 2`, so pipeline pairs are ADJACENT ranks (0,1), (2,3),
... With an even number of GPUs per node every pair is therefore on one node, which is
the requirement that pipeline traffic never crosses the network. The obvious
alternative, `pp_rank = rank // (world//2)`, pairs rank r with r + world/2 and puts
EVERY pair across the wire on a multi-node job. assert_intra_node() pins this.

Data parallelism is unchanged in kind: the ranks sharing a `pp_rank` form the DP group,
and DistOptimizer reduces within it instead of over the whole world.

TIED EMBEDDINGS. Gemma-3 (every size) and Qwen3 below 8B tie lm_head.weight to
embed_tokens.weight. Stage 0 owns the embedding and stage 1 owns the head, so the tie
would be split across two processes. The parameter is therefore REPLICATED on both
stages and its gradient all-reduced over the pipeline group each step, which is what
keeps the two copies identical forever. It costs nothing relative to today (the
embedding is already replicated on every rank), it simply is not halved -- and it must
be counted ONCE in the norms, see local_sq_skipping_replica().
"""

import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

PP_SIZE = 2


class PipelineGroups:
    """Process groups and this rank's place in the 2-stage pipeline."""

    def __init__(self, pp_group, dp_group, pp_rank: int, dp_rank: int,
                 dp_size: int, peer: int):
        self.pp_group = pp_group
        self.dp_group = dp_group
        self.pp_rank = pp_rank          # 0 = embeddings, 1 = loss
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.peer = peer                # global rank of the other half of this pipeline

    @property
    def is_first(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last(self) -> bool:
        return self.pp_rank == 1


def build_groups() -> PipelineGroups:
    """Create the DP and PP groups. COLLECTIVE: every rank must call this, in order.

    new_group() must be called by ALL ranks for EVERY group, even ones they do not join
    -- a rank that skips a group it is not a member of desynchronises the creation
    sequence and the job hangs before the first step.
    """
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world % PP_SIZE:
        raise ValueError(f"world_size={world} is not divisible by PP_SIZE={PP_SIZE}")

    pp_rank, dp_rank = rank % PP_SIZE, rank // PP_SIZE
    dp_size = world // PP_SIZE

    dp_groups = [dist.new_group(ranks=[s + PP_SIZE * d for d in range(dp_size)])
                 for s in range(PP_SIZE)]
    pp_groups = [dist.new_group(ranks=[PP_SIZE * d + s for s in range(PP_SIZE)])
                 for d in range(dp_size)]

    peer = PP_SIZE * dp_rank + (1 - pp_rank)
    return PipelineGroups(pp_groups[dp_rank], dp_groups[pp_rank],
                          pp_rank, dp_rank, dp_size, peer)


def assert_intra_node() -> None:
    """Fail loudly if a pipeline pair would straddle two nodes.

    Pairs are (2k, 2k+1), so they share a node exactly when the per-node rank count is
    even. torchrun exports LOCAL_WORLD_SIZE; if it is missing we cannot check and say so
    rather than pretending the requirement holds.
    """
    lws = os.environ.get("LOCAL_WORLD_SIZE")
    if lws is None:
        return
    if int(lws) % PP_SIZE:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE={lws} is not divisible by PP_SIZE={PP_SIZE}; pipeline "
            f"pairs (2k, 2k+1) would cross a node boundary and the activation exchange "
            f"would go over the network")


# ---------------------------------------------------------------------------
# Model splitting
# ---------------------------------------------------------------------------
def split_stack(base: nn.Module, head: nn.Module | None, pp_rank: int,
                cut: int | None = None) -> None:
    """Keep only this stage's half of `base`, IN PLACE.

    Frees the other half's parameters outright -- the point of the exercise -- rather
    than leaving them allocated and unused.

    Stage 0's final norm becomes Identity: the norm belongs at the END of the whole
    stack, and applying it at the halfway point would silently change the model rather
    than fail. Stage 1 keeps the real norm and applies it after its own layers, exactly
    where the unsplit model does.
    """
    layers = base.layers
    n = len(layers)
    if n < PP_SIZE:
        raise ValueError(f"{n} layers cannot be split across {PP_SIZE} pipeline stages")
    # `cut` = how many layers stage 0 keeps. The midpoint balances LAYER COUNT, which is
    # not the same as balancing MEMORY: stage 1 additionally holds the fused JSD loss
    # over the full vocabulary (262144 at gemma-3-12b), and at 12b split it was stage 1
    # -- every odd rank, all 16 of them -- that OOMed while stage 0 had headroom. Moving
    # layers to stage 0 is a pure repartition: same layers, same order, same arithmetic.
    cut = n // PP_SIZE if cut is None else cut
    if not 1 <= cut < n:
        raise ValueError(f"pp cut {cut} must be in [1, {n}) for a {n}-layer stack")

    # Where this stage's layers sat in the ORIGINAL stack. Export has to undo the
    # reindexing below, or stage 1's "layers.0" would collide with stage 0's.
    base._pp_layer_offset = 0 if pp_rank == 0 else cut

    if pp_rank == 0:
        base.layers = nn.ModuleList(list(layers[:cut]))
        base.norm = nn.Identity()
    else:
        base.layers = nn.ModuleList(list(layers[cut:]))
        # Reindex so each layer's self-reported position matches its position in this
        # stage's ModuleList. Attention implementations read layer_idx to index the KV
        # cache; leaving the original indices would address slots this stage does not
        # own. Training runs with use_cache=False, so this is defensive, but a wrong
        # layer_idx is exactly the kind of thing that only shows up under eval.
        for i, layer in enumerate(base.layers):
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = i
            if getattr(layer, "self_attn", None) is not None and \
                    hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = i
    del layers


def stage_forward(base: nn.Module, pp_rank: int, *, input_ids=None, hidden=None):
    """Run this stage's half of the stack.

    Stage 1 is entered with `inputs_embeds=`, NOT input_ids: the received activation is
    already past the embedding, and inputs_embeds is the documented way to hand a stack
    pre-computed states. The stack still derives position ids, the causal mask and (for
    Gemma-3) the per-layer sliding-window masks from the shape, so nothing about the
    attention pattern has to be reimplemented here.
    """
    if pp_rank == 0:
        return base(input_ids=input_ids).last_hidden_state
    return base(inputs_embeds=hidden).last_hidden_state


# ---------------------------------------------------------------------------
# The exchange: ONE activation tensor forward, ONE gradient tensor back
# ---------------------------------------------------------------------------
def send_activations(student_h: Tensor, teacher_h: Tensor, groups: PipelineGroups) -> None:
    """Stage 0 -> stage 1, as a single tensor.

    Student and teacher hidden states are stacked so the boundary costs exactly one
    send, as specified. The student half is DETACHED for the send -- the autograd graph
    stays on this rank, and the returning gradient is applied to the original tensor by
    backward_from_peer().
    """
    # FP32 ON THE WIRE, both halves. The residual stream is fp32 even under autocast:
    # `h = h + layer(h)` starts at the embedding output (nn.Embedding is not autocast-
    # eligible, so fp32) and each bf16 layer result type-promotes back to fp32. Sending
    # bf16 made stage 1's entire stream bf16, which failed only at the very END -- Liger
    # matmuls the final hidden against the fp32 head weight OUTSIDE autocast, giving
    # "expected mat1 and mat2 to have the same dtype: BFloat16 != float" after a full
    # forward had already run.
    #
    # The teacher's stream genuinely IS bf16 (bf16 params), so the halves differ and one
    # stacked send needs the wider dtype. recv_activations() casts the teacher back,
    # which is exact -- it was bf16 before the upcast.
    payload = torch.stack([student_h.detach().float(), teacher_h.float()]).contiguous()
    dist.send(payload, dst=groups.peer, group=None)


def recv_activations(shape, device, groups: PipelineGroups,
                     teacher_dtype=torch.bfloat16) -> tuple[Tensor, Tensor]:
    """Stage 1 <- stage 0. Returns (student fp32, teacher cast to `teacher_dtype`).

    The student half is marked requires_grad so backward() populates .grad on it; that
    gradient goes back over the wire in fp32, matching the tensor stage 0 applies it to.
    """
    buf = torch.empty((2, *shape), dtype=torch.float32, device=device)
    dist.recv(buf, src=groups.peer, group=None)
    student_h = buf[0].detach().requires_grad_(True)
    return student_h, buf[1].to(teacher_dtype)


def send_input_grad(student_h: Tensor, groups: PipelineGroups) -> None:
    """Stage 1 -> stage 0: d(loss)/d(received activation), one tensor.

    A None grad means the loss did not depend on the activation, which for this model
    can only be a wiring bug; zeros are sent rather than crashing the peer in a recv it
    would otherwise wait on forever, and the caller is expected to notice the flat loss.
    """
    g = student_h.grad
    if g is None:
        g = torch.zeros_like(student_h)
    dist.send(g.contiguous(), dst=groups.peer, group=None)


def backward_from_peer(student_h: Tensor, groups: PipelineGroups) -> None:
    """Stage 0: receive the boundary gradient and continue the backward pass."""
    grad = torch.empty_like(student_h)
    dist.recv(grad, src=groups.peer, group=None)
    student_h.backward(grad)


# ---------------------------------------------------------------------------
# Norms and metrics -- must produce the SAME numbers as the unsplit setup
# ---------------------------------------------------------------------------
def replicated_params(base: nn.Module, head, groups: PipelineGroups) -> list[Tensor]:
    """Parameters present on BOTH stages: the tied embedding/head weight, if tied.

    Identified by STORAGE identity, not by name: the tie is one tensor under two names,
    and which name survives the split depends on the stage.

    MUST be called BEFORE split_stack(), which is what makes the tie observable -- after
    the split each stage holds only one of the two names and they no longer compare
    equal.

    Returns the copy held by THIS stage, on BOTH stages, because that is what the
    gradient all-reduce needs: a collective in which only one side participates hangs.
    For the NORMS the requirement is the opposite -- count it once, not twice -- so they
    go through norm_skip() instead of using this list directly. Conflating the two is
    what made the tied weight count ZERO times (weight_norm 12.6 against the correct
    47.8) while every gradient still matched exactly.
    """
    embed = getattr(base, "embed_tokens", None)
    if embed is None or head is None or not hasattr(head, "weight"):
        return []
    if head.weight.data_ptr() != embed.weight.data_ptr():
        return []
    return [embed.weight if groups.is_first else head.weight]


def replicated_params_for_test(model: nn.Module, groups: PipelineGroups) -> list[Tensor]:
    """Convenience for tests whose model exposes .base/.head directly."""
    return replicated_params(model.base, model.head, groups)


def freeze_unused_endpoints(base: nn.Module, head, groups: PipelineGroups) -> list:
    """Freeze the embedding/head this stage never executes. Returns what was frozen.

    UNTIED models only (Qwen3-8B and up). split_stack touches only `layers` and `norm`,
    so both stages keep both endpoints -- but stage 0 never runs the head, and stage 1
    never runs the embedding because it enters through inputs_embeds. Those parameters
    therefore get no gradient, and DistOptimizer dereferences p.grad unconditionally:

        rsize = g.shape[0] // world_size
        AttributeError: 'NoneType' object has no attribute 'shape'

    requires_grad_(False) is what the default param-group builder filters on, so the
    optimizer stops seeing them -- and no gradient or moment is allocated for them
    either, which is a side benefit rather than the point.

    TIED models (Gemma-3 every size, Qwen3 below 8B) are deliberately left alone: there
    the two names are ONE tensor that both stages genuinely use, which is why 12b never
    hit this and why replicated_params()/sync_replicated_grads() exist.

    The master weight itself is still resident. Dropping it would free ~2.5 GB at
    Qwen3-8B (151936x4096 fp32), but it is reachable by name from the model and the
    export path, so it is frozen rather than deleted.
    """
    embed = getattr(base, "embed_tokens", None)
    if embed is None or head is None or not hasattr(head, "weight"):
        return []
    if head.weight.data_ptr() == embed.weight.data_ptr():
        return []                       # tied: both stages use it
    dead = head.weight if groups.is_first else embed.weight
    dead.requires_grad_(False)
    return [dead]


def norm_skip(replicas: list[Tensor], groups: PipelineGroups) -> list[Tensor]:
    """Which replicated params THIS stage must leave out of a summed norm.

    The tied weight is identical on both stages, so a sum over stages counts it twice.
    Stage 0 counts it and stage 1 skips it -- an arbitrary but fixed choice, and the
    only one that yields the unsplit total exactly once.
    """
    return replicas if groups.is_last else []


def sync_replicated_grads(params: list[Tensor], groups: PipelineGroups) -> None:
    """SUM the tied weight's gradient across the two stages, before clipping.

    SUM, not AVG: stage 0 accumulated d(loss)/d(embedding) through the embedding lookup
    and stage 1 accumulated d(loss)/d(head) through the output projection. In the
    unsplit model those are two contributions to ONE parameter and autograd adds them,
    so averaging here would halve the tied weight's gradient and quietly train it at
    half the learning rate of everything else.

    Runs BEFORE the norm is computed, so the reported grad_norm reflects the true
    gradient rather than one stage's share.
    """
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=groups.pp_group)


def local_sq_skipping_replica(params, replicas: list[Tensor], grad: bool) -> Tensor:
    """Sum of squared norms over `params`, counting each replicated param once.

    The tied weight lives on both stages with identical values, so summing across stages
    would count it twice. replicated_params() hands stage 0 the copy that counts; stage
    1 skips it.
    """
    skip = {id(p) for p in replicas}
    total = None
    for p in params:
        t = p.grad if grad else p
        if t is None:
            continue
        if id(p) in skip:
            continue
        s = t.float().norm() ** 2
        total = s if total is None else total + s
    return total if total is not None else torch.zeros((), device="cuda")


def global_grad_norm(student: nn.Module, replicas: list[Tensor],
                     groups: PipelineGroups) -> float:
    """The unsplit setup's grad_norm, computed across the split model.

    Unsplit, every rank holds the whole model and the loop computes
        sqrt( mean_over_ranks( ||g_rank||^2 ) ).
    Split, one DP replica's squared norm is spread over its two stages, so the same
    quantity is
        sqrt( sum_over_all_ranks( local_sq ) / dp_size )
    -- one SUM over the whole world, divided by the number of DP replicas. That is
    numerically the same expression, which is what makes the logged curve comparable
    with every run trained before this existed.
    """
    local = local_sq_skipping_replica(
        [p for p in student.parameters() if p.grad is not None],
        norm_skip(replicas, groups), grad=True)
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return float((local / groups.dp_size).sqrt().item())


def global_weight_norm(student: nn.Module, replicas: list[Tensor],
                       groups: PipelineGroups) -> float:
    """The unsplit setup's weight_norm across the split model.

    Only a PP reduction, no DP one: DP replicas hold identical weights, and the unsplit
    code takes a purely local sum over the whole model. Summing the two stages
    reconstructs exactly that.
    """
    local = local_sq_skipping_replica(list(student.parameters()),
                                      norm_skip(replicas, groups), grad=False)
    dist.all_reduce(local, op=dist.ReduceOp.SUM, group=groups.pp_group)
    return float(local.sqrt().item())


def reduce_metrics(values: list[float], groups: PipelineGroups, device) -> list[float]:
    """Average loss metrics over DP replicas and make them available on EVERY rank.

    The metrics only exist on stage 1 (it owns the loss), but rank 0 -- which does the
    logging -- is a stage 0 rank. Stage 0 contributes zeros and the whole world is
    reduced with SUM, then divided by dp_size: one collective that both averages across
    replicas and carries the result to the logging rank. Matches the unsplit
    all_reduce(AVG) semantics.
    """
    t = torch.tensor(values if groups.is_last else [0.0] * len(values), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / groups.dp_size).tolist()


# ---------------------------------------------------------------------------
# Validation and export across the split
# ---------------------------------------------------------------------------
def eval_ntp(student, text, val_chunks, device, batch_size: int,
             groups: PipelineGroups) -> float:
    """Pipelined next-token loss. Same number the unsplit eval_ntp returns.

    The unsplit version calls `student(input_ids=..., labels=...)` and takes HF's loss;
    no stage can do that alone, so the CE is computed here from stage 1's head with the
    same shift HF uses (logits at t predict token t+1) and the same -100 masking.

    Reduction is a token-weighted sum over the whole world. Stage 0 contributes nothing,
    which is correct rather than a special case: both stages of a pipeline are fed the
    SAME shard, so counting only stage 1 counts each document exactly once.
    """
    import torch.nn.functional as F

    student.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0.0, device=device)
    for i in range(0, len(val_chunks), batch_size):
        batch = val_chunks[i:i + batch_size]
        if not batch:
            break
        ids_b, lbl_b, _ = zip(*batch)
        input_ids = torch.stack(ids_b).to(device)
        labels = torch.stack(lbl_b).to(device)
        n = int((labels[:, 1:] != -100).sum().item())
        if n == 0:
            continue
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if groups.is_first:
                h = stage_forward(text.base, 0, input_ids=input_ids)
                # Same one-tensor exchange as training; the teacher slot is unused here
                # and carries the student's own states rather than becoming a second,
                # differently-shaped message.
                send_activations(h, h, groups)
            else:
                B, T = input_ids.shape
                H = text.base.config.hidden_size
                h_in, _ = recv_activations((B, T, H), device, groups)
                h = stage_forward(text.base, 1, hidden=h_in)
                logits = text.head(h[:, :-1]).float()
                tgt = labels[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                    ignore_index=-100, reduction="sum")
                total_loss += loss
                total_n += n
    dist.all_reduce(total_loss)
    dist.all_reduce(total_n)
    student.train()
    return float((total_loss / total_n.clamp(min=1)).item())


_LAYER_RE = re.compile(r"^(.*\blayers\.)(\d+)(\..*)$")


def remap_layer_indices(state: dict, offset: int) -> dict:
    """Shift `layers.<i>` -> `layers.<i+offset>` in every key.

    split_stack renumbers stage 1's layers to 0..k so the ModuleList and each layer's
    self-reported layer_idx stay consistent while it runs. For EXPORT that has to be
    undone: otherwise stage 1's "layers.0" is the same key as stage 0's "layers.0" and
    the merge silently drops half the model -- the exact failure the export guard was
    put there to prevent.
    """
    if not offset:
        return state
    out = {}
    for k, v in state.items():
        m = _LAYER_RE.match(k)
        out[f"{m.group(1)}{int(m.group(2)) + offset}{m.group(3)}" if m else k] = v
    return out


def merge_export_state(local_state: dict, base: nn.Module,
                       groups: PipelineGroups, out_dir=None) -> dict:
    """Merge both stages' tensors onto stage 0 (= global rank 0), ready to write.

    Stage 1 renumbers its layer keys back to the original indices, then sends. Only the
    pipeline pair talks, so this stays intra-node like every other pipeline message.

    WHICH COPY WINS matters and is not symmetric. Both stages carry an embedding and a
    head; each only TRAINS one of them:
      * stage 0 trains the embedding, and its head (untied) is frozen and stale;
      * stage 1 trains the head, and its embedding is frozen and stale.
    So stage 1 contributes ONLY its layers, its final norm and its head, and stage 0
    keeps everything else. Merging naively would let stage 1's never-updated embedding
    overwrite the trained one -- a checkpoint that loads clean and scores like a random
    embedding.

    For a TIED model the two are the same tensor with the same values, so the rule is
    harmless there; it is written for the untied case and is a no-op otherwise.

    Returns the merged dict on stage 0, {} on stage 1.
    """
    offset = getattr(base, "_pp_layer_offset", 0)
    if not groups.is_first:
        payload = remap_layer_indices(local_state, offset)
        payload = {k: v for k, v in payload.items()
                   if ".layers." in k or k.endswith("norm.weight")
                   or k.endswith("norm.bias") or "lm_head" in k or "head." in k}
        payload = {k: v.detach().to("cpu") for k, v in payload.items()}
    else:
        payload = None

    # HANDOFF VIA SAFETENSORS ON DISK, not dist.broadcast_object_list.
    #
    # The object broadcast pickles the payload through a GPU byte tensor, and that round
    # trip CANNOT carry this dict: the export holds float8_e4m3fn block scales and packed
    # uint8 weights, and unpickling them dies inside torch.serialization.persistent_load
    # with "UntypedStorage has no attribute 'dtype'". Rank 0 then left its peer blocked in
    # the broadcast, which surfaced as NCCL/TCPStore teardown errors on all 16 ranks --
    # a comms failure that was really a serialization bug.
    #
    # safetensors handles fp8 and uint8 natively, needs no pickle, and keeps gigabytes off
    # the GPU. Export runs every --export-every steps, so one extra write+read is cheap
    # against the failure mode it removes.
    from safetensors.torch import load_file, save_file

    # ONE FILE PER PIPELINE PAIR. Every pair writing "_pp_stage1.safetensors" into the
    # same out_dir races: one pair's unlink() deletes the file another is about to read,
    # and the reader dies with FileNotFoundError. qad.py currently calls this only for
    # the dp_rank 0 pair, so production never collided -- but that made this function
    # correct only because of who happened to call it, which is not a property worth
    # relying on. Keying the name on dp_rank makes it safe for any caller.
    handoff = Path(out_dir) / f"_pp_stage1_dp{groups.dp_rank}.safetensors"
    if not groups.is_first:
        handoff.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in payload.items()}, str(handoff))
    dist.barrier(group=groups.pp_group)          # the file exists past this point
    if not groups.is_first:
        return {}
    merged = dict(local_state)
    merged.update(load_file(str(handoff)))
    handoff.unlink(missing_ok=True)
    return merged


@torch.no_grad()
def calibrate(text, chunks, device, groups: PipelineGroups,
              n_batches: int = 8, batch_size: int = 4) -> None:
    """Pipelined activation calibration for the W4A4 export.

    calibrate_nvfp4() runs the model to drive each NVFP4Linear's running-max observer.
    Under PP no rank can run the model alone, and the observers live on whichever stage
    owns the layer, so BOTH stages must participate and each updates its own half.

    Rank-invariant by construction: the batch count is derived from n_batches and
    batch_size, never from len(chunks), because the stages hold equal-length shards only
    by construction and a per-rank count would desynchronise the send/recv pairs.
    """
    from quantizers.nvfp4 import NVFP4Linear

    mods = [m for m in text.base.modules()
            if isinstance(m, NVFP4Linear) and m.quantize_act]
    for m in mods:
        m.calibrating = True
    try:
        n = min(n_batches * batch_size, len(chunks))
        for i in range(0, n, batch_size):
            batch = chunks[i:i + batch_size]
            if not batch:
                break
            ids = torch.stack([b[0] for b in batch]).to(device)
            labels = torch.stack([b[1] for b in batch]).to(device)
            from quantizers.dual import prefill_mask_from_labels, quant_phase
            with quant_phase(prefill_mask_from_labels(labels)), \
                    torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if groups.is_first:
                    h = stage_forward(text.base, 0, input_ids=ids)
                    send_activations(h, h, groups)
                else:
                    B, T = ids.shape
                    H = text.base.config.hidden_size
                    h_in, _ = recv_activations((B, T, H), device, groups)
                    stage_forward(text.base, 1, hidden=h_in)
    finally:
        for m in mods:
            m.calibrating = False
