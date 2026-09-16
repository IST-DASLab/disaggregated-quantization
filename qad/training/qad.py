"""
Quantization-aware distillation (QAD) via KL(student‖teacher).

Student linear weights and activations are fake-quantized to FP8 E4M3 using
per-tensor absmax scaling + straight-through estimator (STE). The teacher model
is identical but frozen in bf16. KL(student‖teacher) is minimized using
LigerFusedLinearJSDLoss (beta=0 collapses JSD to KL(teacher‖student)) which fuses both
lm_head projections with the KL computation for memory efficiency.
Gradient sync uses nanochat DistAdamW (ZeRO-2 style) — no DDP wrapper needed.

Launch:
    torchrun --nproc_per_node=8 qad.py --model <hf-model-id> [options]

Notes:
    - The text stack is addressed through training/models.py (`load_model` +
      `text_stack`), never through a hardcoded `.model` / `.lm_head`. For a plain
      CausalLM (Qwen3, gemma-3-270m/1b) those resolve to exactly `.model` and
      `.lm_head`; for the multimodal wrappers (gemma-3-4b/12b are
      Gemma3ForConditionalGeneration) `.model` is the vision+text CONTAINER, so
      running it would take the multimodal path and quantizing it would swap the
      SigLIP tower's linears. Everything model-shaped here goes through the stack.
    - DistAdamW requires shape[0] of large params (numel >= 1024) to be divisible
      by world_size. Holds for all standard Llama-family vocab/hidden sizes.
    - Quantization: fp8.py — fake_fp8 (STE), FP8Linear, apply_fp8_linear.
    - Optimizer:    dist_optim.py — DistOptimizer (ZeRO-2, per-group AdamW or Lion).
"""

import argparse
import hashlib
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

_QAD = Path(__file__).resolve().parent.parent   # training/ -> qad
sys.path.insert(0, str(_QAD))   # quantizers/, export/, training/ are rooted here
sys.path.insert(
    0,
    str(_QAD.parent / "third_party" / "Liger-Kernel" / "src"),
)
from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

from export.save import build_state_dict, export_variants, save_checkpoint
from quantizers import (REGISTRY, build_quantizer_params, calibrate_nvfp4,
                        load_frozen_decode, prefill_mask_from_labels, quant_phase,
                        uses_compressed_tensors)
from training.checkpoint import (find_latest, load_training_state,
                                 save_training_state)
from training.data import build_chunks, get_reasoning_train_val, get_tulu_train_val
from training.models import _PLAIN, _TEXT_PATHS, load_model, text_stack
from quantizers.full_disag import (apply_full_disag, dual_lm_head_weights,
                                   full_disag_hash)
from training.dist_optim import DistOptimizer
from training import pipeline as pp_mod


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, total_steps: int, lr_max: float, warmup_steps: int,
          schedule: str = "cosine") -> float:
    """Linear warmup, then either a cosine decay to 0 or a constant plateau.

    `constant` (warmup then flat, no cooldown) keeps every checkpoint along the run
    directly comparable, since none of them sit at a different point of a decay.
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    if schedule == "constant":
        return lr_max
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
# Resumable training state (FP32 weights + un-sharded optimizer moments + RNG) is
# handled by training/checkpoint.py — see save_training_state / load_training_state.


def save_weights(student: nn.Module, step: int, args: argparse.Namespace,
                 val_chunks=None, device=None) -> None:
    """Save an eval-ready checkpoint for the current quantizer.

    Two export paths, selected by the registry's "export" field:
      * compressed_tensors – a real quantized checkpoint (packed FP4 weights, FP8
        block scales, and for W4A4 a static input_global_scale calibrated here on a
        few val batches) that vLLM serves with true 4-bit kernels.
      * dequantized        – pseudo-quantization: a plain HF bf16 model whose
        weights carry the quantization error, served as an ordinary model.
    Either way evaluation is a single fast from_pretrained().
    """

    base_dir = Path(args.ckpt_dir) / args.ckpt_tag / "weights" / f"step_{step:07d}"

    # Calibrate the static activation scale before rank 0 writes (W4A4 only; a no-op
    # for weight-only formats, which quantize no activations). Done ONCE, outside the
    # variant loop: the observer is a property of the trained model, not of whichever
    # view is about to be serialized.
    groups = getattr(args, "_pp_groups", None)
    # Under PP only ONE pipeline pair does the export: each pp_group's collectives are
    # independent, so the (dp_rank 0) pair can calibrate and merge without the other
    # replicas taking part. They hold identical weights, so nothing is lost -- and it
    # avoids every rank assembling a full CPU state dict (~24 GB each at 12b). This also
    # mirrors the non-PP path, where rank 0 alone calibrates on its own val shard.
    if groups is not None and groups.dp_rank != 0:
        dist.barrier()
        return
    if uses_compressed_tensors(args.quantizer) and val_chunks is not None:
        if groups is not None:
            # BOTH stages, because neither can run the model alone and each owns the
            # observers for its own layers.
            pp_mod.calibrate(text_stack(student), val_chunks, device, groups)
        elif dist.get_rank() == 0:
            calibrate_nvfp4(student, val_chunks, device)

    rank0 = dist.get_rank() == 0
    if groups is not None or rank0:
        # The layers own their format, so there is one code path here regardless of
        # whether it ends up packed FP4 or a plain bf16 HF model. Most formats emit a
        # single checkpoint straight to base_dir; the prefill/decode formats emit one per
        # phase under base_dir/<variant>/, each standalone so a disaggregated deployment
        # can load them on different workers.
        for v in export_variants(student):
            out_dir = base_dir / v if v else base_dir
            if groups is not None:
                # Assemble on BOTH stages, merge onto rank 0, write there. The merge
                # renumbers stage 1's layers back to their original indices -- without
                # that they collide with stage 0's and half the model vanishes.
                local = build_state_dict(student, v)
                merged = pp_mod.merge_export_state(
                    local, text_stack(student).base, groups, out_dir=out_dir)
                n = save_checkpoint(student, out_dir, variant=v, step=step,
                                    state=merged, write=rank0)
            else:
                n = save_checkpoint(student, out_dir, variant=v, step=step)
            if rank0:
                print(f"[rank0] checkpoint → {out_dir}  ({n} tensors)", flush=True)
    # Rank 0 alone does the calibration + export (seconds to minutes on a busy
    # filesystem). Without this barrier the other ranks run ahead and queue collectives
    # that rank 0 cannot service, and once the lag exceeds the NCCL watchdog the whole
    # job dies with a reduce_scatter timeout.
    dist.barrier()


# ---------------------------------------------------------------------------
# Distillation loss with a per-phase LM head (--full-disag)
# ---------------------------------------------------------------------------
def _split_head_loss(kl_loss_fn, s_hidden, w_pair, t_hidden, t_lm_w, labels, head_prefill):
    """Distillation loss when the LM head is dual (--full-disag).

    Which head applies at shifted position t is decided by t's OWN phase, while whether t
    is scored is decided by the label at t+1. Under SFT those two nearly coincide: the
    scored positions are the assistant tokens (decode) plus, per sequence, exactly ONE
    boundary token -- the last prompt position, which is prefill but predicts the first
    assistant token. Every other prefill position is unscored, so the prefill head is
    almost absent from the loss and the fused kernel only has to cover the decode bulk.

    The boundary slice is a handful of tokens (one per sequence), so it goes through the
    same fused call rather than a separate unfused path: materializing B x V logits would
    be affordable, but reusing the kernel keeps the arithmetic identical to the bulk,
    which matters more than saving a kernel launch on B tokens.

    Liger divides by the number of non-ignored tokens it is given. Both calls are handed
    pre-filtered labels, so each returns a mean over its own slice and the two are
    recombined by count -- otherwise the boundary token would carry the same weight as the
    entire decode half.
    """
    w_p, w_d = w_pair
    valid = labels != -100
    sel_d = valid & ~head_prefill
    sel_p = valid & head_prefill
    n_d, n_p = int(sel_d.sum()), int(sel_p.sum())

    zero = torch.zeros((), device=s_hidden.device, dtype=torch.float32)
    parts = []
    for sel, n, w in ((sel_d, n_d, w_d), (sel_p, n_p, w_p)):
        if n == 0:
            parts.append((zero, zero, zero, 0))
            continue
        l, kl, ntp = kl_loss_fn(s_hidden[sel].contiguous(), w,
                                t_hidden[sel].contiguous(), t_lm_w,
                                true_labels=labels[sel])
        parts.append((l, kl, ntp, n))

    total = n_d + n_p
    if total == 0:
        return zero, zero, zero
    return tuple(sum(part[i] * part[3] for part in parts) / total for i in range(3))


# ---------------------------------------------------------------------------
# Validation — per-token NTP cross-entropy on the student
# ---------------------------------------------------------------------------
@torch.no_grad()
def _ntp_loss(model: nn.Module, val_chunks, device, batch_size) -> float:
    """Per-token NTP CE loss for model over val_chunks."""
    total_loss = torch.tensor(0.0, device=device)
    total_n    = torch.tensor(0,   device=device)
    for i in range(0, len(val_chunks), batch_size):
        batch = val_chunks[i : i + batch_size]
        if not batch:
            break
        ids_b, lbl_b, _ = zip(*batch)
        input_ids   = torch.stack(ids_b).to(device)
        labels_data = torch.stack(lbl_b).to(device)
        n = (labels_data[:, 1:] != -100).sum().item()
        if n == 0:
            continue
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels_data)
        total_loss += out.loss * n
        total_n    += n
    dist.all_reduce(total_loss)
    dist.all_reduce(total_n)
    return (total_loss / total_n.clamp(min=1)).item()


@torch.no_grad()
def eval_ntp(
    student: nn.Module,
    val_chunks: list[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    batch_size: int,
) -> float:
    student.eval()
    val_steps = range(0, len(val_chunks), batch_size)
    total_loss = torch.tensor(0.0, device=device)
    total_n    = torch.tensor(0,   device=device)
    for i in tqdm(val_steps, desc="val", unit="batch", disable=dist.get_rank() != 0):
        batch = val_chunks[i : i + batch_size]
        if not batch:
            break
        ids_b, lbl_b, _ = zip(*batch)
        input_ids   = torch.stack(ids_b).to(device)
        labels_data = torch.stack(lbl_b).to(device)
        n = (labels_data[:, 1:] != -100).sum().item()
        if n == 0:
            continue
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = student(input_ids=input_ids, labels=labels_data)
        total_loss += out.loss * n
        total_n    += n
    dist.all_reduce(total_loss)
    dist.all_reduce(total_n)
    student.train()
    return (total_loss / total_n.clamp(min=1)).item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="QAD: quantization-aware distillation")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=100_000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dataset", choices=("tulu", "reasoning"), default="tulu",
                        help="tulu = allenai/tulu-3-sft-mixture (no reasoning traces; the "
                             "chat template then supervises an EMPTY <think> block). "
                             "reasoning = faunix/Qwen3.8-27B-Distillation-40K, traced by "
                             "the teacher itself. Its median example is 2322 tokens and "
                             "half of that is the think block, so it needs a much larger "
                             "--max-seq-len than tulu: rows longer than --max-seq-len are "
                             "DROPPED, not truncated, because the answer follows the think "
                             "block and a truncated row would train reasoning that never "
                             "concludes.")
    parser.add_argument("--recompute-wq", action="store_true",
                        help="recompute the hard-quantized weight every forward instead of "
                             "caching it. EXACT (same function of the same master), "
                             "trades throughput for memory: frees the unsharded _wq/"
                             "_wq_dec buffers, 19.8 GiB per stage at 12b split.")
    parser.add_argument("--pp-cut", type=int, default=0,
                        help="layers kept by stage 0 (0 = midpoint). The midpoint "
                             "balances layer COUNT, not memory: stage 1 also holds the "
                             "fused loss over the whole vocabulary, and at 12b split it "
                             "was stage 1 that OOMed while stage 0 had room. Raising "
                             "this moves layers to stage 0; it repartitions only, and "
                             "changes no arithmetic.")
    parser.add_argument("--pp", type=int, default=1, choices=[1, 2],
                        help="pipeline stages. 2 splits the decoder layers (student AND "
                             "teacher) in half across adjacent, same-node rank pairs: "
                             "stage 0 holds the embeddings, stage 1 the loss. No "
                             "overlap -- one activation send forward, one gradient send "
                             "back, per micro-batch.")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="sequences per GPU per forward pass")
    parser.add_argument("--global-batch-size", type=int, default=64, help="total sequences per optimizer step across all GPUs")
    parser.add_argument("--eval-batch-size",  type=int, default=1, help="sequences per GPU during validation")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scale-ratio", type=float, default=0.5,
                        help="scale LR multiplier for quantizers that use separate scale LR")
    parser.add_argument("--quantizer", type=str, default="nvfp4",
                        choices=list(REGISTRY),
                        help="quantization scheme to apply to the student")
    parser.add_argument("--quantizer-params", type=str, default="",
                        help="JSON string of quantizer hyperparameter overrides, "
                             "e.g. '{\"groupsize\": 64}'")
    parser.add_argument("--full-disag", action="store_true",
                        help="also make the embedding, every RMSNorm and the LM head "
                             "dual (one tensor per phase), so the prefill and decode "
                             "checkpoints share nothing but the architecture. Tying is "
                             "preserved per phase. COSTS a second copy of the embedding "
                             "table plus its gradient and optimizer state -- at 4B that "
                             "is ~389M extra parameters; check the memory budget first.")
    parser.add_argument("--include-prefill-loss", action="store_true",
                        help="compute KL loss on all token positions (paper behaviour); "
                             "default is assistant-reply tokens only")
    parser.add_argument("--run-name", type=str, default="qad")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--chunk-cache-dir", type=str, default=None, help="directory to cache tokenized chunks")
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["cosine", "constant"],
                        help="constant = linear warmup then a flat plateau (no cooldown)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="steps between resumable training-state saves. States are "
                             "pruned to --keep-last (default 1), so this costs write "
                             "bandwidth, not quota. 250 was chosen when a chain link ran "
                             "far past one save; at 40 s/step a 4 h link covers ~360 "
                             "steps, so a restart threw away up to 249 of them (~2.8 h). "
                             "Measured on the 27B: a state save is ~150 s (295 GiB), so "
                             "50 costs ~7% of wall-clock and caps the loss at ~33 min.")
    parser.add_argument("--keep-last", type=int, default=1,
                        help="how many training states to retain (they are ~12 bytes/param)")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "never"],
                        help="auto: continue from the newest training state for this tag")
    # Validation stays FREQUENT: it is the loss curve, and it is cheap. It used to double
    # as the tail export cadence, which is why it looks like it should be coarse -- it
    # should not. The export cadences below are separate knobs precisely so this one can
    # stay at 25.
    parser.add_argument("--val-every", type=int, default=25)
    # BOTH export cadences must be MULTIPLES of --val-every: the export is nested in the
    # validation branch, so a non-multiple would silently never fire. 250 matches the
    # step grid the eval sweep uses; the final step is always exported regardless.
    #
    # Keep --export-tail-every in step with prune_checkpoints.py --tail-grid: one decides
    # what is WRITTEN past the threshold, the other what is KEPT. They were mismatched
    # once -- retention was generous past 2000 while the writer had already thinned the
    # tail to the 250 grid, so the rule protected nothing on newly-trained runs.
    parser.add_argument("--export-dense-after", type=int, default=0,
                        help="past this step, export at --export-tail-every instead of "
                             "--export-every. DISABLED (0) by default: a denser tail "
                             "wrote checkpoints at steps no figure plots and no eval "
                             "grid asks for (submit_missing_evals GRID is 0,250,...,2250), "
                             "and at ~27 GB per 12b export that is pure quota. Every "
                             "export is now on the single --export-every cadence.")
    parser.add_argument("--export-tail-every", type=int, default=250,
                        help="tail export cadence past --export-dense-after. Previously "
                             "this was the --val-every cadence (25), which wrote ~19 "
                             "checkpoints per run that nothing plots -- about two thirds "
                             "of a run's exported bytes (~340 GB of a 4B run's 502 GB). "
                             "125 still lands on 2250, the last plotted step.")
    parser.add_argument("--export-every", type=int, default=250,
                        help="steps between exported HF checkpoints (weights/step_N). "
                             "Separate from --val-every: validation is cheap, a full "
                             "checkpoint is ~15 GB for a 4B dual format.")
    args = parser.parse_args()
    for _name, _v in (("--export-every", args.export_every),
                      ("--export-tail-every", args.export_tail_every)):
        if _v % args.val_every:
            parser.error(f"{_name} ({_v}) must be a multiple of --val-every "
                         f"({args.val_every}); it is checked inside the validation "
                         f"branch and would otherwise never fire.")

    # BEFORE any quantizer is constructed: __init__ decides whether to allocate the
    # caches, so this cannot be flipped later without leaving a model half in each mode.
    from quantizers.base import set_recompute_wq
    set_recompute_wq(args.recompute_wq)

    quant_params, quant_hash = build_quantizer_params(args.quantizer, args.quantizer_params)
    if args.full_disag:
        # --full-disag changes the model, so it MUST change the checkpoint tag. Without
        # this a full-disag run and a plain run of the same quantizer share a checkpoint
        # directory, and `--resume auto` would happily continue one from the other's
        # state -- a corrupt run that looks entirely normal. Folded into the hash rather
        # than added as a tag segment so the <run>-<quant>-<hash> shape every other tool
        # parses stays intact.
        quant_hash = full_disag_hash(quant_hash)
    quant_entry = REGISTRY[args.quantizer]
    # Embed quantizer name + param hash into identifiers for traceability
    run_tag = f"{args.run_name}-{args.quantizer}"
    ckpt_tag = f"{run_tag}-{quant_hash}"
    args.run_tag  = run_tag   # used for wandb run name
    args.ckpt_tag = ckpt_tag  # used for checkpoint path
    # The wandb name carries the format, not the hash: a --full-disag run would otherwise
    # be indistinguishable from its plain counterpart in the UI, since run_tag is only
    # (run, model, quantizer). The checkpoint path stays keyed on the hash.
    wandb_name = run_tag + ("-fulldisag" if args.full_disag else "") \


    # 10-minute default is too tight: rank 0 writes multi-GB checkpoints while the
    # other ranks wait at a barrier, and lustre is slow when many jobs write at once.
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # PIPELINE GROUPS FIRST, because the DATA SHARDING depends on them: both stages of a
    # pipeline must be fed the SAME documents, or stage 1 would score its own tokens
    # against activations stage 0 computed for different ones. Sharding by global rank --
    # correct without a pipeline -- does exactly that. The MODEL is split much later
    # (after the teacher val baseline, which needs the whole teacher).
    groups = None
    replicas: list = []
    if args.pp > 1:
        pp_mod.assert_intra_node()
        groups = pp_mod.build_groups()          # COLLECTIVE, every rank
    # Data-parallel width: what the batch maths and the data shards are keyed on.
    dp_rank = groups.dp_rank if groups is not None else rank
    dp_size = groups.dp_size if groups is not None else world_size

    if rank == 0:
        # id + resume="allow": a restarted job continues logging into the same run
        # instead of creating a duplicate.
        wandb.init(
            project="prefill-decode-distill",
            name=wandb_name,
            # id=hashlib.md5(ckpt_tag.encode()).hexdigest(),
            resume="allow",
            config={**vars(args), "quantizer_params": quant_params, "quantizer_hash": quant_hash},
        )

    # Teacher — frozen bf16 reference; no gradients anywhere.
    # NOTE the teacher is loaded further down, AFTER the student has been quantized.
    # See the comment there: it is 21.9 GiB at gemma-3-12b and is not needed until the
    # val baseline, so holding it across quantizer construction wastes exactly that much
    # at the peak.

    # Student — same checkpoint, quantized linears, gradient checkpointing.
    student = load_model(args.model, torch.float32,
                         attn_implementation="sdpa").to(device)
    student_text = text_stack(student)
    # The quantizer is scoped to the TEXT stack, never to the whole model. On a plain
    # CausalLM that is the same set of linears as before (the head lives outside
    # `.model`, and replace_linears skipped it by name anyway), so Qwen behaviour is
    # unchanged. On a multimodal wrapper it is the difference between quantizing the
    # 34*7 text projections and also quantizing the 163 linears of the SigLIP tower.
    quant_entry["apply"](student_text.base, **quant_params)
    # Anything outside the text stack (i.e. a vision tower and its projector) is loaded
    # but never runs and is never trained: freeze it so it stays out of the optimizer
    # below. DistOptimizer dereferences p.grad unconditionally, so a parameter that never
    # receives a gradient is not merely wasteful — it crashes the first step().
    # No-op on a plain CausalLM: base + head already cover every parameter.
    _text_params = {id(p) for p in student_text.base.parameters()}
    _text_params |= {id(p) for p in student_text.head.parameters()}
    n_frozen = sum(p.numel() for p in student.parameters() if id(p) not in _text_params)
    for p in student.parameters():
        if id(p) not in _text_params:
            p.requires_grad_(False)
    if n_frozen and rank == 0:
        print(f"frozen (outside the text stack): {n_frozen / 1e6:.1f}M params", flush=True)
    if args.quantizer == "nvfp4frozendec":
        # The LM head is SHARED and FROZEN for this format, and it lives outside
        # student_text.base, so the quantizer's own apply() cannot reach it. Sharing
        # rather than duplicating is the point: at 248320 x 5120, untied, a per-phase copy
        # would cost 1.27B parameters plus gradient and optimizer state to represent two
        # tensors that are both frozen and identical. The embedding is frozen inside
        # apply_nvfp4frozendec, which does have it in scope.
        _head_n = sum(p.numel() for p in student_text.head.parameters())
        for p in student_text.head.parameters():
            p.requires_grad_(False)
        if rank == 0:
            print(f"frozen (shared lm_head): {_head_n / 1e6:.1f}M params", flush=True)
    if args.full_disag:
        # --full-disag is DEPRECATED (docs/GEMMA3_PLAN.md 2.7) and was only ever
        # validated on Qwen3. Two separate things break it elsewhere: Gemma3RMSNorm is
        # `(1 + weight)` with a zeros init, which DualRMSNorm does not implement (2.1),
        # and apply_full_disag's discovery walks the WHOLE model, so on a multimodal
        # wrapper the first nn.Embedding it finds is the SigLIP position embedding (2.2).
        # Refuse rather than silently train something wrong.
        if str(getattr(student.config, "model_type", "")).startswith("gemma") \
                or hasattr(student.config, "vision_config"):
            raise SystemExit(
                "--full-disag is deprecated and is not supported for Gemma-3 or any "
                "multimodal wrapper; see docs/GEMMA3_PLAN.md 2.7")
        # After the quantizer, so the linears are already dual and this only has to deal
        # with what it skipped: the embedding, the norms and the head.
        n = apply_full_disag(student)
        student_text = text_stack(student)   # lm_head was swapped for a DualLMHead
        if rank == 0:
            print(f"full-disag: {n['embedding']} embedding, {n['norm']} norms, "
                  f"{n['lm_head']} lm_head -> dual"
                  f"{' (head tied to embedding, per phase)' if n['tied'] else ''}",
                  flush=True)
    student.config.use_cache = False
    student_text.base.config.use_cache = False   # see the teacher, loaded below
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()

    # NOTE: torch.compile is NOT disabled by the environment. That was true of an older
    # container; as of torch 2.10.0a0 / triton 3.6.0 here, compiled forward AND backward
    # both work (verified directly, including the sign x bucketize pattern in
    # e2m1_round). What defeated an attempt to compile the quantizers was GUARD CHURN,
    # not Triton: one entry point served fp32 weights and bf16 activations, no_grad
    # post_update and the grad-enabled forward, and view inputs whose x._base.size()
    # became a guard. Dynamo hit recompile_limit and fell back to eager permanently, so
    # the run paid compilation AND lost fusion. Compiling here means splitting the units
    # by caller first.

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        n_params = sum(p.numel() for p in student.parameters()) / 1e9
        print(f"Model: {args.model}  ({n_params:.1f}B params)  quantizer={args.quantizer}  hash={quant_hash}", flush=True)
        print("Loading and tokenizing dataset …", flush=True)

    # Rank 0 alone computes+caches the split first. datasets' cross-process cache
    # locking relies on flock, which Lustre does not reliably serialize across
    # separate NODES (only within one) -- with every rank racing this call at once,
    # multiple ranks write the same fingerprinted .arrow cache file concurrently and
    # one loses with FileNotFoundError mid-chmod. Once rank 0's write lands, every
    # other rank's call just reads the now-complete cache instead of computing it.
    def _load_raw():
        if args.dataset == "reasoning":
            return get_reasoning_train_val()
        return get_tulu_train_val()

    if rank == 0:
        train_raw, val_raw = _load_raw()
    dist.barrier()
    if rank != 0:
        train_raw, val_raw = _load_raw()
    cache_dir = Path(args.chunk_cache_dir) if args.chunk_cache_dir else None
    # The dataset goes in the `split` label because it is part of the chunk cache key
    # (model, split, tokens, seq, rank, world) -- without it two datasets tokenized at
    # the same --max-seq-len would share one cache path and silently serve each other's
    # chunks.
    tag = "" if args.dataset == "tulu" else f"{args.dataset}-"
    drop_overlong = args.dataset == "reasoning"
    train_chunks = build_chunks(
        tokenizer, train_raw, args.train_tokens, args.max_seq_len, dp_rank, dp_size,
        cache_dir=cache_dir, split=f"{tag}train", model_name=args.model,
        drop_overlong=drop_overlong,
    )
    val_chunks = build_chunks(
        tokenizer, val_raw, args.val_tokens, args.max_seq_len, dp_rank, dp_size,
        cache_dir=cache_dir, split=f"{tag}val", model_name=args.model,
        drop_overlong=drop_overlong,
    )
    if rank == 0:
        print(f"Train: {len(train_chunks)} docs/rank  ({len(train_chunks) * dp_size} total)", flush=True)
        print(f"Val:   {len(val_chunks)} docs/rank", flush=True)

    # TEACHER, loaded HERE rather than before the student. It is frozen, unused until
    # this line, and large -- 21.9 GiB at gemma-3-12b in bf16. Loading it first meant it
    # sat resident through the student load AND the whole quantizer construction, which is
    # where the setup peak is: teacher 21.9 + student fp32 43.8 + _wq 20.0 (+ _wq_dec 20.0
    # for upcast/split) on every rank. Deferring it takes that peak down by its full size.
    #
    # load_model loads the class the checkpoint DECLARES; see training/models.py for why
    # AutoModelForCausalLM is not safe here (it maps gemma3 -> Gemma3ForCausalLM, which
    # loads a 4b/12b repo into a half-random model without complaining).
    teacher = load_model(args.model, torch.bfloat16,
                         attn_implementation="sdpa").to(device)
    teacher_text = text_stack(teacher)
    teacher.config.use_cache = False
    # A wrapper's decoder stack carries its OWN config object (config.text_config), and
    # that is the one the forward we actually call reads. For a plain CausalLM it is the
    # same object, so this line is a no-op there.
    teacher_text.base.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Teacher val loss is a constant (frozen model) — compute once here.
    if rank == 0:
        print("Computing teacher val baseline …", flush=True)
    teacher_val_ntp = _ntp_loss(teacher, val_chunks, device, args.eval_batch_size)
    if rank == 0:
        print(f"Teacher val_ntp: {teacher_val_ntp:.4f}", flush=True)
        wandb.log({"val/teacher_ntp": teacher_val_ntp}, step=0)

    # PIPELINE SPLIT. Deliberately after the teacher val baseline above, which runs the
    # whole teacher and cannot be computed once the stack is halved.
    if groups is not None:
        # nvr2bit's post_update round-robins refresh_buffers() across ranks and then
        # BROADCASTS each layer's _wq with src=i % world_size. Under PP the two stages
        # hold DIFFERENT layers, so the loop length differs per stage and src names a
        # rank that does not own that layer: it deadlocks, or worse, overwrites one
        # stage's cache with the other's. Refuse up front rather than hang at step 1.
        if "nvr2bit" in args.quantizer:
            raise NotImplementedError(
                f"--pp is not supported for {args.quantizer}: post_update_nvr2bit "
                "broadcasts layer caches across the whole world with src=i % world, "
                "which assumes every rank holds every layer. Train it with --pp 1.")
        # The tied embedding/head must be identified BEFORE splitting: afterwards each
        # stage holds only one of the two names and the tie is no longer observable.
        replicas = pp_mod.replicated_params(student_text.base, student_text.head,
                                            groups)
        # SAME cut for student and teacher: stage 1 is fed the teacher's half-way
        # hidden state too, so a different boundary would pair mismatched depths.
        _cut = args.pp_cut or None
        pp_mod.split_stack(student_text.base, student_text.head, groups.pp_rank, _cut)
        pp_mod.split_stack(teacher_text.base, teacher_text.head, groups.pp_rank, _cut)
        if rank == 0:
            _n = len(student_text.base.layers)
            print(f"PP cut: stage {groups.pp_rank} holds {_n} layers", flush=True)
        # BEFORE the param groups are built: an untied model leaves each stage an
        # endpoint it never runs, and the optimizer would dereference its absent grad.
        _dead = pp_mod.freeze_unused_endpoints(student_text.base, student_text.head,
                                               groups)
        if rank == 0 and _dead:
            print(f"PP: froze {len(_dead)} unused endpoint tensor(s) (untied model)",
                  flush=True)
        # save_weights() is called from several places and only receives `args`; hang
        # the groups off it rather than threading a parameter through every call site.
        args._pp_groups = groups
        torch.cuda.empty_cache()                # hand back the half we just dropped
        if rank == 0:
            n = sum(p.numel() for p in student.parameters())
            print(f"PP={args.pp}: stage {groups.pp_rank} holds {n/1e9:.2f}B student "
                  f"params, dp_size={groups.dp_size}", flush=True)

    # Frozen external decode half. AFTER split_stack on purpose: each rank then fills only
    # the layers it actually holds, and load_frozen_decode undoes split_stack's layer
    # renumbering so stage 1 reads original layer `cut`, not layer 0. Before the first
    # forward, which is what the layer's own has_decode guard enforces.
    if args.quantizer == "nvfp4frozendec":
        _dm = quant_params.get("decode_model") or ""
        if not _dm:
            raise SystemExit(
                "nvfp4frozendec needs the external decode checkpoint: pass "
                '--quantizer-params \'{"decode_model": "/path/to/...-bf16"}\'')
        _base_path, _ = _TEXT_PATHS.get(type(student).__name__, _PLAIN)
        _n = load_frozen_decode(student_text.base, _dm, prefix=f"{_base_path}.")
        if rank == 0:
            print(f"frozen decode: filled {_n['linear']} linears and {_n['norm']} norms "
                  f"from {_dm}", flush=True)

    # DistAdamW handles gradient reduction — no DDP wrapper required.
    # Quantizers that need separate LR / weight_decay per parameter class
    # supply a param_groups builder; others fall back to a single group over all params.
    _pg_fn = quant_entry["param_groups"]
    if _pg_fn is not None:
        param_groups = _pg_fn(student, lr=args.lr, lr_scale_ratio=args.lr_scale_ratio)
    else:
        # requires_grad filter, matching what the quantizer builders already do: the only
        # frozen parameters are the ones outside the text stack (a vision tower), which
        # get no gradient and would crash DistOptimizer.step(). No-op on Qwen3, where
        # every parameter is trainable.
        param_groups = [{"params": [p for p in student.parameters() if p.requires_grad]}]

    optimizer = DistOptimizer(
            param_groups,
            process_group=(groups.dp_group if groups is not None else None),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    # beta=0 → JSD degenerates to KL(teacher‖student) = Σ p_teacher·log(p_teacher/p_student),
    # matching the paper's objective. beta=1 was KL(student‖teacher) — the wrong direction.
    # return_soft_hard_loss=True: also return the raw CE (NTP) loss for logging without
    # affecting gradients (weight_hard_loss=0 keeps it out of the backward).
    # compiled=False: the NeMo container's Triton is incompatible with inductor (cluster_dims).
    kl_loss_fn = LigerFusedLinearJSDLoss(
        weight_hard_loss=0.0,
        weight_soft_loss=1.0,
        beta=0.0,
        ignore_index=-100,
        compiled=False,
        chunk_size=256,
        return_soft_hard_loss=True,
    )
    # Teacher CE loss on training batches: pass teacher hidden as both "student" and "teacher"
    # with weight_hard=1, weight_soft=0 — only the chunked CE is computed, no KL.
    teacher_ce_fn = LigerFusedLinearJSDLoss(
        weight_hard_loss=1.0,
        weight_soft_loss=0.0,
        beta=0.0,
        ignore_index=-100,
        compiled=False,
        chunk_size=256,
    )

    # (w_prefill, w_decode) under --full-disag, else None. Resolved once: the modules do
    # not change during training, and the training loop only needs the tensors.
    head_pair = dual_lm_head_weights(student)


    mbs = args.micro_batch_size
    gbs = args.global_batch_size
    grad_accum = gbs // (mbs * dp_size)
    assert grad_accum >= 1, (
        f"global_batch_size={gbs} < micro_batch_size={mbs} * dp_size={dp_size}"
    )
    # chunks consumed per rank per optimizer step
    chunks_per_step = mbs * grad_accum
    # len(train_chunks) is LOCAL: each rank tokenizes its own dp_rank/dp_size shard
    # of the dataset, and shard sizes need not tokenize to identical chunk counts.
    # An unsynchronized total_steps lets ranks disagree on which step is "the last
    # one" -- the loop's `step == total_steps - 1` special-cases (final validation,
    # final export) then fire on DIFFERENT steps per rank, which permanently
    # desyncs the collective sequence (observed: one rank stuck issuing an
    # ALLGATHER_COALESCED that no other rank ever calls, everything after hangs).
    # MIN, not e.g. rank 0's value, so no rank ever runs past the data it has.
    total_steps_t = torch.tensor(len(train_chunks) // chunks_per_step, device=device)
    dist.all_reduce(total_steps_t, op=dist.ReduceOp.MIN)
    total_steps = int(total_steps_t.item())
    if rank == 0:
        print(
            f"Steps: {total_steps}  "
            f"(gbs={gbs}, mbs={mbs}, grad_accum={grad_accum}, world={world_size})",
            flush=True,
        )

    # Resume: restore weights + optimizer moments + RNG from the newest training
    # state for this tag. The data position needs no bookkeeping — chunks are
    # indexed as step*chunks_per_step + acc*mbs, so starting the loop at start_step
    # fast-forwards the dataset to exactly where it left off.
    start_step = 0
    if args.resume == "auto":
        latest = find_latest(args.ckpt_dir, args.ckpt_tag)
        if latest is not None:
            start_step = load_training_state(latest[0], student, optimizer, device,
                                             groups=groups)
            if start_step >= total_steps:
                if rank == 0:
                    print(f"Nothing to do: state at step {latest[1]} >= total {total_steps}",
                          flush=True)
                dist.barrier()
                return
        elif rank == 0:
            print("No training state found — starting from scratch.", flush=True)

    # Remember each group's configured LR. The schedule is applied below as a
    # MULTIPLIER on these, not as an absolute value: quantizers whose parameters do
    # not live in weight units set their own per-group LRs (GSQ's assignment logits
    # and block-scale deltas), and overwriting every group with the weight LR
    # silently discarded them — a --quantizer-params logit_lr sweep then trained
    # every arm at args.lr and produced three identical runs.
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])

    pbar = tqdm(range(start_step, total_steps), desc="train", unit="step",
                initial=start_step, total=total_steps, disable=rank != 0)
    for step in pbar:
        lr_frac = lr_at(step, total_steps, 1.0, args.warmup_steps, args.lr_schedule)
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lr_frac
        lr = args.lr * lr_frac          # weight LR, for logging

        optimizer.zero_grad()
        accum_kl          = 0.0
        accum_ntp         = 0.0
        accum_teacher_ntp = 0.0

        for acc in range(grad_accum):
            chunk_start = step * chunks_per_step + acc * mbs
            chunk_slice = train_chunks[chunk_start : chunk_start + mbs]
            ids_list, lbl_list, msk_list = zip(*chunk_slice)
            input_ids   = torch.stack(ids_list).to(device)  # [mbs, T]
            labels_data = torch.stack(lbl_list).to(device)  # [mbs, T]

            # Route every position to the format it will run under at inference:
            # prompt/system tokens through the prefill format, assistant tokens
            # through the decode format. A no-op for single-format quantizers.
            #
            # This context MUST also enclose backward() below — gradient
            # checkpointing recomputes the forward during the backward pass, and if
            # the mask were already cleared the recomputation would silently run
            # every position as decode, so the gradients would belong to a model that
            # was never evaluated.
            prefill_mask = prefill_mask_from_labels(labels_data)
            with quant_phase(prefill_mask):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if groups is not None and groups.is_first:
                        # STAGE 0: embeddings + first half, for both models, then ONE
                        # send. Nothing else happens on this rank until the boundary
                        # gradient comes back -- no overlap, by design.
                        with torch.no_grad():
                            t_h0 = pp_mod.stage_forward(teacher_text.base, 0,
                                                        input_ids=input_ids)
                        s_h0 = pp_mod.stage_forward(student_text.base, 0,
                                                    input_ids=input_ids)
                        pp_mod.send_activations(s_h0, t_h0, groups)
                        s_hidden = t_hidden = None
                    elif groups is not None:
                        # STAGE 1: receive, finish both stacks, own the loss.
                        B, T = input_ids.shape
                        H = student_text.base.config.hidden_size
                        s_in, t_in = pp_mod.recv_activations(
                            (B, T, H), device, groups,
                            teacher_dtype=teacher_text.base.embed_tokens.weight.dtype)
                        with torch.no_grad():
                            t_hidden = pp_mod.stage_forward(teacher_text.base, 1,
                                                            hidden=t_in)
                        s_hidden = pp_mod.stage_forward(student_text.base, 1,
                                                        hidden=s_in)
                    else:
                        # Teacher: hidden states only, no gradients. `.base` is the
                        # decoder stack itself, so this never enters a wrapper's
                        # multimodal forward.
                        with torch.no_grad():
                            t_hidden = teacher_text.base(input_ids=input_ids).last_hidden_state

                        # Student: base transformer; lm_head handled by Liger below
                        s_hidden = student_text.base(input_ids=input_ids).last_hidden_state

                # Causal shift: hidden state at t predicts token at t+1.
                # labels_data[t+1] is the target (assistant token or -100).
                if groups is not None and groups.is_first:
                    # The whole loss/metrics block below belongs to stage 1. Stage 0's
                    # backward is driven entirely by the returning boundary gradient.
                    pp_mod.backward_from_peer(s_h0, groups)
                    continue

                B, T, H = s_hidden.shape
                N = B * (T - 1)
                s_hidden_q = s_hidden[:, :-1].reshape(N, H).contiguous()
                t_hidden_f = t_hidden[:, :-1].reshape(N, H).contiguous()
                # With --include-prefill-loss: KL on all positions (paper behaviour).
                # Default: assistant reply tokens only (labels_data has -100 elsewhere).
                if args.include_prefill_loss:
                    labels = input_ids[:, 1:].reshape(N)
                else:
                    labels = labels_data[:, 1:].reshape(N)

                t_lm_w = teacher_text.head.weight

                if head_pair is None:
                    loss, kl_soft, ntp_hard = kl_loss_fn(
                        s_hidden_q, student_text.head.weight, t_hidden_f, t_lm_w,
                        true_labels=labels
                    )
                else:
                    # --full-disag: the head is dual, so the loss splits by which head
                    # each scored position uses. See _split_head_loss.
                    head_prefill = prefill_mask[:, :-1].reshape(N)
                    loss, kl_soft, ntp_hard = _split_head_loss(
                        kl_loss_fn, s_hidden_q, head_pair, t_hidden_f, t_lm_w,
                        labels, head_prefill
                    )
                (loss / grad_accum).backward()
                if groups is not None:
                    # ONE gradient tensor back to stage 0, immediately after the local
                    # backward that produced it.
                    pp_mod.send_input_grad(s_in, groups)

            # Teacher NTP on this training batch: CE(teacher_logits, labels), no grad.
            with torch.no_grad():
                t_ntp = teacher_ce_fn(t_hidden_f, t_lm_w, t_hidden_f, t_lm_w, true_labels=labels)
            accum_teacher_ntp += t_ntp.item() / grad_accum
            accum_kl  += kl_soft.item()  / grad_accum
            accum_ntp += ntp_hard.item() / grad_accum

        # Global gradient clipping before DistAdamW reduces across ranks.
        # Each rank computes its local squared norm; we all_reduce (avg) across ranks
        # to get an estimate of the global norm, then apply the same clip factor on
        # every rank so the reduced gradients stay within max_norm.
        # sqrt(avg(||g_i||²)) >= ||avg(g_i)|| by Jensen, so this never under-clips.
        if groups is not None:
            # SUM the tied embedding's gradient across stages FIRST: stage 0 accumulated
            # it through the embedding lookup and stage 1 through the head, and in the
            # unsplit model autograd adds those. Doing it before the norm means the
            # logged grad_norm and the clip both see the true gradient.
            pp_mod.sync_replicated_grads(replicas, groups)
            grad_norm = pp_mod.global_grad_norm(student, replicas, groups)
        else:
            local_sq = sum(
                p.grad.float().norm() ** 2
                for p in student.parameters() if p.grad is not None
            )
            dist.all_reduce(local_sq, op=dist.ReduceOp.AVG)
            grad_norm = local_sq.sqrt().item()
        clip_factor = args.grad_clip / max(grad_norm, args.grad_clip)
        if clip_factor < 1.0:
            for p in student.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_factor)

        optimizer.step()

        if quant_entry["post_update"] is not None:
            quant_entry["post_update"](student, step, total_steps)

        with torch.no_grad():
            if groups is not None:
                weight_norm = pp_mod.global_weight_norm(student, replicas, groups)
            else:
                weight_norm = torch.sqrt(
                    sum(p.float().norm() ** 2 for p in student.parameters())
                ).item()

        # PER-STAGE PEAK, once. Under PP the two stages hold different things -- stage 1
        # additionally carries the loss over the whole vocabulary -- so a single number
        # from rank 0 says nothing about which stage is the constraint. Every 12b OOM so
        # far was stage 1 while stage 0 had room, and without this the balance had to be
        # guessed. Printed at the first step only, from one rank of each stage.
        if step == 0 and groups is not None and groups.dp_rank == 0:
            print(f"[mem] stage {groups.pp_rank}: peak "
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB of 178.4, "
                  f"{len(student_text.base.layers)} layers", flush=True)

        # Reduce all three train metrics across ranks before logging
        if groups is not None:
            # Only stage 1 has the loss; stage 0 contributes zeros and the SUM/dp_size
            # both averages over replicas and carries the result to rank 0, which does
            # the logging and is a stage 0 rank.
            accum_kl, accum_ntp, teacher_ntp_step = pp_mod.reduce_metrics(
                [accum_kl, accum_ntp, accum_teacher_ntp], groups, device)
        else:
            metrics_t = torch.tensor(
                [accum_kl, accum_ntp, accum_teacher_ntp], device=device
            )
            dist.all_reduce(metrics_t, op=dist.ReduceOp.AVG)
            accum_kl, accum_ntp, teacher_ntp_step = metrics_t.tolist()
        train_ntp_delta = accum_ntp - teacher_ntp_step

        pbar.set_postfix(kl=f"{accum_kl:.4f}", ntp=f"{accum_ntp:.4f}", Δ=f"{train_ntp_delta:+.4f}", lr=f"{lr:.2e}")

        if rank == 0:
            wandb.log({
                "train/kl":           accum_kl,
                "train/ntp":          accum_ntp,
                "train/teacher_ntp":  teacher_ntp_step,
                "train/ntp_delta":    train_ntp_delta,
                "train/grad_norm":    grad_norm,
                "train/weight_norm":  weight_norm,
                "train/lr":           lr,
            }, step=step)

        if step % args.val_every == 0 or step == total_steps - 1:
            ntp = (pp_mod.eval_ntp(student, student_text, val_chunks, device,
                             args.eval_batch_size, groups)
                   if groups is not None else
                   eval_ntp(student, val_chunks, device, args.eval_batch_size))
            # Export on its OWN cadence. Validation is cheap and wants to be frequent
            # (it is the loss curve); exporting a full HF checkpoint is not. Tying the
            # two together wrote 99 checkpoints per run when the eval sweep only ever
            # reads 10 of them -- 90% of 20 TB was never evaluated by anything.
            #
            # But the TAIL is where the curves are actually read: the tail-averaged
            # recovery bars take the last few steps, and prune_checkpoints.py keeps the
            # tail at --tail-grid for the same reason. A flat 250 cadence left the tail
            # as sparse as the start, so past --export-dense-after this switches to the
            # finer --export-tail-every. NOT to the validation cadence: that was every
            # 25, i.e. ~19 tail checkpoints per run, none of which anything plots.
            dense = args.export_dense_after and step >= args.export_dense_after
            cadence = args.export_tail_every if dense else args.export_every
            # EVERY exported step is a multiple of the cadence (250). The old
            # `or step == total_steps - 1` also exported e.g. 2457, which no figure
            # plots and no eval grid asks for -- submit_missing_evals' GRID is
            # 0,250,...,2250 -- so it only consumed quota.
            if step % cadence == 0:
                save_weights(student, step, args, val_chunks, device)
            if rank == 0:
                delta = ntp - teacher_val_ntp
                pbar.write(f"step {step:5d} | val_ntp={ntp:.4f}  teacher={teacher_val_ntp:.4f}  Δ={delta:+.4f}")
                stats_fn = quant_entry.get("stats")
                wandb.log({
                    "val/ntp_loss":     ntp,
                    "val/ntp_delta":    delta,
                    **(stats_fn(student) if stats_fn else {}),
                }, step=step)

        if step > 0 and step % args.save_every == 0:
            save_training_state(student, optimizer, step, args, args.keep_last)

    ntp = (pp_mod.eval_ntp(student, student_text, val_chunks, device,
                             args.eval_batch_size, groups)
                   if groups is not None else
                   eval_ntp(student, val_chunks, device, args.eval_batch_size))
    # ALWAYS export the finished model, even off the 250 grid. This used to be gated on
    # `total_steps % export_every == 0` to avoid one export per run that no figure plots
    # and no eval grid asks for. That reasoning held while totals were ~2458 and the gate
    # cost the last 208 steps, but it does not survive a short run: the reasoning corpus
    # gives 980 steps, so the gate silently discarded the last 230 -- 23% of training,
    # including the only checkpoint anyone actually wants to evaluate. One off-grid
    # directory is much cheaper than re-deriving the final weights from the state.
    save_weights(student, total_steps, args, val_chunks, device)
    save_training_state(student, optimizer, total_steps, args, args.keep_last)
    if rank == 0:
        delta = ntp - teacher_val_ntp
        print(f"Final    | val_ntp={ntp:.4f}  Δ={delta:+.4f}", flush=True)
        wandb.log({"val/ntp_loss": ntp, "val/ntp_delta": delta}, step=total_steps)
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
