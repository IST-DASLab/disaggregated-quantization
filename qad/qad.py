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
    - student.model / teacher.model assumes the standard HF CausalLM layout where
      .model is the base transformer (Llama, Mistral, Qwen, etc.) and .lm_head is
      the output projection. GPT-2-style models use .transformer instead.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "third_party" / "Liger-Kernel" / "src"),
)
from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

from export.save import export_variants, save_checkpoint
from quantizers import (REGISTRY, build_quantizer_params, calibrate_nvfp4,
                        prefill_mask_from_labels, quant_phase,
                        uses_compressed_tensors)
from training.checkpoint import (find_latest, load_training_state,
                                 save_training_state)
from training.data import build_chunks, get_tulu_train_val
from training.dist_optim import DistOptimizer


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
    if uses_compressed_tensors(args.quantizer) and val_chunks is not None \
            and dist.get_rank() == 0:
        calibrate_nvfp4(student, val_chunks, device)

    if dist.get_rank() == 0:
        # The layers own their format, so there is one code path here regardless of
        # whether it ends up packed FP4 or a plain bf16 HF model. Most formats emit a
        # single checkpoint straight to base_dir; the prefill/decode formats emit one
        # per phase under base_dir/<variant>/, each standalone so a disaggregated
        # deployment can load them on different workers.
        for v in export_variants(student):
            out_dir = base_dir / v if v else base_dir
            n = save_checkpoint(student, out_dir, variant=v, step=step)
            print(f"[rank0] checkpoint → {out_dir}  ({n} tensors)", flush=True)
    # Rank 0 alone does the calibration + export (seconds to minutes on a busy
    # filesystem). Without this barrier the other ranks run ahead and queue
    # collectives that rank 0 cannot service, and once the lag exceeds the NCCL
    # watchdog the whole job dies with a reduce_scatter timeout.
    dist.barrier()


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
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--val-tokens", type=int, default=100_000)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=8, help="sequences per GPU per forward pass")
    parser.add_argument("--global-batch-size", type=int, default=64, help="total sequences per optimizer step across all GPUs")
    parser.add_argument("--eval-batch-size",  type=int, default=1, help="sequences per GPU during validation")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scale-ratio", type=float, default=0.5,
                        help="scale LR multiplier for quantizers that use separate scale LR")
    parser.add_argument("--quantizer", type=str, default="gsq2bit",
                        choices=list(REGISTRY),
                        help="quantization scheme to apply to the student")
    parser.add_argument("--quantizer-params", type=str, default="",
                        help="JSON string of quantizer hyperparameter overrides, "
                             "e.g. '{\"groupsize\": 64}'")
    parser.add_argument("--include-prefill-loss", action="store_true",
                        help="compute KL loss on all token positions (paper behaviour); "
                             "default is assistant-reply tokens only")
    parser.add_argument("--run-name", type=str, default="qad")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--chunk-cache-dir", type=str, default=None, help="directory to cache tokenized chunks")
    parser.add_argument("--lr-schedule", type=str, default="cosine",
                        choices=["cosine", "constant"],
                        help="constant = linear warmup then a flat plateau (no cooldown)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="steps between resumable training-state saves")
    parser.add_argument("--keep-last", type=int, default=1,
                        help="how many training states to retain (they are ~12 bytes/param)")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "never"],
                        help="auto: continue from the newest training state for this tag")
    parser.add_argument("--val-every", type=int, default=25)
    args = parser.parse_args()

    quant_params, quant_hash = build_quantizer_params(args.quantizer, args.quantizer_params)
    quant_entry = REGISTRY[args.quantizer]
    # Embed quantizer name + param hash into identifiers for traceability
    run_tag = f"{args.run_name}-{args.quantizer}"
    ckpt_tag = f"{run_tag}-{quant_hash}"
    args.run_tag  = run_tag   # used for wandb run name
    args.ckpt_tag = ckpt_tag  # used for checkpoint path

    # 10-minute default is too tight: rank 0 writes multi-GB checkpoints while the
    # other ranks wait at a barrier, and lustre is slow when many jobs write at once.
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        # id + resume="allow": a restarted job continues logging into the same run
        # instead of creating a duplicate.
        wandb.init(
            project="prefill-decode-distill",
            name=run_tag,
            # id=hashlib.md5(ckpt_tag.encode()).hexdigest(),
            resume="allow",
            config={**vars(args), "quantizer_params": quant_params, "quantizer_hash": quant_hash},
        )

    # Teacher — frozen bf16 reference; no gradients anywhere
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    teacher.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student — same checkpoint, quantized linears, gradient checkpointing.
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="flash_attention_2"
    ).to(device)
    quant_entry["apply"](student, **quant_params)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()

    # torch.compile disabled: the NeMo container's Triton version is incompatible with
    # torch.inductor (KernelMetadata missing cluster_dims), crashing both the forward
    # and backward compiled passes.

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        n_params = sum(p.numel() for p in student.parameters()) / 1e9
        print(f"Model: {args.model}  ({n_params:.1f}B params)  quantizer={args.quantizer}  hash={quant_hash}", flush=True)
        print("Loading and tokenizing dataset …", flush=True)

    train_raw, val_raw = get_tulu_train_val()
    cache_dir = Path(args.chunk_cache_dir) if args.chunk_cache_dir else None
    train_chunks = build_chunks(
        tokenizer, train_raw, args.train_tokens, args.max_seq_len, rank, world_size,
        cache_dir=cache_dir, split="train", model_name=args.model,
    )
    val_chunks = build_chunks(
        tokenizer, val_raw, args.val_tokens, args.max_seq_len, rank, world_size,
        cache_dir=cache_dir, split="val", model_name=args.model,
    )
    if rank == 0:
        print(f"Train: {len(train_chunks)} docs/rank  ({len(train_chunks) * world_size} total)", flush=True)
        print(f"Val:   {len(val_chunks)} docs/rank", flush=True)

    # Teacher val loss is a constant (frozen model) — compute once here.
    if rank == 0:
        print("Computing teacher val baseline …", flush=True)
    teacher_val_ntp = _ntp_loss(teacher, val_chunks, device, args.eval_batch_size)
    if rank == 0:
        print(f"Teacher val_ntp: {teacher_val_ntp:.4f}", flush=True)
        wandb.log({"val/teacher_ntp": teacher_val_ntp}, step=0)

    # DistAdamW handles gradient reduction — no DDP wrapper required.
    # Quantizers that need separate LR / weight_decay per parameter class (e.g. gsq2bit)
    # supply a param_groups builder; others fall back to a single group over all params.
    _pg_fn = quant_entry["param_groups"]
    if _pg_fn is not None:
        param_groups = _pg_fn(student, lr=args.lr, lr_scale_ratio=args.lr_scale_ratio)
    else:
        param_groups = [{"params": list(student.parameters())}]
    optimizer = DistOptimizer(
        param_groups,
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


    mbs = args.micro_batch_size
    gbs = args.global_batch_size
    grad_accum = gbs // (mbs * world_size)
    assert grad_accum >= 1, (
        f"global_batch_size={gbs} < micro_batch_size={mbs} * world_size={world_size}"
    )
    # chunks consumed per rank per optimizer step
    chunks_per_step = mbs * grad_accum
    total_steps = len(train_chunks) // chunks_per_step
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
            start_step = load_training_state(latest[0], student, optimizer, device)
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
            with quant_phase(prefill_mask_from_labels(labels_data)):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Teacher: hidden states only, no gradients
                    with torch.no_grad():
                        t_hidden = teacher.model(input_ids=input_ids).last_hidden_state

                    # Student: base transformer; lm_head handled by Liger below
                    s_hidden = student.model(input_ids=input_ids).last_hidden_state

                # Causal shift: hidden state at t predicts token at t+1.
                # labels_data[t+1] is the target (assistant token or -100).
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

                s_lm_w = student.lm_head.weight
                t_lm_w = teacher.lm_head.weight

                loss, kl_soft, ntp_hard = kl_loss_fn(
                    s_hidden_q, s_lm_w, t_hidden_f, t_lm_w, true_labels=labels
                )
                (loss / grad_accum).backward()

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

        # Refresh hard-quantized weight buffers and advance annealing schedules.
        if quant_entry["post_update"] is not None:
            quant_entry["post_update"](student, step, total_steps)

        with torch.no_grad():
            weight_norm = torch.sqrt(sum(
                p.float().norm() ** 2
                for p in student.parameters()
            )).item()

        # Reduce all three train metrics across ranks before logging
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
            ntp = eval_ntp(student, val_chunks, device, args.eval_batch_size)
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

    ntp = eval_ntp(student, val_chunks, device, args.eval_batch_size)
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
