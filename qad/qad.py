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
    - Optimizer:    dist_adamw.py — DistAdamW (ZeRO-2, from karpathy/nanochat).
"""

import argparse
import math
import os
import sys
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

from data_utils import get_tulu_train_val
from dist_adamw import DistAdamW
from fp8 import apply_fp8_linear


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _tokenize_with_labels(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Tokenize a chat example. Returns (input_ids, labels) where labels[t] equals
    input_ids[t] for tokens that belong to an assistant reply and -100 otherwise.

    Uses prefix-diff: for each assistant turn, the span is
    [len(tokens up to start-of-assistant-turn), len(tokens up to end-of-turn)).
    add_generation_prompt=True on the prefix captures the turn-start marker so we
    predict from the first content token (not the role header).
    """
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(full_ids)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True
        )
        start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        suffix = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        end = len(tokenizer(suffix, add_special_tokens=False)["input_ids"])
        for j in range(start, min(end, len(full_ids))):
            labels[j] = full_ids[j]

    return full_ids, labels


def _chunk_cache_path(
    cache_dir: Path, model: str, split: str,
    target_tokens: int, max_seq_len: int, rank: int, world_size: int, seed: int,
) -> Path:
    tag = (
        f"{model.replace('/', '__')}.{split}.sft.nopacking"
        f".tok{target_tokens}.seq{max_seq_len}.r{rank}of{world_size}.s{seed}"
    )
    return cache_dir / f"chunks_{tag}.pt"


def build_chunks(
    tokenizer,
    raw_dataset,
    target_tokens: int,
    max_seq_len: int,
    rank: int,
    world_size: int,
    seed: int = 42,
    cache_dir: Path | None = None,
    split: str = "data",
    model_name: str = "",
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """One chunk per document: tokenize, truncate to max_seq_len, pad shorter docs.

    Returns a list of (input_ids, labels, attention_mask) triples.
      - labels[t] = token id if position t is part of an assistant reply, else -100.
      - attention_mask[t] = 1 for real tokens, 0 for padding.
    Stops once target_tokens // world_size tokens have been collected for this rank.
    """
    if cache_dir is not None:
        cache_file = _chunk_cache_path(
            cache_dir, model_name, split, target_tokens, max_seq_len, rank, world_size, seed
        )
        if cache_file.exists():
            if rank == 0:
                print(f"Loading cached chunks from {cache_file}", flush=True)
            return torch.load(cache_file, weights_only=False)

    shard = raw_dataset.select(range(rank, len(raw_dataset), world_size))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    chunks: list[tuple[Tensor, Tensor, Tensor]] = []
    total_tokens = 0
    per_rank_target = target_tokens // world_size
    pbar = tqdm(shard, desc=f"tokenize rank{rank}", unit="ex", disable=rank != 0)
    for example in pbar:
        try:
            ids, lbls = _tokenize_with_labels(tokenizer, example["messages"])
        except Exception:
            continue

        # Truncate
        ids = ids[:max_seq_len]
        lbls = lbls[:max_seq_len]
        real_len = len(ids)

        # Pad to max_seq_len
        pad_len = max_seq_len - real_len
        ids_t = torch.tensor(ids + [pad_id] * pad_len, dtype=torch.long)
        lbl_t = torch.tensor(lbls + [-100] * pad_len, dtype=torch.long)
        msk_t = torch.tensor([1] * real_len + [0] * pad_len, dtype=torch.long)
        chunks.append((ids_t, lbl_t, msk_t))

        total_tokens += real_len
        pbar.set_postfix(tokens=f"{total_tokens/1e3:.0f}k/{per_rank_target/1e3:.0f}k")
        if total_tokens >= per_rank_target:
            break

    g = torch.Generator()
    g.manual_seed(seed + rank)
    perm = torch.randperm(len(chunks), generator=g).tolist()
    chunks = [chunks[i] for i in perm]

    if cache_dir is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(chunks, cache_file)

    return chunks


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int, lr_max: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    student: nn.Module,
    optimizer: DistAdamW,
    step: int,
    args: argparse.Namespace,
) -> None:
    if dist.get_rank() != 0:
        return
    path = Path(args.ckpt_dir) / args.run_name / f"step_{step:07d}" / "ckpt.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )
    print(f"[rank0] checkpoint → {path}", flush=True)


# ---------------------------------------------------------------------------
# Validation — per-token NTP cross-entropy on the student
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ntp(
    student: nn.Module,
    val_chunks: list[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    batch_size: int,
) -> float:
    student.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0, device=device)

    val_steps = range(0, len(val_chunks), batch_size)
    for i in tqdm(val_steps, desc="val", unit="batch", disable=dist.get_rank() != 0):
        batch = val_chunks[i : i + batch_size]
        if not batch:
            break
        ids_b, lbl_b, msk_b = zip(*batch)
        input_ids   = torch.stack(ids_b).to(device)
        labels_data = torch.stack(lbl_b).to(device)
        attn_mask   = torch.stack(msk_b).to(device)
        # No attention_mask: causal attention ensures real tokens only attend to earlier
        # real tokens; padding predictions are masked by labels=-100.
        out = student(input_ids=input_ids, labels=labels_data)
        n = (labels_data[:, 1:] != -100).sum().item()
        total_loss += out.loss * n
        total_n += n

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
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--micro-batch-size", type=int, default=8, help="sequences per GPU per forward pass")
    parser.add_argument("--global-batch-size", type=int, default=64, help="total sequences per optimizer step across all GPUs")
    parser.add_argument("--eval-batch-size",  type=int, default=1, help="sequences per GPU during validation")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--include-prefill-loss", action="store_true",
                        help="compute KL loss on all token positions (paper behaviour); "
                             "default is assistant-reply tokens only")
    parser.add_argument("--run-name", type=str, default="qad")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--chunk-cache-dir", type=str, default=None, help="directory to cache tokenized chunks")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=100)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        wandb.init(
            project="prefill-decode-distill",
            name=args.run_name,
            config=vars(args),
        )

    # Teacher — frozen bf16 reference; no gradients anywhere
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    teacher.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student — same checkpoint, FP8 fake-quantized linears, gradient checkpointing.
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    apply_fp8_linear(student)
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
        print(f"Model: {args.model}  ({n_params:.1f}B params)", flush=True)
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

    # DistAdamW handles gradient reduction — no DDP wrapper required
    optimizer = DistAdamW(
        [{"params": list(student.parameters())}],
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
        chunk_size=256,          # smaller chunks → lower peak memory per torch.func VJP
        return_soft_hard_loss=True,
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

    pbar = tqdm(range(total_steps), desc="train", unit="step", disable=rank != 0)
    for step in pbar:
        lr = cosine_lr(step, total_steps, args.lr, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        accum_kl = 0.0
        accum_ntp = 0.0

        for acc in range(grad_accum):
            chunk_start = step * chunks_per_step + acc * mbs
            chunk_slice = train_chunks[chunk_start : chunk_start + mbs]
            ids_list, lbl_list, msk_list = zip(*chunk_slice)
            input_ids   = torch.stack(ids_list).to(device)  # [mbs, T]
            labels_data = torch.stack(lbl_list).to(device)  # [mbs, T]

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
            # With --include-prefill-loss: compute KL on all positions (paper behaviour).
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

        with torch.no_grad():
            weight_norm = torch.sqrt(sum(
                p.float().norm() ** 2
                for p in student.parameters()
            )).item()

        pbar.set_postfix(kl=f"{accum_kl:.4f}", ntp=f"{accum_ntp:.4f}", lr=f"{lr:.2e}")

        if rank == 0:
            wandb.log({
                "train/kl":          accum_kl,
                "train/ntp":         accum_ntp,
                "train/grad_norm":   grad_norm,
                "train/weight_norm": weight_norm,
                "train/lr":          lr,
            }, step=step)

        if step > 0 and step % args.val_every == 0:
            ntp = eval_ntp(student, val_chunks, device, args.eval_batch_size)
            if rank == 0:
                pbar.write(f"step {step:5d} | val_ntp={ntp:.4f}")
                wandb.log({"val/ntp_loss": ntp}, step=step)

        if step % args.save_every == 0:
            save_checkpoint(student, optimizer, step, args)

    ntp = eval_ntp(student, val_chunks, device, args.eval_batch_size)
    if rank == 0:
        print(f"Final    | val_ntp={ntp:.4f}", flush=True)
        wandb.log({"val/ntp_loss": ntp}, step=total_steps)
        wandb.finish()
    save_checkpoint(student, optimizer, total_steps, args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
