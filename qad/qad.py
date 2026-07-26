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
from gsq2bit import apply_gsq2bit, apply_gsq3bit, gsq_param_groups, post_update_all
from ste_quant import apply_ste2bit, apply_ste3bit, apply_ste4bit
from quest import apply_quest2bit, apply_quest3bit, apply_quest4bit
from nvfp4 import apply_nvfp4, apply_nvfp4a16, calibrate_nvfp4, save_nvfp4_checkpoint
from quant import QuantizedLinear


# ---------------------------------------------------------------------------
# Quantizer registry
# ---------------------------------------------------------------------------
# Each entry maps a name to:
#   apply(model, **params)            – replace linears in-place
#   param_groups(model, lr, **params) – DistAdamW-compatible param groups (or None → all params)
#   post_update                       – callable(model, step, total_steps) or None
#   defaults                          – default hyperparameter dict merged with --quantizer-params

_QUANTIZER_REGISTRY: dict = {
    "fp8": {
        "apply":        lambda model, **_: apply_fp8_linear(model),
        "param_groups": None,          # all params equally
        "post_update":  None,          # FP8 recomputes each forward via STE; no buffer to refresh
        "defaults":     {},
    },
    "gsq2bit": {
        "apply":        apply_gsq2bit,
        "param_groups": gsq_param_groups,
        "post_update":  post_update_all,
        "defaults": {
            "groupsize":   128,
            "std":         0.01,
            "strength":    6.0,
            "temp_start":  2.0,
            "temp_end":    0.05,
            "scale_start": 100.0,
            "scale_end":   500.0,
        },
    },
    "gsq3bit": {
        "apply":        apply_gsq3bit,
        "param_groups": gsq_param_groups,
        "post_update":  post_update_all,
        "defaults": {
            "groupsize":   128,
            "std":         0.01,
            "strength":    6.0,
            "temp_start":  2.0,
            "temp_end":    0.05,
            "scale_start": 100.0,
            "scale_end":   500.0,
        },
    },
    "ste2bit": {
        "apply":        apply_ste2bit,
        "param_groups": None,          # single weight param, no split needed
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
    },
    "ste3bit": {
        "apply":        apply_ste3bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
    },
    "ste4bit": {
        "apply":        apply_ste4bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
    },
    "quest2bit": {
        "apply":        apply_quest2bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
    },
    "quest3bit": {
        "apply":        apply_quest3bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
    },
    "quest4bit": {
        "apply":        apply_quest4bit,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {"groupsize": 128},
    },
    "nvfp4": {
        # W4A4: NVFP4 fake-quant on both weights and activations (STE), block=16.
        "apply":        apply_nvfp4,
        "param_groups": None,          # single weight param per layer
        "post_update":  post_update_all,
        "defaults":     {},
    },
    "nvfp4a16": {
        # W4A16: NVFP4 weights only — activations stay bf16 (no input_global_scale).
        "apply":        apply_nvfp4a16,
        "param_groups": None,
        "post_update":  post_update_all,
        "defaults":     {},
    },
}


def _build_quantizer_params(name: str, overrides_json: str) -> dict:
    import json, hashlib
    entry = _QUANTIZER_REGISTRY[name]
    params = dict(entry["defaults"])
    if overrides_json:
        params.update(json.loads(overrides_json))
    # stable hash of the effective overrides for checkpoint naming
    h = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return params, h


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

        # Skip documents whose assistant reply was entirely cut off by truncation —
        # they produce all-(-100) labels, causing CE(mean of empty set) = NaN.
        if all(l == -100 for l in lbls):
            continue

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
    path = Path(args.ckpt_dir) / args.ckpt_tag / f"step_{step:07d}" / "ckpt.pt"
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


def build_hf_state_dict(student: nn.Module) -> dict[str, torch.Tensor]:
    """Remap the quantized student into a vanilla HF state dict.

    Each QuantizedLinear's hard-quantized weight (_wq) is written to the standard
    `<module>.weight` slot, so the result is bit-identical in structure to a plain
    Qwen3ForCausalLM. All quant-internal tensors (master weight, _mask, scales,
    quant_logits, schedule buffers, _values, _idx) are dropped. Everything else
    (embeddings, norms, lm_head, rotary buffers) is kept at its native dtype.
    """
    quant_paths: set[str] = {
        name for name, mod in student.named_modules()
        if isinstance(mod, QuantizedLinear)
    }
    out: dict[str, torch.Tensor] = {}
    for key, tensor in student.state_dict().items():
        parent, _, leaf = key.rpartition(".")
        if parent in quant_paths:
            if leaf == "_wq":
                out[f"{parent}.weight"] = tensor.detach().to(torch.bfloat16).cpu()
            elif leaf == "bias":
                out[key] = tensor.detach().to(torch.bfloat16).cpu()
            # drop all other quant-internal tensors
        else:
            out[key] = tensor.detach().cpu()
    return out


def save_weights(student: nn.Module, step: int, args: argparse.Namespace,
                 val_chunks=None, device=None) -> None:
    """Save an eval-ready HuggingFace checkpoint (dequantized weights in `weight`).

    Written as a standard HF model directory so evaluation is a single, fast
    from_pretrained() — no quantizer wrapping, no base-model read, no manual
    load_state_dict.

    For the NVFP4 quantizers a real compressed-tensors checkpoint is written instead:
    packed FP4 weights + FP8 block scales (+ a static per-layer input_global_scale for
    W4A4, calibrated here on a few val batches) so vLLM serves true W4A4 / W4A16.
    """
    if args.quantizer.startswith("nvfp4"):
        # Calibrate the static activation scale on every rank's model, but only
        # rank 0 writes (rank 0's calibration is what gets exported).
        # No-op for nvfp4a16 (weight-only): activations aren't quantized.
        if val_chunks is not None and dist.get_rank() == 0 and args.quantizer == "nvfp4":
            calibrate_nvfp4(student, val_chunks, device)
        if dist.get_rank() != 0:
            return
        out_dir = Path(args.ckpt_dir) / args.ckpt_tag / "weights" / f"step_{step:07d}"
        n = save_nvfp4_checkpoint(student, out_dir)
        print(f"[rank0] NVFP4 checkpoint → {out_dir}  ({n} tensors)", flush=True)
        return

    if dist.get_rank() != 0:
        return
    from safetensors.torch import save_file
    out_dir = Path(args.ckpt_dir) / args.ckpt_tag / "weights" / f"step_{step:07d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    state = build_hf_state_dict(student)
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt", "step": str(step)})
    # Write config (+ generation_config) so from_pretrained rebuilds the architecture.
    student.config.save_pretrained(out_dir)
    if getattr(student, "generation_config", None) is not None:
        student.generation_config.save_pretrained(out_dir)

    print(f"[rank0] HF checkpoint → {out_dir}  ({len(state)} tensors)", flush=True)


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
                        choices=list(_QUANTIZER_REGISTRY),
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
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=25)
    args = parser.parse_args()

    quant_params, quant_hash = _build_quantizer_params(args.quantizer, args.quantizer_params)
    quant_entry = _QUANTIZER_REGISTRY[args.quantizer]
    # Embed quantizer name + param hash into identifiers for traceability
    run_tag = f"{args.run_name}-{args.quantizer}"
    ckpt_tag = f"{run_tag}-{quant_hash}"
    args.run_tag  = run_tag   # used for wandb run name
    args.ckpt_tag = ckpt_tag  # used for checkpoint path

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        wandb.init(
            project="prefill-decode-distill",
            name=run_tag,
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
    optimizer = DistAdamW(
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

    pbar = tqdm(range(total_steps), desc="train", unit="step", disable=rank != 0)
    for step in pbar:
        lr = cosine_lr(step, total_steps, args.lr, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

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
                wandb.log({
                    "val/ntp_loss":     ntp,
                    "val/ntp_delta":    delta,
                }, step=step)

        if step % args.save_every == 0:
            save_checkpoint(student, optimizer, step, args)

    ntp = eval_ntp(student, val_chunks, device, args.eval_batch_size)
    save_weights(student, total_steps, args, val_chunks, device)
    if rank == 0:
        delta = ntp - teacher_val_ntp
        print(f"Final    | val_ntp={ntp:.4f}  Δ={delta:+.4f}", flush=True)
        wandb.log({"val/ntp_loss": ntp, "val/ntp_delta": delta}, step=total_steps)
        wandb.finish()
    save_checkpoint(student, optimizer, total_steps, args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
