"""Tokenization and chunk building for SFT distillation.

One chunk per document (no packing): each chat example is tokenized with
assistant-only labels, truncated to max_seq_len and padded. Chunks are sharded
across ranks and cached to disk, since tokenizing the full mixture is slow.
"""

from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from .tulu import get_tulu_train_val   # noqa: F401  (re-exported for callers)
# NOT .datasets: qad.py lives in this directory, so Python puts it first on sys.path
# and a module named datasets.py here shadows HuggingFace's `datasets` for the whole
# process -- which made it import itself and die on a circular import.


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


