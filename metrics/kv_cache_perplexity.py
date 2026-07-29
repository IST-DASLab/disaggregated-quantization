#!/usr/bin/env python3
"""Measure prefill/decode perplexity using quant/main.py's evaluation split."""

from __future__ import annotations

import argparse
import gc
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from metrics.kv_cache_acceptance import add_activation_quantization, load_model


TULU_DATASET = "allenai/tulu-3-sft-mixture"
EvalConversation = Tuple[torch.Tensor, int]


def get_tulu_conversations(
    tokenizer,
    num_conversations: int,
    seed: int,
    shuffle_buffer: int = 10_000,
    min_assistant_tokens: int = 1_000,
) -> List[EvalConversation]:
    """Reproduce quant/main.py's long, single-turn Tulu evaluation data."""
    raw = load_dataset(TULU_DATASET, split="train", streaming=True).shuffle(
        seed=seed,
        buffer_size=shuffle_buffer,
    )
    conversations = []

    for example in raw:
        messages = example["messages"]
        if not (
            len(messages) == 2
            and messages[0]["role"] == "user"
            and messages[1]["role"] == "assistant"
        ):
            continue

        prefill_ids = tokenizer.apply_chat_template(
            messages[:1],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
        )["input_ids"]
        prefill_len = len(prefill_ids)

        if prefill_len >= len(full_ids):
            continue
        if len(full_ids) - prefill_len < min_assistant_tokens:
            continue

        conversations.append(
            (torch.tensor(full_ids, dtype=torch.long).unsqueeze(0), prefill_len)
        )
        if len(conversations) == num_conversations:
            break

    if len(conversations) < num_conversations:
        print(
            f"Warning: collected only {len(conversations)}/"
            f"{num_conversations} Tulu conversations"
        )
    if not conversations:
        raise RuntimeError("No matching Tulu conversations were found")
    return conversations


def _perplexity(total_nll: float, total_tokens: int) -> float:
    return (
        math.exp(total_nll / total_tokens)
        if total_tokens
        else float("nan")
    )


@torch.no_grad()
def compute_perplexity_prefill_decode(
    prefill_model,
    decode_model,
    eval_data: List[EvalConversation],
    device: torch.device,
    description: str = "Evaluating",
) -> Dict:
    """Match quant/main.py's prefill/decode PPL computation.

    The prefill model consumes the user-side chat prefix and produces the KV
    cache. Its loss covers targets at positions ``1..prefill_len``, including
    the first assistant token. The decode model reuses that cache and consumes
    the complete assistant suffix; its loss covers the remaining assistant
    targets at positions ``prefill_len + 1..sequence_end``.
    """
    prefill_model.eval()
    decode_model.eval()
    prefill_nll = 0.0
    decode_nll = 0.0
    prefill_tokens = 0
    decode_tokens = 0

    for ids, prefill_len in tqdm(eval_data, desc=description):
        ids = ids.to(device)
        sequence_len = ids.shape[-1]

        prefill_output = prefill_model(
            ids[:, :prefill_len],
            use_cache=True,
        )
        cache = prefill_output.past_key_values
        num_prefill_targets = min(prefill_len, sequence_len - 1)
        if num_prefill_targets > 0:
            logits = prefill_output.logits[
                :, :num_prefill_targets, :
            ].reshape(-1, prefill_output.logits.shape[-1]).float()
            targets = ids[:, 1 : num_prefill_targets + 1].reshape(-1)
            prefill_nll += F.cross_entropy(
                logits,
                targets,
                reduction="none",
            ).sum().item()
            prefill_tokens += num_prefill_targets

        num_decode_targets = sequence_len - 1 - prefill_len
        if num_decode_targets > 0:
            decode_output = decode_model(
                ids[:, prefill_len:],
                past_key_values=cache,
                use_cache=True,
            )
            logits = decode_output.logits[
                :, :num_decode_targets, :
            ].reshape(-1, decode_output.logits.shape[-1]).float()
            targets = ids[:, prefill_len + 1 :].reshape(-1)
            decode_nll += F.cross_entropy(
                logits,
                targets,
                reduction="none",
            ).sum().item()
            decode_tokens += num_decode_targets

    total_nll = prefill_nll + decode_nll
    total_tokens = prefill_tokens + decode_tokens
    return {
        "prefill_ppl": _perplexity(prefill_nll, prefill_tokens),
        "prefill_mean_nll": prefill_nll / prefill_tokens,
        "prefill_nll": prefill_nll,
        "prefill_tokens": prefill_tokens,
        "decode_ppl": _perplexity(decode_nll, decode_tokens),
        "decode_mean_nll": (
            decode_nll / decode_tokens if decode_tokens else float("nan")
        ),
        "decode_nll": decode_nll,
        "decode_tokens": decode_tokens,
        "overall_ppl": _perplexity(total_nll, total_tokens),
        "overall_mean_nll": total_nll / total_tokens,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Tulu PPL using quant/main.py's phase split."
    )
    parser.add_argument("--fp16-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--prefill-model",
        default="models/Qwen3-8B-independent-identity-gptq-prefill",
    )
    parser.add_argument(
        "--decode-model",
        default="models/Qwen3-8B-independent-identity-gptq-decode",
    )
    parser.add_argument(
        "--prefill-act-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle dynamic NVFP4 activation quantization for prefill",
    )
    parser.add_argument(
        "--decode-act-quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle dynamic NVFP4 activation quantization for decode",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=256,
        help="Number of long single-turn Tulu conversations",
    )
    parser.add_argument(
        "--min-assistant-tokens",
        type=int,
        default=1_000,
        help="Minimum assistant suffix length, matching quant/main.py by default",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def print_perplexity(results: Dict[str, Dict]) -> None:
    first_result = next(iter(results.values()))
    print("\nTulu perplexity — quant/main.py prefill/decode split")
    print(
        f"Tokens: prefill={first_result['prefill_tokens']:,}, "
        f"decode={first_result['decode_tokens']:,}"
    )
    print(
        f"{'Model':<16} {'Prefill/user':>16} "
        f"{'Decode/assistant':>18} {'Overall':>14}"
    )
    for label, result in results.items():
        print(
            f"{label:<16} {result['prefill_ppl']:>16.6f} "
            f"{result['decode_ppl']:>18.6f} "
            f"{result['overall_ppl']:>14.6f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.num_conversations < 1:
        raise ValueError("--num-conversations must be at least 1")
    if args.min_assistant_tokens < 1:
        raise ValueError("--min-assistant-tokens must be at least 1")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model)
    eval_data = get_tulu_conversations(
        tokenizer,
        args.num_conversations,
        args.seed,
        min_assistant_tokens=args.min_assistant_tokens,
    )

    print(f"Loading FP16 model: {args.fp16_model}")
    fp16_model = load_model(args.fp16_model, args.device)
    fp16_result = compute_perplexity_prefill_decode(
        fp16_model,
        fp16_model,
        eval_data,
        device,
        description="Evaluating FP16",
    )

    del fp16_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading custom prefill model: {args.prefill_model}")
    prefill_model = load_model(args.prefill_model, args.device)
    if args.prefill_act_quant:
        module_count = add_activation_quantization(prefill_model)
        print(f"Enabled activation quantization on {module_count} prefill linears")

    print(f"Loading custom decode model: {args.decode_model}")
    decode_model = load_model(args.decode_model, args.device)
    if args.decode_act_quant:
        module_count = add_activation_quantization(decode_model)
        print(f"Enabled activation quantization on {module_count} decode linears")

    mixed_result = compute_perplexity_prefill_decode(
        prefill_model,
        decode_model,
        eval_data,
        device,
        description="Evaluating mixed model",
    )
    print_perplexity({"FP16": fp16_result, "Mixed": mixed_result})


if __name__ == "__main__":
    main()
