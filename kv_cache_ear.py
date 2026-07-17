#!/usr/bin/env python3
"""Measure top-k expected acceptance rate for a mixed prefill/decode model."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from kv_cache_acceptance import (
    PAPER_PROMPTS,
    GeneratedSequence,
    add_activation_quantization,
    build_parser as build_acceptance_parser,
    generate_anchor_sequences,
    get_paper_prompts,
    get_tulu_prompts,
    load_model,
    verify_with_prefill_decode,
)


@dataclass
class TopKAnchorSequence:
    """An FP16 path plus reference top-k logits for every generated token."""

    prompt_ids: torch.Tensor
    draft_chunks: List[torch.Tensor]
    fp16_topk_values: List[torch.Tensor]
    fp16_topk_indices: List[torch.Tensor]
    total_tokens: int


@torch.no_grad()
def collect_fp16_topk(
    fp16_model,
    sequences: List[GeneratedSequence],
    top_k: int = 10,
    description: str = "Collecting FP16 top-k distributions",
) -> List[TopKAnchorSequence]:
    """Collect FP16 logits aligned with every token in the anchor sequences."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    fp16_model.eval()
    device = next(fp16_model.parameters()).device
    topk_sequences = []

    for sequence in tqdm(sequences, desc=description):
        reference_tokens = torch.cat(sequence.draft_chunks, dim=1).to(device)
        prompt_ids = sequence.prompt_ids.to(device)
        full_ids = torch.cat([prompt_ids, reference_tokens], dim=1)
        prompt_len = prompt_ids.shape[1]

        logits = fp16_model(full_ids, use_cache=False).logits[
            :, prompt_len - 1 : -1, :
        ].float()
        if top_k > logits.shape[-1]:
            raise ValueError(
                f"top_k={top_k} exceeds vocabulary size {logits.shape[-1]}"
            )
        topk_values, topk_indices = logits.topk(top_k, dim=-1)
        chunk_lengths = [chunk.shape[1] for chunk in sequence.draft_chunks]

        topk_sequences.append(
            TopKAnchorSequence(
                prompt_ids=sequence.prompt_ids,
                draft_chunks=sequence.draft_chunks,
                fp16_topk_values=list(
                    torch.split(topk_values.cpu(), chunk_lengths, dim=1)
                ),
                fp16_topk_indices=list(
                    torch.split(topk_indices.cpu(), chunk_lengths, dim=1)
                ),
                total_tokens=reference_tokens.shape[1],
            )
        )

    return topk_sequences


def topk_distribution_metrics(
    fp16_topk_values: torch.Tensor,
    fp16_topk_indices: torch.Tensor,
    candidate_logits: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-token EAR and KL on the FP16 top-k support.

    This matches the supplied reference: candidate logits are gathered at the
    FP16 top-k indices, then both sets of k logits are independently normalized.
    EAR is ``sum(min(p, q))`` over that shared, renormalized support.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if fp16_topk_values.shape != fp16_topk_indices.shape:
        raise ValueError("FP16 top-k values and indices must have the same shape")
    if candidate_logits.shape[:-1] != fp16_topk_indices.shape[:-1]:
        raise ValueError("Candidate logits are not aligned with FP16 positions")

    indices = fp16_topk_indices.to(candidate_logits.device)
    fp16_logits = fp16_topk_values.to(candidate_logits.device).float()
    candidate_topk_logits = candidate_logits.float().gather(-1, indices)

    p = F.softmax(fp16_logits / temperature, dim=-1)
    q = F.softmax(candidate_topk_logits / temperature, dim=-1)
    ear = torch.minimum(p, q).sum(dim=-1)

    eps = 1e-10
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    return ear, kl


@torch.no_grad()
def evaluate_topk_ear(
    prefill_model,
    decode_model,
    sequences: List[TopKAnchorSequence],
    device: torch.device,
    temperature: float = 1.0,
    description: str = "Evaluating top-k EAR",
) -> Dict:
    """Evaluate mixed-model distribution overlap on the FP16-corrected path."""
    total_ear = 0.0
    total_kl = 0.0
    total_tokens = 0
    prefill_ear = 0.0
    prefill_kl = 0.0
    prefill_tokens = 0
    decode_ear = 0.0
    decode_kl = 0.0
    decode_tokens = 0
    per_prompt_ear = []

    for sequence in tqdm(sequences, desc=description):
        corrected_prefix_ids = sequence.prompt_ids.to(device)
        prompt_ear = 0.0
        prompt_tokens = 0

        for reference_tokens, fp16_values, fp16_indices in zip(
            sequence.draft_chunks,
            sequence.fp16_topk_values,
            sequence.fp16_topk_indices,
        ):
            reference_tokens = reference_tokens.to(device)
            mixed_logits = verify_with_prefill_decode(
                prefill_model,
                decode_model,
                corrected_prefix_ids,
                reference_tokens,
            )
            token_ear, token_kl = topk_distribution_metrics(
                fp16_values,
                fp16_indices,
                mixed_logits,
                temperature=temperature,
            )

            chunk_tokens = reference_tokens.shape[1]
            chunk_ear = token_ear.sum().item()
            total_ear += chunk_ear
            total_kl += token_kl.sum().item()
            total_tokens += chunk_tokens
            prompt_ear += chunk_ear
            prompt_tokens += chunk_tokens

            # Position 0 comes from the prefill model's final prefix logit.
            prefill_ear += token_ear[:, 0].sum().item()
            prefill_kl += token_kl[:, 0].sum().item()
            prefill_tokens += token_ear[:, 0].numel()

            # All later positions are produced by decode while extending the
            # cache initialized by prefill.
            if chunk_tokens > 1:
                decode_ear += token_ear[:, 1:].sum().item()
                decode_kl += token_kl[:, 1:].sum().item()
                decode_tokens += token_ear[:, 1:].numel()

            # Every chunk starts from the complete FP16-corrected prefix.
            corrected_prefix_ids = torch.cat([corrected_prefix_ids, reference_tokens], dim=1)

        per_prompt_ear.append(prompt_ear / prompt_tokens)

    return {
        "ear": total_ear / total_tokens,
        "mean_topk_kl": total_kl / total_tokens,
        "total_tokens": total_tokens,
        "prefill_ear": prefill_ear / prefill_tokens,
        "prefill_mean_topk_kl": prefill_kl / prefill_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_ear": decode_ear / decode_tokens if decode_tokens else float("nan"),
        "decode_mean_topk_kl": decode_kl / decode_tokens if decode_tokens else float("nan"),
        "decode_tokens": decode_tokens,
        "per_prompt_ear": per_prompt_ear,
        "top_k": sequences[0].fp16_topk_values[0].shape[-1],
        "temperature": temperature,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_acceptance_parser()
    parser.description = (
        "Measure mixed-model expected acceptance rate on the FP16 top-k support."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="FP16 reference support size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for both top-k distributions",
    )
    return parser


def print_ear(result: Dict) -> None:
    print(
        f"\nTop-{result['top_k']} expected acceptance rate "
        f"(temperature={result['temperature']:g})"
    )
    print(f"FP16 reference EAR: {1.0:.4%}")
    print(f"{'Phase':<22} {'Tokens':>12} {'Mixed EAR':>14} {'Top-k KL':>14}")
    rows = (
        ("Overall", "total_tokens", "ear", "mean_topk_kl"),
        (
            "Chunk first/prefill",
            "prefill_tokens",
            "prefill_ear",
            "prefill_mean_topk_kl",
        ),
        (
            "Within chunk/decode",
            "decode_tokens",
            "decode_ear",
            "decode_mean_topk_kl",
        ),
    )
    for label, tokens_key, ear_key, kl_key in rows:
        print(
            f"{label:<22} {result[tokens_key]:>12,} "
            f"{result[ear_key]:>13.4%} {result[kl_key]:>14.6f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.num_prompts is None:
        args.num_prompts = len(PAPER_PROMPTS) if args.prompt_source == "paper" else 5
    if args.max_new_tokens is None:
        args.max_new_tokens = 512 if args.prompt_source == "paper" else 5
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model)
    if args.prompt_source == "paper":
        prompts = get_paper_prompts(args.num_prompts)
    else:
        prompts = get_tulu_prompts(tokenizer, args.num_prompts, args.seed)

    print(f"Loading FP16 model: {args.fp16_model}")
    fp16_model = load_model(args.fp16_model, args.device)
    anchor_sequences = generate_anchor_sequences(
        fp16_model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
    )
    topk_sequences = collect_fp16_topk(
        fp16_model,
        anchor_sequences,
        top_k=args.top_k,
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

    result = evaluate_topk_ear(
        prefill_model,
        decode_model,
        topk_sequences,
        device,
        temperature=args.temperature,
    )
    print_ear(result)


if __name__ == "__main__":
    main()
