#!/usr/bin/env python3
"""Measure custom prefill/decode acceptance against an FP16 model."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quant.quantizer import NVFP_GROUPSIZE, Quantizer


TULU_DATASET = "allenai/tulu-3-sft-mixture"

# Faithful reconstructions of the 15 prompt topics in Appendix C.5, Table 7 of
# "Statistically-Lossless Quantization of Large Language Models". The paper
# publishes full wording for only three prompts. ``paper_output_tokens`` is the
# observed FP16 response length reported by the paper, not a generation limit.
PAPER_PROMPTS = [
    {
        "id": "cookies",
        "domain": "everyday",
        "paper_output_tokens": 268,
        "prompt": (
            "Give a complete recipe for homemade cookies, including ingredients, "
            "quantities, oven temperature, baking time, and numbered instructions."
        ),
    },
    {
        "id": "photosynthesis",
        "domain": "everyday",
        "paper_output_tokens": 138,
        "prompt": (
            "What is photosynthesis, how does it work, and why is it important "
            "for plants?"
        ),
    },
    {
        "id": "binary_search_tree",
        "domain": "code",
        "paper_output_tokens": 512,
        "prompt": (
            "Implement a binary search tree in Python with insertion, search, and "
            "deletion operations. Include a usage example and explain the time "
            "complexity of each operation."
        ),
    },
    {
        "id": "determinant_intuition",
        "domain": "mathematics",
        "paper_output_tokens": 512,
        "prompt": (
            "Explain the geometric intuition behind the determinant of a square "
            "matrix. Discuss volume scaling, orientation, invertibility, and the "
            "meaning of a zero determinant."
        ),
    },
    {
        "id": "sql_top_five_customers",
        "domain": "code",
        "paper_output_tokens": 446,
        "prompt": (
            "Consider tables customers(id, name) and orders(id, customer_id, "
            "amount). Write an SQL query that returns the five customers with the "
            "greatest total order value, including each customer's name and total, "
            "ordered from highest to lowest."
        ),
    },
    {
        "id": "geometric_series",
        "domain": "mathematics",
        "paper_output_tokens": 512,
        "prompt": (
            "Starting from a + ar + ar^2 + ar^3 + ..., derive the formula for its "
            "infinite sum and state precisely the values of r for which the series "
            "converges."
        ),
    },
    {
        "id": "merge_intervals",
        "domain": "code",
        "paper_output_tokens": 448,
        "prompt": (
            "Write a Python function that merges all overlapping intervals in a "
            "list of [start, end] pairs. Explain the algorithm, analyze its time "
            "and space complexity, and provide example test cases."
        ),
    },
    {
        "id": "wolf_goat_cabbage",
        "domain": "logic",
        "paper_output_tokens": 512,
        "prompt": (
            "A farmer must take a wolf, a goat, and a cabbage across a river. The "
            "boat carries only the farmer and one item. The wolf cannot be left "
            "alone with the goat, and the goat cannot be left alone with the "
            "cabbage. Find a valid crossing sequence and explain why it works."
        ),
    },
    {
        "id": "fdr_vs_reagan",
        "domain": "essay",
        "paper_output_tokens": 512,
        "prompt": (
            "Write a comparative essay about Franklin D. Roosevelt and Ronald "
            "Reagan. Compare their political philosophies, economic policies, "
            "leadership styles, historical circumstances, and long-term legacies."
        ),
    },
    {
        "id": "python_vs_java",
        "domain": "code",
        "paper_output_tokens": 512,
        "prompt": (
            "Compare Python and Java, covering syntax, type systems, performance, "
            "memory management, concurrency, ecosystems, and common use cases. "
            "Include a small equivalent code example in both languages."
        ),
    },
    {
        "id": "themes_of_isolation",
        "domain": "essay",
        "paper_output_tokens": 512,
        "prompt": (
            "Write an analytical essay about the theme of isolation in literature. "
            "Discuss its causes, psychological effects, and how authors use setting "
            "and relationships to develop the theme."
        ),
    },
    {
        "id": "byzantine_generals",
        "domain": "logic",
        "paper_output_tokens": 512,
        "prompt": (
            "Describe the Byzantine Generals Problem in distributed computing. "
            "Under an unauthenticated-message model, explain why three participants "
            "cannot guarantee consensus when one participant may be traitorous."
        ),
    },
    {
        "id": "romeo_and_juliet",
        "domain": "everyday",
        "paper_output_tokens": 259,
        "prompt": (
            "Summarize Shakespeare's Romeo and Juliet, covering the central "
            "conflict, major turning points, and conclusion."
        ),
    },
    {
        "id": "car_engine_for_child",
        "domain": "everyday",
        "paper_output_tokens": 364,
        "prompt": (
            "Describe how a gasoline car engine works in language that a "
            "ten-year-old can understand."
        ),
    },
    {
        "id": "mrna_vaccines",
        "domain": "reasoning",
        "paper_output_tokens": 512,
        "prompt": (
            "Explain step by step how mRNA vaccines work, from injection through "
            "protein production, immune response, and immune memory. Also explain "
            "why the vaccine does not alter a person's DNA."
        ),
    },
]

PAPER_OUTPUT_TOKENS = sum(
    prompt["paper_output_tokens"] for prompt in PAPER_PROMPTS
)
assert PAPER_OUTPUT_TOKENS == 6_531


@dataclass
class GeneratedSequence:
    """FP16-generated tokens split into independently verified chunks."""

    prompt_ids: torch.Tensor
    draft_chunks: List[torch.Tensor]
    total_drafted: int


def _count_accepted_tokens(
    verify_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> int:
    """Count the consecutive greedy matches at the start of a draft chunk."""
    accepted = 0
    for i in range(draft_tokens.shape[1]):
        verify_token = verify_logits[:, i, :].argmax(dim=-1)
        if verify_token.item() != draft_tokens[0, i].item():
            break
        accepted += 1
    return accepted


def _chat_input_ids(tokenizer, messages, add_generation_prompt: bool) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    return input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids


def get_paper_prompts(num_prompts: int) -> List[list]:
    """Return the reconstructed prompt suite as single-turn chat messages."""
    if not 1 <= num_prompts <= len(PAPER_PROMPTS):
        raise ValueError(
            f"--num-prompts must be between 1 and {len(PAPER_PROMPTS)} "
            "for --prompt-source paper"
        )
    return [
        [{"role": "user", "content": prompt["prompt"]}]
        for prompt in PAPER_PROMPTS[:num_prompts]
    ]


def get_tulu_prompts(tokenizer, num_prompts: int, seed: int) -> List[list]:
    """Sample the same long, single-turn Tulu v3 examples as quant/main.py."""
    raw = load_dataset(TULU_DATASET, split="train", streaming=True).shuffle(
        seed=seed,
        buffer_size=10_000,
    )
    prompts = []

    for example in raw:
        messages = example["messages"]
        if not (
            len(messages) == 2
            and messages[0]["role"] == "user"
            and messages[1]["role"] == "assistant"
        ):
            continue

        prefill_len = _chat_input_ids(
            tokenizer, messages[:1], add_generation_prompt=True
        ).shape[-1]
        full_len = _chat_input_ids(
            tokenizer, messages, add_generation_prompt=False
        ).shape[-1]
        if full_len - prefill_len < 1_000:
            continue

        prompts.append(messages[:1])
        if len(prompts) == num_prompts:
            break

    if len(prompts) < num_prompts:
        raise RuntimeError(f"Found only {len(prompts)}/{num_prompts} Tulu prompts")
    return prompts


@torch.no_grad()
def generate_anchor_sequences(
    fp16_model,
    tokenizer,
    prompts: List[list],
    max_new_tokens: int = 128,
    draft_len: int = 5,
    description: str = "Generating FP16 anchor sequences",
) -> List[GeneratedSequence]:
    """Greedily generate with FP16 and divide the resulting path into chunks."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if draft_len < 1:
        raise ValueError("draft_len must be at least 1")

    fp16_model.eval()
    fp16_device = next(fp16_model.parameters()).device
    sequences = []

    for messages in tqdm(prompts, desc=description):
        prompt_ids = _chat_input_ids(
            tokenizer, messages, add_generation_prompt=True
        )
        input_ids = prompt_ids.to(fp16_device)

        draft_chunks = []
        chunk_tokens = []
        output = fp16_model(input_ids, use_cache=True)
        cache = output.past_key_values

        for token_index in range(max_new_tokens):
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            chunk_tokens.append(next_token.cpu())

            if len(chunk_tokens) == draft_len or token_index + 1 == max_new_tokens:
                draft_chunks.append(torch.cat(chunk_tokens, dim=1))
                chunk_tokens = []

            if token_index + 1 < max_new_tokens:
                output = fp16_model(
                    next_token,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values

        sequences.append(
            GeneratedSequence(
                prompt_ids=prompt_ids,
                draft_chunks=draft_chunks,
                total_drafted=max_new_tokens,
            )
        )

    return sequences


@torch.no_grad()
def evaluate_model_on_sequences(
    verify: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sequences: List[GeneratedSequence],
    device: torch.device,
    description: str = "Evaluating mixed model",
) -> Dict:
    """Evaluate chunks independently while always following the FP16 token path."""
    total_model_accepted = 0
    total_drafted = 0
    per_prompt_acceptance = []

    for seq in tqdm(sequences, desc=description):
        corrected_prefix_ids = seq.prompt_ids.to(device)
        prompt_model_accepted = 0
        prompt_drafted = 0

        for draft_tokens in seq.draft_chunks:
            draft_tokens = draft_tokens.to(device)
            model_verify_logits = verify(corrected_prefix_ids, draft_tokens)
            model_accepted = _count_accepted_tokens(model_verify_logits, draft_tokens)

            prompt_model_accepted += model_accepted
            prompt_drafted += draft_tokens.shape[1]

            # A rejection never changes the next chunk's context: rebuild from
            # the complete FP16-corrected path before verifying that chunk.
            corrected_prefix_ids = torch.cat([corrected_prefix_ids, draft_tokens], dim=1)

        total_model_accepted += prompt_model_accepted
        total_drafted += prompt_drafted
        per_prompt_acceptance.append(prompt_model_accepted / prompt_drafted)

    acceptance_rate = total_model_accepted / total_drafted
    return {
        "acceptance_rate": acceptance_rate,
        "etl": 1.0 - acceptance_rate,
        "per_prompt_acceptance": per_prompt_acceptance,
        "total_model_accepted": total_model_accepted,
        "total_drafted": total_drafted,
    }


@torch.no_grad()
def verify_with_prefill_decode(
    prefill_model,
    decode_model,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Prefill the shared prefix, then verify the linear draft with decode."""
    prefill_output = prefill_model(input_ids, use_cache=True)
    cache = prefill_output.past_key_values
    verify_logits = [prefill_output.logits[:, -1:, :]]

    # Consuming draft token i produces the logits that verify draft token i + 1.
    for i in range(draft_tokens.shape[1] - 1):
        decode_output = decode_model(
            draft_tokens[:, i : i + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cache = decode_output.past_key_values
        verify_logits.append(decode_output.logits[:, -1:, :])

    return torch.cat(verify_logits, dim=1)

def load_model(model_name: str, device: str):
    """Load an FP16 model or one of quant/main.py's exported checkpoints."""
    config = AutoConfig.from_pretrained(model_name)
    load_args = {}
    quantization_config = getattr(config, "quantization_config", None)

    if (
        isinstance(quantization_config, dict)
        and quantization_config.get("quant_method") == "fp_quant"
    ):
        # Prefill weights are already fake-quantized. Load dqweight as nn.Linear
        # weight; the activation hook below supplies the exported model's A4 path.
        config.quantization_config = None
        load_args["key_mapping"] = {r"\.dqweight$": ".weight"}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        **load_args,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = True
    return model


def add_activation_quantization(model) -> int:
    """Apply quant/main.py's dynamic NVFP4 quantization to model activations."""
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not (
            "mlp" in name or "attn" in name
        ):
            continue

        quantizer = Quantizer(format="nvfp", bits=4, symmetric=True, group_size=NVFP_GROUPSIZE)

        def quantize_input(_module, args, quantizer=quantizer):
            inputs = args[0]
            scales, zeros = quantizer.get_quantization_params(inputs, dynamic=True)
            quantized = quantizer.quantize_dequantize(inputs, scales, zeros)
            return (quantized, *args[1:])

        handles.append(module.register_forward_pre_hook(quantize_input))

    # Keep the hook handles alive for the lifetime of the model.
    model._activation_quantization_handles = handles
    return len(handles)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prefill-model", default=str("models/Qwen3-8B-downcast-identity-gptq-prefill"))
    parser.add_argument("--decode-model", default=str("models/Qwen3-8B-downcast-identity-gptq-decode"))
    parser.add_argument("--prefill-act-quant", action=argparse.BooleanOptionalAction, default=False, help="Toggle dynamic NVFP4 activation quantization for cache prefill")
    parser.add_argument("--decode-act-quant", action=argparse.BooleanOptionalAction, default=False, help="Toggle dynamic NVFP4 activation quantization for chunk decoding",)
    parser.add_argument("--prompt-source", choices=("tulu", "paper"), default="paper", help="Use sampled Tulu prompts or the reconstructed 15-prompt paper suite")
    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompts (default: 5 for Tulu, all 15 for paper)")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="FP16 reference tokens per prompt (default: 5 for Tulu, 512 for paper)")
    parser.add_argument("--draft-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def print_acceptance(result: Dict) -> None:
    print("\nAcceptance — mixed model on FP16-generated chunks")
    accepted = f"{result['total_model_accepted']}/{result['total_drafted']}"
    print(f"{'Format':<16} {'Accepted':>15} {'Acceptance':>12} {'ETL':>12}")
    print(f"{'Mixed':<16} {accepted:>15} {result['acceptance_rate']:>11.4%} {result['etl']:>11.4%}")

def main() -> None:
    args = build_parser().parse_args()
    if args.num_prompts is None:
        args.num_prompts = len(PAPER_PROMPTS) if args.prompt_source == "paper" else 5
    if args.max_new_tokens is None:
        args.max_new_tokens = 512 if args.prompt_source == "paper" else 5

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model)
    if args.prompt_source == "paper":
        prompts = get_paper_prompts(args.num_prompts)
    else:
        prompts = get_tulu_prompts(tokenizer, args.num_prompts, args.seed)

    print(f"Loading FP16 model: {args.fp16_model}")
    fp16_model = load_model(args.fp16_model, args.device)
    sequences = generate_anchor_sequences(
        fp16_model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        draft_len=args.draft_len,
    )

    del fp16_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading custom prefill model: {args.prefill_model}")
    prefill_model = load_model(args.prefill_model, args.device)
    if args.prefill_act_quant:
        add_activation_quantization(prefill_model)

    print(f"Loading custom decode model: {args.decode_model}")
    decode_model = load_model(args.decode_model, args.device)
    if args.decode_act_quant:
        add_activation_quantization(decode_model)

    def verify_mixed(input_ids, draft_tokens):
        return verify_with_prefill_decode(
            prefill_model, decode_model, input_ids, draft_tokens
        )

    result = evaluate_model_on_sequences(verify_mixed, sequences, device)
    print_acceptance(result)


if __name__ == "__main__":
    main()
