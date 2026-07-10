from collections import defaultdict
import csv
import math
import os

from tqdm import tqdm

from export_fpquant import export_decode_bf16, export_prefill_pseudoquant

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import get_data
from rtn import rtn_quantization
from gptq import gptq_quantization
from model_utils import QuantizedLinear, clear_device_cache
from datasets import load_dataset

MODEL = "Qwen/Qwen3-8B"
QUANT_METHOD = "rtn"  # "gptq" or "rtn"
QUANT_SCHEME = "downcast"  # "nvfp", "int", "independent", "downcast" (downcast is gptq-only)
ACT_QUANT = True       # dynamic NVFP4 activation quantization (prefill path only)
SEQUENCE_LENGTH = 2048
DECODE_BITS = 3
NUM_CALIBRATION_SEQUENCES = 64
NUM_EVAL_CONVERSATIONS = 256   # Tulu-3 chats used for the prefill/decode ppl split
SEED = 42
SEEDS = [0]
RESULTS_CSV_PATH = "results_" + QUANT_METHOD + "_" + QUANT_SCHEME + ".csv"

def _get_tulu_chat(tokenizer, num_conversations, seed, shuffle_buffer=10_000):
    raw = load_dataset(
        "allenai/tulu-3-sft-mixture", split="train", streaming=True
    ).shuffle(seed=seed, buffer_size=shuffle_buffer)
    data = []
    for example in raw:
        messages = example["messages"]
        # Single-turn only, so there is exactly one prefill->decode boundary.
        if not (len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"):
            continue

        prefill_len = len(
            tokenizer.apply_chat_template(
                messages[:1], 
                add_generation_prompt=True,
                tokenize=True, 
                return_dict=True
            )["input_ids"]
        )
        ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=False,
            tokenize=True, 
            return_dict=True
        )["input_ids"]
        
        if prefill_len >= len(ids):
            continue
        if len(ids) - prefill_len < 1000:
            continue

        data.append((torch.tensor(ids, dtype=torch.long).unsqueeze(0), prefill_len))
        if len(data) >= num_conversations:
            break
    if len(data) < num_conversations:
        print(f"Warning: collected only {len(data)}/{num_conversations} Tulu chats for seed {seed}.")
    return data


def get_tulu_chats(tokenizer, num_conversations, seeds, shuffle_buffer=10_000):
    return [
        _get_tulu_chat(tokenizer, num_conversations, seed, shuffle_buffer)
        for seed in seeds
    ]



def set_quantization_mode(model, mode: str, act: bool) -> None:
    """Select the weight set (``mode``) and toggle activation quantization (``act``)."""
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.set_mode(mode)
            module.act_quant = act


def set_auto_mode(model, auto: bool) -> None:
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.auto_mode = auto


def _topk_kl(logits, ref_vals, ref_idx):
    """Top-k KL(ref || model): both distributions restricted to the reference's
    top-k tokens and renormalized, as in eval.py. ``logits``: [T, V], refs: [T, k]."""
    q = F.softmax(logits.gather(-1, ref_idx), dim=-1)
    p = F.softmax(ref_vals, dim=-1)
    eps = 1e-10
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1).sum().item()


@torch.no_grad()
def compute_perplexity_prefill_decode(model, eval_data, device, prefill=("prefill", True), decode=("decode", False),
                                      reference_topk=None, top_k=10):
    """``prefill`` / ``decode`` are ``(mode, act_quant)`` pairs applied in each phase.

    If ``reference_topk`` is None, top-k logits are collected per conversation and
    returned as the last element (run this on the full-precision model, before
    quantization). Pass that object back as ``reference_topk`` on quantized runs
    to also get the top-k KL divergence vs. full precision, split by phase.

    Returns ``(prefill_ppl, decode_ppl, prefill_kl, decode_kl, collected_topk)``;
    the KL entries are NaN on the collection pass, ``collected_topk`` is None
    when a reference was supplied.
    """
    set_auto_mode(model, auto=False)
    prefill_nll = decode_nll = 0.0
    prefill_kl = decode_kl = 0.0
    prefill_tokens = decode_tokens = 0
    collected = [] if reference_topk is None else None

    for conv_idx, (ids, prefill_len) in enumerate(tqdm(eval_data, desc="Evaluating")):
        ids = ids.to(device)
        seq_len = ids.shape[-1]
        conv_ref = {} if collected is not None else reference_topk[conv_idx]

        # Prefill phase.
        set_quantization_mode(model, *prefill)
        out = model(ids[:, :prefill_len], use_cache=True)
        cache = out.past_key_values
        n = min(prefill_len, seq_len - 1)
        if n > 0:
            logits = out.logits[:, :n, :].reshape(-1, out.logits.shape[-1]).float()
            targets = ids[:, 1 : n + 1].reshape(-1)
            prefill_nll += F.cross_entropy(logits, targets, reduction="none").sum().item()
            prefill_tokens += n
            if collected is not None:
                vals, idx = logits.topk(top_k, dim=-1)
                conv_ref["prefill"] = (vals.cpu(), idx.cpu())
            else:
                vals, idx = conv_ref["prefill"]
                prefill_kl += _topk_kl(logits, vals.to(device), idx.to(device))

        # Decode phase: positions prefill_len..L-1 predict tokens prefill_len+1..L
        # (drop the last, whose target is past the end), reusing the prefill cache.
        m = seq_len - 1 - prefill_len
        if m > 0:
            # Decode phase, reusing the prefill cache.
            set_quantization_mode(model, *decode)
            out = model(ids[:, prefill_len:], past_key_values=cache, use_cache=True)
            logits = out.logits[:, :m, :].reshape(-1, out.logits.shape[-1]).float()
            targets = ids[:, prefill_len + 1 :].reshape(-1)
            decode_nll += F.cross_entropy(logits, targets, reduction="none").sum().item()
            decode_tokens += m
            if collected is not None:
                vals, idx = logits.topk(top_k, dim=-1)
                conv_ref["decode"] = (vals.cpu(), idx.cpu())
            else:
                vals, idx = conv_ref["decode"]
                decode_kl += _topk_kl(logits, vals.to(device), idx.to(device))

        if collected is not None:
            collected.append(conv_ref)
        ids.to("cpu")

    prefill_ppl = math.exp(prefill_nll / prefill_tokens) if prefill_tokens else float("nan")
    decode_ppl = math.exp(decode_nll / decode_tokens) if decode_tokens else float("nan")
    if collected is not None:
        prefill_kl = decode_kl = float("nan")
    else:
        prefill_kl = prefill_kl / prefill_tokens if prefill_tokens else float("nan")
        decode_kl = decode_kl / decode_tokens if decode_tokens else float("nan")

    return prefill_ppl, decode_ppl, prefill_kl, decode_kl, collected


def save_results_csv(results, seeds, path):
    # Index into the (prefill_ppl, decode_ppl, prefill_kl, decode_kl) tuples.
    rows = [("prefill", "ppl", 0), ("decode", "ppl", 1), ("prefill", "kl", 2), ("decode", "kl", 3)]
    header = ["label", "phase", "metric"] + [f"seed_{s}" for s in seeds] + ["mean"]
    print("### Average across seeds (label x phase x metric) ###")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, per_seed in results.items():
            for phase, metric, idx in rows:
                vals = [per_seed[s][idx] for s in seeds]
                mean = sum(vals) / len(vals) if vals else float("nan")
                writer.writerow([label, phase, metric] + [f"{v:.6f}" for v in vals] + [f"{mean:.6f}"])
                print(f"[{label}] {phase} {metric}: {mean:.4f}")
    print(f"Wrote results to {path}")


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    chat_evals = get_tulu_chats(tokenizer, NUM_EVAL_CONVERSATIONS, SEEDS)

    clear_device_cache(True)
    results = defaultdict(dict)
    # Full-precision pass: must run BEFORE quantization (weights are replaced
    # in place); it also collects the top-k logit reference for the KL metric.
    kl_refs = {}
    for seed, evals in zip(SEEDS, chat_evals):
        prefill_ppl, decode_ppl, _, _, kl_refs[seed] = compute_perplexity_prefill_decode(model, evals, device)
        print(f"[full-precision] SEED {seed}: Tulu ppl — prefill/user: {prefill_ppl:.3f}, decode/assistant: {decode_ppl:.3f}")
        results["full-precision"][seed] = (prefill_ppl, decode_ppl, float("nan"), float("nan"))

    calibration_data = get_data("open-thoughts", tokenizer, SEQUENCE_LENGTH, NUM_CALIBRATION_SEQUENCES, SEED)
    calibration_data = [s.to(device) for s in calibration_data]
    if QUANT_METHOD == "gptq":
        gptq_quantization(model, calibration_data, wbits=DECODE_BITS, device=device, act_quant=ACT_QUANT, scheme=QUANT_SCHEME)
    else:
        rtn_quantization(model, calibration_data, wbits=DECODE_BITS, device=device, act_quant=ACT_QUANT, scheme=QUANT_SCHEME)

    model.config.use_cache = True

    for seed, evals in zip(SEEDS, chat_evals):
        prefill_ppl, decode_ppl, prefill_kl, decode_kl, _ = compute_perplexity_prefill_decode(
            model, evals, device, ("prefill", True), ("decode", False), reference_topk=kl_refs[seed])
        print(f"[mixed-precision] SEED {seed}: Tulu ppl — prefill/user: {prefill_ppl:.3f}, decode/assistant: {decode_ppl:.3f} | kl — prefill: {prefill_kl:.4f}, decode: {decode_kl:.4f}")
        results["mixed-precision"][seed] = (prefill_ppl, decode_ppl, prefill_kl, decode_kl)

    # for label, phase in [
    #     ("NVFP4", ("prefill", False)),
    #     ("NVFP4+A", ("prefill", True)),
    #     ("INT", ("decode", False))
    # ]:
    #     for seed, evals in zip(SEEDS, chat_evals):
    #         prefill_ppl, decode_ppl, prefill_kl, decode_kl, _ = compute_perplexity_prefill_decode(
    #             model, evals, device, phase, phase, reference_topk=kl_refs[seed])
    #         print(f"[{label}] SEED {seed}: Tulu ppl — prefill/user: {prefill_ppl:.3f}, decode/assistant: {decode_ppl:.3f} | kl — prefill: {prefill_kl:.4f}, decode: {decode_kl:.4f}")
    #         results[label][seed] = (prefill_ppl, decode_ppl, prefill_kl, decode_kl)

    # save_results_csv(results, SEEDS, RESULTS_CSV_PATH)

    prefill_dir = os.path.join("../models", f"{MODEL.split('/')[-1]}-{QUANT_SCHEME}-identity-{QUANT_METHOD}-prefill")
    decode_dir = os.path.join("../models", f"{MODEL.split('/')[-1]}-{QUANT_SCHEME}-identity-{QUANT_METHOD}-decode")
    print(f"[4/4] Exporting checkpoints ...", flush=True)
    export_prefill_pseudoquant(model, tokenizer, prefill_dir)
    print(f"  prefill (pseudoquant NVFP4, W4A4)  -> {os.path.abspath(prefill_dir)}", flush=True)
    export_decode_bf16(model, tokenizer, decode_dir)
    print(f"  decode  (fake-quant bf16, W16A16)  -> {os.path.abspath(decode_dir)}", flush=True)
    print("EXPORT DONE", flush=True)


if __name__ == "__main__":
    main()
