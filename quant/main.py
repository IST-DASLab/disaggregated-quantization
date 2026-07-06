from collections import defaultdict
import csv
import math
import sys
import types

from tqdm import tqdm
# lm_eval 0.4.x / transformers 5.x: hf_vlms uses AutoModelForVision2Seq which was removed
sys.modules.setdefault("lm_eval.models.hf_vlms", types.ModuleType("lm_eval.models.hf_vlms"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
import lm_eval.tasks
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

from data import get_data
from rtn import rtn_quantization
from gptq import gptq_quantization
from model_utils import QuantizedLinear, clear_device_cache
from datasets import load_dataset

MODEL = "Qwen/Qwen3-8B"
QUANT_METHOD = "gptq"  # "gptq" or "rtn"
QUANT_SCHEME = "downcast"  # "nvfp", "int", "independent", "downcast" (downcast is gptq-only)
ACT_QUANT = True       # dynamic NVFP4 activation quantization (prefill path only)
SEQUENCE_LENGTH = 2048
DECODE_BITS = 3
NUM_CALIBRATION_SEQUENCES = 128
NUM_EVAL_CONVERSATIONS = 256   # Tulu-3 chats used for the prefill/decode ppl split
SEED = 42
SEEDS = [0, 42, 1234, 2026, 9999]
LMEVAL_OUTPUT_PATH = "results.json"
RESULTS_CSV_PATH = "results_ppl_" + QUANT_METHOD + "_" + QUANT_SCHEME + ".csv"
LMEVAL_TASKS = ["gsm8k", "arc_challenge_llama"]
LMEVAL_MAX_LENGTH = 4*1024
LMEVAL_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_gen_toks": 1024,
}

def eval_lmeval(model, tokenizer):

    set_auto_mode(model, auto=True)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=128,
        max_length=LMEVAL_MAX_LENGTH,
        enable_thinking=False,
    )
    task_manager = lm_eval.tasks.TaskManager()
    evaluation_tracker = EvaluationTracker(output_path=LMEVAL_OUTPUT_PATH)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=LMEVAL_TASKS,
        batch_size=128,
#        limit=100,
        task_manager=task_manager,
        evaluation_tracker=evaluation_tracker,
        apply_chat_template=True,
        fewshot_as_multiturn=True
    )

    if results is None:
        return

    print("### Final results ###")
    samples = results.pop("samples", None)
    evaluation_tracker.save_results_aggregated(results=results, samples=samples)
    if samples:
        for task_name, task_samples in samples.items():
            evaluation_tracker.save_results_samples(task_name=task_name, samples=task_samples)
    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))


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


@torch.no_grad()
def compute_perplexity_prefill_decode(model, eval_data, device, prefill=("prefill", True), decode=("decode", False)):
    """``prefill`` / ``decode`` are ``(mode, act_quant)`` pairs applied in each phase."""
    set_auto_mode(model, auto=False)
    prefill_nll = decode_nll = 0.0
    prefill_tokens = decode_tokens = 0

    for ids, prefill_len in tqdm(eval_data, desc="Evaluating"):
        ids = ids.to(device)
        seq_len = ids.shape[-1]

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

        ids.to("cpu")

    prefill_ppl = math.exp(prefill_nll / prefill_tokens) if prefill_tokens else float("nan")
    decode_ppl = math.exp(decode_nll / decode_tokens) if decode_tokens else float("nan")

    return prefill_ppl, decode_ppl


def save_results_csv(results, seeds, path):
    phases = ["prefill", "decode"]
    header = ["label", "phase"] + [f"seed_{s}" for s in seeds] + ["mean"]
    print("### Average ppl across seeds (label x phase) ###")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, per_seed in results.items():
            for p_idx, phase in enumerate(phases):
                vals = [per_seed[s][p_idx] for s in seeds]
                mean = sum(vals) / len(vals) if vals else float("nan")
                writer.writerow([label, phase] + [f"{v:.6f}" for v in vals] + [f"{mean:.6f}"])
                print(f"[{label}] {phase}: {mean:.3f}")
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

    # chat_evals = get_tulu_chats(tokenizer, NUM_EVAL_CONVERSATIONS, SEEDS)

    lm_eval_results = eval_lmeval(model, tokenizer)

    clear_device_cache(True)
    results = defaultdict(dict)
    # for seed, evals in zip(SEEDS, chat_evals):        
    #     prefill_ppl, decode_ppl = compute_perplexity_prefill_decode(model, evals, device)
    #     print(f"[full-precision] SEED {seed}: Tulu ppl — prefill/user: {prefill_ppl:.3f}, decode/assistant: {decode_ppl:.3f}")
    #     results["full-precision"][seed] = (prefill_ppl, decode_ppl)

    calibration_data = get_data("open-thoughts", tokenizer, SEQUENCE_LENGTH, NUM_CALIBRATION_SEQUENCES, SEED)
    calibration_data = [s.to(device) for s in calibration_data]
    if QUANT_METHOD == "gptq":
        gptq_quantization(model, calibration_data, wbits=DECODE_BITS, device=device, act_quant=ACT_QUANT, scheme=QUANT_SCHEME)
    else:
        rtn_quantization(model, calibration_data, wbits=DECODE_BITS, device=device, act_quant=ACT_QUANT, scheme=QUANT_SCHEME)

    model.config.use_cache = True

    lm_eval_results = eval_lmeval(model, tokenizer)

    #for seed, evals in zip(SEEDS, chat_evals):
    #    prefill_ppl, decode_ppl = compute_perplexity_prefill_decode(model, evals, device, ("prefill", True), ("decode", False))
    #    print(f"[mixed-precision] SEED {seed}: Tulu ppl — prefill/user: {prefill_ppl:.3f}, decode/assistant: {decode_ppl:.3f}")
    #    results["mixed-precision"][seed] = (prefill_ppl, decode_ppl)
    
    # for label, phase in [
    #     ("NVFP4", ("prefill", False)), 
    #     ("NVFP4+A", ("prefill", True)), 
    #     ("INT", ("decode", False))
    # ]:
    #     for seed, evals in zip(SEEDS, chat_evals):
    #         prefill_ppl, decode_ppl = compute_perplexity_prefill_decode(model, evals, device, phase, phase)
    #         print(f"[{label}] SEED {seed}: Tulu ppl — prefill/user: {prefill_ppl:.3f}, decode/assistant: {decode_ppl:.3f}")
    #         results[label][seed] = (prefill_ppl, decode_ppl)

    save_results_csv(results, SEEDS, RESULTS_CSV_PATH)


if __name__ == "__main__":
    main()
