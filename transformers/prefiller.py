"""Trainer-compatible QAD model; the objective and routing follow qad/training/qad.py.

Frozen decode weights are Parameters so FSDP2 can shard them. Quantized prefill
weights have an optional lazy BF16 buffer cache reused until the optimizer update.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Trainer, TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qad"))
from quantizers.base import set_recompute_wq
from quantizers.dual import quant_phase, prefill_mask_from_labels
from quantizers.frozen_decode import (
    apply_nvfp4frozendec, NVFP4FrozenDecodeLinear,
    FrozenDecodeNorm,
)
from training.models import text_stack


class TextOnly(nn.Module):
    """Keep just the text stack; no vision forward or vision parameter residency."""
    def __init__(self, model):
        super().__init__()
        stack = text_stack(model)
        self.model, self.lm_head = stack.base, stack.head
        self.config = getattr(model.config, "text_config", model.config)
        self.config._name_or_path = model.config._name_or_path
        if self.config.model_type == "qwen3_5_text":
            self.config.architectures = ["Qwen3_5ForCausalLM"]
        self.generation_config = getattr(model, "generation_config", None)
        self.config.use_cache = False
        self.model.config.use_cache = False


def prepare_student(student, decode_path, decode_prefix="model.language_model.", load_weights=True,
                    cache_quantized_weights=True, weight_cache_max_gib=0):
    """FP32 masters/shared state and BF16 shardable frozen decode matrices."""
    set_recompute_wq(True)
    if load_weights:
        print("Installing the prefill/decode quantizer on the text stack", flush=True)
    # from_pretrained(dequantize=True, dtype=float32) supplies FP32 initial weights.
    apply_nvfp4frozendec(student.model)
    student.lm_head.requires_grad_(False)
    if cache_quantized_weights:
        budget = int(weight_cache_max_gib * 1024**3) if weight_cache_max_gib else None
        used = 0
        linears = [m for m in student.modules() if isinstance(m, NVFP4FrozenDecodeLinear)]
        # Spend limited replicated-cache memory on the largest weight tensors.
        for module in sorted(linears, key=lambda m: m.weight.numel(), reverse=True):
            size = module.weight.numel() * 2
            if budget is None or used + size <= budget:
                module.enable_weight_cache()
                used += size
        if load_weights:
            print(f"BF16 weight cache budget: {used / 1024**3:.3f} GiB per rank (lazy)", flush=True)
    for module in student.modules():
        if isinstance(module, NVFP4FrozenDecodeLinear):
            value = torch.empty_like(module.weight, dtype=torch.bfloat16)
            del module.decode_frozen
            module.register_parameter("decode_frozen", nn.Parameter(value, requires_grad=False))
        for name, parameter in module.named_parameters(recurse=False):
            if name != "decode_frozen":
                parameter.data = parameter.data.float()
    # The sharing/trainable partition matches the reference. Shared frozen tensors
    # are initialized from the external decoder, which is the inference source of
    # truth. Prefill masters and specialized norm scales retain teacher initialization.
    if not load_weights:
        return
    print(f"Streaming frozen decoder tensors from {decode_path}", flush=True)
    decode_path = Path(decode_path)
    if decode_path.suffix == ".gguf":
        from gguf_decode import dense_gguf_weights
        weights = dense_gguf_weights(decode_path, student.config)
    else:
        weights = dense_safetensors_weights(decode_path, decode_prefix)
    targets = {}
    for name, parameter in student.named_parameters():
        if parameter.requires_grad or ".inner." in name:
            continue
        key = name.replace(".decode_frozen", ".weight").replace(".weight_decode", ".weight")
        targets[key] = parameter
    seen = set()
    with torch.no_grad():
        for key, value in weights:
            if key not in targets:
                continue
            parameter = targets[key]
            if value.shape != parameter.shape:
                raise ValueError(f"Decoder shape mismatch for {key}: {value.shape} != {parameter.shape}")
            parameter.copy_(value)
            seen.add(key)
            if len(seen) % 64 == 0:
                print(f"Frozen decoder: {len(seen)}/{len(targets)} tensors loaded", flush=True)
    missing = set(targets) - seen
    if missing:
        raise ValueError(f"Frozen decoder missing {len(missing)} tensors: {sorted(missing)[:8]}")
    assert_trainable_prefill_only(student)


def assert_trainable_prefill_only(student):
    allowed = set()
    for module in student.modules():
        if isinstance(module, NVFP4FrozenDecodeLinear):
            allowed.add(id(module.weight))
            if module.bias is not None:
                raise ValueError("Biased dual linears need separate frozen decode biases")
        elif isinstance(module, FrozenDecodeNorm):
            allowed.add(id(module.weight_prefill))
    actual = {id(p) for p in student.parameters() if p.requires_grad}
    if actual != allowed:
        raise ValueError("Trainable parameters differ from the prefill-only allowlist")


def dense_safetensors_weights(path, prefix):
    from safetensors import safe_open
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as reader:
            for key in reader.keys():
                name = "model." + key[len(prefix):] if key.startswith(prefix) else key
                yield name, reader.get_tensor(key)


def chunked_kl(student_hidden, teacher_hidden, student_head, teacher_head,
               labels, attention_mask, chunk_size=1024, temperature=1.0,
               include_prefill_loss=False):
    """KL(teacher || student), shifted and scored on non-padding target positions.

    Checkpoint each projection/loss so backward holds O(chunk_size * vocab) logits,
    not O(sequence_length * vocab). Head forwards preserve FSDP hooks.
    """
    valid = attention_mask[:, 1:].bool()
    if not include_prefill_loss:
        valid = valid & labels[:, 1:].ne(-100)
    s = student_hidden[:, :-1][valid]
    t = teacher_hidden[:, :-1][valid]
    if s.shape[0] == 0:
        raise ValueError("Batch has no supervised next-token positions")

    def loss_part(sh, th):
        with torch.no_grad():
            tl = teacher_head(th).float() / temperature
            tlog = F.log_softmax(tl, dim=-1)
        sl = student_head(sh).float() / temperature
        slog = F.log_softmax(sl, dim=-1)
        return (tlog.exp() * (tlog - slog)).sum() * temperature**2

    total = s.new_zeros((), dtype=torch.float32)
    for start in range(0, s.shape[0], chunk_size):
        args = (s[start:start + chunk_size], t[start:start + chunk_size])
        if torch.is_grad_enabled():
            total = total + checkpoint(loss_part, *args, use_reentrant=False)
        else:
            total = total + loss_part(*args)
    return total / s.shape[0]


class Prefiller(nn.Module):
    def __init__(self, student, teacher, chunk_size=1024, temperature=1.0,
                 include_prefill_loss=False):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.config = student.config
        self.chunk_size, self.temperature = chunk_size, temperature
        self.include_prefill_loss = include_prefill_loss
        # Both text stacks are inside the Trainer's FSDP root. Shard each decoder
        # block and each embedding; keep heads on the root for chunked projections.
        self._no_split_modules = sorted({type(layer).__name__ for model in (student, teacher)
                                        for layer in model.model.layers} | {"Embedding"})

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.student.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False})

    def forward(self, input_ids, attention_mask, labels):
        # FSDP2 normally relies on its param_dtype policy for mixed precision.
        # Ours intentionally leaves FP32 masters and FP8 teacher storage intact,
        # so Trainer's bf16 flag alone does not supply compute autocast here.
        # Checkpointing captures this autocast context for backward recomputation.
        with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16,
                            enabled=input_ids.is_cuda):
            return self._forward(input_ids, attention_mask, labels)

    def _forward(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            teacher_hidden = self.teacher.model(input_ids=input_ids,
                attention_mask=attention_mask, use_cache=False).last_hidden_state
        with quant_phase(prefill_mask_from_labels(labels)):
            student_hidden = self.student.model(input_ids=input_ids,
                attention_mask=attention_mask, use_cache=False).last_hidden_state
        loss = chunked_kl(student_hidden, teacher_hidden, self.student.lm_head,
                          self.teacher.lm_head, labels, attention_mask,
                          self.chunk_size, self.temperature, self.include_prefill_loss)
        return {"loss": loss}


class WeightCacheCallback(TrainerCallback):
    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        # This event follows optimizer.step, before evaluation or checkpoint save.
        # Never invalidate between microbatches or during backward recomputation.
        for module in model.modules():
            if isinstance(module, NVFP4FrozenDecodeLinear):
                module.invalidate_weight_cache()


class PrefillerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(WeightCacheCallback())
        self.model_accepts_loss_kwargs = False
        if self.is_fsdp_enabled:
            from torch.distributed.fsdp import MixedPrecisionPolicy
            # Never cast stored FP8 teacher weights to BF16 without their scales.
            # Autocast handles GEMMs; student masters and reductions remain FP32.
            self.accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecisionPolicy(
                param_dtype=None, reduce_dtype=torch.float32)

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs.update(self.__dict__.pop("_utilization_metrics", {}))
        return super().log(logs, start_time)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # The phase mask MUST remain active through backward/checkpoint recompute.
        if hasattr(self, "_step_token_counts"):
            ids, labels, mask = inputs["input_ids"], inputs["labels"], inputs["attention_mask"]
            batch, seq = ids.shape
            counts = [ids.numel(), int(mask.sum()),
                      int((labels[:, 1:].ne(-100) & mask[:, 1:].bool()).sum()), batch * seq * seq]
            self._step_token_counts = [a + b for a, b in zip(self._step_token_counts, counts)]
        inputs = self._prepare_inputs(inputs)
        with quant_phase(prefill_mask_from_labels(inputs["labels"])):
            return super().training_step(model, inputs, num_items_in_batch)

    def _save(self, output_dir=None, state_dict=None):
        # Plain nn.Module may contain tied teacher weights. torch.save supports ties;
        # these are resumable training states, not deployable HF prefiller exports.
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict if state_dict is not None else self.model.state_dict(), output / "pytorch_model.bin")
        torch.save(self.args, output / "training_args.bin")
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output)


def collate_chunks(examples):
    return {key: torch.stack([example[key] for example in examples])
            for key in ("input_ids", "labels", "attention_mask")}


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No usable training examples fit max_seq_len")
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        ids, labels, mask = self.chunks[index]
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
