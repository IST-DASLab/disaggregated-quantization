"""Train Qwen3.8 NVFP4 prefillers with Hugging Face Trainer and FSDP2."""
import argparse
import json
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, FineGrainedFP8Config, TrainingArguments, set_seed
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from prefiller import (ChunkDataset, Prefiller, PrefillerTrainer, TextOnly,
                       collate_chunks, prepare_student, assert_trainable_prefill_only)
from training.data import build_chunks, get_reasoning_train_val, _chunk_cache_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--decode-model", default="ISTA-DASLab/GSQ-RCO-small")
    p.add_argument("--decode-gguf", default="Qwen3.8-27B-GSQ-RCO-IQ1_S.gguf")
    p.add_argument("--decode-prefix", default="model.language_model.")
    p.add_argument("--output-dir", default="transformers/outputs/qwen38-iq1s")
    p.add_argument("--cache-dir", default="transformers/.cache")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--train-tokens", type=int, default=200_000_000)
    p.add_argument("--val-tokens", type=int, default=200_000)
    p.add_argument("--max-steps", type=int, default=980)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--sync-each-batch", action=argparse.BooleanOptionalAction, default=True,
                   help="Reduce-scatter gradients per microbatch instead of retaining full gradients")
    p.add_argument("--optimizer", choices=["adamw_torch", "adamw_torch_fused"], default="adamw_torch")
    p.add_argument("--activation-checkpoint-every", type=int, default=1,
                   help="Checkpoint every Nth student block; 0 disables block recomputation")
    p.add_argument("--fsdp-prefetch-depth", type=int, default=0,
                   help="Explicit FSDP2 block prefetch depth; 0 keeps implicit prefetch")
    p.add_argument("--fsdp-retain-student-layers", type=int, default=0,
                   help="Keep first N student blocks gathered between forward and backward")
    p.add_argument("--profile-step", type=int, default=0,
                   help="Capture rank-zero CPU/CUDA trace for this absolute optimizer step")
    p.add_argument("--compile-blocks", action="store_true", help="Compile repeated decoder blocks in place with Inductor")
    p.add_argument("--compile-mode", choices=["default", "max-autotune-no-cudagraphs"], default="default")
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                   help="Disable for disposable throughput benchmarks")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--logit-chunk-size", type=int, default=1024)
    p.add_argument("--cache-quantized-weights", action=argparse.BooleanOptionalAction, default=True,
                   help="Reuse replicated BF16 weight buffers until the optimizer update")
    p.add_argument("--weight-cache-max-gib", type=float, default=0,
                   help="Cap replicated BF16 caches per GPU, prioritizing largest weights; 0 is unlimited")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--include-prefill-loss", action="store_true")
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke-test", action="store_true", help="Tiny random hybrid Qwen with a real FP8 teacher")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16-peak-tflops", type=float, default=2250.0, help="Dense per-GPU peak for MFU estimates (B200)")
    p.add_argument("--fp8-peak-tflops", type=float, default=4500.0)
    args = p.parse_args()
    for key in ("max_seq_len", "max_steps", "batch_size", "gradient_accumulation_steps", "logit_chunk_size"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.temperature <= 0:
        p.error("--temperature must be positive")
    if args.weight_cache_max_gib < 0:
        p.error("--weight-cache-max-gib must be nonnegative")
    for key in ("activation_checkpoint_every", "fsdp_prefetch_depth", "fsdp_retain_student_layers", "profile_step"):
        if getattr(args, key) < 0:
            p.error(f"--{key.replace('_', '-')} must be nonnegative")
    if not args.fsdp and (args.fsdp_prefetch_depth or args.fsdp_retain_student_layers):
        p.error("FSDP tuning controls require --fsdp")
    if not args.fsdp and args.activation_checkpoint_every > 1:
        p.error("Selective block checkpointing requires --fsdp")
    if args.compile_blocks and not args.fsdp:
        p.error("--compile-blocks currently requires --fsdp")
    if args.profile_step > args.max_steps:
        p.error("--profile-step must not exceed --max-steps")
    return args


def tiny_models():
    """Synthetic architecture smoke test; never substitutes for a full-model run."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from transformers.integrations.finegrained_fp8 import FP8Linear
    config = Qwen3_5TextConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32,
        layer_types=["linear_attention", "full_attention"],
        pad_token_id=0, tie_word_embeddings=False)
    student = TextOnly(Qwen3_5ForCausalLM(config).float())
    teacher = TextOnly(Qwen3_5ForCausalLM(config).bfloat16())
    teacher.load_state_dict(student.state_dict())
    # Use HF's real FP8 layer and kernel for MLP projections; scales are exact here.
    for name, module in list(teacher.model.named_modules()):
        if isinstance(module, torch.nn.Linear) and ".mlp." in "." + name:
            fp8 = FP8Linear(module.in_features, module.out_features, block_size=(128, 128))
            fp8.weight = torch.nn.Parameter(module.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
            fp8.weight_scale_inv = torch.nn.Parameter(torch.ones_like(fp8.weight_scale_inv), requires_grad=False)
            parent, _, child = name.rpartition(".")
            setattr(teacher.model.get_submodule(parent), child, fp8)
    return student, teacher


def load_text_model(model_id, dequantize, cache_dir, metadata_only=False):
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    cls = getattr(transformers, config.architectures[0])
    quant = dict(config.quantization_config)
    quant.pop("quant_method", None)
    quant["dequantize"] = dequantize
    if metadata_only:
        # HF's rank-zero-only loading excludes quantized checkpoints, including
        # dequantize=True. Never call from_pretrained on nonzero ranks here.
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=False):
            model = cls._from_config(config,
                dtype=torch.float32 if dequantize else torch.bfloat16,
                attn_implementation="sdpa")
            if not dequantize:
                from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
                quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(**quant), pre_quantized=True)
                quantizer._process_model_before_weight_loading(model)
        return TextOnly(model)
    model = cls.from_pretrained(model_id, config=config, cache_dir=cache_dir,
        dtype=torch.float32 if dequantize else torch.bfloat16,
        quantization_config=FineGrainedFP8Config(**quant), attn_implementation="sdpa")
    return TextOnly(model)


def allocate_empty_parameters(model):
    """Unique empty CPU storage for Accelerate's optimizer pointer mapping.

    No tensor data is read, zeroed, or initialized here. Linux reserves address
    space without resident pages; FSDP broadcasts actual parameter shards later.
    """
    replacements = {}
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if parameter.device.type != "meta":
                continue
            key = id(parameter)
            if key not in replacements:
                replacements[key] = torch.nn.Parameter(
                    torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu"),
                    requires_grad=parameter.requires_grad)
            module.register_parameter(name, replacements[key])
        for name, value in list(module.named_buffers(recurse=False)):
            if value.device.type == "meta":
                module.register_buffer(name, torch.zeros(value.shape, dtype=value.dtype, device="cpu"),
                                       persistent=name not in module._non_persistent_buffers_set)


def verify_rank_loading(model, rank):
    """Fail before FSDP if ranks disagree or a nonzero rank loaded real weights."""
    import psutil
    rss = psutil.Process().memory_info().rss / 1024**3
    print(f"[rank {rank}] pre-sharding resident RAM: {rss:.2f} GiB", flush=True)
    if rank != 0 and rss > 8:
        raise RuntimeError("Nonzero rank unexpectedly materialized full model weights")
    signature = [(name, tuple(p.shape), str(p.dtype), p.requires_grad)
                 for name, p in model.named_parameters()]
    signatures = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(signatures, signature)
    if any(item != signatures[0] for item in signatures[1:]):
        differences = [rank_index for rank_index, item in enumerate(signatures) if item != signatures[0]]
        raise RuntimeError(f"Rank model schemas differ: {differences}")


def resolve_decoder(args, hf_cache):
    path = Path(args.decode_model)
    if path.is_file():
        return path
    if path.is_dir():
        return path / args.decode_gguf if (path / args.decode_gguf).exists() else path
    if args.decode_gguf:
        return Path(hf_hub_download(args.decode_model, args.decode_gguf, cache_dir=hf_cache))
    return Path(snapshot_download(args.decode_model, cache_dir=hf_cache,
                                 allow_patterns=["*.safetensors", "*.json"]))


def load_datasets(tokenizer, args, cache):
    specifications = [("train", args.train_tokens), ("val", args.val_tokens)]
    paths = [_chunk_cache_path(cache / "chunks", args.teacher_model, split, tokens,
                              args.max_seq_len, 0, 1, args.seed)
             for split, tokens in specifications]
    if not all(path.exists() for path in paths):
        raw = get_reasoning_train_val()
        for (split, tokens), path, dataset in zip(specifications, paths, raw):
            if not path.exists():
                build_chunks(tokenizer, dataset, tokens, args.max_seq_len, rank=0,
                    world_size=1, seed=args.seed, cache_dir=cache / "chunks",
                    split=split, model_name=args.teacher_model, drop_overlong=True)
    # Share cache pages between ranks rather than copying gigabytes into each
    # worker. Trainer/Accelerate, not this dataset, shards the batches.
    return tuple(ChunkDataset(torch.load(path, mmap=True, weights_only=False)) for path in paths)


def main():
    args = parse_args()
    cache = Path(args.cache_dir).resolve()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.fsdp:
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_OFFLOAD_PARAMS"] = "false"
    if args.fsdp and world == 1:
        raise SystemExit("FSDP needs torchrun with at least two processes; use --no-fsdp for a single GPU test")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. On this host, launch outside the filesystem sandbox.")
    fsdp_config = {"version": 2, "cpu_offload": False,
        "cpu_ram_efficient_loading": not bool(args.resume_from_checkpoint), "state_dict_type": "SHARDED_STATE_DICT",
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP", "reshard_after_forward": True,
        "activation_checkpointing": True, "activation_checkpointing_offload": False}
    training_args = TrainingArguments(output_dir=args.output_dir,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant_with_warmup", weight_decay=0.0,
        bf16=True, gradient_checkpointing=not args.fsdp and args.activation_checkpoint_every != 0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=args.fsdp,
        fsdp_config=fsdp_config if args.fsdp else None,
        optim=args.optimizer, remove_unused_columns=False,
        accelerator_config={"gradient_accumulation_kwargs": {"sync_each_batch": args.sync_each_batch}},
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps" if args.save_checkpoints else "no", save_steps=args.save_steps, save_total_limit=1,
        logging_steps=1, report_to="none", prediction_loss_only=True,
        seed=args.seed, dataloader_pin_memory=True)
    set_seed(args.seed)
    if args.compile_blocks:
        from quantizers.nvfp4 import use_exact_quant_op
        use_exact_quant_op()
    if args.fsdp:
        from fsdp_compat import install_buffer_loading_fix
        install_buffer_loading_fix()
    rank = training_args.process_index
    metadata_only = args.fsdp and (rank != 0 or bool(args.resume_from_checkpoint))
    if args.smoke_test:
        if metadata_only:
            from accelerate import init_empty_weights
            with init_empty_weights(include_buffers=False):
                student, teacher = tiny_models()
        else:
            student, teacher = tiny_models()
        # A local dense decoder fixture makes all frozen tensor sources explicit.
        decode = cache / "tiny-decoder"
        with training_args.main_process_first():
            if rank == 0 and not args.resume_from_checkpoint:
                from safetensors.torch import save_file
                decode.mkdir(parents=True, exist_ok=True)
                save_file({k: v.clone() for k, v in student.state_dict().items()}, str(decode / "model.safetensors"))
        prepare_student(student, decode, "model.", load_weights=not metadata_only,
                        cache_quantized_weights=args.cache_quantized_weights,
                        weight_cache_max_gib=args.weight_cache_max_gib)
        g = torch.Generator().manual_seed(args.seed)
        chunks = []
        for _ in range(max(16, world * args.batch_size * args.gradient_accumulation_steps * 2)):
            ids = torch.randint(1, 256, (32,), generator=g)
            labels = ids.clone()
            labels[:8] = -100
            chunks.append((ids, labels, torch.ones_like(ids)))
        train_data, val_data, tokenizer = ChunkDataset(chunks), ChunkDataset(chunks[:world * 2]), None
    else:
        hf_cache = HF_HUB_CACHE
        with training_args.main_process_first():
            decode = None if args.resume_from_checkpoint else resolve_decoder(args, hf_cache)
        print(f"[rank {rank}] {'building student metadata' if metadata_only else 'loading FP32 student from dequantized FP8 teacher'}", flush=True)
        student = load_text_model(args.teacher_model, True, hf_cache, metadata_only=metadata_only)
        prepare_student(student, decode, args.decode_prefix, load_weights=not metadata_only,
                        cache_quantized_weights=args.cache_quantized_weights,
                        weight_cache_max_gib=args.weight_cache_max_gib)
        print(f"[rank {rank}] {'building FP8 teacher metadata' if metadata_only else 'loading real FP8 teacher'}", flush=True)
        teacher = load_text_model(args.teacher_model, False, hf_cache, metadata_only=metadata_only)
        fp8_count = sum(p.numel() for p in teacher.parameters() if p.dtype == torch.float8_e4m3fn)
        if fp8_count == 0:
            raise RuntimeError("Teacher was dequantized unexpectedly; real FP8 residency is required")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, cache_dir=hf_cache)
        with training_args.main_process_first():
            train_data, val_data = load_datasets(tokenizer, args, cache)
    assert_trainable_prefill_only(student)
    model = Prefiller(student, teacher, args.logit_chunk_size, args.temperature, args.include_prefill_loss)
    if metadata_only:
        allocate_empty_parameters(model)
    if args.fsdp:
        verify_rank_loading(model, rank)
    trainer = PrefillerTrainer(model=model, args=training_args, train_dataset=train_data,
        eval_dataset=val_data, data_collator=collate_chunks, processing_class=tokenizer)
    trainer.qad_checkpoint_every = args.activation_checkpoint_every
    trainer.qad_profile_step = args.profile_step
    if args.fsdp:
        from fsdp_tuning import install_fsdp_tuning
        install_fsdp_tuning(trainer, args.activation_checkpoint_every,
                            args.fsdp_prefetch_depth, args.fsdp_retain_student_layers,
                            args.compile_mode if args.compile_blocks else None)
    from metrics import UtilizationCallback
    trainer.add_callback(UtilizationCallback(trainer, args.bf16_peak_tflops, args.fp8_peak_tflops))
    if args.profile_step:
        from profiling import ProfileStepCallback
        trainer.add_callback(ProfileStepCallback(args.profile_step))
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "run_config.json").write_text(json.dumps(vars(args), indent=2))
        print(f"Training on {world} GPUs; global batch = {world * args.batch_size * args.gradient_accumulation_steps}", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.evaluate()
    trainer.save_state()
    # Trainer normally saves at the final step. Avoid rewriting hundreds of GB.
    final = Path(args.output_dir) / f"checkpoint-{trainer.state.global_step}" / "trainer_state.json"
    needs_save = [not final.is_file() if rank == 0 else None]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(needs_save, src=0)
    if args.save_checkpoints and needs_save[0]:
        trainer._save_checkpoint(trainer.model, trial=None)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
