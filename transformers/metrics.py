"""Wall-clock utilization estimates for the dual-path QAD training workload."""
import time

import torch
from transformers import TrainerCallback


class UtilizationCallback(TrainerCallback):
    def __init__(self, trainer, bf16_peak_tflops=2250.0, fp8_peak_tflops=4500.0):
        from quantizers.frozen_decode import NVFP4FrozenDecodeLinear
        self.trainer = trainer
        model = trainer.model
        self.config = model.config
        self.base_params = sum(p.numel() for name, p in model.student.named_parameters()
            if not name.endswith(("decode_frozen", "weight_decode")) and ".inner." not in name)
        self.dual = sum(m.weight.numel() for m in model.student.modules()
                        if isinstance(m, NVFP4FrozenDecodeLinear))
        self.shared = sum(m.weight.numel() for m in model.student.model.modules()
                          if isinstance(m, torch.nn.Linear) and not isinstance(m, NVFP4FrozenDecodeLinear))
        every = getattr(trainer, "qad_checkpoint_every", 1)
        checkpointed = [layer for index, layer in enumerate(model.student.model.layers)
                        if every and index % every == 0]
        self.checkpointed_dual = sum(m.weight.numel() for layer in checkpointed for m in layer.modules()
                                    if isinstance(m, NVFP4FrozenDecodeLinear))
        self.checkpointed_shared = sum(m.weight.numel() for layer in checkpointed for m in layer.modules()
                                      if isinstance(m, torch.nn.Linear) and not isinstance(m, NVFP4FrozenDecodeLinear))
        self.checkpointed_full_layers = sum(t == "full_attention" for index, t in enumerate(self.config.layer_types)
                                           if every and index % every == 0)
        teacher_linears = [m for m in model.teacher.model.modules() if isinstance(m, torch.nn.Linear)]
        self.teacher_fp8 = sum(m.weight.numel() for m in teacher_linears if m.weight.dtype == torch.float8_e4m3fn)
        self.teacher_bf16 = sum(m.weight.numel() for m in teacher_linears if m.weight.dtype != torch.float8_e4m3fn)
        self.head = model.student.lm_head.weight.numel()
        self.bf16_peak, self.fp8_peak = bf16_peak_tflops * 1e12, fp8_peak_tflops * 1e12
        self.first_step = True

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.trainer._step_token_counts = [0, 0, 0, 0]

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        counts = torch.tensor(self.trainer._step_token_counts, device=args.device, dtype=torch.float64)
        timing = torch.tensor([elapsed, torch.cuda.max_memory_allocated() / 1024**3],
                              device=args.device, dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
        padded, real, scored, attention_square = counts.tolist()
        seconds, peak_gib = timing.tolist()
        world = args.world_size
        # Standard 6N student proxy; QAD performs more work than ordinary training.
        padded_mfu = 6 * self.base_params * padded / (world * self.bf16_peak * seconds)
        real_mfu = 6 * self.base_params * real / (world * self.bf16_peak * seconds)
        # Trainable branch 6N + frozen branch 4N; checkpointing adds 4N.
        # Shared frozen linears cost 4N, plus 2N when checkpointed.
        full_layers = sum(t == "full_attention" for t in self.config.layer_types)
        causal_attention = 2 * attention_square * full_layers * self.config.num_attention_heads * self.config.head_dim
        bf16_work = (10 * self.dual + 4 * self.checkpointed_dual + 4 * self.shared
                     + 2 * self.checkpointed_shared + 2 * self.teacher_bf16) * padded
        attention_recompute = 2 * attention_square * self.checkpointed_full_layers * self.config.num_attention_heads * self.config.head_dim
        bf16_work += 10 * self.head * scored + 4 * causal_attention + attention_recompute
        fp8_work = 2 * self.teacher_fp8 * padded
        ideal_seconds = bf16_work / (world * self.bf16_peak) + fp8_work / (world * self.fp8_peak)
        # Logging here would reset DefaultFlowCallback.should_log before Trainer
        # records the loss. Merge into Trainer's normal log instead.
        self.trainer._utilization_metrics = {"step_seconds": seconds, "tokens_per_second": real / seconds,
            "mfu_student_6n_padded": padded_mfu, "mfu_student_6n_real": real_mfu,
            "hfu_qad_execution_estimate": ideal_seconds / seconds,
            "peak_gpu_allocated_gib": peak_gib, "cold_step": self.first_step,
            "profiled_step": state.global_step == getattr(self.trainer, "qad_profile_step", 0)}
        self.first_step = False
        control.should_log = True
