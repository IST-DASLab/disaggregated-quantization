"""Capture one warmed optimizer step, separately from throughput measurements."""
import json
from pathlib import Path

import torch
from transformers import TrainerCallback


class ProfileStepCallback(TrainerCallback):
    def __init__(self, step):
        self.step = step
        self.profiler = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step + 1 == self.step and args.process_index == 0:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != self.step:
            return
        if self.profiler is not None:
            self.profiler.stop()
            output = Path(args.output_dir) / "profile"
            output.mkdir(parents=True, exist_ok=True)
            self.profiler.export_chrome_trace(str(output / f"rank0-step{self.step}.json"))
            events = self.profiler.key_averages()
            (output / "operators.txt").write_text(events.table(sort_by="self_device_time_total", row_limit=60))
            (output / "operators.json").write_text(json.dumps([
                {"operator": event.key, "count": event.count,
                 "self_cpu_us": event.self_cpu_time_total,
                 "self_device_us": event.self_device_time_total,
                 "device_us": event.device_time_total} for event in events], indent=2))
            self.profiler = None
        # Keep trace-export time out of the following step on every rank.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
