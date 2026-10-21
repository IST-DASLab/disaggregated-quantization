"""Validate and synchronize the replicated NVFP4 activation observers."""
import torch
import torch.distributed as dist
from transformers import TrainerCallback


@torch.no_grad()
def activation_observers(model, *, synchronize=False, reconcile=False):
    """Synchronize BEFORE DCP deduplicates buffers; optionally repair legacy groups.

    Reconciliation takes the maximum of the surviving observations. It cannot
    recover rank-local values that an old distributed checkpoint never wrote.
    Weight tensors and weight quantization scales are never changed.
    """
    from quantizers.frozen_decode import NVFP4FrozenDecodeLinear

    layers = [(name, m) for name, m in model.named_modules()
              if isinstance(m, NVFP4FrozenDecodeLinear)]
    if not layers:
        return []
    values = torch.stack([m.act_amax.detach().reshape(()) for _, m in layers])
    if synchronize and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    indices = {id(m): i for i, (_, m) in enumerate(layers)}
    changes = []
    for name, module in layers:
        group = module._group or [module]
        positions = [indices[id(member)] for member in group]
        observed = values[positions]
        if not torch.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"Invalid activation observer at {name}")
        if not torch.equal(observed, observed[0].expand_as(observed)):
            if not reconcile:
                raise ValueError(f"Inconsistent fused activation observers at {name}; "
                                 "legacy FSDP checkpoints require explicit reconciliation")
            target = observed.max()
            for index in positions:
                changes.append({"name": layers[index][0], "old_amax": float(values[index]),
                                "new_amax": float(target)})
                values[index] = target
    for (_, module), value in zip(layers, values):
        module.act_amax.copy_(value)
        module._observed = bool(value > 0)
    return changes


class StaticActivationCalibration(TrainerCallback):
    """Calibrate once on all ranks after FSDP setup, then freeze global scales."""

    def __init__(self, trainer, batches=8, resume=False, recalibrate=False):
        self.trainer, self.batches = trainer, batches
        self.resume, self.recalibrate = resume, recalibrate

    @torch.no_grad()
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from quantizers.frozen_decode import NVFP4FrozenDecodeLinear
        layers = [m for m in model.modules() if isinstance(m, NVFP4FrozenDecodeLinear)]
        if not self.resume or self.recalibrate:
            # Explicit recalibration starts a new calibration contract. Ordinary
            # resume retains exactly the fixed scales saved in its checkpoint.
            for module in layers:
                module.act_amax.zero_()
                module._observed = False
                module.calibrating = True
            was_training = model.training
            model.eval()
            try:
                for index, batch in enumerate(self.trainer.get_eval_dataloader()):
                    if index >= self.batches:
                        break
                    model(**self.trainer._prepare_inputs(batch))
            finally:
                for module in layers:
                    module.calibrating = False
                model.train(was_training)
        activation_observers(model, synchronize=True, reconcile=True)
        for module in layers:
            if not bool(module.act_amax > 0):
                raise ValueError("Static activation calibration left an unobserved projection")
            module.observer_frozen = True
        if self.trainer.is_world_process_zero():
            print("NVFP4 input globals calibrated, synchronized and frozen for training/export", flush=True)
