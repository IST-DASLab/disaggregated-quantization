"""Scoped FSDP2 checkpointing and runtime prefetch controls for Accelerate 1.15."""
from contextlib import contextmanager

import torch


def apply_student_checkpointing(accelerator, model, every, compile_mode=None):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    # Teacher is frozen/no_grad. Checkpoint only student decoder blocks, keeping
    # the routing context supplied by PrefillerTrainer through recomputation.
    if every:
        for index, layer in enumerate(model.student.model.layers):
            if index % every == 0:
                model.student.model.layers[index] = checkpoint_wrapper(layer, preserve_rng_state=False)
    if compile_mode is not None:
        # In-place compilation preserves nn.Module hooks installed later by
        # fully_shard; wrapping in OptimizedModule can bypass those hooks.
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        for stack in (model.student, model.teacher):
            for layer in stack.model.layers:
                layer.compile(backend="inductor", mode=compile_mode, dynamic=False, fullgraph=False)
    return model


def block_units(stack):
    return [getattr(layer, "_checkpoint_wrapped_module", layer) for layer in stack.model.layers]


def configure_prefetch(model, depth, retain_layers):
    from torch.distributed.fsdp import FSDPModule

    for stack in (model.teacher, model.student):
        units = block_units(stack)
        if not all(isinstance(unit, FSDPModule) for unit in units):
            raise RuntimeError("Expected each decoder block to be an FSDP2 unit")
        if depth:
            for index, unit in enumerate(units):
                unit.set_modules_to_forward_prefetch(units[index + 1:index + 1 + depth])
                if stack is model.student:
                    unit.set_modules_to_backward_prefetch(list(reversed(units[max(0, index - depth):index])))
        if stack is model.student:
            for unit in units[:retain_layers]:
                unit.set_reshard_after_forward(False, recurse=False)


@contextmanager
def checkpoint_policy(every, compile_mode=None):
    import accelerate.accelerator as accelerator_module

    original = accelerator_module.fsdp2_apply_ac
    accelerator_module.fsdp2_apply_ac = lambda accelerator, model: apply_student_checkpointing(accelerator, model, every, compile_mode)
    try:
        yield
    finally:
        accelerator_module.fsdp2_apply_ac = original


def install_fsdp_tuning(trainer, every=1, prefetch_depth=0, retain_layers=0, compile_mode=None):
    original = trainer.accelerator._prepare_fsdp2

    def prepare(*args):
        model = next((obj for obj in args if isinstance(obj, torch.nn.Module)), None)
        if model is None:
            return original(*args)
        with checkpoint_policy(every, compile_mode):
            result = original(*args)
        configure_prefetch(model, prefetch_depth, retain_layers)
        if trainer.args.process_index == 0:
            print(f"FSDP2: student checkpoint every={every}, explicit prefetch depth={prefetch_depth}, "
                  f"retain after forward={retain_layers} student blocks", flush=True)
        return result

    trainer.accelerator._prepare_fsdp2 = prepare
