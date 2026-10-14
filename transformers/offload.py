"""Per-unit FSDP2 offload policies, scoped to Accelerate model preparation."""
from contextlib import contextmanager

import torch


def resident_unit(name, teacher, student_layers):
    name = name.replace("._checkpoint_wrapped_module", "")
    if teacher and name.startswith("teacher."):
        return True
    prefix = "student.model.layers."
    if name.startswith(prefix):
        index = name[len(prefix):].split(".", 1)[0]
        return index.isdigit() and int(index) < student_layers
    return False


@contextmanager
def selective_offload(model, teacher=False, student_layers=0):
    """Override individual fully_shard calls without editing installed packages."""
    import torch.distributed.fsdp as fsdp
    from accelerate.utils import fsdp_utils

    original_shard = fsdp.fully_shard
    original_load = fsdp_utils.fsdp2_load_full_state_dict
    resident_params = set()

    def shard(module, **kwargs):
        names = {id(child): name for name, child in model.named_modules()}
        units = module if isinstance(module, list) else [module]
        if all(resident_unit(names.get(id(unit), ""), teacher, student_layers) for unit in units):
            kwargs["offload_policy"] = fsdp.OffloadPolicy()
            for unit in units:
                prefix = names[id(unit)]
                for local, _ in unit.named_parameters():
                    resident_params.add(f"{prefix}.{local}".replace("._checkpoint_wrapped_module", ""))
        return original_shard(module, **kwargs)

    def load(accelerator, wrapped, full_sd, cpu_offload=False):
        original_load(accelerator, wrapped, full_sd, cpu_offload=cpu_offload)
        # Accelerate's rank-zero loader has a global CPU-offload switch. Correct
        # placement for the resident units before FSDP performs its lazy init.
        updates = {name: value.to(accelerator.device) for name, value in wrapped.state_dict().items()
                   if name.replace("._checkpoint_wrapped_module", "") in resident_params}
        wrapped.load_state_dict(updates, strict=False, assign=True)

    fsdp.fully_shard = shard
    fsdp_utils.fsdp2_load_full_state_dict = load
    try:
        yield
    finally:
        fsdp.fully_shard = original_shard
        fsdp_utils.fsdp2_load_full_state_dict = original_load


def install_selective_offload(trainer, teacher=False, student_layers=0):
    original = trainer.accelerator._prepare_fsdp2

    def prepare(*args):
        model = next((obj for obj in args if isinstance(obj, torch.nn.Module)), None)
        if model is None:
            return original(*args)
        with selective_offload(model, teacher, student_layers):
            result = original(*args)
        devices = {}
        for param in model.parameters():
            local = param.to_local() if hasattr(param, "to_local") else param
            key = local.device.type
            devices[key] = devices.get(key, 0) + local.numel() * local.element_size()
        rank = trainer.args.process_index
        print(f"[rank {rank}] local parameter shards GiB: "
              f"{ {device: round(size / 1024**3, 3) for device, size in devices.items()} }",
              flush=True)
        return result

    trainer.accelerator._prepare_fsdp2 = prepare
