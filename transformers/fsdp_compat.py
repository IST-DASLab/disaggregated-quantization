"""Local workaround for Accelerate 1.15's FSDP2 full-state buffer loading.

Its loader assumes every state_dict entry is a DTensor, but quantizers have
persistent observer buffers. Keep the upstream parameter loader and separately
broadcast replicated buffers. No installed packages are modified.
"""
import torch


def install_buffer_loading_fix():
    from accelerate.utils import fsdp_utils
    original = fsdp_utils.fsdp2_load_full_state_dict
    if getattr(original, "_prefiller_buffer_fix", False):
        return

    def load(accelerator, model, full_sd, cpu_offload=False):
        import torch.distributed as dist
        persistent = []
        for module_name, module in model.named_modules():
            for name, value in module.named_buffers(recurse=False):
                if name in module._non_persistent_buffers_set:
                    continue
                fqn = f"{module_name}.{name}" if module_name else name
                fqn = fqn.replace("._checkpoint_wrapped_module", "")
                persistent.append((module, name, fqn, value))
                module._non_persistent_buffers_set.add(name)
        try:
            original(accelerator, model, full_sd, cpu_offload=cpu_offload)
        finally:
            for module, name, fqn, value in persistent:
                buffer = (full_sd[fqn].to(accelerator.device) if accelerator.is_main_process
                          else torch.empty(value.shape, dtype=value.dtype, device=accelerator.device))
                if buffer.numel():
                    dist.broadcast(buffer, src=0)
                module.register_buffer(name, buffer, persistent=True)
    load._prefiller_buffer_fix = True
    fsdp_utils.fsdp2_load_full_state_dict = load
