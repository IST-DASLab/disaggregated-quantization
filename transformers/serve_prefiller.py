"""Serve real NVFP4 while preserving each trained projection's global scales.

vLLM normally collapses fused projections' global scales with max(). Split only
incompatible groups into separate native NVFP4 GEMMs, then concatenate outputs.
No weights or scales are requantized. BF16 decoder execution stays native.
This entry point installs the same patch in spawned engine workers.
"""
import torch
from torch import nn

from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4


def projection_groups(widths, weights, activations):
    """Maximal adjacent groups sharing BOTH original tensor-global scales."""
    if not (len(widths) == len(weights) == len(activations)):
        raise ValueError("Projection widths and scales disagree")
    groups = []
    offset = 0
    for width, weight, activation in zip(widths, weights, activations):
        if width <= 0 or weight <= 0 or activation <= 0:
            raise ValueError("Invalid projection width or scale")
        if groups and groups[-1][2:] == (weight, activation):
            start, _, _, _ = groups[-1]
            groups[-1] = (start, offset + width, weight, activation)
        else:
            groups.append((offset, offset + width, weight, activation))
        offset += width
    return groups


def install():
    cls = CompressedTensorsW4A4Fp4
    if getattr(cls, "_qad_preserves_scales", False):
        return
    original_process = cls.process_weights_after_loading
    original_apply = cls.apply_weights

    def process(self, layer):
        widths = layer.logical_widths
        weights = layer.weight_global_scale.detach().float().tolist()
        activations = ([1.0] * len(widths) if self.use_a16 else
                       layer.input_global_scale.detach().float().tolist())
        groups = projection_groups(widths, weights, activations)
        if len(groups) == 1:
            return original_process(self, layer)
        children = nn.ModuleList()
        for start, end, weight, activation in groups:
            child = nn.Module()
            child.logical_widths = [end - start]
            child.input_size_per_partition = layer.input_size_per_partition
            child.output_size_per_partition = end - start
            child.weight_packed = nn.Parameter(layer.weight_packed[start:end].detach().contiguous(), requires_grad=False)
            child.weight_scale = nn.Parameter(layer.weight_scale[start:end].detach().contiguous(), requires_grad=False)
            child.weight_global_scale = nn.Parameter(torch.tensor([weight], device=layer.weight_global_scale.device), requires_grad=False)
            if not self.use_a16:
                child.input_global_scale = nn.Parameter(torch.tensor([activation], device=layer.input_global_scale.device), requires_grad=False)
            child.scheme = cls(use_a16=self.use_a16)
            original_process(child.scheme, child)
            children.append(child)
        layer._qad_projection_parts = children
        layer.requires_unquantized_input = True
        for name in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
            if hasattr(layer, name):
                delattr(layer, name)
        print(f"QAD NVFP4: preserved {len(widths)} projection scales using {len(groups)} native GEMMs", flush=True)

    def apply(self, layer, x, bias=None):
        if not hasattr(layer, "_qad_projection_parts"):
            return original_apply(self, layer, x, bias)
        outputs = []
        offset = 0
        for child in layer._qad_projection_parts:
            end = offset + child.output_size_per_partition
            outputs.append(original_apply(child.scheme, child, x,
                None if bias is None else bias[offset:end]))
            offset = end
        return torch.cat(outputs, dim=-1)

    cls.process_weights_after_loading = process
    cls.apply_weights = apply
    cls._qad_preserves_scales = True
    print("QAD serving: preserve all trained NVFP4 weight/activation global scales", flush=True)


install()

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    main()
