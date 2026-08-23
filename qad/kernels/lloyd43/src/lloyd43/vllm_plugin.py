"""Serve lloyd43 in vLLM: `--quantization lloyd43` on any bf16 checkpoint.

    from lloyd43 import vllm_plugin          # registers the config as a side effect
    llm = LLM(model="Qwen/Qwen3-8B", quantization="lloyd43")

or, without touching the caller, `VLLM_PLUGINS=...` / `--quantization lloyd43` once this
module has been imported. `vllm_serve.py` in this directory does the import for you.

QUANTIZES ON THE FLY, ON PURPOSE
--------------------------------
QAD has no packed-lloyd43 checkpoint format yet -- `pack_from_weight` is the only producer
and it works off a live weight tensor (HANDOFF 8). So this method loads an ordinary bf16
checkpoint and packs each linear in `process_weights_after_loading`, then drops the bf16
copy. That is the same shape of thing vLLM's on-the-fly fp8 does, it needs no new
checkpoint format, and it means the numbers below are measured against the *same weights*
the bf16 baseline runs.

The cost is load time (packing is a few seconds per model) and a transient 2x weight
memory during conversion, layer by layer. Neither affects steady-state decode.

WHAT vLLM GIVES US FOR FREE
---------------------------
vLLM already merges q/k/v into one QKVParallelLinear and gate/up into one
MergedColumnParallelLinear, so a decoder layer issues 4 weight GEMMs, not 7. Those fused
shapes are exactly the ones in `cuda_gemv.CUDA_SHAPE_TABLE` -- they were tuned for this.

BATCH 1 IS THE POINT, AND ONLY BATCH 1
--------------------------------------
`apply` dispatches on the token count: M == 1 takes the GEMV kernel, everything else
(prefill, and any batched decode) falls back to dequantizing the weight and running a
dense matmul. That fallback is correct but SLOW -- it materializes the weight it exists to
avoid. This is a decode-latency kernel for single-user serving; do not read throughput
numbers off it at batch > 1 without reading this paragraph first.
"""

from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.parameter import ModelWeightParameter

from .cuda_gemv import auto_config, linear_op, load_extension
from .format import GROUP, LLOYD21, LLOYD43, pack_from_weight

__all__ = ["LloydConfig", "Lloyd43Config", "Lloyd21Config",
           "LloydLinearMethod", "register"]


def register() -> None:
    """vLLM plugin entry point (group `vllm.general_plugins`, see pyproject.toml).

    Registration happens as a side effect of importing this module -- the decorator on
    Lloyd43Config below does it -- so this function only has to exist and be importable.
    vLLM calls it in every process it spawns, which is the point: the EngineCore
    subprocess is where the model is actually built, and it would otherwise have never
    heard of "lloyd43".
    """


class LloydConfig(QuantizationConfig):
    """LUT weight-only quantization, blocks of 16, two-level scaling.

    One class, two registrations: lloyd43 (3-bit) and lloyd21 (2-bit) differ only in
    `FORMAT`, exactly as the kernels do. `--quantization lloyd21` on any bf16 checkpoint.
    """

    FORMAT = LLOYD43

    def __init__(self, skip_modules: list[str] | None = None):
        super().__init__()
        # lm_head stays bf16, matching quantizers.blocked.replace_linears' skip list --
        # so the served model is the one QAD actually trains and evaluates.
        self.skip_modules = skip_modules or ["lm_head"]

    @classmethod
    def get_name(cls) -> str:
        return cls.FORMAT.name

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # __shfl_sync + fp8 e4m3 conversion; sm_89 is the honest floor for the fp8
        # intrinsic. Only ever measured on sm_121 (GB10).
        return 89

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # on-the-fly: nothing to read from the checkpoint

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Lloyd43Config":
        return cls(skip_modules=config.get("skip_modules"))

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if any(s in prefix for s in self.skip_modules):
            return None
        return LloydLinearMethod(self)


class LloydLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: LloydConfig):
        self.quant_config = quant_config
        self.fmt = quant_config.FORMAT
        load_extension()  # fail at load time, not mid-decode

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # Allocate the bf16 weight the checkpoint loader expects; it is replaced by the
        # packed form in process_weights_after_loading.
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition,
                             dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        N, K = w.shape
        if K % GROUP:
            # Leave it dense rather than silently producing a wrong layout.
            layer.lloyd43_packed = None
            return

        dev = w.device
        packed, bscale, gscale = pack_from_weight(w.float(), fmt=self.fmt)
        # Every buffer must land on the weight's device. A stray CPU tensor here does not
        # fail loudly -- it fails as "Cannot copy between CPU and CUDA tensors during CUDA
        # graph capture", thrown from inside inductor's generated code, which points
        # nowhere near this file. The LUT is the easy one to get wrong: it is a
        # module-level CPU constant in format.py.
        bufs = {
            "lloyd43_packed": packed.to(dev).contiguous(),
            "lloyd43_bscale_u8": bscale.to(dev).contiguous().view(torch.uint8),
            "lloyd43_gscale": gscale.to(dev, torch.float32).reshape(()).contiguous(),
        }
        for name, t in bufs.items():
            if t.device.type != dev.type:
                raise RuntimeError(f"lloyd43: {name} landed on {t.device}, want {dev}")
            layer.register_buffer(name, t)
        layer.lloyd43_cfg = auto_config(N, K, self.fmt)
        layer.lloyd43_K = K
        layer.lloyd43_N = N
        # Drop the dense copy -- holding it would defeat the entire point.
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device),
                                          requires_grad=False)
        del w
        torch.cuda.empty_cache()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(layer, "lloyd43_packed", None) is None:
            raise RuntimeError("lloyd43: layer was not packed (K not a multiple of 32)")

        K, N = layer.lloyd43_K, layer.lloyd43_N
        out = linear_op(x.reshape(-1, K), layer.lloyd43_packed, layer.lloyd43_bscale_u8,
                        layer.lloyd43_gscale, K, layer.lloyd43_cfg)
        out = out.reshape(*x.shape[:-1], N)
        return out if bias is None else out + bias


@register_quantization_config("lloyd43")
class Lloyd43Config(LloydConfig):
    """3 bits, 8 levels, 0.4375 bytes per weight."""

    FORMAT = LLOYD43


@register_quantization_config("lloyd21")
class Lloyd21Config(LloydConfig):
    """2 bits, 4 levels, 0.3125 bytes per weight, 1.4x less traffic than lloyd43."""

    FORMAT = LLOYD21
