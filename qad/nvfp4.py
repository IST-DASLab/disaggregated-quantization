"""NVFP4 (W4A4) fake-quantization + real-checkpoint export for QAD.

NVFP4 is NVIDIA's 4-bit block floating-point format used by Blackwell tensor
cores and served by vLLM via the ModelOpt quantization path. Layout:

  * elements  : E2M1 4-bit floats — representable magnitudes
                {0, .5, 1, 1.5, 2, 3, 4, 6}  (absmax = 6).
  * block     : 16 contiguous elements along the contraction (K / in_features)
                dim share one scale.
  * block scale: stored in FP8 E4M3 (absmax 448).
  * global scale (a.k.a. weight_scale_2 / input_scale): one FP32 scalar per
    tensor that normalises the block scales into the E4M3 range.

Two-level dequant for one block:            w ≈ q_e2m1 * s_block_e4m3 * s_global
with   s_global = amax_tensor / (6 * 448)   (so block scales ≤ 448)
       s_block  = to_e4m3( (amax_block / 6) / s_global )

Both weights AND activations are quantized this way (W4A4). Training uses the
straight-through estimator: forward sees the fake-quantized value, gradients
pass through unchanged to the FP32 master weight.

`build_nvfp4_state_dict` re-derives the exact same scales and emits a real
ModelOpt-format checkpoint (packed uint8 FP4 weights + FP8 block scales + FP32
global scales + per-layer static `input_scale`) that vLLM serves as true W4A4.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from quant import QuantizedLinear

_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_GLOBAL_DEN = _E2M1_MAX * _E4M3_MAX  # 2688
BLOCK = 16

# E2M1 positive levels and the midpoint thresholds used for round-to-nearest.
_E2M1_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]  # midpoints between levels


def _e2m1_round(x: Tensor) -> Tensor:
    """Round to the nearest signed E2M1 magnitude (values ≥5 clamp to 6)."""
    levels = torch.tensor(_E2M1_LEVELS, device=x.device, dtype=x.dtype)
    bounds = torch.tensor(_E2M1_BOUNDS, device=x.device, dtype=x.dtype)
    return torch.sign(x) * levels[torch.bucketize(x.abs(), bounds)]


def _e2m1_codes(x: Tensor) -> Tensor:
    """Map already-normalised values to 4-bit E2M1 codes (0..15):
    bit3 = sign, bits0-2 = magnitude index. code 8 (=-0) collapses to 0."""
    bounds = torch.tensor(_E2M1_BOUNDS, device=x.device, dtype=torch.float32)
    mag = torch.bucketize(x.float().abs(), bounds).to(torch.uint8)  # 0..7
    neg = (x < 0).to(torch.uint8)
    code = mag | (neg << 3)
    return torch.where(mag == 0, torch.zeros_like(code), code)


def _to_e4m3(x: Tensor) -> Tensor:
    """Emulate an FP8 E4M3 cast (grid rounding via a real f8 round-trip)."""
    return x.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def nvfp4_quantize(x: Tensor, block: int = BLOCK,
                   global_scale: Tensor | None = None
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Two-level NVFP4 fake-quant along the last dim.

    Returns (dequantized_x, block_scale_e4m3, global_scale). `global_scale` may be
    supplied (e.g. a calibrated static activation scale) or derived from the
    tensor's own absmax when None.  block_scale_e4m3 is a float32 tensor holding
    E4M3-grid values, shape (..., n_blocks).
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    n_blocks = (K + block - 1) // block
    pad = n_blocks * block - K
    xf = F.pad(x.float(), (0, pad)).reshape(*orig_shape[:-1], n_blocks, block)

    if global_scale is None:
        amax = xf.abs().amax().clamp(min=1e-8)
        global_scale = amax / _GLOBAL_DEN
    global_scale = global_scale.clamp(min=1e-8)

    block_amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    block_scale = _to_e4m3((block_amax / _E2M1_MAX) / global_scale)  # e4m3 grid
    eff = (block_scale * global_scale).clamp(min=1e-8)               # per-block dequant scale

    q = _e2m1_round(xf / eff)
    deq = (q * eff).reshape(*orig_shape[:-1], n_blocks * block)[..., :K].to(x.dtype)
    return deq, block_scale.squeeze(-1), global_scale


def _fake_quant_ste(x: Tensor, block: int, global_scale: Tensor | None) -> Tensor:
    deq, _, _ = nvfp4_quantize(x, block, global_scale)
    return x + (deq - x).detach()


def pack_nvfp4_weight(w: Tensor, block: int = BLOCK,
                      global_scale: Tensor | None = None
                      ) -> tuple[Tensor, Tensor, Tensor]:
    """Encode a weight matrix into the real NVFP4 checkpoint tensors, using the
    SAME scales as nvfp4_quantize so the packed weight is bit-identical to the
    fake-quantized _wq the model was trained with.  `global_scale` may be a shared
    per-fused-group scale (see NVFP4Linear._wgroup_scale) so vLLM's per-shard
    weight_global_scale.max() collapse is a no-op.

    Returns (packed_uint8 [O, K//2], weight_scale float8_e4m3fn [O, K//block],
             weight_scale_2 float32 [1]).
    """
    O, K = w.shape
    assert K % block == 0, f"in_features {K} not divisible by block {block}"
    _, block_scale, global_scale = nvfp4_quantize(w, block, global_scale=global_scale)  # e4m3-grid f32, scalar
    eff = (block_scale.unsqueeze(-1).float() * global_scale).clamp(min=1e-8)  # [O, nB, 1]
    wb = w.float().reshape(O, K // block, block)
    codes = _e2m1_codes(wb / eff).reshape(O, K)                      # [O, K] uint8 0..15
    packed = (codes[:, 1::2] << 4) | codes[:, 0::2]                  # low=even, high=odd
    weight_scale = block_scale.to(torch.float8_e4m3fn)              # [O, K//block]
    weight_scale_2 = global_scale.reshape(1).float()
    return packed.to(torch.uint8).contiguous(), weight_scale.contiguous(), weight_scale_2


class NVFP4Linear(QuantizedLinear):
    """W4A4 NVFP4 fake-quantized linear.

    Weight: NVFP4 with a global scale derived from the master weight, cached in
    _wq and refreshed each optimizer step (STE gradient to the FP32 master).
    Activation: NVFP4 with a STATIC per-tensor global scale from a running-max
    observer (act_amax), so training sees exactly the fixed input_scale vLLM uses
    at inference (per-block e4m3 scales stay dynamic). The observer updates every
    training/calibration forward; the same act_amax is exported as input_scale.
    Before any observation act_amax is 0 → we fall back to a dynamic per-forward
    scale for that first pass (self._observed flips true once amax>0).
    """

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK,
                 quantize_act: bool = True):
        out, in_ = weight.shape
        super().__init__(out, in_, bias, dtype=weight.dtype, device=weight.device)
        self.weight = nn.Parameter(weight.clone())
        self.block_size = block_size
        # W4A4 (True) vs weight-only W4A16 (False: activations stay bf16, and no
        # input_global_scale is exported → vLLM picks CompressedTensorsW4A16Fp4).
        self.quantize_act = quantize_act
        self.calibrating = False
        self._observed = False  # python flag: has act_amax ever been set (>0)?
        self._group = None      # fused-group siblings (set by link_fused_groups)
        # running-max |activation| = the static activation global-scale observer.
        self.register_buffer("act_amax", torch.zeros((), device=weight.device))
        with torch.no_grad():
            self._wq.copy_(self._compute_wq())

    def _wgroup_scale(self) -> Tensor:
        """Weight global scale (= amax/2688). Shared across a fused group (q/k/v,
        gate/up) using the group-MAX amax, so vLLM's weight_global_scale.max()
        collapse over the fused shards reproduces exactly this scale (no weight
        corruption). Non-fused layers use their own amax."""
        members = self._group if (self._group and len(self._group) > 1) else [self]
        amax = max(m.weight.detach().float().abs().amax() for m in members)
        return (amax / _GLOBAL_DEN).clamp(min=1e-8)

    @torch.no_grad()
    def _compute_wq(self) -> Tensor:
        deq, _, _ = nvfp4_quantize(self.weight, self.block_size,
                                   global_scale=self._wgroup_scale())
        return deq

    def _differentiable_weight(self) -> Tensor:
        # STE: forward uses the cached hard-quant weight, grad flows to master.
        return self.weight + (self._wq - self.weight).detach()

    def forward(self, x: Tensor) -> Tensor:
        w = self._wq if not self.training else self._differentiable_weight()
        if not self.quantize_act:
            return F.linear(x, w, self.bias)   # W4A16: activations stay bf16
        if self.training or self.calibrating:
            with torch.no_grad():
                self.act_amax = torch.maximum(
                    self.act_amax, x.detach().float().abs().amax())
            self._observed = True
        # Static observed activation scale (matches export/inference). The running
        # max is ≥ this batch's amax, so activations never saturate; falls back to
        # a dynamic per-forward scale only until the observer has seen data.
        gscale = (self.act_amax / _GLOBAL_DEN) if self._observed else None
        xq = _fake_quant_ste(x, self.block_size, global_scale=gscale)
        return F.linear(xq, w, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "NVFP4Linear":
        return cls(linear.weight.data, linear.bias, **kwargs)


_FUSED_GROUPS = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]


def link_fused_groups(model: nn.Module) -> None:
    """Link the NVFP4Linears that vLLM fuses (q/k/v→qkv_proj, gate/up→gate_up_proj)
    so they share one weight global scale. vLLM collapses the per-projection
    weight_global_scale via .max() after loading; sharing the scale (group-max amax)
    makes that a no-op instead of corrupting the smaller-amax projections."""
    for parent in model.modules():
        children = dict(parent.named_children())
        for group in _FUSED_GROUPS:
            members = [children[g] for g in group
                       if isinstance(children.get(g), NVFP4Linear)]
            if len(members) >= 2:
                for m in members:
                    m._group = members
    for m in model.modules():
        if isinstance(m, NVFP4Linear):
            if m._group is None:
                m._group = [m]
            with torch.no_grad():          # refresh _wq with the (now shared) scale
                m._wq.copy_(m._compute_wq())


def apply_nvfp4(model: nn.Module, **kwargs) -> None:
    """Replace every nn.Linear (except lm_head) with NVFP4Linear, in-place, and
    link fused groups so weight global scales are shared (matches vLLM). W4A4."""
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, NVFP4Linear.from_linear(module, **kwargs))
    link_fused_groups(model)


def apply_nvfp4a16(model: nn.Module, **kwargs) -> None:
    """Weight-only NVFP4 (W4A16): 4-bit NVFP4 weights, activations left in bf16."""
    apply_nvfp4(model, quantize_act=False, **kwargs)


# ---------------------------------------------------------------------------
# Activation calibration (static input_scale for W4A4 export)
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_nvfp4(model: nn.Module, chunks, device, n_batches: int = 8,
                    batch_size: int = 4) -> None:
    """Record per-layer activation absmax over a few batches so the exporter can
    write a static input_scale. Safe to call on a single rank (no DDP sync)."""
    mods = [m for m in model.modules() if isinstance(m, NVFP4Linear)]
    if not mods:
        return
    was_training = model.training
    model.eval()
    # Do NOT zero act_amax: the running-max observer already holds the training-time
    # activation ranges; calibration only augments it with val-set ranges (max), so
    # the exported static input_scale never underestimates and clips at inference.
    for m in mods:
        m.calibrating = True
    n = min(n_batches * batch_size, len(chunks))
    for i in range(0, n, batch_size):
        batch = chunks[i:i + batch_size]
        if not batch:
            break
        ids = torch.stack([b[0] for b in batch]).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=ids)
    for m in mods:
        m.calibrating = False
    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# Real NVFP4 checkpoint export (compressed-tensors `nvfp4-pack-quantized`)
# ---------------------------------------------------------------------------
# vLLM auto-detects this from config.json's quantization_config (quant_method
# "compressed-tensors") and serves it with CompressedTensorsW4A4Fp4 — true W4A4
# on Blackwell — with NO forced --quantization flag. (Both this and the ModelOpt
# format were verified to load as W4A4 on this container; we use compressed-tensors
# because it's fully self-describing/auto-detected.) Per linear vLLM reads:
#   weight_packed        uint8      [O, K//2]   packed FP4 (== our packed weight)
#   weight_scale         fp8_e4m3   [O, K//16]  per-block scale (== our block scale)
#   weight_global_scale  fp32 [1]   = 2688/amax        (RECIPROCAL of ModelOpt's)
#   input_global_scale   fp32 [1]   = 2688/act_amax    (W4A4 activation global)
# dequant = e2m1 * weight_scale / weight_global_scale.
_CT_GRP = {
    "actorder": None, "block_structure": None, "group_size": BLOCK,
    "num_bits": 4, "observer": "minmax", "observer_kwargs": {},
    "strategy": "tensor_group", "symmetric": True, "type": "float",
}


def ct_quant_config(quantize_act: bool = True) -> dict:
    """compressed-tensors quantization_config for NVFP4 (group_size 16).

    W4A4 (quantize_act=True): input_activations.dynamic="local" → static per-tensor
    input_global_scale with dynamic per-block FP4 activation quant; vLLM selects
    CompressedTensorsW4A4Fp4. Field set matches a real RedHatAI NVFP4 W4A4 checkpoint.

    W4A16 (quantize_act=False): input_activations=None — vLLM's _is_fp4a16_nvfp4()
    keys on `input_quant is None` to select the weight-only CompressedTensorsW4A16Fp4,
    which registers NO input_global_scale param (so we must not emit that tensor).

    `quant_method` and the absence of a `weight_shape` tensor are both required.
    """
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {**_CT_GRP, "dynamic": "local"} if quantize_act else None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {**_CT_GRP, "dynamic": False},
            }
        },
        "format": "nvfp4-pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def build_nvfp4_state_dict(student: nn.Module) -> dict[str, Tensor]:
    """Emit a compressed-tensors nvfp4-pack-quantized state dict: per NVFP4Linear
    weight_packed (uint8), weight_scale (fp8 e4m3 block scales), weight_global_scale
    and input_global_scale (fp32 reciprocal per-tensor globals, 2688/amax); nothing
    named weight_shape (that breaks vLLM's loader). Everything else kept native."""
    quant_mods = {name for name, m in student.named_modules()
                  if isinstance(m, NVFP4Linear)}
    out: dict[str, Tensor] = {}
    for name, m in student.named_modules():
        if not isinstance(m, NVFP4Linear):
            continue
        gscale = m._wgroup_scale()          # shared across fused group (matches vLLM .max())
        packed, wscale, wscale2 = pack_nvfp4_weight(m.weight.data, m.block_size,
                                                    global_scale=gscale)
        out[f"{name}.weight_packed"] = packed.cpu()
        out[f"{name}.weight_scale"] = wscale.cpu()
        out[f"{name}.weight_global_scale"] = (1.0 / wscale2).reshape(1).cpu()  # 2688/group_amax
        if m.quantize_act:   # W4A4 only — the W4A16 scheme registers no such param
            act = m.act_amax.float().clamp(min=1e-8)
            out[f"{name}.input_global_scale"] = (_GLOBAL_DEN / act).reshape(1).cpu()  # 2688/act_amax
        if m.bias is not None:
            out[f"{name}.bias"] = m.bias.data.to(torch.bfloat16).cpu()
    # non-quantized tensors (embed/norm/lm_head/rotary + dropped quant internals)
    for key, tensor in student.state_dict().items():
        parent, _, _ = key.rpartition(".")
        if parent in quant_mods:
            continue  # weight master, _wq, act_amax, bias handled above
        out[key] = tensor.detach().cpu()
    return out


def save_nvfp4_checkpoint(student: nn.Module, out_dir: Path) -> int:
    """Write a real compressed-tensors NVFP4 checkpoint dir (W4A4 or W4A16, matching
    how the model was quantized). Returns tensor count."""
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_nvfp4_state_dict(student)
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    student.config.save_pretrained(out_dir)
    if getattr(student, "generation_config", None) is not None:
        student.generation_config.save_pretrained(out_dir)
    # Self-contained: embed the quantization_config so vLLM auto-detects (no flag).
    quantize_act = next(m.quantize_act for m in student.modules()
                        if isinstance(m, NVFP4Linear))
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = ct_quant_config(quantize_act)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return len(state)
