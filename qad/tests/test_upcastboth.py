import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from quantizers import REGISTRY, build_quantizer_params, variants
from quantizers.dual import (NVFP4Lloyd21UpcastBothLinear, NVFP4Lloyd43UpcastBothLinear,
                             prefill_mask_from_labels, quant_phase)
from quantizers.nvfp4 import unpack_nvfp4_weight

NAMES = ["nvfp4lloyd43upcastboth", "nvfp4lloyd21upcastboth"]


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def build(name):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, False
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).cuda().float()
    params, _ = build_quantizer_params(name, "")
    REGISTRY[name]["apply"](m, **params)
    for mod in m.modules():
        if hasattr(mod, "act_amax"):
            mod.act_amax.fill_(3.5); mod._observed = True
    return m


def test_registry_says_single_checkpoint():
    for n in NAMES:
        check(f"{n}: one variant", variants(n) == [None], f"{variants(n)}")
        check(f"{n}: exports compressed_tensors",
              REGISTRY[n]["export"] == "compressed_tensors")


def test_both_phases_get_the_same_weight():
    """The defining property: no phase branch, so the mask changes nothing."""
    for n, cls in zip(NAMES, (NVFP4Lloyd43UpcastBothLinear, NVFP4Lloyd21UpcastBothLinear)):
        m = build(n)
        lin = m.model.layers[0].mlp.gate_proj
        check(f"{n}: layer class", isinstance(lin, cls))
        check(f"{n}: prefill and decode ship one tensor",
              lin._variant_weight("prefill") is lin._variant_weight("decode"))
        check(f"{n}: activations quantized on both phases",
              lin._variant_quantize_act("prefill") and lin._variant_quantize_act("decode"))

        B, T, K = 2, 8, m.config.hidden_size
        x = torch.randn(B, T, K, device="cuda")
        labels = torch.full((B, T), -100, device="cuda"); labels[:, T // 2:] = 1
        mask = prefill_mask_from_labels(labels)
        with quant_phase(mask):
            y_mixed = lin(x)
        with quant_phase(torch.ones_like(mask)):
            y_pre = lin(x)
        with quant_phase(~torch.ones_like(mask)):
            y_dec = lin(x)
        check(f"{n}: output does not depend on the phase mask",
              torch.equal(y_mixed, y_pre) and torch.equal(y_mixed, y_dec))


def test_weight_is_the_upcast_of_the_lut_weight():
    """It must still be LUT-constrained -- otherwise it is just plain nvfp4."""
    for n in NAMES:
        m = build(n)
        lin = m.model.layers[0].mlp.gate_proj
        # The LUT source is RECOMPUTED, not stored. upcastboth never serves it (forward
        # and export both use _wq), and a second full-size fp32 buffer costs ~43 GB at
        # gemma-3-12b, which OOMed those runs. Asserting the buffer's ABSENCE keeps that
        # from being reintroduced; the property that actually matters -- that the served
        # weight is LUT-constrained -- is asserted below, independently of storage.
        check(f"{n}: no persistent LUT buffer", not hasattr(lin, "_wq_dec"))
        src = lin._upcast_source()
        levels = torch.unique(src).numel()
        want = 8 if "43" in n else 4
        check(f"{n}: LUT source is {want}-level per block",
              levels <= want * src.shape[0] * 2, f"{levels} distinct values")
        check(f"{n}: served weight differs from the LUT source (it is upcast)",
              not torch.equal(lin._wq, src))
        # The invariant the docstring is really about, and the one the old buffer check
        # never made: the served weight must NOT equal a plain NVFP4 quantization of the
        # raw master. If it did, this format would be indistinguishable from nvfp4 and
        # every comparison it exists to support would be vacuous.
        from quantizers.blocked import blocked_quantize
        plain = blocked_quantize(lin.weight, lin.rounder, lin.block_size,
                                 signed=lin.signed,
                                 global_scale=lin.group_global_scale())[0]
        check(f"{n}: served weight is NOT plain NVFP4 of the master",
              not torch.equal(lin._wq, plain))


def test_export_is_one_checkpoint_and_round_trips():
    for n in NAMES:
        m = build(n)
        lin = m.model.layers[0].self_attn.q_proj
        check(f"{n}: export_variants", type(lin).export_variants() == [None])
        t = lin.export_tensors(None)
        check(f"{n}: exports packed NVFP4", "weight_packed" in t)
        check(f"{n}: carries an input scale (W4A4)", "input_global_scale" in t)
        check(f"{n}: has a quantization config", lin.export_config(None) is not None)
        w = unpack_nvfp4_weight(t["weight_packed"], t["weight_scale"],
                                t["weight_global_scale"]).cuda().float()
        ref = lin._wq.float()
        rel = float((w - ref).abs().max() / ref.abs().max())
        # BF16 tolerance, and the gap is expected rather than tolerated: the checkpoint
        # is packed from the fp32 MASTER (_variant_weight returns self.weight), so it is
        # exact, while _wq is the bf16 training cache of the same quantization. vLLM
        # dequantizes to bf16 too, so the SERVED value matches the cache; this compares
        # an fp32-exact tensor against its bf16 rounding. bf16 has 8 mantissa bits ->
        # ~4e-3 relative. The fp32-era bound was 1e-5.
        check(f"{n}: packed checkpoint == _wq (to bf16)", rel < 5e-3,
              f"max rel {rel:.2e}")


def test_differs_from_plain_nvfp4():
    """If the LUT bottleneck did nothing, this format would be pointless."""
    import torch.nn as nn
    from quantizers.nvfp4 import NVFP4Linear
    torch.manual_seed(0)
    w = torch.randn(64, 128).cuda()
    lin = nn.Linear(128, 64, bias=False).cuda(); lin.weight.data.copy_(w)
    plain = NVFP4Linear.from_linear(lin, block_size=16, quantize_act=True)
    both = NVFP4Lloyd43UpcastBothLinear.from_linear(lin, block_size=16)
    d = (plain._wq - both._wq).abs().max().item()
    check("upcastboth != plain nvfp4", d > 1e-4, f"max|diff| {d:.4e}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    print("  all upcastboth tests passed")
