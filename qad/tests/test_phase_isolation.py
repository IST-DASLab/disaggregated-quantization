"""Gates for the phase-isolation ablation formats.

    nvfp4decode : BF16 prefill, NVFP4 (W4A4) decode
    nvfp4prefill  : NVFP4 (W4A4) prefill, BF16 decode

They exist to answer "which phase does quantization hurt more?", which is only a
valid question if the two arms differ in NOTHING but the phase that pays. Three
claims are defended here:

 1. The BF16 half is EXACT. Its forward is the untouched master weight, and its
    exported directory is a plain HF bf16 model with no quantization_config -- if
    any rounding leaked in, the ablation would be measuring two quantized phases.
 2. The NVFP4 half is an ordinary `nvfp4` checkpoint, packed and carrying a static
    input_global_scale, so vLLM serves it on its own worker.
 3. The activation observer sees ONLY the quantized phase. The static scale is baked
    into that half's checkpoint; folding the BF16 phase's activations into the
    running max would inflate it and cost precision where it is actually used. This
    is the failure that unmasked calibration used to cause.

    python tests/test_phase_isolation.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from export.save import load_into, save_checkpoint
from quantizers import (REGISTRY, build_quantizer_params, prefill_mask_from_labels,
                        variants,
                        quant_phase)
from quantizers.dual import (BF16PrefillNVFP4DecodeLinear, NVFP4PrefillBF16DecodeLinear)

TMP = Path("/tmp/qad_phase_iso_test")

# name -> (which phase runs NVFP4, which runs BF16)
FORMATS = {
    "nvfp4decode": ("decode", "prefill"),
    "nvfp4prefill":  ("prefill", "decode"),
}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def build(name):
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
            mod.act_amax.fill_(3.5)
            mod._observed = True
    return m


def _layer(name):
    m = build(name)
    return m, m.model.layers[0].mlp.gate_proj


def test_registry_wiring():
    """The two formats must be registered as dual (two variants) and packed."""
    for name, (nv_phase, _) in FORMATS.items():
        e = REGISTRY[name]
        check(f"{name}: two variants", e.get("variants") == ["prefill", "decode"])
        check(f"{name}: exports compressed-tensors", e["export"] == "compressed_tensors")
        m, lin = _layer(name)
        check(f"{name}: NVFP4 phase is {nv_phase}", lin.NVFP4_PHASE == nv_phase,
              f"got {lin.NVFP4_PHASE}")
    check("classes are distinct",
          BF16PrefillNVFP4DecodeLinear is not NVFP4PrefillBF16DecodeLinear)


def test_bf16_phase_is_exact():
    """The BF16 phase must be the untouched master -- bit-exact against F.linear.

    If this drifts, both phases are quantized and the ablation compares nothing.
    """
    for name, (nv_phase, bf_phase) in FORMATS.items():
        m, lin = _layer(name)
        m.eval()
        x = torch.randn(2, 6, lin.in_features, device="cuda") * 0.5
        allbf = torch.full((2, 6), bf_phase == "prefill", dtype=torch.bool, device="cuda")
        allnv = ~allbf
        with quant_phase(allbf):
            y_bf = lin(x)
        with quant_phase(allnv):
            y_nv = lin(x)
        exact = F.linear(x, lin.weight, lin.bias)
        check(f"{name}: {bf_phase} half is exact BF16",
              torch.equal(y_bf, exact),
              f"max|diff| {(y_bf - exact).abs().max().item():.2e}")
        check(f"{name}: {nv_phase} half is actually quantized",
              (y_nv - exact).abs().max().item() > 1e-3,
              f"max|diff| {(y_nv - exact).abs().max().item():.2e}")


def test_phase_routing():
    """A mixed mask must select per POSITION: BF16 rows exact, NVFP4 rows quantized."""
    for name, (nv_phase, bf_phase) in FORMATS.items():
        m, lin = _layer(name)
        m.eval()
        x = torch.randn(2, 8, lin.in_features, device="cuda") * 0.5
        prefill = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
        prefill[:, :4] = True                       # first half prompt, second half generated
        with quant_phase(prefill):
            y = lin(x)
        exact = F.linear(x, lin.weight, lin.bias)
        bf_rows = slice(0, 4) if bf_phase == "prefill" else slice(4, 8)
        nv_rows = slice(4, 8) if bf_phase == "prefill" else slice(0, 4)
        check(f"{name}: {bf_phase} rows exact under a mixed mask",
              torch.allclose(y[:, bf_rows], exact[:, bf_rows], atol=1e-6),
              f"max|diff| {(y[:, bf_rows] - exact[:, bf_rows]).abs().max().item():.2e}")
        check(f"{name}: {nv_phase} rows quantized under a mixed mask",
              (y[:, nv_rows] - exact[:, nv_rows]).abs().max().item() > 1e-3)


def test_generate_shape_inference():
    """With no mask, shape decides the phase: multi-token = prefill, 1 token = decode.
    That is how HF generate() and a real deployment behave."""
    for name, (nv_phase, bf_phase) in FORMATS.items():
        m, lin = _layer(name)
        m.eval()
        exact = lambda t: F.linear(t, lin.weight, lin.bias)
        xp = torch.randn(1, 5, lin.in_features, device="cuda") * 0.5   # prefill pass
        xd = torch.randn(1, 1, lin.in_features, device="cuda") * 0.5   # decode step
        dp = (lin(xp) - exact(xp)).abs().max().item()
        dd = (lin(xd) - exact(xd)).abs().max().item()
        quantized_pass, exact_pass = (dp, dd) if nv_phase == "prefill" else (dd, dp)
        check(f"{name}: {nv_phase} pass quantized by shape alone", quantized_pass > 1e-3)
        check(f"{name}: {bf_phase} pass exact by shape alone", exact_pass < 1e-6,
              f"max|diff| {exact_pass:.2e}")


def test_observer_sees_quantized_phase_only():
    """act_amax must track the NVFP4 phase and ignore the BF16 one.

    The static input_global_scale is baked into the quantized half's checkpoint; the
    BF16 half never passes activations through it, so letting them into the running
    max inflates the scale on exactly the tensors it governs.
    """
    for name, (nv_phase, bf_phase) in FORMATS.items():
        m, lin = _layer(name)
        m.train()
        lin.act_amax.zero_()
        lin._observed = False
        x = torch.randn(2, 8, lin.in_features, device="cuda") * 0.1
        prefill = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
        prefill[:, :4] = True
        bf_rows = slice(0, 4) if bf_phase == "prefill" else slice(4, 8)
        nv_rows = slice(4, 8) if bf_phase == "prefill" else slice(0, 4)
        x[:, bf_rows] *= 500.0                  # enormous activations on the BF16 phase
        with quant_phase(prefill):
            lin(x)
        nv_amax = x[:, nv_rows].abs().max().item()
        check(f"{name}: act_amax tracks the {nv_phase} (NVFP4) phase",
              abs(lin.act_amax.item() - nv_amax) < 1e-4,
              f"got {lin.act_amax.item():.3f}, expected {nv_amax:.3f}")
        check(f"{name}: act_amax ignores the {bf_phase} (BF16) phase",
              lin.act_amax.item() < 0.1 * x.abs().max().item(),
              f"full amax was {x.abs().max().item():.1f}")


def test_exported_halves_are_plain_checkpoints():
    """One half must be an ordinary bf16 HF model, the other an ordinary NVFP4 one.

    That is what lets two independent vLLM workers load them for disaggregated
    serving -- neither engine knows this is an ablation.
    """
    if TMP.exists():
        shutil.rmtree(TMP)
    for name, (nv_phase, bf_phase) in FORMATS.items():
        m = build(name)
        out = TMP / name
        for variant in variants(name):
            save_checkpoint(m, out / variant, variant=variant, step=0)

        bf_cfg = json.loads((out / bf_phase / "config.json").read_text())
        nv_cfg = json.loads((out / nv_phase / "config.json").read_text())
        check(f"{name}: {bf_phase} half has NO quantization_config",
              "quantization_config" not in bf_cfg)
        check(f"{name}: {nv_phase} half has a quantization_config",
              "quantization_config" in nv_cfg)

        bf = load_file(str(out / bf_phase / "model.safetensors"))
        nv = load_file(str(out / nv_phase / "model.safetensors"))
        k = "model.layers.0.mlp.gate_proj"
        check(f"{name}: {bf_phase} half stores a plain bf16 weight",
              f"{k}.weight" in bf and bf[f"{k}.weight"].dtype == torch.bfloat16
              and f"{k}.weight_packed" not in bf)
        check(f"{name}: {nv_phase} half stores packed FP4",
              f"{k}.weight_packed" in nv and f"{k}.weight_scale" in nv
              and f"{k}.weight_global_scale" in nv)
        # W4A4 -> the static activation scale must be present on the quantized half
        check(f"{name}: {nv_phase} half carries input_global_scale",
              f"{k}.input_global_scale" in nv)
        check(f"{name}: {bf_phase} half carries NO input_global_scale",
              f"{k}.input_global_scale" not in bf)

        # the BF16 half must be the master itself, not a dequantized copy of it
        lin = m.model.layers[0].mlp.gate_proj
        check(f"{name}: {bf_phase} weight equals the master",
              torch.equal(bf[f"{k}.weight"].cuda().float(),
                          lin.weight.detach().float()),
              "exported bf16 half differs from the trained master")


def test_reload_round_trip():
    """Loading either half back must restore the layer; the BF16 half is exact."""
    for name, (nv_phase, bf_phase) in FORMATS.items():
        m = build(name)
        out = TMP / name
        ref = m.model.layers[0].mlp.gate_proj.weight.detach().clone()
        m2 = build(name)
        with torch.no_grad():                       # scramble, then restore
            for mod in m2.modules():
                if hasattr(mod, "weight") and mod.weight is not None:
                    mod.weight.add_(torch.randn_like(mod.weight) * 0.05)
        n = load_into(m2, load_file(str(out / bf_phase / "model.safetensors")),
                      variant=bf_phase)
        got = m2.model.layers[0].mlp.gate_proj.weight.detach()
        check(f"{name}: reload from the {bf_phase} (BF16) half is exact",
              torch.allclose(got.float(), ref.float(), atol=1e-2),
              f"{n} tensors, max|diff| {(got - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    for fn in [test_registry_wiring, test_bf16_phase_is_exact, test_phase_routing,
               test_generate_shape_inference, test_observer_sees_quantized_phase_only,
               test_exported_halves_are_plain_checkpoints, test_reload_round_trip]:
        print(f"\n{fn.__name__}")
        fn()
    if TMP.exists():
        shutil.rmtree(TMP)
    print("\nall phase-isolation tests passed")
