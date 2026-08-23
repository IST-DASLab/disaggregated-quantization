"""Gates for NVFP4 prefill + Lloyd43 decode (shared and split masters).

This format is the first to combine three things that have each bitten before, so each
gets an assertion rather than an assumption:

  1. The `signed` flag on nvfp4_quantize/pack_nvfp4_weight. E2M1 is symmetric, so a
     signed block scale changes nothing REPRESENTABLE -- the two encodings dequantize
     identically -- but only if pack/unpack honour it. A flag that were silently
     ignored would still round-trip against itself, so the tests below also assert the
     two encodings DIFFER. (This is why signed weights were once believed free to ship;
     see 4 for why they are not.)
  2. MIXED export: prefill is packed compressed-tensors, decode is pseudo-quantized bf16
     (no 3-bit Lloyd kernel exists). Two ordinary checkpoints, one per engine.
  3. Two grids under DIFFERENT normalisations: unsigned E2M1 on prefill, signed
     0/6-pinned Lloyd43 on decode. Lloyd43's grid is asymmetric, so its block scale must
     carry the sign for the extreme to land on the pinned +6.0; E2M1 gains nothing from
     it. So even on a shared master the halves' block scales agree in MAGNITUDE only.
  4. The prefill export must have NO negative block scales. NVFP4's scale factor is
     UE4M3 -- unsigned, because that is what the Blackwell block-scaled MMA consumes --
     so a byte >= 0x80 is read as exponent/mantissa and the magnitude comes out wrong.
     A signed checkpoint served GSM8K at 0.0 against lloyd43's 0.569.

    python tests/test_nvfp4lloyd43.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from export.save import load_into, save_checkpoint
from quantizers import REGISTRY, build_quantizer_params, quant_phase, variants
from quantizers.dual import NVFP4Lloyd43SharedLinear, NVFP4Lloyd43SplitLinear
from quantizers.grids import LLOYD43_SIGNED_3BIT
from quantizers.nvfp4 import pack_nvfp4_weight, unpack_nvfp4_weight

TMP = Path("/tmp/qad_nvfp4lloyd43_test")
NAMES = ["nvfp4lloyd43shared", "nvfp4lloyd43split"]


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


def test_signed_nvfp4_round_trips_through_the_packer():
    """The claim that E2M1 can go signed 'for free' is only true if pack/unpack agree.

    Packs the SAME weight both ways and requires each to reconstruct itself exactly. A
    signed pack that silently fell back to unsigned would still round-trip against
    itself, so the test also asserts the two disagree -- otherwise it proves nothing.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 128, device="cuda")
    for signed in (False, True):
        p, s, g = pack_nvfp4_weight(w, 16, signed=signed)
        deq = unpack_nvfp4_weight(p, s, (1.0 / g).reshape(1), 16)
        rel = float(((deq - w) ** 2).mean() / (w ** 2).mean())
        check(f"signed={signed}: pack/unpack round-trips", rel < 0.02, f"rel MSE {rel:.4f}")
    pu, su, _ = pack_nvfp4_weight(w, 16, signed=False)
    ps, ss, _ = pack_nvfp4_weight(w, 16, signed=True)
    check("signed pack DIFFERS from unsigned", not torch.equal(su, ss),
          "otherwise the flag is being ignored")
    check("signed block scales can be negative",
          bool((ss.float() < 0).any()), "that is where the sign is absorbed")
    check("unsigned block scales are all >= 0", bool((su.float() >= 0).all()))


def test_signed_and_unsigned_dequantize_identically():
    """E2M1 is symmetric, so negating both the code and the block scale cancels.

    If this holds, a signed checkpoint carries the SAME weights as an unsigned one and
    can be re-packed rather than retrained -- which is the difference between a re-export
    and six 4-hour runs.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 128, device="cuda")
    du = unpack_nvfp4_weight(*(lambda p, s, g: (p, s, (1.0 / g).reshape(1)))(
        *pack_nvfp4_weight(w, 16, signed=False)), 16)
    ds = unpack_nvfp4_weight(*(lambda p, s, g: (p, s, (1.0 / g).reshape(1)))(
        *pack_nvfp4_weight(w, 16, signed=True)), 16)
    check("signed and unsigned dequantize to the SAME weights",
          torch.allclose(du, ds, atol=1e-6),
          f"max|diff| {float((du - ds).abs().max()):.2e}")


def test_no_negative_scales_in_a_servable_export():
    """NVFP4 block scales are UE4M3 -- unsigned by hardware spec.

    That is the scale-factor format the Blackwell block-scaled MMA (tcgen05.mma)
    consumes, so there is no sign bit: a byte >= 0x80 is read as exponent/mantissa and
    the magnitude is wrong, not just the sign. Measured before this was understood, a
    signed checkpoint served GSM8K at 0.0 (same-size lloyd43: 0.569) with 50% of its
    block scales negative; every format that serves correctly has 0%. So any half that
    ships as packed NVFP4 must export non-negative scales, whatever the training-time
    fake-quant does internally.
    """
    shutil.rmtree(TMP, ignore_errors=True)
    for name in NAMES:
        m = build(name)
        out = TMP / name
        save_checkpoint(m, out / "prefill", variant="prefill", step=0)
        P = load_file(str(out / "prefill" / "model.safetensors"))
        ws = [v for k, v in P.items() if k.endswith("weight_scale")]
        neg = sum(int((v.float() < 0).any()) for v in ws)
        check(f"{name}: prefill export has NO negative block scales", neg == 0,
              f"{neg}/{len(ws)} tensors would be misread as UE4M3 magnitudes")


def test_registry_and_layer_wiring():
    for name in NAMES:
        e = REGISTRY[name]
        check(f"{name}: two variants", e.get("variants") == ["prefill", "decode"])
        check(f"{name}: exports compressed-tensors", e["export"] == "compressed_tensors")
        m = build(name)
        lin = m.model.layers[0].mlp.gate_proj
        want = NVFP4Lloyd43SharedLinear if "shared" in name else NVFP4Lloyd43SplitLinear
        check(f"{name}: layer class", isinstance(lin, want))
        # The NVFP4 half must stay UNSIGNED: its block scale ships as UE4M3, which has
        # no sign bit (see test_no_negative_scales_in_a_servable_export). The signed
        # normalisation survives only on the decode half, which is pseudo-quantized to
        # bf16 and so never reaches an FP4 kernel.
        check(f"{name}: NVFP4 half is unsigned (UE4M3 has no sign bit)",
              lin.signed is False)
        check(f"{name}: split has a second master",
              hasattr(lin, "decode_weight") == ("split" in name))


def test_decode_half_uses_the_lloyd43_grid():
    """Decode must be Lloyd43, not a second copy of the NVFP4 half."""
    for name in NAMES:
        m = build(name)
        m.eval()
        lin = m.model.layers[0].mlp.gate_proj
        x = torch.randn(2, 6, lin.in_features, device="cuda") * 0.5
        allp = torch.ones(2, 6, dtype=torch.bool, device="cuda")
        with quant_phase(allp):
            y_pre = lin(x)
        with quant_phase(~allp):
            y_dec = lin(x)
        check(f"{name}: the phases give different outputs",
              float((y_pre - y_dec).abs().max()) > 1e-4)
        # the decode cache must lie on lloyd43 * (its block scale); check the normalised
        # values land on the grid by dividing out the per-block scale
        wq = lin._wq_dec.detach().float()
        blk = wq.reshape(wq.shape[0], -1, lin.block_size)
        # Undo the SIGNED normalisation the way blocked_quantize applied it: divide by
        # the block's max-abs element WITH ITS SIGN, so the extreme comes back to +6.
        # Dividing by the magnitude instead sign-flips every negatively-scaled block,
        # and Lloyd43 is asymmetric, so a flipped block cannot lie on the grid.
        idx = blk.abs().argmax(-1, keepdim=True)
        peak = blk.take_along_dim(idx, -1)
        peak = torch.where(peak.abs() < 1e-12, torch.ones_like(peak), peak)
        norm = (blk / peak) * 6.0
        g = LLOYD43_SIGNED_3BIT.to("cuda")
        off = float((norm.unsqueeze(-1) - g).abs().min(-1).values.max())
        # BF16 round-off, not slack: _wq is stored bf16 (quantizers/base.py), so a value
        # exactly on the grid in fp32 lands within one bf16 ulp of it. Normalised values
        # reach ~6.0 and bf16 carries 8 mantissa bits -> ~6*2^-8 = 2.3e-2. The fp32-era
        # bound was 1e-3. Same bound as tests/test_lloyd21.py.
        check(f"{name}: decode cache lies on the Lloyd43 grid (within bf16 round-off)",
              off < 2.5e-2,
              f"max off-grid {off:.2e}")


def test_mixed_export_shapes():
    """prefill = packed FP4 + input_global_scale; decode = plain bf16, no quant config."""
    shutil.rmtree(TMP, ignore_errors=True)
    for name in NAMES:
        m = build(name)
        out = TMP / name
        for v in variants(name):
            save_checkpoint(m, out / v, variant=v, step=0)
        pre = json.loads((out / "prefill" / "config.json").read_text())
        dec = json.loads((out / "decode" / "config.json").read_text())
        check(f"{name}: prefill HAS quantization_config", "quantization_config" in pre)
        check(f"{name}: decode has NO quantization_config",
              "quantization_config" not in dec, "no 3-bit kernel exists to serve it")

        P = load_file(str(out / "prefill" / "model.safetensors"))
        D = load_file(str(out / "decode" / "model.safetensors"))
        k = "model.layers.0.mlp.gate_proj"
        check(f"{name}: prefill is packed FP4",
              f"{k}.weight_packed" in P and f"{k}.weight_scale" in P)
        check(f"{name}: prefill carries input_global_scale (W4A4)",
              f"{k}.input_global_scale" in P)
        check(f"{name}: decode is plain bf16",
              f"{k}.weight" in D and D[f"{k}.weight"].dtype == torch.bfloat16
              and f"{k}.weight_packed" not in D)
        check(f"{name}: decode carries NO input_global_scale",
              f"{k}.input_global_scale" not in D)

        # the exported decode weights must be the QUANTIZED cache, not the raw master
        lin = m.model.layers[0].mlp.gate_proj
        check(f"{name}: decode export is the Lloyd43-quantized weight",
              torch.allclose(D[f"{k}.weight"].cuda().float(),
                             lin._wq_dec.detach().float(), atol=1e-2))


def test_reload_round_trip():
    for name in NAMES:
        m = build(name)
        out = TMP / name
        ref = m.model.layers[0].mlp.gate_proj._wq_dec.detach().clone()
        m2 = build(name)
        with torch.no_grad():
            for mod in m2.modules():
                if isinstance(mod, (NVFP4Lloyd43SharedLinear, NVFP4Lloyd43SplitLinear)):
                    mod._wq_dec.zero_()
        n = load_into(m2, load_file(str(out / "decode" / "model.safetensors")),
                      variant="decode")
        got = m2.model.layers[0].mlp.gate_proj._wq_dec.detach()
        check(f"{name}: decode half reloads", torch.allclose(got.float(), ref.float(),
                                                             atol=1e-2),
              f"{n} tensors, max|diff| {(got - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    for fn in [test_signed_nvfp4_round_trips_through_the_packer,
               test_signed_and_unsigned_dequantize_identically,
               test_no_negative_scales_in_a_servable_export,
               test_registry_and_layer_wiring,
               test_decode_half_uses_the_lloyd43_grid,
               test_mixed_export_shapes,
               test_reload_round_trip]:
        print(f"\n{fn.__name__}")
        fn()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nall nvfp4lloyd43 tests passed")
