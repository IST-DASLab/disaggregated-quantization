"""Gates for nvfp4lloyd43upcast: prefill NVFP4 derived FROM the Lloyd43 decode weight.

    shared:  W -> NVFP4                 W -> Lloyd43
    upcast:  W -> Lloyd43 -> NVFP4      W -> Lloyd43

The claim being tested is a storage claim: the prefill checkpoint is a pure function of
the 3-bit decode weight, so only 3 bits per weight need to be kept. Three things can
silently break that and each gets an assertion rather than an assumption:

  1. The upcast could be a NO-OP. If `quantize_weight` still read the master, this
     format would be byte-identical to nvfp4lloyd43shared and every result would look
     plausible. So the test asserts prefill DIFFERS from the shared format's prefill,
     and separately that it EQUALS the upcast of the decode weight. Self-consistency
     alone proves nothing -- something silently ignored round-trips against itself.
  2. STALE ORDERING. The base refreshes _wq (prefill) before _wq_dec (decode), so an
     un-overridden post_update would upcast LAST step's Lloyd43 weight forever. The
     test mutates the master, calls post_update, and requires the two to agree.
  3. The prefill half must still be REAL W4A4 NVFP4 for vLLM -- packed weights, an
     input_global_scale, and non-negative (UE4M3) block scales -- while decode stays
     pseudo-quantized bf16. Rerouting the weight source must not disturb any of that.

    python tests/test_nvfp4lloyd43upcast.py
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

from export.save import save_checkpoint
from quantizers import REGISTRY, build_quantizer_params, quant_phase, variants
from quantizers.blocked import blocked_quantize
from quantizers.dual import NVFP4Lloyd43UpcastLinear
from quantizers.grids import LLOYD43_SIGNED_3BIT
from quantizers.nvfp4 import unpack_nvfp4_weight

TMP = Path("/tmp/qad_nvfp4lloyd43upcast_test")
NAME = "nvfp4lloyd43upcast"


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


def test_prefill_is_an_upcast_of_the_decode_weight():
    """_wq must be NVFP4(_wq_dec), not NVFP4(master)."""
    m = build(NAME)
    lin = m.model.layers[0].mlp.gate_proj
    check(f"{NAME}: layer class", isinstance(lin, NVFP4Lloyd43UpcastLinear))

    # what the layer actually cached for prefill
    got = lin._wq.detach().float()
    # what an upcast of the decode weight should give
    want = blocked_quantize(lin._wq_dec, lin.rounder, lin.block_size,
                            signed=lin.signed,
                            global_scale=lin.group_global_scale())[0].float()
    check("prefill IS the upcast of the decode weight",
          torch.allclose(got, want, atol=1e-5),
          f"max|diff| {float((got - want).abs().max()):.2e}")

    # ...and is NOT simply the master quantized directly, which is the shared format
    direct = blocked_quantize(lin.weight, lin.rounder, lin.block_size,
                              signed=lin.signed,
                              global_scale=lin.group_global_scale())[0].float()
    d = float((got - direct).abs().max())
    check("prefill DIFFERS from quantizing the master directly", d > 1e-6,
          f"max|diff| {d:.2e} -- otherwise this is just nvfp4lloyd43shared")


def test_upcast_source_lies_on_the_lloyd43_grid():
    """The tensor prefill upcasts FROM must be a genuine 3-bit weight.

    This is the storage claim: if the source were not on the Lloyd43 grid, the prefill
    checkpoint would not be derivable from 3 bits per weight.
    """
    m = build(NAME)
    lin = m.model.layers[0].mlp.gate_proj
    wq = lin._wq_dec.detach().float()
    blk = wq.reshape(wq.shape[0], -1, lin.block_size)
    # undo the SIGNED normalisation: divide by the block's max-abs element WITH its
    # sign, so the extreme comes back to +6. Dividing by the magnitude would sign-flip
    # negatively-scaled blocks, and Lloyd43 is asymmetric, so those cannot lie on it.
    idx = blk.abs().argmax(-1, keepdim=True)
    peak = blk.take_along_dim(idx, -1)
    peak = torch.where(peak.abs() < 1e-12, torch.ones_like(peak), peak)
    norm = (blk / peak) * 6.0
    g = LLOYD43_SIGNED_3BIT.to("cuda")
    off = float((norm.unsqueeze(-1) - g).abs().min(-1).values.max())
    # BF16 round-off, same reasoning as tests/test_lloyd21.py: the value is exact on the
    # grid in fp32 and lands within one bf16 ulp of it once stored. Normalised values
    # reach ~6.0 and bf16 has 8 mantissa bits -> ~6*2^-8 = 2.3e-2. The fp32-era bound
    # was 1e-3.
    check("upcast source lies on the Lloyd43 grid (within bf16 round-off)", off < 2.5e-2,
          f"max off-grid {off:.2e}")


def test_refresh_order_is_not_stale():
    """post_update must refresh decode BEFORE prefill.

    The inherited order is the opposite, so without the override every prefill weight
    would be an upcast of the PREVIOUS step's Lloyd43 weight -- wrong by one step, and
    entirely plausible-looking in a loss curve.
    """
    m = build(NAME)
    lin = m.model.layers[0].mlp.gate_proj
    before = lin._wq_dec.detach().clone()

    with torch.no_grad():                      # simulate an optimizer step
        lin.weight.add_(torch.randn_like(lin.weight) * 0.35)
    lin.post_update(step=1, total_steps=10)

    moved = float((lin._wq_dec.detach() - before).abs().max())
    check("decode weight actually moved", moved > 1e-6, f"max|diff| {moved:.2e}")

    want = blocked_quantize(lin._wq_dec, lin.rounder, lin.block_size,
                            signed=lin.signed,
                            global_scale=lin.group_global_scale())[0].float()
    got = lin._wq.detach().float()
    check("prefill upcasts the CURRENT decode weight, not the previous one",
          torch.allclose(got, want, atol=1e-5),
          f"max|diff| {float((got - want).abs().max()):.2e}")


def test_only_one_master_weight():
    """The whole point is 3-bit storage, so there must be no second master."""
    m = build(NAME)
    lin = m.model.layers[0].mlp.gate_proj
    check("no second master weight", not hasattr(lin, "decode_weight"),
          "a split master would defeat the storage claim")
    names = {n for n, _ in lin.named_parameters()}
    check("exactly one weight parameter", names <= {"weight", "bias"}, f"{sorted(names)}")


def test_phases_differ():
    m = build(NAME)
    m.eval()
    lin = m.model.layers[0].mlp.gate_proj
    x = torch.randn(2, 6, lin.in_features, device="cuda") * 0.5
    allp = torch.ones(2, 6, dtype=torch.bool, device="cuda")
    with quant_phase(allp):
        y_pre = lin(x)
    with quant_phase(~allp):
        y_dec = lin(x)
    check("the phases give different outputs",
          float((y_pre - y_dec).abs().max()) > 1e-4)


def test_export_is_real_nvfp4_w4a4_for_prefill():
    """prefill = packed FP4 + input_global_scale (hardware NVFP4); decode = plain bf16."""
    shutil.rmtree(TMP, ignore_errors=True)
    m = build(NAME)
    out = TMP / NAME
    for v in variants(NAME):
        save_checkpoint(m, out / v, variant=v, step=0)

    pre = json.loads((out / "prefill" / "config.json").read_text())
    dec = json.loads((out / "decode" / "config.json").read_text())
    check("prefill HAS quantization_config", "quantization_config" in pre)
    check("prefill format is nvfp4-pack-quantized",
          pre["quantization_config"].get("format") == "nvfp4-pack-quantized")
    check("prefill quantizes ACTIVATIONS too (W4A4, not W4A16)",
          pre["quantization_config"]["config_groups"]["group_0"]["input_activations"]
          is not None, "weight-only would not use the NVFP4 tensor cores")
    check("decode has NO quantization_config", "quantization_config" not in dec,
          "no 3-bit kernel exists to serve it")

    P = load_file(str(out / "prefill" / "model.safetensors"))
    D = load_file(str(out / "decode" / "model.safetensors"))
    k = "model.layers.0.mlp.gate_proj"
    check("prefill is packed FP4",
          f"{k}.weight_packed" in P and f"{k}.weight_scale" in P)
    check("prefill carries input_global_scale (W4A4)", f"{k}.input_global_scale" in P)
    check("prefill emits no weight_shape", f"{k}.weight_shape" not in P)
    check("decode is plain bf16",
          f"{k}.weight" in D and D[f"{k}.weight"].dtype == torch.bfloat16
          and f"{k}.weight_packed" not in D)
    check("decode carries NO input_global_scale", f"{k}.input_global_scale" not in D)

    # UE4M3 has no sign bit -- a negative scale is unservable, not merely unusual
    ws = [v for kk, v in P.items() if kk.endswith("weight_scale")]
    neg = sum(int((v.float() < 0).any()) for v in ws)
    check("prefill export has NO negative block scales", neg == 0,
          f"{neg}/{len(ws)} tensors would be misread as UE4M3 magnitudes")

    # the SHIPPED prefill weight must be the upcast the model trained on
    lin = m.model.layers[0].mlp.gate_proj
    deq = unpack_nvfp4_weight(P[f"{k}.weight_packed"].cuda(),
                              P[f"{k}.weight_scale"].cuda(),
                              P[f"{k}.weight_global_scale"].cuda(), lin.block_size)
    d = float((deq.float() - lin._wq.detach().float()).abs().max())
    check("shipped prefill weight == the trained upcast", d < 1e-2, f"max|diff| {d:.2e}")

    # and the decode half ships the 3-bit weight it trained on
    d2 = float((D[f"{k}.weight"].cuda().float() - lin._wq_dec.detach().float()).abs().max())
    check("decode export is the Lloyd43-quantized weight", d2 < 1e-2, f"max|diff| {d2:.2e}")


def test_registry_entry():
    e = REGISTRY[NAME]
    check("two variants", e.get("variants") == ["prefill", "decode"])
    check("exports compressed-tensors", e["export"] == "compressed_tensors")
    check("block_size default is 16", e["defaults"] == {"block_size": 16})


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"  all {NAME} tests passed")
