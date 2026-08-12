"""Gates for nvfp4lloyd21split: NVFP4 W4A4 prefill + Lloyd21 W2A16 decode, two masters.

Everything is inherited from the 3-bit split layer, so the assertions here are the ones
that inheritance can silently get wrong:

  GRID      a DECODE_GRID that failed to override leaves the layer on lloyd43 and
            produces a plausible 3-bit curve filed under a 2-bit name.
  SPLIT     two masters that actually DIVERGE. If decode_weight were aliased to weight
            (or never trained), this would be the upcast/shared format wearing a
            different name and every comparison against it would be meaningless.
  EXPORT    prefill packed FP4 with input_global_scale, decode plain bf16, and no
            negative block scales (UE4M3 has no sign bit).

    python tests/test_nvfp4lloyd21split.py
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
from quantizers import REGISTRY, build_quantizer_params, variants
from quantizers.dual import NVFP4Lloyd21SplitLinear, NVFP4Lloyd43SplitLinear
from quantizers.grids import LLOYD21_SIGNED_2BIT

TMP = Path("/tmp/qad_nvfp4lloyd21split_test")
NAME = "nvfp4lloyd21split"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def build():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, False
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).cuda().float()
    params, _ = build_quantizer_params(NAME, "")
    REGISTRY[NAME]["apply"](m, **params)
    for mod in m.modules():
        if hasattr(mod, "act_amax"):
            mod.act_amax.fill_(3.5); mod._observed = True
    return m


def test_decode_grid_is_the_2bit_one():
    m = build()
    lin = m.model.layers[0].mlp.gate_proj
    check("layer class", isinstance(lin, NVFP4Lloyd21SplitLinear))
    check("decode grid has 4 levels", lin._lloyd_grid.numel() == 4,
          f"{lin._lloyd_grid.numel()} levels")
    check("grid IS lloyd21", torch.allclose(lin._lloyd_grid.cpu(), LLOYD21_SIGNED_2BIT,
                                            atol=1e-6))
    check("the 3-bit sibling still has 8 levels (base untouched)",
          NVFP4Lloyd43SplitLinear.DECODE_GRID.numel() == 8)


def test_two_masters_that_can_diverge():
    """A split format whose masters are aliased is the shared format in disguise."""
    m = build()
    lin = m.model.layers[0].mlp.gate_proj
    check("has a second master", hasattr(lin, "decode_weight"))
    check("decode master is a separate Parameter",
          lin.decode_weight.data_ptr() != lin.weight.data_ptr(),
          "aliased masters would make this the shared format")
    check("both masters are trainable",
          lin.weight.requires_grad and lin.decode_weight.requires_grad)
    # move ONE master and check only its half responds
    before_pre, before_dec = lin._wq.clone(), lin._wq_dec.clone()
    with torch.no_grad():
        lin.decode_weight.add_(torch.randn_like(lin.decode_weight) * 0.3)
    lin.post_update(step=1, total_steps=10)
    check("perturbing the decode master moves the DECODE half",
          not torch.equal(lin._wq_dec, before_dec))
    check("...and leaves the PREFILL half alone", torch.equal(lin._wq, before_pre),
          "otherwise the two phases are not actually independent")


def test_mixed_export():
    shutil.rmtree(TMP, ignore_errors=True)
    m = build()
    for v in variants(NAME):
        save_checkpoint(m, TMP / v, variant=v, step=0)
    pre = json.loads((TMP / "prefill" / "config.json").read_text())
    dec = json.loads((TMP / "decode" / "config.json").read_text())
    check("prefill HAS quantization_config", "quantization_config" in pre)
    check("prefill is W4A4 (activations quantized)",
          pre["quantization_config"]["config_groups"]["group_0"]["input_activations"]
          is not None)
    check("decode has NO quantization_config", "quantization_config" not in dec,
          "no 2-bit kernel exists to serve it")
    P = load_file(str(TMP / "prefill" / "model.safetensors"))
    D = load_file(str(TMP / "decode" / "model.safetensors"))
    k = "model.layers.0.mlp.gate_proj"
    check("prefill is packed FP4", f"{k}.weight_packed" in P)
    check("prefill carries input_global_scale", f"{k}.input_global_scale" in P)
    check("decode is plain bf16",
          D[f"{k}.weight"].dtype == torch.bfloat16 and f"{k}.weight_packed" not in D)
    ws = [v for kk, v in P.items() if kk.endswith("weight_scale")]
    neg = sum(int((v.float() < 0).any()) for v in ws)
    check("prefill export has NO negative block scales", neg == 0,
          f"{neg}/{len(ws)} would be misread as UE4M3 magnitudes")
    shutil.rmtree(TMP, ignore_errors=True)


def test_registry_entry():
    e = REGISTRY[NAME]
    check("two variants", e.get("variants") == ["prefill", "decode"])
    check("exports compressed-tensors", e["export"] == "compressed_tensors")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    print(f"  all {NAME} tests passed")
