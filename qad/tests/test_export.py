"""The export layer must keep producing exactly what it produced before.

Serialization moved from external per-format exporters into the layers themselves
(QuantizedLinear.export_tensors / export_config / load_tensors). vLLM resolves every
tensor with a bare params_dict[name] lookup, so a single renamed key silently breaks
loading for every checkpoint of that format. tests/_export_golden/ holds reference
directories captured BEFORE the refactor; this compares key sets and tensor VALUES
(not file bytes, which depend on serialization order).

    python tests/test_export.py
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from export.save import build_state_dict, export_variants, load_into, save_checkpoint
from quantizers import REGISTRY, build_quantizer_params

GOLD = Path(__file__).parent / "_export_golden"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def tiny_model():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, False
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).cuda().float()


def build(name):
    m = tiny_model()
    params, _ = build_quantizer_params(name, "")
    REGISTRY[name]["apply"](m, **params)
    for mod in m.modules():                      # deterministic stand-in for calibration
        if hasattr(mod, "act_amax"):
            mod.act_amax.fill_(3.5); mod._observed = True
    return m


def test_matches_golden():
    for name in ["nvfp4", "nvfp4a16", "lloyd3bit", "ste3bit", "fp8"]:
        ref = load_file(str(GOLD / name / "model.safetensors"))
        got = build_state_dict(build(name))
        check(f"{name}: same tensor keys", set(got) == set(ref),
              f"+{sorted(set(got)-set(ref))[:3]} -{sorted(set(ref)-set(got))[:3]}")
        worst, bad = 0.0, None
        for k in ref:
            a, b = ref[k].float(), got[k].float().cpu()
            if a.shape != b.shape:
                bad = f"{k} shape {tuple(b.shape)} != {tuple(a.shape)}"; break
            worst = max(worst, (a - b).abs().max().item())
        check(f"{name}: tensor values identical", bad is None and worst == 0.0,
              bad or f"max|Δ|={worst:.3e}")
        rc = json.loads((GOLD / name / "config.json").read_text()).get("quantization_config")
        gc = build(name).model.layers[0].mlp.gate_proj.export_config(None) \
            if REGISTRY[name]["export"] == "compressed_tensors" else None
        check(f"{name}: quantization_config unchanged", rc == gc)


def test_roundtrip():
    """export -> load must restore the quantized weight to the format's precision.

    Not bit-exact, and cannot be: the packed formats store weight_global_scale
    RECIPROCAL (2688/amax, which is what vLLM reads) and 1/(1/x) != x in fp32, so
    ~1e-7 relative error is inherent to the checkpoint. Pseudo-quant stores bf16, so
    its bound is ~2^-8. The point is that the error stays at representation level
    rather than signalling a mis-decoded nibble order or a swapped scale.
    """
    TOL = {"nvfp4": 1e-6, "nvfp4a16": 1e-6, "ste3bit": 8e-3}
    for name, tol in TOL.items():
        src = build(name)
        state = build_state_dict(src)
        dst = build(name)
        for mod in dst.modules():                # zero first, so a no-op load shows up
            if hasattr(mod, "_wq"):
                mod._wq.zero_()
        n = load_into(dst, state)
        srcs = [m for m in src.modules() if hasattr(m, "_wq")]
        dsts = [m for m in dst.modules() if hasattr(m, "_wq")]
        worst = max((a._wq - b._wq).abs().max().item() for a, b in zip(srcs, dsts))
        scale = max(a._wq.abs().max().item() for a in srcs)
        rel = worst / max(scale, 1e-12)
        check(f"{name}: {n} layers round-trip (rel < {tol:g})", rel < tol,
              f"rel={rel:.2e} abs={worst:.2e}")
        check(f"{name}: load actually wrote weights", worst < scale, "still zeroed")


def test_variants():
    check("single-format model reports [None]", export_variants(build("nvfp4")) == [None])
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        check(f"{name} reports two variants",
              export_variants(build(name)) == ["prefill", "decode"])


if __name__ == "__main__":
    for fn in (test_matches_golden, test_roundtrip, test_variants):
        print(f"\n{fn.__name__}:")
        fn()
    print("\nPASS: export")
