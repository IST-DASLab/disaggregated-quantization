"""The export layer must keep producing exactly what it produced before.

Serialization moved from external per-format exporters into the layers themselves
(QuantizedLinear.export_tensors / export_config / load_tensors). vLLM resolves every
tensor with a bare params_dict[name] lookup, so a single renamed key silently breaks
loading for every checkpoint of that format. tests/_export_golden/ holds reference
directories captured BEFORE the refactor; this compares key sets and tensor VALUES
(not file bytes, which depend on serialization order).

    python tests/test_export.py
"""
import json, os, sys, tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from export.save import build_state_dict, load_into, save_checkpoint
from quantizers import variants as _variants
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
    check("single-format model reports [None]", _variants("nvfp4") == [None])
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        check(f"{name} reports two variants",
              _variants(name) == ["prefill", "decode"])


# ---------------------------------------------------------------------------
# Text-only export of a multimodal wrapper
# ---------------------------------------------------------------------------
def tiny_wrapper():
    """A 2-layer Gemma3ForConditionalGeneration, built from the 4b config.

    Down-scaled rather than loaded: the point is the module TREE and the config, and an
    8 GB load proves nothing extra about either. The vision tower is kept (shrunk), so
    the export really does have wrapper tensors to drop.
    """
    from transformers import Gemma3ForConditionalGeneration
    cfg = AutoConfig.from_pretrained("google/gemma-3-4b-it")
    t = cfg.text_config
    t.num_hidden_layers, t.hidden_size, t.intermediate_size = 2, 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.vocab_size = 512
    # layer_types is derived from the FULL layer count and is validated against it on
    # every re-read, so it has to shrink with num_hidden_layers.
    t.layer_types = ["sliding_attention", "full_attention"]
    v = cfg.vision_config
    v.num_hidden_layers, v.hidden_size, v.intermediate_size = 2, 64, 128
    v.num_attention_heads, v.image_size, v.patch_size = 4, 32, 16
    torch.manual_seed(0)
    return Gemma3ForConditionalGeneration(cfg).float()


def test_text_only_export():
    from export.save import text_only_arch, to_model_keys
    from training.models import text_stack

    m = tiny_wrapper()
    check("wrapper is exported as Gemma3ForCausalLM",
          text_only_arch(m) == "Gemma3ForCausalLM", str(text_only_arch(m)))
    check("a plain CausalLM is exported unchanged", text_only_arch(tiny_model()) is None)

    params, _ = build_quantizer_params("nvfp4", "")
    REGISTRY["nvfp4"]["apply"](text_stack(m).base, **params)
    for mod in m.modules():
        if hasattr(mod, "act_amax"):
            mod.act_amax.fill_(3.5); mod._observed = True

    out = Path(tempfile.mkdtemp()) / "step"
    n = save_checkpoint(m, out, step=7)
    state = load_file(str(out / "model.safetensors"))
    cfg = json.loads((out / "config.json").read_text())

    check("no vision / projector / language_model tensor survives",
          not [k for k in state if "vision" in k or "multi_modal" in k
               or "language_model" in k],
          str([k for k in state if "vision" in k][:2]))
    check("text weights are renamed to the plain layout",
          "model.embed_tokens.weight" in state and
          all(k.startswith(("model.layers.", "model.embed_tokens", "model.norm",
                            "model.rotary", "lm_head.")) for k in state),
          str(sorted(k for k in state if not k.startswith("model.layers."))[:6]))
    check("every quantized layer is under model.layers.N",
          sum(k.endswith(".weight_packed") for k in state) == 2 * 7,
          str(sum(k.endswith(".weight_packed") for k in state)))
    check("model.safetensors written with all tensors", n == len(state), f"{n}")
    files = sorted(p.name for p in out.iterdir())
    check("tokenizer + chat template land in the export",
          "tokenizer.json" in files and "tokenizer_config.json" in files
          and any(f.startswith("chat_template") for f in files), str(files))

    check("architectures rewritten", cfg["architectures"] == ["Gemma3ForCausalLM"],
          str(cfg["architectures"]))
    check("model_type is gemma3_text", cfg["model_type"] == "gemma3_text",
          cfg["model_type"])
    check("no vision_config / text_config in the export",
          "vision_config" not in cfg and "text_config" not in cfg)
    check("quantization_config survives", "quantization_config" in cfg)
    check("tie_word_embeddings preserved", cfg.get("tie_word_embeddings") is True,
          str(cfg.get("tie_word_embeddings")))
    check("rope is written in the stock (flat) form",
          cfg.get("rope_scaling") == {"rope_type": "linear", "factor": 8.0}
          and cfg.get("rope_theta") == 1000000.0
          and cfg.get("rope_local_base_freq") == 10000.0
          and "rope_parameters" not in cfg,
          str({k: cfg.get(k) for k in ("rope_theta", "rope_scaling",
                                       "rope_local_base_freq", "rope_parameters")}))
    check("token ids present", all(k in cfg for k in
                                   ("bos_token_id", "eos_token_id", "pad_token_id")),
          str({k: cfg.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")}))

    # the export must be readable as the class it claims to be
    from transformers import AutoConfig
    back = AutoConfig.from_pretrained(out)
    check("re-reads as a gemma3_text config",
          type(back).__name__ == "Gemma3TextConfig", type(back).__name__)
    check("layer count survives", back.num_hidden_layers == 2, str(back.num_hidden_layers))
    # the rope the SERVER will see must be the rope the wrapper had -- this is what the
    # flat/nested round-trip is for, and vLLM reads exactly this attribute
    check("rope_parameters round-trip unchanged",
          back.rope_parameters == m.config.text_config.rope_parameters,
          f"{back.rope_parameters} vs {m.config.text_config.rope_parameters}")
    check("sliding window survives", back.sliding_window == 1024, str(back.sliding_window))
    check("layer_types survives", back.layer_types == ["sliding_attention",
                                                       "full_attention"],
          str(back.layer_types))

    # the checkpoint must load back into the wrapper it came from
    rt = to_model_keys(m, state)
    check("to_model_keys is the inverse of the export rename",
          all(k.startswith(("model.language_model.", "lm_head.")) for k in rt),
          str([k for k in rt if not k.startswith("model.language_model.")][:3]))
    dst = tiny_wrapper()
    REGISTRY["nvfp4"]["apply"](text_stack(dst).base, **params)
    for mod in dst.modules():
        if hasattr(mod, "_wq"):
            mod._wq.zero_()
    loaded = load_into(dst, state)
    worst = max((a._wq - b._wq).abs().max().item()
                for a, b in zip([x for x in m.modules() if hasattr(x, "_wq")],
                                [x for x in dst.modules() if hasattr(x, "_wq")]))
    check(f"{loaded} layers round-trip into the wrapper", loaded == 14 and worst < 1e-6,
          f"max|Δ|={worst:.2e}")


def test_text_only_config_of_the_real_wrappers():
    """The config transform, on the REAL 4b/12b configs (cheap: no weights)."""
    from transformers import AutoConfig
    from export.save import text_only_arch, text_only_config

    ref = AutoConfig.from_pretrained("google/gemma-3-1b-it")     # the shape we target
    for repo in ("google/gemma-3-4b-it", "google/gemma-3-12b-it"):
        src = AutoConfig.from_pretrained(repo)
        got = text_only_config(src, text_only_arch(src))
        check(f"{repo}: model_type", got.model_type == ref.model_type, got.model_type)
        check(f"{repo}: architectures", got.architectures == ref.architectures,
              str(got.architectures))
        check(f"{repo}: no vision_config", not hasattr(got, "vision_config"))
        check(f"{repo}: text dims preserved",
              (got.num_hidden_layers, got.hidden_size, got.vocab_size)
              == (src.text_config.num_hidden_layers, src.text_config.hidden_size,
                  src.text_config.vocab_size))
        check(f"{repo}: tie_word_embeddings", got.tie_word_embeddings is True)
        check(f"{repo}: eos_token_id taken from the wrapper",
              got.eos_token_id == src.eos_token_id, str(got.eos_token_id))
        # Diffed against the stock 1b repo (the shape we are matching). Exactly two of
        # its keys are absent, both information-preserving:
        #   sliding_window_pattern (a stride) -> layer_types, the explicit per-layer list
        #     transformers derives from it and the one vLLM indexes (gemma3.py:163);
        #   cache_implementation -> generation_config.json, which the export also writes.
        missing = {k for k in ref.to_diff_dict() if not hasattr(got, k)}
        check(f"{repo}: only the two superseded keys are absent",
              missing == {"cache_implementation", "sliding_window_pattern"},
              str(sorted(missing)))
        check(f"{repo}: layer_types carries the pattern explicitly",
              len(got.layer_types) == got.num_hidden_layers
              and set(got.layer_types) == {"sliding_attention", "full_attention"},
              f"{len(got.layer_types)} entries")


if __name__ == "__main__":
    for fn in (test_matches_golden, test_roundtrip, test_variants,
               test_text_only_export, test_text_only_config_of_the_real_wrappers):
        print(f"\n{fn.__name__}:")
        fn()
    print("\nPASS: export")
