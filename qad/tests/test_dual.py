"""Gates for the prefill/decode dual formats (nvfp4pdshared, nvfp4pdsplit).

Two claims this file defends:

 1. Each exported directory is an ORDINARY single-format NVFP4 checkpoint —
    prefill indistinguishable from `nvfp4`, decode from `nvfp4a16`, down to the
    config and the tensor key set. That is what lets a disaggregated deployment
    load them on separate vLLM workers.
 2. Positions are routed to the format they will actually run under at inference,
    both under an explicit training mask and under generate()'s shape convention.

    python tests/test_dual.py
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
from quantizers import (REGISTRY, build_quantizer_params, post_update_all, variants,
                        prefill_mask_from_labels, quant_phase)
from quantizers.dual import DualSharedNVFP4Linear, DualSplitNVFP4Linear

GOLD = Path(__file__).parent / "_export_golden"
TMP = Path("/tmp/qad_dual_test")


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


def diverge(m):
    """Move the decode master away from prefill, the way training would.

    Perturb EVERY layer first and refresh caches only afterwards: the fused-group
    global scale is a max over q/k/v (and gate/up), so refreshing a layer's cache
    before its siblings have moved bakes in a scale that export will not reproduce.
    post_update_all does exactly this ordering after optimizer.step().
    """
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, DualSplitNVFP4Linear):
                mod.decode_weight.add_(torch.randn_like(mod.decode_weight) * 0.05)
    post_update_all(m, 0, 1)


# ---------------------------------------------------------------------------
# 1. the checkpoints are ordinary NVFP4 checkpoints
# ---------------------------------------------------------------------------
def test_checkpoints_are_plain_nvfp4():
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m = build(name)
        diverge(m)
        out = TMP / name
        shutil.rmtree(out, ignore_errors=True)
        check(f"{name}: two variants", variants(name) == ["prefill", "decode"])
        for v in variants(name):
            save_checkpoint(m, out / v, variant=v)
        for v, ref in [("prefill", "nvfp4"), ("decode", "nvfp4a16")]:
            cfg = json.loads((out / v / "config.json").read_text())
            rcfg = json.loads((GOLD / ref / "config.json").read_text())
            check(f"{name}/{v}: config identical to pure {ref}", cfg == rcfg)
            keys = set(load_file(str(out / v / "model.safetensors")))
            rkeys = set(load_file(str(GOLD / ref / "model.safetensors")))
            check(f"{name}/{v}: tensor keys identical to pure {ref}", keys == rkeys,
                  f"diff={sorted(keys ^ rkeys)[:3]}")
        a = load_file(str(out / "prefill/model.safetensors"))
        b = load_file(str(out / "decode/model.safetensors"))
        check(f"{name}/prefill: has input_global_scale",
              any("input_global_scale" in k for k in a))
        check(f"{name}/decode: no input_global_scale (W4A16 registers none)",
              not any("input_global_scale" in k for k in b))
        k = "model.layers.0.mlp.gate_proj.weight_packed"
        same = torch.equal(a[k], b[k])
        # shared: one master, so identical by construction. split: two masters that
        # training moved apart, so they MUST differ or the split is a no-op.
        check(f"{name}: packed weights {'identical' if same else 'differ'}",
              same == name.endswith("shared"))


def test_reload_is_order_independent():
    """Both halves must load into one model for dual-format eval, either order."""
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m = build(name)
        diverge(m)
        out = TMP / f"{name}_rl"
        shutil.rmtree(out, ignore_errors=True)
        for v in variants(name):
            save_checkpoint(m, out / v, variant=v)
        m2 = build(name)
        for v in ["decode", "prefill"]:          # reversed on purpose
            load_into(m2, load_file(str(out / v / "model.safetensors")), variant=v)
        src = [x for x in m.modules() if isinstance(x, (DualSharedNVFP4Linear,
                                                        DualSplitNVFP4Linear))]
        dst = [x for x in m2.modules() if isinstance(x, (DualSharedNVFP4Linear,
                                                         DualSplitNVFP4Linear))]
        sc = max(x._wq.abs().max().item() for x in src)
        dp = max((x._wq - y._wq).abs().max().item() for x, y in zip(src, dst)) / sc
        check(f"{name}: prefill weights reload", dp < 1e-6, f"rel={dp:.2e}")
        if isinstance(src[0], DualSplitNVFP4Linear):
            dd = max((x._wq_dec - y._wq_dec).abs().max().item()
                     for x, y in zip(src, dst)) / sc
            check(f"{name}: decode weights reload", dd < 1e-6, f"rel={dd:.2e}")


# ---------------------------------------------------------------------------
# 2. positions run under the right format
# ---------------------------------------------------------------------------
def _layer(name):
    m = build(name)
    return m, m.model.layers[0].mlp.gate_proj


def test_phase_routing():
    """A mixed mask must equal all-prefill on prefill rows and all-decode on decode
    rows — i.e. the mask genuinely selects per position, not per tensor."""
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m, lin = _layer(name)
        m.eval()
        x = torch.randn(2, 6, lin.in_features, device="cuda") * 0.5
        allp = torch.ones(2, 6, dtype=torch.bool, device="cuda")
        with quant_phase(allp):
            y_p = lin(x)
        with quant_phase(~allp):
            y_d = lin(x)
        check(f"{name}: prefill and decode paths differ", not torch.allclose(y_p, y_d),
              f"max|Δ|={(y_p - y_d).abs().max():.3e}")
        mask = torch.zeros(2, 6, dtype=torch.bool, device="cuda")
        mask[:, :3] = True                      # first half prompt, second half reply
        with quant_phase(mask):
            y_m = lin(x)
        check(f"{name}: mixed mask matches all-prefill on prompt rows",
              torch.equal(y_m[:, :3], y_p[:, :3]))
        check(f"{name}: mixed mask matches all-decode on reply rows",
              torch.equal(y_m[:, 3:], y_d[:, 3:]))


def test_generate_shape_inference():
    """With no mask (inference), q_len>1 is prefill and q_len==1 is decode — the
    shapes HF generate() actually produces."""
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m, lin = _layer(name)
        m.eval()
        x = torch.randn(1, 5, lin.in_features, device="cuda") * 0.5
        with quant_phase(torch.ones(1, 5, dtype=torch.bool, device="cuda")):
            ref_p = lin(x)
        with quant_phase(torch.zeros(1, 5, dtype=torch.bool, device="cuda")):
            ref_d = lin(x)
        y_multi = lin(x)                        # no mask -> prefill by shape
        check(f"{name}: multi-token forward takes the prefill path",
              torch.equal(y_multi, ref_p))
        one = x[:, :1]
        with quant_phase(torch.zeros(1, 1, dtype=torch.bool, device="cuda")):
            ref_one_d = lin(one)
        check(f"{name}: single-token forward takes the decode path",
              torch.equal(lin(one), ref_one_d))


def test_observer_sees_prefill_only():
    """input_global_scale is baked into the PREFILL checkpoint as a static scale.
    Decode-phase activations never pass through it in deployment, so letting them
    into the running max would inflate it and cost precision where it is used."""
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m, lin = _layer(name)
        m.train()
        lin.act_amax.zero_()
        lin._observed = False
        x = torch.randn(2, 8, lin.in_features, device="cuda") * 0.1
        x[:, 4:] *= 500.0                       # enormous DECODE-phase activations
        mask = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
        mask[:, :4] = True
        with quant_phase(mask):
            lin(x)
        pre_amax = x[:, :4].abs().max().item()
        check(f"{name}: act_amax tracks prefill positions",
              abs(lin.act_amax.item() - pre_amax) < 1e-4,
              f"got {lin.act_amax.item():.3f}, prefill amax {pre_amax:.3f}")
        check(f"{name}: act_amax ignores decode positions",
              lin.act_amax.item() < 0.1 * x.abs().max().item(),
              f"full amax was {x.abs().max().item():.1f}")


def test_mask_survives_gradient_checkpointing():
    """Gradient checkpointing recomputes the forward during backward. If the phase
    mask were scoped to the forward only, the recomputation would run every position
    as decode and the gradients would belong to a model that was never evaluated."""
    from torch.utils.checkpoint import checkpoint
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m, lin = _layer(name)
        m.train()
        x = torch.randn(2, 6, lin.in_features, device="cuda") * 0.5
        mask = torch.zeros(2, 6, dtype=torch.bool, device="cuda")
        mask[:, :3] = True

        def grads(use_ckpt):
            for p in lin.parameters():
                p.grad = None
            with quant_phase(mask):
                y = (checkpoint(lin, x, use_reentrant=False) if use_ckpt else lin(x))
                y.square().mean().backward()
            return [p.grad.detach().clone() for p in lin.parameters()]

        a, b = grads(False), grads(True)
        worst = max((p - q).abs().max().item() for p, q in zip(a, b))
        scale = max(p.abs().max().item() for p in a)
        check(f"{name}: checkpointed grads match non-checkpointed",
              worst / max(scale, 1e-12) < 1e-5, f"rel={worst / max(scale, 1e-12):.2e}")


def test_compiles_with_dynamic_shapes():
    """torch.compile(dynamic=True) must route phases correctly and match eager.

    Under dynamic shapes `x.shape[-2] > 1` is a SymBool, which is neither the True
    nor the False singleton — so an `is True` identity check silently falls through
    and hands a bool to torch.where(). That is invisible in eager mode and broke
    every compiled eval task with
    `TypeError: where() received an invalid combination of arguments`.
    """
    for name in ["nvfp4pdshared", "nvfp4pdsplit"]:
        m, lin = _layer(name)
        m.eval()
        x = torch.randn(2, 6, lin.in_features, device="cuda") * 0.5
        one = x[:, :1]
        with torch.no_grad():
            ref_multi, ref_one = lin(x), lin(one)
        clin = torch.compile(lin, dynamic=True)
        try:
            with torch.no_grad():
                got_multi, got_one = clin(x), clin(one)
        except Exception as e:
            check(f"{name}: compiles under dynamic shapes", False,
                  f"{type(e).__name__}: {str(e)[:110]}")
            continue
        check(f"{name}: compiled multi-token matches eager (prefill)",
              torch.allclose(got_multi, ref_multi, atol=1e-5),
              f"max|Δ|={(got_multi - ref_multi).abs().max():.2e}")
        check(f"{name}: compiled single-token matches eager (decode)",
              torch.allclose(got_one, ref_one, atol=1e-5),
              f"max|Δ|={(got_one - ref_one).abs().max():.2e}")
        # the two phases must still be genuinely different after compilation
        check(f"{name}: compiled phases remain distinct",
              not torch.allclose(got_multi[:, :1], got_one, atol=1e-6))


def test_prefill_mask_from_labels():
    labels = torch.tensor([[-100, -100, 7, 8, -100]], device="cuda")
    pm = prefill_mask_from_labels(labels)
    check("labels==-100 -> prefill, assistant tokens -> decode",
          pm.tolist() == [[True, True, False, False, True]])


if __name__ == "__main__":
    for fn in (test_checkpoints_are_plain_nvfp4, test_reload_is_order_independent,
               test_phase_routing, test_generate_shape_inference,
               test_observer_sees_prefill_only,
               test_mask_survives_gradient_checkpointing,
               test_compiles_with_dynamic_shapes,
               test_prefill_mask_from_labels):
        print(f"\n{fn.__name__}:")
        fn()
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nPASS: dual prefill/decode formats")
