import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from export.save import build_state_dict
from quantizers import REGISTRY, build_quantizer_params
from quantizers.dual import prefill_mask_from_labels, quant_phase
from quantizers.full_disag import (DualEmbedding, DualLMHead, DualParamModule,
                                   DualRMSNorm, apply_full_disag,
                                   dual_lm_head_weights, is_full_disag)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def build(quant="nvfp4pdsplit", tie=True, full=True):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, tie
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).cuda().float()
    if quant:
        params, _ = build_quantizer_params(quant, "")
        REGISTRY[quant]["apply"](m, **params)
    counts = apply_full_disag(m) if full else {}
    return m, counts


def _mask(B, T, device):
    labels = torch.full((B, T), -100, device=device)
    labels[:, T // 2:] = 1
    return labels, prefill_mask_from_labels(labels)


def test_swaps_embedding_norms_and_head():
    m, n = build()
    check("embedding is dual", isinstance(m.model.embed_tokens, DualEmbedding))
    check("lm_head is dual", isinstance(m.lm_head, DualLMHead))
    check("norms are dual", n["norm"] >= 2 * cfgless_layers(m),
          f"{n['norm']} norms swapped")
    left = [type(x).__name__ for x in m.modules()
            if type(x).__name__.endswith("RMSNorm") and not isinstance(x, DualRMSNorm)]
    check("no plain RMSNorm left", not left, f"{left[:3]}")
    check("is_full_disag", is_full_disag(m))


def cfgless_layers(m):
    return len(m.model.layers)


def test_tying_is_preserved_per_phase():
    m, n = build(tie=True)
    check("reported as tied", n["tied"] == 1)
    e, h = m.model.embed_tokens, m.lm_head
    check("prefill head IS the prefill embedding", h.weight_prefill is e.weight_prefill)
    check("decode head IS the decode embedding", h.weight_decode is e.weight_decode)
    check("the two phases are still separate tensors",
          h.weight_prefill is not h.weight_decode)
    n_tables = len({id(p) for p in m.parameters() if p.dim() == 2 and p.shape[0] == 512})
    check("two vocab tables, not four", n_tables == 2, f"{n_tables} tables")


def test_untied_model_gets_independent_head():
    m, n = build(tie=False)
    check("reported as untied", n["tied"] == 0)
    check("head is not the embedding",
          m.lm_head.weight_prefill is not m.model.embed_tokens.weight_prefill)


def test_phase_routing_selects_the_right_tensor():
    m, _ = build(quant=None)
    B, T = 2, 8
    labels, mask = _mask(B, T, "cuda")
    ids = torch.randint(0, 512, (B, T), device="cuda")

    e = m.model.embed_tokens
    with torch.no_grad():
        e.weight_prefill.fill_(1.0)
        e.weight_decode.fill_(-1.0)
    with quant_phase(mask):
        out = e(ids)
    check("prefill positions use the prefill table", bool((out[mask] == 1.0).all()))
    check("decode positions use the decode table", bool((out[~mask] == -1.0).all()))

    nrm = m.model.norm                       # already swapped by apply_full_disag
    check("final norm is dual", isinstance(nrm, DualRMSNorm))
    with torch.no_grad():
        nrm.weight_prefill.fill_(2.0)
        nrm.weight_decode.fill_(4.0)
    # Compare against the all-prefill and all-decode outputs on the SAME x: the two
    # phases hold different data, so comparing their magnitudes proves nothing.
    x = torch.randn(B, T, m.config.hidden_size, device="cuda")
    ones = torch.ones_like(mask)
    with quant_phase(ones):
        y_p = nrm(x)
    with quant_phase(~ones):
        y_d = nrm(x)
    with quant_phase(mask):
        y = nrm(x)
    check("norm: prefill rows match the all-prefill output", torch.equal(y[mask], y_p[mask]))
    check("norm: decode rows match the all-decode output", torch.equal(y[~mask], y_d[~mask]))
    check("the two phases really differ", not torch.equal(y_p, y_d))


def test_norm_handles_4d_qk_norm():
    """q_norm/k_norm see [B, T, heads, head_dim]; the mask must broadcast, not misalign."""
    m, _ = build(quant=None)
    B, T = 2, 8
    _, mask = _mask(B, T, "cuda")
    qn = m.model.layers[0].self_attn.q_norm
    check("q_norm was swapped", isinstance(qn, DualRMSNorm))
    x = torch.randn(B, T, 4, 32, device="cuda")
    with quant_phase(mask):
        y = qn(x)
    check("4-D input works and keeps shape", y.shape == x.shape)
    bad = torch.randn(B, T + 1, 4, 32, device="cuda")
    try:
        with quant_phase(mask):
            qn(bad)
        check("mismatched [B, T] is rejected", False)
    except RuntimeError:
        check("mismatched [B, T] is rejected", True)


def test_both_phases_receive_gradient():
    m, _ = build()
    B, T = 2, 16
    labels, mask = _mask(B, T, "cuda")
    ids = torch.randint(0, 512, (B, T), device="cuda")
    with quant_phase(mask):
        m(input_ids=ids).logits.square().mean().backward()
    duals = [(n, x) for n, x in m.named_modules() if isinstance(x, DualParamModule)]
    missing = [n for n, x in duals
               for p in x._pair() if p.grad is None or p.grad.abs().sum() == 0]
    check("every dual tensor got gradient", not missing,
          f"{len(missing)} of {2 * len(duals)} without grad: {missing[:3]}")


def test_export_ships_a_different_half_each_time():
    m, _ = build()
    sd_p = build_state_dict(m, "prefill")
    sd_d = build_state_dict(m, "decode")
    keys = ["model.embed_tokens.weight", "lm_head.weight",
            "model.layers.0.input_layernorm.weight"]
    for k in keys:
        check(f"{k} present in both", k in sd_p and k in sd_d)
    with torch.no_grad():          # force the halves apart so the check is meaningful
        m.model.embed_tokens.weight_decode.add_(1.0)
        m.model.layers[0].input_layernorm.weight_decode.add_(1.0)
    sd_p, sd_d = build_state_dict(m, "prefill"), build_state_dict(m, "decode")
    for k in keys:
        check(f"{k} differs between halves", not torch.equal(sd_p[k], sd_d[k]))
    leaked = [k for k in sd_p if k.endswith(("weight_prefill", "weight_decode"))]
    check("no raw dual params leak into the checkpoint", not leaked, f"{leaked[:3]}")


def test_head_pair_lookup():
    m, _ = build()
    pair = dual_lm_head_weights(m)
    check("dual_lm_head_weights finds the pair", pair is not None and len(pair) == 2)
    plain, _ = build(full=False)
    check("returns None without --full-disag", dual_lm_head_weights(plain) is None)


def test_only_the_boundary_token_uses_the_prefill_head():
    """The claim the split loss is built on: one scored prefill position per sequence."""
    B, T = 4, 32
    labels_data = torch.full((B, T), -100)
    labels_data[:, T // 2:] = 1
    prefill = prefill_mask_from_labels(labels_data)

    labels = labels_data[:, 1:].reshape(-1)          # target at t+1
    head_prefill = prefill[:, :-1].reshape(-1)       # head used at t
    scored = labels != -100
    n_pre = int((scored & head_prefill).sum())
    n_dec = int((scored & ~head_prefill).sum())
    check("exactly one scored prefill position per sequence", n_pre == B,
          f"{n_pre} for B={B}")
    check("everything else scored is decode", n_dec == int(scored.sum()) - B,
          f"{n_dec} decode positions")


def test_split_loss_matches_a_single_call_when_the_heads_agree():
    """If both heads hold the same weight, splitting must change nothing."""
    # training.qad first: importing it is what puts third_party/Liger-Kernel on sys.path
    from training.qad import _split_head_loss
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

    torch.manual_seed(0)
    B, T, H, V = 4, 32, 64, 128
    fn = LigerFusedLinearJSDLoss(weight_hard_loss=0.0, weight_soft_loss=1.0, beta=0.0,
                                 ignore_index=-100, compiled=False, chunk_size=256,
                                 return_soft_hard_loss=True)
    s = torch.randn(B * (T - 1), H, device="cuda").requires_grad_()
    t = torch.randn(B * (T - 1), H, device="cuda")
    w = torch.randn(V, H, device="cuda") * 0.05
    t_w = torch.randn(V, H, device="cuda") * 0.05

    labels_data = torch.full((B, T), -100, device="cuda")
    labels_data[:, T // 2:] = torch.randint(0, V, (B, T - T // 2), device="cuda")
    labels = labels_data[:, 1:].reshape(-1)
    head_prefill = prefill_mask_from_labels(labels_data)[:, :-1].reshape(-1)

    want = fn(s, w, t, t_w, true_labels=labels)
    got = _split_head_loss(fn, s, (w, w), t, t_w, labels, head_prefill)
    for name, a, b in zip(("loss", "kl", "ntp"), got, want):
        rel = abs(a.item() - b.item()) / max(abs(b.item()), 1e-9)
        check(f"split {name} == single call", rel < 1e-5,
              f"{a.item():.6f} vs {b.item():.6f}")

    # and the split really did exercise both branches
    check("both branches were used",
          int((labels != -100).logical_and(head_prefill).sum()) == B)


def test_training_loop_shape_produces_a_backwardable_loss():
    """Mirror the real loop: gradient checkpointing + autocast + the split-head loss.

    The unit test above feeds _split_head_loss tensors that already require grad. This one
    goes through the model the way training does, which is where a detached loss would
    actually show up.
    """
    from training.qad import _split_head_loss
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

    m, _ = build("nvfp4lloyd21split")
    m.config.use_cache = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()

    B, T = 2, 32
    labels_data = torch.full((B, T), -100, device="cuda")
    labels_data[:, T // 2:] = torch.randint(0, 512, (B, T - T // 2), device="cuda")
    ids = torch.randint(0, 512, (B, T), device="cuda")
    prefill = prefill_mask_from_labels(labels_data)

    fn = LigerFusedLinearJSDLoss(weight_hard_loss=0.0, weight_soft_loss=1.0, beta=0.0,
                                 ignore_index=-100, compiled=False, chunk_size=256,
                                 return_soft_hard_loss=True)
    with quant_phase(prefill):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = m.model(input_ids=ids).last_hidden_state
        check("hidden state carries grad", hidden.requires_grad)

        N, H = B * (T - 1), hidden.shape[-1]
        s = hidden[:, :-1].reshape(N, H).contiguous()
        t = s.detach().clone()
        labels = labels_data[:, 1:].reshape(N)
        head_prefill = prefill[:, :-1].reshape(N)
        pair = dual_lm_head_weights(m)

        loss, _, _ = _split_head_loss(fn, s, pair, t, pair[1].detach(),
                                      labels, head_prefill)
        check("split loss carries grad", loss.requires_grad,
              f"n_pre={int((labels != -100).logical_and(head_prefill).sum())} "
              f"n_dec={int((labels != -100).logical_and(~head_prefill).sum())}")
        loss.backward()
    check("embedding got gradient", m.model.embed_tokens.weight_decode.grad is not None)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    print("  all full-disag tests passed")
