"""Smoke tests for GSQ2BitLinear and the quantizer registry in qad.py."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist

# ── unit tests (no distributed) ──────────────────────────────────────────────

def test_construction():
    lin = nn.Linear(256, 128, bias=False)
    from quantizers.gsq import GSQ2BitLinear
    gsq = GSQ2BitLinear.from_linear(lin, groupsize=64)

    assert gsq.quant_logits.shape == (128 * 4, 256), gsq.quant_logits.shape
    assert gsq.scales.shape == (128, 256 // 64), gsq.scales.shape
    assert gsq._wq.shape == (128, 256), gsq._wq.shape
    print("  [OK] construction + shapes")


def test_eval_forward():
    lin = nn.Linear(256, 128, bias=True)
    from quantizers.gsq import GSQ2BitLinear
    gsq = GSQ2BitLinear.from_linear(lin)
    gsq.eval()

    x = torch.randn(4, 256)
    with torch.no_grad():
        out = gsq(x)
    assert out.shape == (4, 128), out.shape
    # eval uses hard _wq — no Gumbel noise, fully deterministic
    out2 = gsq(x)
    assert torch.allclose(out, out2), "eval forward should be deterministic"
    print("  [OK] eval forward (deterministic, uses _wq)")


def test_train_forward_and_backward():
    lin = nn.Linear(64, 32, bias=False)
    from quantizers.gsq import GSQ2BitLinear
    gsq = GSQ2BitLinear.from_linear(lin, groupsize=32)
    gsq.train()

    x = torch.randn(2, 64)
    out = gsq(x)
    assert out.shape == (2, 32)
    loss = out.sum()
    loss.backward()

    assert gsq.quant_logits.grad is not None, "quant_logits must have grad"
    assert gsq.scales.grad is not None, "scales must have grad"
    print("  [OK] train forward + backward (grads on logits and scales)")


def test_post_update_refreshes_wq():
    lin = nn.Linear(64, 32, bias=False)
    from quantizers.gsq import GSQ2BitLinear
    gsq = GSQ2BitLinear.from_linear(lin)

    wq_before = gsq._wq.clone()
    # Perturb logits to change the argmax on a few positions
    with torch.no_grad():
        gsq.quant_logits.data += torch.randn_like(gsq.quant_logits) * 10
    gsq.post_update(step=0, total_steps=100)
    wq_after = gsq._wq.clone()

    assert not torch.allclose(wq_before, wq_after), "_wq should change after logit perturbation"
    print("  [OK] post_update refreshes _wq buffer")


def test_schedule_annealing():
    lin = nn.Linear(64, 32)
    from quantizers.gsq import GSQ2BitLinear
    gsq = GSQ2BitLinear.from_linear(lin, temp_start=2.0, temp_end=0.05,
                                    scale_start=100.0, scale_end=500.0)

    gsq.post_update(step=0, total_steps=10)
    assert abs(gsq._temp.item() - 2.0) < 1e-4, gsq._temp.item()
    assert abs(gsq._scale_val.item() - 100.0) < 1e-4, gsq._scale_val.item()

    gsq.post_update(step=9, total_steps=10)
    assert abs(gsq._temp.item() - 0.05) < 1e-4, gsq._temp.item()
    assert abs(gsq._scale_val.item() - 500.0) < 1e-4, gsq._scale_val.item()

    gsq.post_update(step=5, total_steps=10)
    mid_temp  = 2.0 + (0.05 - 2.0)  * 5 / 9
    mid_scale = 100.0 + (500.0 - 100.0) * 5 / 9
    assert abs(gsq._temp.item() - mid_temp)  < 1e-3, gsq._temp.item()
    assert abs(gsq._scale_val.item() - mid_scale) < 1e-1, gsq._scale_val.item()
    print("  [OK] linear schedule annealing (T and scale_val)")


def test_apply_and_registry():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model("llama")
    cfg.hidden_size      = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.vocab_size = 256
    cfg.max_position_embeddings = 128

    from quantizers.gsq import apply_gsq2bit, apply_gsq3bit, GSQLinear

    for bits, apply_fn in [(2, apply_gsq2bit), (3, apply_gsq3bit)]:
        model = AutoModelForCausalLM.from_config(cfg)
        apply_fn(model, groupsize=32)
        layers = [m for m in model.modules() if isinstance(m, GSQLinear)]
        assert len(layers) > 0, f"no GSQ layers for bits={bits}"
        assert isinstance(model.lm_head, nn.Linear), "lm_head should not be quantized"
        assert layers[0].n_levels == 2 ** bits, f"wrong n_levels for bits={bits}"
        assert layers[0].quant_logits.shape[0] == layers[0]._out * (2 ** bits)
    print(f"  [OK] apply_gsq2bit/apply_gsq3bit: layers replaced, lm_head untouched, n_levels correct")


def test_param_groups():
    lin = nn.Linear(128, 64)
    from quantizers.gsq import GSQLinear, gsq_param_groups
    class M(nn.Module):
        def __init__(self): super().__init__(); self.fc = GSQLinear.from_linear(lin, bits=2); self.lm_head = nn.Linear(64, 32)
    m = M()
    groups = gsq_param_groups(m, lr=1e-4)
    assert len(groups) == 3, f"expected 3 groups, got {len(groups)}"
    assert groups[1]["weight_decay"] == 0.0, "scales must have weight_decay=0"
    assert abs(groups[1]["lr"] - 5e-5) < 1e-8, groups[1]["lr"]
    print("  [OK] gsq_param_groups: 3 groups, scales have wd=0 and lr=lr*0.5")


# ── distributed integration test ─────────────────────────────────────────────

def test_dist_integration():
    """End-to-end: load Qwen3-4B, apply GSQ, one forward+backward step."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantizers.gsq import apply_gsq2bit, post_update_all, gsq_param_groups
    from dist_adamw import DistAdamW

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    MODEL = os.environ.get("QAD_MODEL", "Qwen/Qwen3-4B")
    HF_HOME = os.environ.get("HF_HOME", "")

    if rank == 0:
        print(f"  Loading {MODEL} …", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to(device)
    model.config.use_cache = False

    apply_gsq2bit(model, groupsize=128)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    from quantizers.gsq import GSQ2BitLinear
    n_gsq = sum(1 for m in model.modules() if isinstance(m, GSQ2BitLinear))
    if rank == 0:
        print(f"  GSQ layers: {n_gsq}", flush=True)

    groups = gsq_param_groups(model, lr=1e-6)
    opt = DistAdamW(groups, lr=1e-6, betas=(0.9, 0.95), weight_decay=0.1)

    # Tiny forward + backward
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = torch.randint(0, 1000, (1, 32), device=device)

    opt.zero_grad()
    hidden = model.model(input_ids=ids).last_hidden_state  # [1, 32, H]
    loss = hidden.float().norm()
    loss.backward()
    opt.step()
    post_update_all(model, step=0, total_steps=10)

    if rank == 0:
        print("  [OK] dist integration: forward + backward + optimizer step + post_update", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_dist:
        dist.init_process_group("nccl")

    rank = dist.get_rank() if use_dist else 0

    if rank == 0:
        print("=== Unit tests (CPU) ===")
        test_construction()
        test_eval_forward()
        test_train_forward_and_backward()
        test_post_update_refreshes_wq()
        test_schedule_annealing()
        test_apply_and_registry()
        test_param_groups()
        print("\nAll unit tests passed.\n")

    if use_dist:
        print(f"=== Distributed integration test (rank {rank}) ===", flush=True)
        test_dist_integration()

    if use_dist:
        dist.destroy_process_group()
