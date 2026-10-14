import copy
import inspect
import os
from pathlib import Path
import sys

os.environ["USE_HUB_KERNELS"] = "NO"
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

sys.path.insert(0, str(Path(__file__).parent))
from prefiller import TextOnly, prepare_student, chunked_kl
from quantizers.base import set_recompute_wq
from quantizers.dual import quant_phase
from quantizers.frozen_decode import apply_nvfp4frozendec, load_frozen_decode


def base_model():
    from transformers.models.qwen3_5 import modeling_qwen3_5
    # Unit tests run on CPU even when optional GPU attention packages are installed.
    modeling_qwen3_5.torch_chunk_gated_delta_rule = inspect.unwrap(
        modeling_qwen3_5.torch_chunk_gated_delta_rule)
    modeling_qwen3_5.causal_conv1d_fn = inspect.unwrap(modeling_qwen3_5.causal_conv1d_fn)
    modeling_qwen3_5.causal_conv1d_update = inspect.unwrap(modeling_qwen3_5.causal_conv1d_update)
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"], tie_word_embeddings=False)
    return TextOnly(Qwen3_5ForCausalLM(config).float())


def test_reference_sharing_and_frozen_decode_after_update(tmp_path):
    base = base_model()
    student, reference = copy.deepcopy(base), copy.deepcopy(base)
    # Deliberately change EVERY external tensor. Shared tensors must match the
    # external decoder while preserving the reference's trainable partition.
    external = {k: v.clone() + 0.01 for k, v in base.state_dict().items()}
    save_file(external, str(tmp_path / "model.safetensors"))
    set_recompute_wq(True)
    apply_nvfp4frozendec(reference.model)
    reference.lm_head.requires_grad_(False)
    load_frozen_decode(reference.model, str(tmp_path), prefix="model.")
    prepare_student(student, tmp_path, "model.")
    ref_params = dict(reference.named_parameters())
    for name, p in student.named_parameters():
        if name.endswith("decode_frozen"):
            assert not p.requires_grad
        else:
            assert p.requires_grad == ref_params[name].requires_grad
            if p.requires_grad or ".inner." in name or name.endswith("weight_decode"):
                assert torch.equal(p, ref_params[name])
            else:
                assert torch.equal(p, external[name])
    assert torch.equal(student.model.embed_tokens.weight, external["model.embed_tokens.weight"])
    assert torch.equal(student.lm_head.weight, external["lm_head.weight"])
    frozen = {name: p.detach().clone() for name, p in student.named_parameters() if not p.requires_grad}
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    def decode_output():
        with torch.no_grad(), quant_phase(torch.zeros_like(ids, dtype=torch.bool)):
            return student.model(input_ids=ids, use_cache=False).last_hidden_state.clone()
    before = decode_output()
    # Decode-only execution also matches a standalone dequantized decoder, with
    # the same BF16 storage for specialized linear matrices.
    standalone = copy.deepcopy(base)
    standalone.load_state_dict(external)
    for name, module in standalone.model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b")):
            module.weight.data = module.weight.data.bfloat16().float()
    with torch.no_grad():
        expected = standalone.model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(before, expected)
    trainable_before = {n: p.detach().clone() for n, p in student.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
    mask = torch.tensor([[True, True, True, True, False, False, False, False]])
    with quant_phase(mask):
        student.model(input_ids=ids, use_cache=False).last_hidden_state[:, 4:].square().mean().backward()
    optimizer.step()
    assert any(not torch.equal(p, trainable_before[n]) for n, p in student.named_parameters() if p.requires_grad)
    assert all(torch.equal(p, frozen[n]) for n, p in student.named_parameters() if not p.requires_grad)
    assert torch.equal(before, decode_output())


def test_chunked_loss_and_gradient_match_dense():
    torch.manual_seed(7)
    s = torch.randn(2, 6, 16, requires_grad=True)
    t = torch.randn_like(s)
    sh, th = torch.nn.Linear(16, 64, bias=False), torch.nn.Linear(16, 64, bias=False)
    sh.requires_grad_(False)
    th.requires_grad_(False)
    labels = torch.tensor([[-100, -100, 3, 4, 5, -100], [-100, 2, 3, 4, 5, 6]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    valid = (labels[:, 1:] != -100) & attention[:, 1:].bool()
    tl = torch.log_softmax(th(t[:, :-1][valid]).float(), dim=-1)
    sl = torch.log_softmax(sh(s[:, :-1][valid]).float(), dim=-1)
    dense = (tl.exp() * (tl - sl)).sum(-1).mean()
    expected = torch.autograd.grad(dense, s)[0]
    chunked = chunked_kl(s, t, sh, th, labels, attention, chunk_size=2)
    actual = torch.autograd.grad(chunked, s)[0]
    torch.testing.assert_close(chunked, dense)
    torch.testing.assert_close(actual, expected)


def test_missing_decoder_tensor_is_rejected(tmp_path):
    base = base_model()
    save_file({"lm_head.weight": base.lm_head.weight.detach()}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Frozen decoder missing"):
        prepare_student(base, tmp_path, "model.")


def test_gguf_head_layout_round_trip():
    from gguf_decode import restore_heads, restore_layout
    config = Qwen3_5TextConfig(linear_num_key_heads=2, linear_num_value_heads=6,
                              linear_value_head_dim=4, linear_key_head_dim=4)
    hf = torch.arange(24 * 8).reshape(24, 8).float()
    tiled = hf.reshape(2, 3, 4, 8).transpose(0, 1).reshape(24, 8)
    assert torch.equal(restore_heads(tiled, config), hf)
    norm = torch.tensor([0.5, 1.5])
    assert torch.equal(restore_layout("model.norm.weight", norm, config), norm - 1)
