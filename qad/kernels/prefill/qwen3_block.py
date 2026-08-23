"""A Qwen3 decoder block written for benchmarking, not for generality.

transformers' `Qwen3DecoderLayer` is built to serve every cache type, every attention
backend, every mask convention and a pile of kwargs. All of that is dead weight here and
some of it is actively in the way:

  * SEVEN separate projections per layer (q, k, v, o, gate, up, down). q/k/v read the same
    activation and gate/up read the same activation, so that is four GEMM launches wasted
    on splitting work that could be one call each. Fusing gives 4 projections per layer.
  * a Cache object threaded through every call, whose `layer_idx` dynamo guards on -- which
    is what forces one compilation per layer unless you run cacheless.
  * mask construction, `**kwargs: TransformersKwargs`, output tuples, and attention-backend
    dispatch, all of which either graph-break or add guards.

This block is the same arithmetic with none of that: fused qkv and gate_up, SDPA with
`is_causal` and `enable_gqa` (so no seq x seq mask is ever materialized), fixed positional
arguments, a single tensor in and out. That makes it clean to `torch.compile(fullgraph=True)`
and clean to capture into a CUDA graph.

Numerically it must agree with transformers, and `python qwen3_block.py` checks that it
does against the real model. Qwen3 specifics that are easy to get wrong and are pinned by
that check:

  * q_norm / k_norm are RMSNorms over HEAD_DIM, applied per head after the projection and
    before RoPE. This is the thing Qwen3 added over Qwen2; skipping it gives plausible
    garbage.
  * GQA: n_kv_heads < n_heads, handled by `enable_gqa=True` rather than materializing
    repeated K/V.
  * RoPE is applied to q and k only, with the standard rotate-half convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_geglu_quant as fused


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """cos/sin broadcast as (1, 1, S, head_dim) against (B, H, S, head_dim)."""
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Qwen3Block(nn.Module):
    """One decoder layer: fused qkv, fused gate_up, SDPA, RMSNorm throughout."""

    def __init__(self, cfg, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        q_dim, kv_dim = self.n_heads * self.head_dim, self.n_kv * self.head_dim

        kw = dict(bias=False, dtype=dtype, device=device)
        self.qkv = nn.Linear(self.hidden, q_dim + 2 * kv_dim, **kw)
        self.o = nn.Linear(q_dim, self.hidden, **kw)
        self.gate_up = nn.Linear(self.hidden, 2 * self.inter, **kw)
        self.down = nn.Linear(self.inter, self.hidden, **kw)

        p = lambda n: nn.Parameter(torch.ones(n, dtype=dtype, device=device))
        self.in_norm = p(self.hidden)
        self.post_norm = p(self.hidden)
        self.q_norm = p(self.head_dim)     # per-head norms: the Qwen3 addition
        self.k_norm = p(self.head_dim)
        self.split = (q_dim, kv_dim, kv_dim)
        # Fold the activation quantization into SwiGLU when the MLP is W4A4. A plain
        # attribute rather than an implicit capability check, so the fusion can be switched
        # off to measure what it is worth.
        self.fuse_mlp_quant = True

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
                ) -> torch.Tensor:
        B, S, _ = x.shape
        h = F.rms_norm(x, (self.hidden,), self.in_norm, self.eps)
        q, k, v = self.qkv(h).split(self.split, dim=-1)

        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_kv, self.head_dim)
        v = v.view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (self.head_dim,), self.q_norm, self.eps).transpose(1, 2)
        k = F.rms_norm(k, (self.head_dim,), self.k_norm, self.eps).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        # is_causal lets SDPA apply the mask internally; enable_gqa avoids expanding K/V.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, -1))

        h = F.rms_norm(x, (self.hidden,), self.post_norm, self.eps)
        gu = self.gate_up(h)
        if self.fuse_mlp_quant and hasattr(self.down, "forward_prequantized"):
            # W4A4: emit fp4 straight out of SwiGLU rather than materializing the
            # intermediate in bf16 and quantizing it as a separate pass. Same fusion as
            # Gemma's, with SiLU instead of GeLU -- see fused_geglu_quant.py.
            y4, ybs = fused.geglu_nvfp4(gu.reshape(-1, 2 * self.inter), self.down.x_gs,
                                        "silu")
            return x + self.down.forward_prequantized(y4, ybs, gu.shape[:-1])
        gate, up = gu.chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)

    @classmethod
    def from_hf(cls, layer, cfg, dtype=torch.bfloat16, device="cuda") -> "Qwen3Block":
        """Build from a transformers Qwen3DecoderLayer, concatenating the fused pairs."""
        blk = cls(cfg, dtype=dtype, device=device)
        a, m = layer.self_attn, layer.mlp
        with torch.no_grad():
            blk.qkv.weight.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight,
                                            a.v_proj.weight], 0))
            blk.o.weight.copy_(a.o_proj.weight)
            blk.gate_up.weight.copy_(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0))
            blk.down.weight.copy_(m.down_proj.weight)
            blk.in_norm.copy_(layer.input_layernorm.weight)
            blk.post_norm.copy_(layer.post_attention_layernorm.weight)
            blk.q_norm.copy_(a.q_norm.weight)
            blk.k_norm.copy_(a.k_norm.weight)
        return blk


def rope_tables(cfg, seq_len, dtype=torch.bfloat16, device="cuda"):
    """(cos, sin) shaped (1, 1, S, head_dim), ready to broadcast over (B, H, S, D).

    Computed once outside the timed region -- they depend only on position, so recomputing
    them per forward would be measuring the wrong thing, and a fixed tensor is what CUDA
    graph capture needs anyway.
    """
    base = getattr(cfg, "rope_theta", 1000000.0)
    d = cfg.head_dim
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None])


def _selftest():
    """Agreement with transformers' own layer, which is the only thing that matters."""
    import transformers
    from transformers import AutoModel
    transformers.logging.set_verbosity_error()

    name = "Qwen/Qwen3-0.6B"
    model = AutoModel.from_pretrained(name, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").cuda().eval()
    cfg = model.config
    S = 256
    torch.manual_seed(0)
    x = torch.randn(1, S, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(S, device="cuda")[None]

    with torch.no_grad():
        hf_pe = model.rotary_emb(x, pos)
        ref = model.layers[0](x, attention_mask=None, position_ids=pos,
                              past_key_values=None, use_cache=False,
                              position_embeddings=hf_pe)
        ref = ref[0] if isinstance(ref, tuple) else ref

        blk = Qwen3Block.from_hf(model.layers[0], cfg)
        cos, sin = rope_tables(cfg, S)
        got = blk(x, cos, sin)

    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    print(f"{name} layer 0, S={S}: rel vs transformers = {rel:.3e}")
    assert rel < 2e-2, f"custom block disagrees with transformers ({rel:.2e})"

    n_hf = sum(p.numel() for p in model.layers[0].parameters())
    n_ours = sum(p.numel() for p in blk.parameters())
    print(f"parameters: transformers {n_hf:,} vs fused {n_ours:,} "
          f"({'match' if n_hf == n_ours else 'MISMATCH'})")
    assert n_hf == n_ours
    print("custom block matches transformers")


if __name__ == "__main__":
    _selftest()
