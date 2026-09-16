"""Fused Qwen3.5 decoder blocks: the linear-attention and full-attention variants.

    python qwen35_block.py        # checks BOTH variants against transformers

Qwen3.5 (`model_type: qwen3_5`, e.g. Qwen/Qwen3.8-27B) is not a Qwen3 with different
numbers. Three layers in four are `linear_attention` -- a gated delta net with a depthwise
causal conv and a recurrent state -- and every fourth is `full_attention` with a gated
output. `qwen3_block.py` cannot represent either one, which is why this file exists.

WHY THIS IS HARDER FOR THE HARNESS THAN GEMMA'S TWO VARIANTS
------------------------------------------------------------
Gemma 3 alternates sliding and global attention, but both layer types hold the SAME
parameters in the same byte layout; only the attention kwargs differ, which is why
offload_forward.py can carry them as one template with a variant axis. Qwen3.5's two types
hold genuinely different parameters -- `in_proj_qkv/z/b/a`, `conv1d`, `A_log`, `dt_bias`,
`norm` against `qkv`/`o` -- so they have different byte counts and cannot share a slot
spec. The harness needs one template, one plan() and one set of slots PER VARIANT.

WHERE THE KERNELS COME FROM
---------------------------
The chunked scan is `vllm.third_party.flash_linear_attention.ops.chunk_gated_delta_rule`,
the Triton FLA kernel vLLM vendors and uses for this architecture -- the same sourcing as
FA2 in gemma3_block.py and the NVFP4 kernels in nvfp4_linear.py. Measured against the
transformers reference at (S, num_v_heads) of (512, 4), (2048, 48) and (8192, 48): rel
4.5e-3, which is bf16 rounding.

NOT FlashInfer, on this box. `flashinfer.gdn_prefill.chunk_gated_delta_rule` is the other
obvious candidate and vLLM does dispatch to it on some architectures, but on GB10 (sm121)
it returns **all-NaN** for every configuration tried -- use_cp true/false/auto, with and
without an explicit zero initial_state, with and without output_final_state, at four shapes
including this model's real head geometry, while the torch reference on the identical
inputs is clean. Its sm120 CUTE-DSL path appears not to work here. Re-test before switching:
it would likely be faster if it worked.

The pure-torch `torch_chunk_gated_delta_rule` in transformers is the correctness reference
and NOT a benchmarking path: it is a Python loop over ceil(S/64) chunks, which at S=32768
is 512 iterations of four matmuls each. Under the fullgraph=True this harness compiles
with, that unrolls into thousands of nodes per layer, once per sequence length.

TWO NORM CONVENTIONS, IN ONE MODEL
----------------------------------
`Qwen3_5RMSNorm` (the block's input/post-attention norms, and q_norm/k_norm) is
ZERO-CENTERED -- `x_normed * (1 + w)`, with w initialized to zeros, as in Gemma 3. The
`Qwen3_5RMSNormGated` inside the delta net is NOT -- it is `w * x_normed * silu(z)` with w
initialized to ones. Getting either backwards leaves the model running and the output
wrong, which the self-test below is the only defence against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_geglu_quant as fused

try:
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as _gdn
except Exception as _exc:                       # noqa: BLE001 - reported by the self-test
    _gdn, _GDN_ERR = None, _exc

if _gdn is not None:
    # A custom op for the same reason gemma3_block.py wraps FA2: the kernel dispatches in
    # Python and has no fake-tensor rule, so tracing it under fullgraph=True fails outright.
    @torch.library.custom_op("qad_prefill::gdn_chunk", mutates_args=())
    def gdn_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                  beta: torch.Tensor) -> torch.Tensor:
        out = _gdn(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                   output_final_state=False, use_qk_l2norm_in_kernel=True)
        return out[0] if isinstance(out, tuple) else out

    @gdn_chunk.register_fake
    def _(q, k, v, g, beta):
        return torch.empty_like(v)


def rms_zero_centered(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm: normalize in fp32, scale by (1 + w), cast back."""
    f = x.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return (f * (1.0 + w.float())).type_as(x)


class Qwen35MLP(nn.Module):
    """SwiGLU with gate/up fused, as in qwen3_block.py -- 2 GEMMs, not 3."""

    def __init__(self, cfg, **kw):
        super().__init__()
        self.hidden, self.inter = cfg.hidden_size, cfg.intermediate_size
        self.gate_up = nn.Linear(self.hidden, 2 * self.inter, bias=False, **kw)
        self.down = nn.Linear(self.inter, self.hidden, bias=False, **kw)
        # A plain attribute, as in qwen3_block.py, so the fusion can be switched off to
        # measure what it is worth rather than being an implicit capability check.
        self.fuse_mlp_quant = True

    def forward(self, x):
        gu = self.gate_up(x)
        if self.fuse_mlp_quant and hasattr(self.down, "forward_prequantized"):
            # W4A4: emit fp4 straight out of SwiGLU instead of materialising the
            # intermediate in bf16 and quantizing it in a second pass. `down`'s input is
            # the widest activation in the block -- (S, 17408) here -- so quantizing it
            # separately costs a full bf16 write plus a bf16 read of the same tensor. The
            # dense Qwen3 and Gemma blocks have always done this; this block was written
            # without it, which showed up in the breakdown as a standalone act-quant bucket
            # and an inflated elementwise one.
            y4, ybs = fused.geglu_nvfp4(gu.reshape(-1, 2 * self.inter),
                                        self.down.x_gs, "silu")
            return self.down.forward_prequantized(y4, ybs, gu.shape[:-1])
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

    def load_hf(self, mlp):
        self.gate_up.weight.data.copy_(
            torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], 0))
        self.down.weight.data.copy_(mlp.down_proj.weight.data)


class Qwen35FullBlock(nn.Module):
    """The every-fourth layer: GQA with a sigmoid output gate.

    Differs from Qwen3 in two ways that matter. `q_proj` emits 2x head_dim per head -- the
    query and a gate, interleaved per head -- and the attention output is multiplied by
    sigmoid(gate) before `o`. RoPE is PARTIAL: only the first `rotary_dim` of each head is
    rotated (config partial_rotary_factor 0.25, so 64 of 256), the rest passes through.
    """

    def __init__(self, cfg, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.eps = cfg.rms_norm_eps
        self.hidden = cfg.hidden_size
        self.heads, self.kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = cfg.head_dim
        q_out, kv_out = self.heads * self.d * 2, self.kv * self.d
        # q(+gate), k, v in one GEMM. The split is [q|gate] per head, then k, then v.
        self.qkv = nn.Linear(self.hidden, q_out + 2 * kv_out, bias=cfg.attention_bias, **kw)
        self.o = nn.Linear(self.heads * self.d, self.hidden, bias=cfg.attention_bias, **kw)
        self.split = (q_out, kv_out, kv_out)
        self.q_norm = nn.Parameter(torch.zeros(self.d, **kw))
        self.k_norm = nn.Parameter(torch.zeros(self.d, **kw))
        self.in_norm = nn.Parameter(torch.zeros(self.hidden, **kw))
        self.post_norm = nn.Parameter(torch.zeros(self.hidden, **kw))
        self.mlp = Qwen35MLP(cfg, **kw)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        h = rms_zero_centered(x, self.in_norm, self.eps)
        qg, k, v = self.qkv(h).split(self.split, dim=-1)
        qg = qg.view(B, S, self.heads, 2 * self.d)
        q, gate = qg[..., :self.d], qg[..., self.d:]
        gate = gate.reshape(B, S, -1)

        q = rms_zero_centered(q, self.q_norm, self.eps).transpose(1, 2)
        k = rms_zero_centered(k.view(B, S, self.kv, self.d), self.k_norm,
                              self.eps).transpose(1, 2)
        v = v.view(B, S, self.kv, self.d).transpose(1, 2)

        r = cos.shape[-1]                       # rotary_dim; the tail is not rotated
        def rope(t):
            rot, keep = t[..., :r], t[..., r:]
            half = torch.cat((-rot[..., r // 2:], rot[..., :r // 2]), dim=-1)
            return torch.cat((rot * cos + half * sin, keep), dim=-1)
        q, k = rope(q), rope(k)

        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        a = a.transpose(1, 2).reshape(B, S, -1) * torch.sigmoid(gate)
        x = x + self.o(a)
        return x + self.mlp(rms_zero_centered(x, self.post_norm, self.eps))

    @classmethod
    def from_hf(cls, layer, cfg, dtype=torch.bfloat16, device="cuda"):
        blk = cls(cfg, dtype=dtype, device=device)
        sa = layer.self_attn
        blk.qkv.weight.data.copy_(torch.cat(
            [sa.q_proj.weight.data, sa.k_proj.weight.data, sa.v_proj.weight.data], 0))
        if sa.q_proj.bias is not None:
            blk.qkv.bias.data.copy_(torch.cat(
                [sa.q_proj.bias.data, sa.k_proj.bias.data, sa.v_proj.bias.data], 0))
        blk.o.weight.data.copy_(sa.o_proj.weight.data)
        blk.q_norm.data.copy_(sa.q_norm.weight.data)
        blk.k_norm.data.copy_(sa.k_norm.weight.data)
        blk.in_norm.data.copy_(layer.input_layernorm.weight.data)
        blk.post_norm.data.copy_(layer.post_attention_layernorm.weight.data)
        blk.mlp.load_hf(layer.mlp)
        return blk


class Qwen35LinearBlock(nn.Module):
    """Three layers in four: a gated delta net.

    in_proj -> depthwise causal conv (kernel 4, silu) -> chunked gated delta rule ->
    gated RMSNorm -> out_proj. q/k/z/b/a all come from one GEMM; only the q/k/v slice goes
    through the conv, which is why the projection is split immediately after.
    """

    def __init__(self, cfg, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        kw = dict(dtype=dtype, device=device)
        self.eps = cfg.rms_norm_eps
        self.hidden = cfg.hidden_size
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.dk * self.nk, self.dv * self.nv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.ksize = cfg.linear_conv_kernel_dim
        self.repeat = self.nv // self.nk

        # qkv|z in one GEMM, b|a in a SEPARATE one that stays bf16.
        #
        # vLLM stacks in_proj_qkv and in_proj_z into a single in_proj_qkvz and quantizes
        # that; it does not fuse in_proj_a/in_proj_b in, and the PTQ recipe ignores them
        # explicitly -- they are [5120 -> 48] each, carry the recurrence's decay and the
        # delta rule's beta, and are ~0.2% of a block's weights in exchange for 4-bit noise
        # on the state update. Fusing all four, as this block used to, meant `convert()`
        # quantized the 96 a/b rows too: only 0.58% of one GEMM and immaterial for speed,
        # but it made the harness measure a quantization the served model does not use.
        self.in_proj_qkvz = nn.Linear(self.hidden, self.conv_dim + self.value_dim,
                                      bias=False, **kw)
        self.in_proj_ba = nn.Linear(self.hidden, 2 * self.nv, bias=False, **kw)
        self.in_split = (self.conv_dim, self.value_dim)
        self.ba_split = (self.nv, self.nv)
        self.conv_w = nn.Parameter(torch.zeros(self.conv_dim, 1, self.ksize, **kw))
        self.dt_bias = nn.Parameter(torch.zeros(self.nv, dtype=torch.float32, device=device))
        self.A_log = nn.Parameter(torch.zeros(self.nv, dtype=torch.float32, device=device))
        self.gate_norm = nn.Parameter(torch.ones(self.dv, **kw))
        self.out = nn.Linear(self.value_dim, self.hidden, bias=False, **kw)
        self.in_norm = nn.Parameter(torch.zeros(self.hidden, **kw))
        self.post_norm = nn.Parameter(torch.zeros(self.hidden, **kw))
        self.mlp = Qwen35MLP(cfg, **kw)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        h = rms_zero_centered(x, self.in_norm, self.eps)
        qkv, z = self.in_proj_qkvz(h).split(self.in_split, dim=-1)
        b, a = self.in_proj_ba(h).split(self.ba_split, dim=-1)

        # Depthwise causal conv: pad left by k-1, drop the tail.
        c = F.conv1d(qkv.transpose(1, 2), self.conv_w, groups=self.conv_dim,
                     padding=self.ksize - 1)[..., :S]
        qkv = F.silu(c).transpose(1, 2)
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)

        beta = b.sigmoid()
        # float32 throughout: A_log.exp() in bf16 can reach -inf, and the kernel wants fp32.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GROUPED, not repeated. transformers expands q/k up to the value-head count, and
        # this block used to copy that -- but the FLA kernel infers the group ratio from the
        # head counts and vLLM's own prefill path passes them grouped. Expanding 16 k-heads
        # to 48 materialises q and k at 3x size (67 -> 201 MiB each at S=16384) and makes
        # the scan read 3x more. Measured at this model's geometry, S=16384: 26.62 ms
        # repeated vs 19.48 ms grouped, and the outputs are BITWISE identical.
        q = q.reshape(B, S, self.nk, self.dk)
        k = k.reshape(B, S, self.nk, self.dk)
        core = torch.ops.qad_prefill.gdn_chunk(
            q.contiguous(), k.contiguous(), v.reshape(B, S, self.nv, self.dv).contiguous(),
            g.reshape(B, S, self.nv).contiguous(),
            beta.reshape(B, S, self.nv).float().contiguous())

        # Gated norm: NOT zero-centered, and the gate is silu(z), not sigmoid.
        f = core.reshape(-1, self.dv).float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)
        f = self.gate_norm.float() * f * F.silu(z.reshape(-1, self.dv).float())
        x = x + self.out(f.type_as(x).reshape(B, S, -1))
        return x + self.mlp(rms_zero_centered(x, self.post_norm, self.eps))

    @classmethod
    def from_hf(cls, layer, cfg, dtype=torch.bfloat16, device="cuda"):
        blk = cls(cfg, dtype=dtype, device=device)
        la = layer.linear_attn
        blk.in_proj_qkvz.weight.data.copy_(torch.cat(
            [la.in_proj_qkv.weight.data, la.in_proj_z.weight.data], 0))
        blk.in_proj_ba.weight.data.copy_(torch.cat(
            [la.in_proj_b.weight.data, la.in_proj_a.weight.data], 0))
        blk.conv_w.data.copy_(la.conv1d.weight.data)
        blk.dt_bias.data.copy_(la.dt_bias.data.float())
        blk.A_log.data.copy_(la.A_log.data.float())
        blk.gate_norm.data.copy_(la.norm.weight.data)
        blk.out.weight.data.copy_(la.out_proj.weight.data)
        blk.in_norm.data.copy_(layer.input_layernorm.weight.data)
        blk.post_norm.data.copy_(layer.post_attention_layernorm.weight.data)
        blk.mlp.load_hf(layer.mlp)
        return blk


def block_for(layer, cfg, dtype=torch.bfloat16, device="cuda"):
    """Whichever variant this layer is, built from its transformers counterpart."""
    cls = Qwen35LinearBlock if layer.block_type == "linear_attention" else Qwen35FullBlock
    return cls.from_hf(layer, cfg, dtype=dtype, device=device)


def rope_tables(cfg, seq_len, dtype=torch.bfloat16, device="cuda"):
    """(cos, sin) for the text grid, taken from transformers' own rotary module.

    Qwen3.5's RoPE is mrope with interleaved sections, so reimplementing it here would be a
    second chance to get it subtly wrong for no benefit -- it is evaluated once, outside the
    timed region, exactly like qwen3_block.rope_tables.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    rot = Qwen3_5TextRotaryEmbedding(cfg).to(device)
    x = torch.zeros(1, seq_len, cfg.hidden_size, dtype=dtype, device=device)
    pos = torch.arange(seq_len, device=device)[None, None].expand(3, 1, seq_len)
    cos, sin = rot(x, pos)
    return cos.to(dtype)[:, None], sin.to(dtype)[:, None]


def _selftest():
    """Both variants against transformers' own layer, on a small random config.

    Deliberately a SMALL config: correctness is a property of the algebra, not of the size,
    and a 27B checkpoint is 52 GiB that would have to be downloaded and held in a unified
    memory pool shared with the GPU. Nothing here needs a checkpoint at all.
    """
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    transformers.logging.set_verbosity_error()

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        hidden_size=512, intermediate_size=1024, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention",
                     "full_attention"],
        full_attention_interval=4, attn_output_gate=True, rms_norm_eps=1e-6,
        vocab_size=256, _attn_implementation="sdpa",
    )
    S = 512
    dev, dt = "cuda", torch.bfloat16
    x = torch.randn(1, S, cfg.hidden_size, dtype=dt, device=dev) * 0.5
    cos, sin = rope_tables(cfg, S, dtype=dt, device=dev)
    pos = torch.arange(S, device=dev)[None, None].expand(3, 1, S)

    print(f"{torch.cuda.get_device_name(0)}  gdn kernel: "
          f"{'vllm FLA (triton)' if _gdn is not None else 'MISSING'}")
    ok = True
    for idx, kind in ((0, "linear_attention"), (3, "full_attention")):
        ref_layer = Qwen3_5DecoderLayer(cfg, idx).to(dev, dt).eval()
        with torch.no_grad():
            ref = ref_layer(x, position_embeddings=(cos[:, 0], sin[:, 0]),
                            attention_mask=None, position_ids=pos)
            ref = ref[0] if isinstance(ref, tuple) else ref
            got = block_for(ref_layer, cfg, dtype=dt, device=dev)(x, cos, sin)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        n_hf = sum(p.numel() for p in ref_layer.parameters())
        n_ours = sum(p.numel() for p in block_for(ref_layer, cfg, dtype=dt,
                                                  device=dev).parameters())
        flag = "OK" if rel < 2e-2 else "FAIL"
        ok &= rel < 2e-2
        print(f"  {kind:<18} rel vs transformers = {rel:.3e}  {flag}   "
              f"params {n_hf:,} vs {n_ours:,} "
              f"({'match' if n_hf == n_ours else 'MISMATCH'})")
    assert ok, "a Qwen3.5 block disagrees with transformers"
    print("both variants match transformers")


if __name__ == "__main__":
    _selftest()
