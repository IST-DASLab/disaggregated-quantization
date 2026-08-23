"""A Gemma 3 decoder block written for benchmarking, in the shape of qwen3_block.py.

Same rationale as that file -- fused qkv and gate_up, fixed positional arguments, one
tensor in and out, clean to `torch.compile(fullgraph=True)` and to capture into a CUDA
graph. What is different is the attention, and it is different in a way that dominates the
numbers, so it gets its own section.

THE 5:1 SLIDING PATTERN IS THE WHOLE STORY AT LONG SEQUENCES
------------------------------------------------------------
Gemma 3 alternates five LOCAL layers (sliding window: 512 tokens on 270m/1b, 1024 on
4b/12b) with one GLOBAL layer, per `config.layer_types`. A local layer at 32k attends to
1024 keys, not 32768, so it does ~32x less attention work than a global one. Running the
stack full-causal throughout -- what you get by default if you hand SDPA `is_causal=True`
and move on -- inflates Gemma 3 prefill by ~3.2x at 32k and turns a model that is mostly
memory-bound into a fake compute-bound one.

BACKEND: MEASURED, NOT ASSUMED (GB10 / sm_121, 4b geometry H=8 KV=4 D=256, window 1024)
----------------------------------------------------------------------------------------
                                        S=8192      S=32768
    SDPA is_causal, full-causal          3.14 ms     50.6 ms    <- best for GLOBAL layers
    FA2 varlen, full-causal              3.75        53.0
    FA4 (flash_attn.cute), full-causal   3.06        49.8
    flex_attention, full-causal mask       --       123.2
    ....................................................................
    flex_attention, sliding BlockMask    2.03         8.7
    FA4, native window_size              3.02        49.0       <- window buys ~nothing
    FA2 varlen, native window_size       1.47         6.0       <- best for LOCAL layers

So this block uses BOTH: SDPA `is_causal=True` for global layers, and FA2's native sliding
window for local ones. No single backend wins both jobs, and the spread is large enough
(5.6x on 5-of-6 layers at 32k) that picking one uniformly would be the dominant error in
the whole measurement.

Notes on the losers, since "use flash attention" is the obvious instinct and it is only
half right here:

  * FA4 is installed (`flash_attn` is a namespace package holding only `cute`) and does run
    on sm121, but ONLY with a hand-forced small tile: its default hd=256 config asks for
    128 KiB of shared memory and this arch allows 101376 bytes, so the launch is rejected.
    With `tile_mn=(64, 64)` it is correct and roughly matches SDPA -- but its `window_size`
    yields 49.0 ms against 49.8 ms unwindowed, i.e. the mask is applied without the
    out-of-window key blocks being skipped. Correct, and no faster.
  * FA2 comes from vLLM's bundled build (`vllm.vllm_flash_attn`); the standalone flash_attn
    wheel here has no `flash_attn_func` at all. FA3 refuses to load (needs 9.x).
  * flex_attention is the fallback if FA2 ever goes away: 8.7 ms against FA2's 6.0, same
    correctness, no custom op needed. It is simply slower, and at full-causal density it is
    a much slower flash kernel.

FA2 needs a custom-op wrapper (see `fa2_sliding`) because it has no fake-tensor rule, and
without one `fullgraph=True` fails with "Operator does not support running with fake
tensors". Wrapping it opaque also keeps inductor from trying to fuse across it, which it
should not do anyway.

Which variant a block runs is decided by the CALLER, by passing the window kwargs or not,
rather than by a flag on the module: the offload harness executes every block through one
template whose weights are rebound per block, so the module cannot know which layer it is.
Two kwarg variants means two compilations and two captured graphs -- not two templates, and
not one per layer.

LAYOUT
------
The FA2 path wants (S, H, D) and the projection already produces (B, S, H*D), so for B=1
the local layers need no transpose at all -- `.view(B, S, H, D)[0]` is exactly it. Only the
SDPA path transposes to (B, H, S, D). RoPE tables are therefore built as (1, S, 1, D) to
broadcast against the (B, S, H, D) layout both paths share before attention.

GEMMA SPECIFICS THAT SILENTLY CORRUPT THE OUTPUT IF MISSED
-----------------------------------------------------------
  * RMSNorm scales by (1 + weight), not by weight. The checkpoint stores weights centred on
    zero, so applying them Llama-style multiplies most activations by ~0 -- and the model
    still runs and still produces tokens. `_selftest` is what catches this.
  * FOUR norms per layer: input and post_attention around the attention branch,
    pre_feedforward and post_feedforward around the MLP. The two post-norms apply to the
    branch OUTPUT before the residual add, which is not where a Llama-shaped block puts
    them.
  * q_norm / k_norm over head_dim, applied before RoPE, as in Qwen3.
  * The attention scale is `query_pre_attn_scalar ** -0.5`, NOT `head_dim ** -0.5`. They
    coincide at 256 for every Gemma 3 size shipped so far, so a wrong one costs nothing
    today and breaks on the next config; it is passed explicitly.
  * TWO RoPE bases: local layers use theta 1e4, global layers 1e6. On 4b and 12b the global
    layers additionally carry linear scaling by 8 (positions divided by 8); on 270m and 1b
    they do not.
  * The MLP is GeGLU with the tanh approximation, not SiLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_geglu_quant as fused

_FA2_ERR = None
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except Exception as exc:                                  # pragma: no cover
    flash_attn_varlen_func, _FA2_ERR = None, exc


@torch.library.custom_op("prefill::fa2_sliding", mutates_args=())
def fa2_sliding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen: int, window: int, scale: float) -> torch.Tensor:
    """Causal FA2 restricted to `window` keys, on (S, H, D) varlen tensors.

    A custom op rather than a direct call: `flash_attn_varlen_func` has no fake-tensor rule,
    so `torch.compile(fullgraph=True)` refuses it. `window - 1` on the left because FA2
    counts the query's own position separately, while Gemma's mask is `q - k < window`
    inclusive of self.
    """
    return flash_attn_varlen_func(
        q, k, v, max_seqlen_q=max_seqlen, cu_seqlens_q=cu_seqlens,
        max_seqlen_k=max_seqlen, cu_seqlens_k=cu_seqlens, causal=True,
        softmax_scale=scale, window_size=(window - 1, 0), fa_version=2)


@fa2_sliding.register_fake
def _(q, k, v, cu_seqlens, max_seqlen, window, scale):
    return torch.empty_like(q)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """cos/sin broadcast as (1, S, 1, head_dim) against (B, S, H, head_dim)."""
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def layer_types(cfg) -> list[str]:
    """['sliding_attention', ..., 'full_attention', ...], one per layer.

    Read from the config rather than reconstructed from a 5:1 rule: 270m, 4b and 12b do not
    all declare `sliding_window_pattern`, and the explicit list is authoritative.
    """
    lt = getattr(cfg, "layer_types", None)
    if lt:
        return list(lt)
    every = getattr(cfg, "sliding_window_pattern", 6) or 6
    return ["full_attention" if (i + 1) % every == 0 else "sliding_attention"
            for i in range(cfg.num_hidden_layers)]


class Gemma3Block(nn.Module):
    """One decoder layer: fused qkv, fused gate_up, four norms, GeGLU.

    Local vs global is a property of the CALL (`window > 0` or not), not of the module --
    see the module docstring.
    """

    def __init__(self, cfg, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.scale = float(getattr(cfg, "query_pre_attn_scalar", self.head_dim)) ** -0.5
        q_dim, kv_dim = self.n_heads * self.head_dim, self.n_kv * self.head_dim

        kw = dict(bias=False, dtype=dtype, device=device)
        self.qkv = nn.Linear(self.hidden, q_dim + 2 * kv_dim, **kw)
        self.o = nn.Linear(q_dim, self.hidden, **kw)
        self.gate_up = nn.Linear(self.hidden, 2 * self.inter, **kw)
        self.down = nn.Linear(self.inter, self.hidden, **kw)

        # Stored ALREADY as (1 + w). Gemma's norm is x_hat * (1 + w); folding the +1 in at
        # load time makes this one fused F.rms_norm call instead of an add per element per
        # forward, and is exactly the same arithmetic. from_hf does the folding.
        p = lambda n: nn.Parameter(torch.ones(n, dtype=dtype, device=device))
        self.in_norm = p(self.hidden)
        self.post_attn_norm = p(self.hidden)
        self.pre_ff_norm = p(self.hidden)
        self.post_ff_norm = p(self.hidden)
        self.q_norm = p(self.head_dim)
        self.k_norm = p(self.head_dim)
        self.split = (q_dim, kv_dim, kv_dim)
        # Fold the activation quantization into the GeGLU when the MLP is W4A4. A plain
        # attribute rather than an implicit capability check, so the fusion can be turned
        # off to measure what it is worth -- and so turning it off is a one-line A/B rather
        # than a different code path.
        self.fuse_mlp_quant = True

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None, max_seqlen: int = 0,
                window: int = 0) -> torch.Tensor:
        B, S, _ = x.shape
        h = F.rms_norm(x, (self.hidden,), self.in_norm, self.eps)
        q, k, v = self.qkv(h).split(self.split, dim=-1)

        q = q.view(B, S, self.n_heads, self.head_dim)
        k = k.view(B, S, self.n_kv, self.head_dim)
        v = v.view(B, S, self.n_kv, self.head_dim)
        q = F.rms_norm(q, (self.head_dim,), self.q_norm, self.eps)
        k = F.rms_norm(k, (self.head_dim,), self.k_norm, self.eps)
        q, k = apply_rope(q, k, cos, sin)

        if window > 0:                # local layer: FA2 skips out-of-window key blocks
            a = fa2_sliding(q[0], k[0], v[0], cu_seqlens, max_seqlen, window,
                            self.scale).unsqueeze(0)
        else:                         # global layer: SDPA owns full-causal here
            a = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, enable_gqa=True, scale=self.scale).transpose(1, 2)

        a = self.o(a.reshape(B, S, -1))
        # Gemma normalizes the BRANCH OUTPUT and then adds the residual.
        x = x + F.rms_norm(a, (self.hidden,), self.post_attn_norm, self.eps)

        h = F.rms_norm(x, (self.hidden,), self.pre_ff_norm, self.eps)
        gu = self.gate_up(h)
        if self.fuse_mlp_quant and hasattr(self.down, "forward_prequantized"):
            # W4A4: emit fp4 straight out of the GeGLU instead of materializing the
            # intermediate in bf16 and quantizing it as a separate pass. Duck-typed rather
            # than isinstance so this module needs no import of the quantized layer; the
            # attribute is resolved once at trace time, so it costs nothing at runtime.
            y4, ybs = fused.geglu_nvfp4(gu.reshape(-1, 2 * self.inter), self.down.x_gs)
            m = self.down.forward_prequantized(y4, ybs, gu.shape[:-1])
        else:
            gate, up = gu.chunk(2, dim=-1)
            m = self.down(F.gelu(gate, approximate="tanh") * up)
        return x + F.rms_norm(m, (self.hidden,), self.post_ff_norm, self.eps)

    @classmethod
    def from_hf(cls, layer, cfg, dtype=torch.bfloat16, device="cuda") -> "Gemma3Block":
        """Build from a transformers Gemma3DecoderLayer, folding +1 into every norm."""
        blk = cls(cfg, dtype=dtype, device=device)
        a, m = layer.self_attn, layer.mlp
        with torch.no_grad():
            blk.qkv.weight.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight,
                                            a.v_proj.weight], 0))
            blk.o.weight.copy_(a.o_proj.weight)
            blk.gate_up.weight.copy_(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0))
            blk.down.weight.copy_(m.down_proj.weight)
            for dst, src in ((blk.in_norm, layer.input_layernorm),
                             (blk.post_attn_norm, layer.post_attention_layernorm),
                             (blk.pre_ff_norm, layer.pre_feedforward_layernorm),
                             (blk.post_ff_norm, layer.post_feedforward_layernorm),
                             (blk.q_norm, a.q_norm), (blk.k_norm, a.k_norm)):
                dst.copy_(1.0 + src.weight.to(dtype))
        return blk


def _rope_theta(cfg, kind: str) -> tuple[float, float]:
    """(theta, position_divisor) for 'sliding_attention' / 'full_attention'.

    transformers 5.x carries these per layer type in `rope_scaling`; older configs used
    `rope_local_base_freq` and `rope_theta`. Both are read, so this does not silently pick
    one base for every layer -- which would be invisible in a latency benchmark and wrong
    in any correctness check.
    """
    rs = getattr(cfg, "rope_scaling", None) or {}
    entry = rs.get(kind) if isinstance(rs, dict) else None
    if isinstance(entry, dict):
        theta = float(entry.get("rope_theta", 10000.0))
        div = float(entry.get("factor", 1.0)) if entry.get("rope_type") == "linear" else 1.0
        return theta, div
    if kind == "sliding_attention":
        return float(getattr(cfg, "rope_local_base_freq", None) or 10000.0), 1.0
    return float(getattr(cfg, "rope_theta", None) or 1000000.0), 1.0


def rope_tables(cfg, seq_len, dtype=torch.bfloat16, device="cuda"):
    """{'sliding_attention': (cos, sin), 'full_attention': (cos, sin)}, (1, S, 1, D) each.

    Two tables, because the two layer types use different bases -- and on 4b/12b the global
    layers also divide positions by 8. Computed once, outside the timed region, so they are
    fixed tensors that CUDA graph capture can point at.
    """
    d = cfg.head_dim
    out = {}
    for kind in ("sliding_attention", "full_attention"):
        theta, div = _rope_theta(cfg, kind)
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=device,
                                            dtype=torch.float32) / d))
        t = torch.arange(seq_len, device=device, dtype=torch.float32) / div
        emb = torch.cat((torch.outer(t, inv),) * 2, dim=-1)
        out[kind] = (emb.cos().to(dtype)[None, :, None], emb.sin().to(dtype)[None, :, None])
    return out


def attention_kwargs(cfg, seq_len, dtype=torch.bfloat16, device="cuda"):
    """(variants, variant_of) for the whole stack.

    variants[j] is a kwargs dict ready to hand to Gemma3Block.forward; variant_of[i] is the
    index of the variant layer i runs. Two variants for Gemma 3, so the offload harness
    compiles twice and captures two graphs per weight slot rather than one per layer.
    """
    rope = rope_tables(cfg, seq_len, dtype=dtype, device=device)
    window = int(getattr(cfg, "sliding_window", 0) or 0)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kinds = ["full_attention", "sliding_attention"]
    variants = []
    for kind in kinds:
        cos, sin = rope[kind]
        kw = dict(cos=cos, sin=sin)
        # A window at least as long as the sequence is not a window: below `window` tokens
        # the two layer types compute the same thing, and SDPA is the faster way to do it.
        if kind == "sliding_attention" and 0 < window < seq_len:
            kw.update(cu_seqlens=cu, max_seqlen=seq_len, window=window)
        variants.append(kw)
    idx = {k: j for j, k in enumerate(kinds)}
    variant_of = [idx.get(t, 0) for t in layer_types(cfg)]
    return variants, variant_of


def _selftest():
    """Agreement with transformers' own layer, for a local AND a global layer.

    Both are checked because they differ in mask and in RoPE base, and a block that gets the
    global layer right can still be wrong about the window -- which is the expensive
    mistake. S is chosen above the window so the window actually bites; at S <= window the
    two layer types are indistinguishable and the test would prove nothing.
    """
    import transformers
    from transformers import AutoConfig, AutoModel
    transformers.logging.set_verbosity_error()
    if flash_attn_varlen_func is None:
        raise SystemExit(f"FA2 unavailable: {_FA2_ERR}")

    name = "google/gemma-3-270m"
    cfg_all = AutoConfig.from_pretrained(name)
    cfg = getattr(cfg_all, "text_config", cfg_all)
    model = AutoModel.from_pretrained(name, dtype=torch.bfloat16,
                                      attn_implementation="eager").cuda().eval()
    model = getattr(model, "language_model", model)
    types = layer_types(cfg)
    S = 1024                          # > sliding_window (512)
    torch.manual_seed(0)
    x = torch.randn(1, S, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(S, device="cuda")[None]

    print(f"{name}: {len(types)} layers, {types.count('sliding_attention')} sliding / "
          f"{types.count('full_attention')} full, window {cfg.sliding_window}, S={S}")

    variants, variant_of = attention_kwargs(cfg, S)
    ok = True
    with torch.no_grad():
        for idx in (types.index("sliding_attention"), types.index("full_attention")):
            kind = types[idx]
            # Feed transformers an explicit causal(+window) bias so the reference mask is
            # unambiguous rather than whatever its mask plumbing decides to build.
            i = torch.arange(S, device="cuda")
            m = i[:, None] >= i[None, :]
            if kind == "sliding_attention":
                m = m & ((i[:, None] - i[None, :]) < cfg.sliding_window)
            bias = torch.zeros(1, 1, S, S, dtype=torch.bfloat16, device="cuda")
            bias.masked_fill_(~m[None, None], torch.finfo(torch.bfloat16).min)

            theta, div = _rope_theta(cfg, kind)
            d = cfg.head_dim
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, device="cuda",
                                                dtype=torch.float32) / d))
            t = torch.arange(S, device="cuda", dtype=torch.float32) / div
            emb = torch.cat((torch.outer(t, inv),) * 2, dim=-1)
            pe = (emb.cos().to(torch.bfloat16)[None], emb.sin().to(torch.bfloat16)[None])

            ref = model.layers[idx](x, position_embeddings=pe, attention_mask=bias,
                                    position_ids=pos, past_key_values=None,
                                    use_cache=False)
            ref = ref[0] if isinstance(ref, tuple) else ref

            blk = Gemma3Block.from_hf(model.layers[idx], cfg)
            got = blk(x, **variants[variant_of[idx]])
            rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
            good = rel < 2e-2
            ok &= good
            print(f"  layer {idx:>2} ({kind:<17}) rel vs transformers = {rel:.3e}  "
                  f"{'OK' if good else 'MISMATCH'}")

        n_hf = sum(p.numel() for p in model.layers[0].parameters())
        n_ours = sum(p.numel() for p in Gemma3Block.from_hf(model.layers[0],
                                                            cfg).parameters())
        print(f"parameters: transformers {n_hf:,} vs fused {n_ours:,} "
              f"({'match' if n_hf == n_ours else 'MISMATCH'})")
        assert n_hf == n_ours
    assert ok, "custom block disagrees with transformers"
    print("custom block matches transformers on both layer types")


if __name__ == "__main__":
    _selftest()
