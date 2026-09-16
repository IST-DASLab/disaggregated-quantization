"""Gates for nvfp4frozendec: NVFP4 W4A4 prefill + FROZEN EXTERNAL decode.

    python tests/test_nvfp4frozendec.py

The property this format lives or dies by is that the decode half NEVER MOVES. Most of
these gates exist to make the failure modes that look healthy fail loudly instead:

  * a decode weight that silently stayed at the base model's values
  * a decode weight that received a gradient anyway
  * the frozen tensors leaking into state_dict, hence into every training state save and
    into an export that is supposed to be prefill-only
  * norms whose math changed because the wrapper reimplemented them
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantizers import REGISTRY, build_quantizer_params            # noqa: E402
from quantizers.dual import prefill_mask_from_labels, quant_phase  # noqa: E402
from quantizers.frozen_decode import (FrozenDecodeNorm,            # noqa: E402
                                      NVFP4FrozenDecodeLinear,
                                      apply_nvfp4frozendec)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


class GatedNorm(nn.Module):
    """Stand-in for Qwen3_5RMSNormGated: takes a gate, scales by `weight` (no 1+)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.variance_epsilon = 1e-6

    def forward(self, x, gate):
        v = x.float().pow(2).mean(-1, keepdim=True)
        h = x.float() * torch.rsqrt(v + self.variance_epsilon)
        return (self.weight * h.to(x.dtype)) * torch.nn.functional.silu(gate.float())


class PlusOneNorm(nn.Module):
    """Stand-in for Qwen3_5RMSNorm: scales by (1.0 + weight), NOT by weight."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        h = x.float() * torch.rsqrt(v + self.variance_epsilon)
        return (h * (1.0 + self.weight.float())).to(x.dtype)


print("== registry ==")
e = REGISTRY["nvfp4frozendec"]
check("variants is prefill-only", e["variants"] == ["prefill"], str(e["variants"]))
check("name is alphanumeric", "nvfp4frozendec".isalnum())
_, h_a = build_quantizer_params("nvfp4frozendec", '{"decode_model": "/a"}')
_, h_b = build_quantizer_params("nvfp4frozendec", '{"decode_model": "/b"}')
check("decode_model changes the tag hash", h_a != h_b, f"{h_a} vs {h_b}")

print("\n== linear: frozen decode ==")
torch.manual_seed(0)
lin = nn.Linear(64, 32, bias=False)
base_w = lin.weight.data.clone()
layer = NVFP4FrozenDecodeLinear.from_linear(lin, block_size=16)
check("decode_frozen starts empty", not layer.has_decode)

external = torch.randn_like(base_w)
layer.set_decode_frozen(external)
check("decode_frozen filled", layer.has_decode)
check("decode DIFFERS from the base weight", not torch.allclose(
    layer.decode_frozen.float(), base_w, atol=1e-6),
    "a silent no-op would leave these equal")

# The gate that matters most: frozen tensors must not reach state_dict, or they land in
# every training state save AND in the prefill-only export.
sd = layer.state_dict()
check("decode_frozen NOT in state_dict", "decode_frozen" not in sd, str(sorted(sd)))
check("decode_frozen is not a Parameter",
      not isinstance(layer.decode_frozen, nn.Parameter))
check("decode_frozen absent from parameters()",
      all(p is not layer.decode_frozen for p in layer.parameters()))

print("\n== linear: gradients reach prefill only ==")
x = torch.randn(2, 6, 64, dtype=torch.float32)
labels = torch.full((2, 6), -100)
labels[:, 3:] = 1                      # second half = decode positions
with quant_phase(prefill_mask_from_labels(labels)):
    out = layer(x)
out.sum().backward()
check("prefill master got a gradient", layer.weight.grad is not None
      and layer.weight.grad.abs().sum().item() > 0)
check("decode_frozen has no .grad", getattr(layer.decode_frozen, "grad", None) is None)
before = layer.decode_frozen.clone()
check("decode_frozen unchanged after backward",
      torch.equal(before, layer.decode_frozen))

print("\n== linear: the two phases really use different weights ==")
lay2 = NVFP4FrozenDecodeLinear.from_linear(nn.Linear(64, 32, bias=False), block_size=16)
lay2.set_decode_frozen(torch.randn(32, 64))
lay2.eval()
xx = torch.randn(1, 4, 64)
allp = torch.full((1, 4), -100)                       # all prefill
alld = torch.ones((1, 4), dtype=torch.long)           # all decode
with quant_phase(prefill_mask_from_labels(allp)):
    y_p = lay2(xx)
with quant_phase(prefill_mask_from_labels(alld)):
    y_d = lay2(xx)
check("prefill output DIFFERS from decode output", not torch.allclose(y_p, y_d, atol=1e-3),
      "identical would mean one half is ignored")

print("\n== norms: delegate, do not reimplement ==")
for cls, name in ((PlusOneNorm, "(1+w) norm"), (GatedNorm, "gated norm")):
    torch.manual_seed(1)
    ref = cls(16)
    dual = FrozenDecodeNorm(cls(16))
    dual.weight_prefill.data.copy_(ref.weight.data)
    dual.weight_decode.data.copy_(ref.weight.data)
    xi = torch.randn(1, 5, 16)
    extra = (torch.randn(1, 5, 16),) if cls is GatedNorm else ()
    allp2 = torch.full((1, 5), -100)
    with quant_phase(prefill_mask_from_labels(allp2)):
        got = dual(xi, *extra)
    want = ref(xi, *extra)
    check(f"{name}: matches the ORIGINAL module exactly",
          torch.allclose(got, want, atol=1e-5),
          f"max|d|={(got - want).abs().max().item():.3e}")

# And the contrast: the old weight*normed formula is NOT what (1+w) computes, so a
# reimplementation would have passed a self-consistency test and still been wrong.
torch.manual_seed(1)
ref = PlusOneNorm(16)
xi = torch.randn(1, 5, 16)
v = xi.float().pow(2).mean(-1, keepdim=True)
normed = (xi.float() * torch.rsqrt(v + 1e-6)).to(xi.dtype)
old_style = ref.weight * normed
check("(1+w) DIFFERS from w*normed", not torch.allclose(ref(xi), old_style, atol=1e-4),
      "this is the silent bug the delegation avoids")

print("\n== norms: 4D (q_norm/k_norm) inputs ==")
# q_norm/k_norm are applied to head-shaped [B, T, heads, head_dim] tensors. dual's
# _phase_for rejects anything but rank 3, so a norm wrapper built on it dies with
# "got a 4D input under an explicit phase mask" the moment real attention runs -- which
# is exactly how the first smoke run failed.
dn4 = FrozenDecodeNorm(PlusOneNorm(8))
x4 = torch.randn(2, 5, 3, 8)                      # [B, T, heads, head_dim]
lab4 = torch.full((2, 5), -100)
lab4[:, 2:] = 1
try:
    with quant_phase(prefill_mask_from_labels(lab4)):
        o4 = dn4(x4)
    ok4, why = o4.shape == x4.shape, f"shape {tuple(o4.shape)}"
except Exception as exc:
    ok4, why = False, f"{type(exc).__name__}: {exc}"
check("4D input with a per-position mask works", ok4, why)

print("\n== norms: flattened [B*T*heads, dim] (Qwen3_5 gated linear-attn norm) ==")
# Qwen3_5's linear attention flattens to 2D before the gated norm and reshapes back
# afterwards, so the mask must be repeat_interleave'd by `heads`. Getting the axis wrong
# still produces correctly-shaped output, so shape alone cannot catch it -- assert the
# ROUTING instead, by giving the two phases wildly different scales and checking which
# rows moved.
B, T, H, D = 1, 4, 3, 8
dnf = FrozenDecodeNorm(PlusOneNorm(D))
dnf.weight_prefill.data.fill_(0.0)      # (1+0) = identity scale
dnf.weight_decode.data.fill_(99.0)      # unmistakably different
labf = torch.full((B, T), -100)
labf[:, 2:] = 1                          # positions 0,1 prefill; 2,3 decode
xf = torch.randn(B * T * H, D)
try:
    with quant_phase(prefill_mask_from_labels(labf)):
        of = dnf(xf)
    rows = of.abs().reshape(T, H, D).mean(dim=(1, 2))      # per POSITION
    routed = (rows[:2] < rows[2:].min()).all().item()      # prefill rows much smaller
    ok_f, why_f = bool(routed), f"per-position magnitudes {[round(v,2) for v in rows.tolist()]}"
except Exception as exc:
    ok_f, why_f = False, f"{type(exc).__name__}: {exc}"
check("flattened 2D input routes phase per POSITION, not per row", ok_f, why_f)

print("\n== norms: decode side frozen ==")
dn = FrozenDecodeNorm(PlusOneNorm(16))
check("weight_decode requires_grad is False", not dn.weight_decode.requires_grad)
check("weight_prefill requires_grad is True", dn.weight_prefill.requires_grad)
check("inner weight neutralised", not dn.inner.weight.requires_grad,
      "otherwise it is a second trainable copy in the optimizer")
trainable = [n for n, p in dn.named_parameters() if p.requires_grad]
check("exactly one trainable norm tensor", trainable == ["weight_prefill"], str(trainable))

print("\n== export_variants agrees across layer types ==")
# export_variants(model) asks whichever quantized layer it finds FIRST, so the linear and
# the norm must answer identically or the variant set depends on module iteration order.
check("linear and norm both report ['prefill']",
      NVFP4FrozenDecodeLinear.export_variants() == FrozenDecodeNorm.export_variants()
      == ["prefill"],
      f"linear={NVFP4FrozenDecodeLinear.export_variants()} "
      f"norm={FrozenDecodeNorm.export_variants()}")

print("\n== apply(): scope and freezing ==")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.q = nn.Linear(16, 16, bias=False)
        self.n1 = PlusOneNorm(16)
        self.n2 = GatedNorm(16)


t = Tiny()
apply_nvfp4frozendec(t, decode_model="", block_size=16)
check("linear replaced", isinstance(t.q, NVFP4FrozenDecodeLinear))
check("plain norm dualized", isinstance(t.n1, FrozenDecodeNorm))
check("gated norm dualized", isinstance(t.n2, FrozenDecodeNorm))
check("embedding frozen (shared, not duplicated)", not t.embed_tokens.weight.requires_grad)
check("embedding NOT duplicated", isinstance(t.embed_tokens, nn.Embedding))

print("\n== in_proj_a/in_proj_b are NOT quantized ==")
# Qwen3.5's linear attention takes its decay/gate terms from these, whose weights are
# (48, 5120) -- 48 = head count, so each output is a per-head scalar driving the
# recurrence. Rounding those through a 4-bit block format perturbs the recurrence rather
# than adding a small weight error, and it compounds over sequence length.


class AttnLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_a = nn.Linear(32, 48, bias=False)
        self.in_proj_b = nn.Linear(32, 48, bias=False)
        self.out_proj = nn.Linear(32, 32, bias=False)


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = AttnLike()
        self.embed_tokens = nn.Embedding(16, 32)
        # Raw module-level parameters, neither Linear nor norm. Qwen3.5's linear
        # attention carries exactly this shape of thing -- A_log (state decay) and
        # dt_bias (timestep) -- and they are SHARED by both phases. An allow-list style
        # freeze misses them completely, which left 96 trainable parameters silently
        # moving the recurrence the frozen decode weights were quantized against.
        self.A_log = nn.Parameter(torch.randn(8))
        self.dt_bias = nn.Parameter(torch.randn(8))


s = Stack()
apply_nvfp4frozendec(s, decode_model="", block_size=16)
check("in_proj_a left unquantized", isinstance(s.linear_attn.in_proj_a, nn.Linear)
      and not isinstance(s.linear_attn.in_proj_a, NVFP4FrozenDecodeLinear))
check("in_proj_b left unquantized", isinstance(s.linear_attn.in_proj_b, nn.Linear)
      and not isinstance(s.linear_attn.in_proj_b, NVFP4FrozenDecodeLinear))
check("a normal projection IS still quantized",
      isinstance(s.linear_attn.out_proj, NVFP4FrozenDecodeLinear))
# Unquantized is not sufficient: these are SHARED by both phases, so a trainable copy
# would move the decode pathway -- the one thing this format holds still.
check("in_proj_a frozen", not s.linear_attn.in_proj_a.weight.requires_grad)
check("in_proj_b frozen", not s.linear_attn.in_proj_b.weight.requires_grad)
check("quantized projection still trains", s.linear_attn.out_proj.weight.requires_grad)

# Excluding layers MUST change the tag, or the new runs silently continue the old
# checkpoints that were trained with these quantized.
_, h_skip = build_quantizer_params("nvfp4frozendec", '{"decode_model":"/x"}')
_, h_none = build_quantizer_params(
    "nvfp4frozendec", '{"decode_model":"/x","skip_linears":""}')
check("skip_linears changes the checkpoint tag", h_skip != h_none, f"{h_skip} vs {h_none}")

# THE decisive gate: after apply(), the ONLY trainable tensors are the prefill halves.
# Stated as a whitelist over the whole module, so a parameter that is neither a Linear
# nor a norm -- A_log, dt_bias, or whatever the next architecture invents -- cannot slip
# through the way it did the first time.
s2 = Stack()
apply_nvfp4frozendec(s2, decode_model="", block_size=16)
trainable = sorted(n for n, p in s2.named_parameters() if p.requires_grad)
expected = sorted(
    [n for n, p in s2.named_parameters()
     if n.endswith(".weight") and isinstance(
         s2.get_submodule(n.rpartition(".")[0]), NVFP4FrozenDecodeLinear)]
    + [n for n, p in s2.named_parameters() if n.endswith("weight_prefill")])
check("ONLY prefill tensors are trainable", trainable == expected,
      f"unexpected: {sorted(set(trainable) - set(expected))}")
check("A_log frozen (shared recurrence term)", not s2.A_log.requires_grad)
check("dt_bias frozen (shared recurrence term)", not s2.dt_bias.requires_grad)

print("\n== pre-existing formats untouched ==")
for n, want in (("nvfp4", "99914b93"), ("nvfp4pdsplit", "2d557910")):
    _, got = build_quantizer_params(n, "")
    check(f"{n} hash unchanged", got == want, f"{got} vs {want}")

print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
