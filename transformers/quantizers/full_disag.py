"""Full disaggregation: dual embeddings, norms and LM head, on top of dual linears.

The dual *linear* formats give the two serving engines different projection weights. The
embedding table, the RMSNorm scales and the LM head are still shared -- one copy, exported
byte-identically into both halves (that is what `build_state_dict` does with anything the
quantizers do not own). `--full-disag` makes those three dual as well, so a prefill engine
and a decode engine share nothing but the architecture.

Each module here holds TWO learnable tensors and routes per position on the same
`_PHASE_MASK` the dual linears use, so a single forward covers both phases and one
backward sends each half only the gradient from the positions it serves.

WEIGHT TYING IS LOAD-BEARING
----------------------------
Qwen3 ties the LM head to the embedding table at 0.6B, 1.7B and 4B (only 8B does not).
Under tying there is ONE parameter serving both roles, so the dual pair must be shared
too: embed_prefill IS head_prefill. Building two independent dual modules would create
four V x H tables instead of two, double the memory this flag already costs, and train an
input embedding that no longer matches the output head. `apply_full_disag` detects tying
and wires the head to the embedding's parameters.

COST
----
This is not free. At 4B the embedding is 151936 x 2560 ~ 389M parameters; duplicating it
adds that again, plus its gradient and optimizer state. Norms are negligible. Check the
memory budget before turning this on above 4B.

THE LM HEAD BARELY RUNS IN PREFILL FORMAT
-----------------------------------------
Worth knowing when reading the training loop. The loss is over shifted positions: hidden
state at t predicts the token at t+1, and SFT scores only assistant tokens. A position t
contributes iff labels[t+1] != -100, and the head used at t is the one for t's own phase.
So the scored positions are the completion positions (decode) plus exactly ONE boundary
token per sequence -- the last prompt token, which predicts the first assistant token
through the PREFILL head. Everything else in prefill is unscored. `training/qad.py` splits
the loss accordingly: the fused kernel over the decode bulk, and a tiny second call for the
boundary. See `dual_lm_head_weights`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import dual as _dual

__all__ = ["DualParamModule", "DualEmbedding", "DualRMSNorm", "DualLMHead",
           "apply_full_disag", "dual_lm_head_weights", "is_full_disag",
           "full_disag_hash"]


def full_disag_hash(quant_hash: str) -> str:
    """Fold --full-disag into a quantizer hash, so the run gets its own checkpoint tag.

    MUST be used by training AND by every eval path. Training alone folding it is how a
    whole eval sweep silently resolved to the plain run's checkpoints, re-measured those,
    and overwrote the plain baseline's results while reporting exit 0.
    """
    import hashlib
    return hashlib.md5(f"{quant_hash}+full_disag".encode()).hexdigest()[:8]


def _phase_like(x: Tensor):
    """True/False (whole tensor is one phase) or a bool mask broadcastable to x.

    Unlike `dual._phase_for` this accepts any rank >= 2: q_norm/k_norm see
    [B, T, heads, head_dim], not just [B, T, H]. The mask is per (batch, position), so
    the first two dims must line up -- checked rather than assumed, because silently
    broadcasting the wrong axis would mis-route every token.
    """
    m = _dual._PHASE_MASK
    if m is None:
        return x.shape[-2] > 1 if x.dim() == 3 else True
    if x.shape[:2] != m.shape:
        raise RuntimeError(
            f"full-disag layer got input {tuple(x.shape)} whose leading dims do not match "
            f"the [B, T] phase mask {tuple(m.shape)}")
    return m.reshape(*m.shape, *([1] * (x.dim() - 2)))


class DualParamModule(nn.Module):
    """Marker + export contract for a module holding one tensor pair.

    `export/save.py` treats these like quantized layers: their raw parameters are kept out
    of the checkpoint and `export_tensors(variant)` decides what each half ships.
    """

    @classmethod
    def export_variants(cls) -> list:
        return ["prefill", "decode"]

    def _pair(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def export_config(self, variant=None) -> dict | None:
        return None

    def export_tensors(self, variant=None) -> dict[str, Tensor]:
        p, d = self._pair()
        w = p if variant != "decode" else d
        return {"weight": w.detach().to(torch.bfloat16).cpu()}

    def load_tensors(self, tensors: dict[str, Tensor], variant=None) -> None:
        p, d = self._pair()
        tgt = p if variant != "decode" else d
        with torch.no_grad():
            tgt.copy_(tensors["weight"].to(device=tgt.device, dtype=tgt.dtype))


class DualEmbedding(DualParamModule):
    """nn.Embedding with one table per phase."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.num_embeddings = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim
        self.padding_idx = emb.padding_idx
        self.weight_prefill = nn.Parameter(emb.weight.data.clone())
        self.weight_decode = nn.Parameter(emb.weight.data.clone())

    def _pair(self):
        return self.weight_prefill, self.weight_decode

    def forward(self, ids: Tensor) -> Tensor:
        phase = _phase_like(ids.unsqueeze(-1))       # ids is [B, T]; give it a feature dim
        if not torch.is_tensor(phase):
            w = self.weight_prefill if phase else self.weight_decode
            return F.embedding(ids, w, self.padding_idx)
        return torch.where(phase,
                           F.embedding(ids, self.weight_prefill, self.padding_idx),
                           F.embedding(ids, self.weight_decode, self.padding_idx))


class DualRMSNorm(DualParamModule):
    """RMSNorm with one scale vector per phase.

    The normalization itself has no parameters, so it is computed ONCE and only the scale
    is selected per position -- no need to run the whole norm twice.
    """

    def __init__(self, norm: nn.Module):
        super().__init__()
        w = norm.weight.data
        self.eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        self.weight_prefill = nn.Parameter(w.clone())
        self.weight_decode = nn.Parameter(w.clone())

    def _pair(self):
        return self.weight_prefill, self.weight_decode

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(dtype)
        phase = _phase_like(x)
        if not torch.is_tensor(phase):
            return (self.weight_prefill if phase else self.weight_decode) * normed
        w = torch.where(phase, self.weight_prefill, self.weight_decode)
        return w * normed


class DualLMHead(DualParamModule):
    """Output projection with one weight per phase.

    `tied_to` shares an existing DualEmbedding's parameters instead of allocating new
    ones, which is what keeps a tied model tied.
    """

    def __init__(self, lin: nn.Linear, tied_to: DualEmbedding | None = None):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.tied = tied_to is not None
        if tied_to is not None:
            self.weight_prefill = tied_to.weight_prefill
            self.weight_decode = tied_to.weight_decode
        else:
            self.weight_prefill = nn.Parameter(lin.weight.data.clone())
            self.weight_decode = nn.Parameter(lin.weight.data.clone())
        self.bias = lin.bias

    def _pair(self):
        return self.weight_prefill, self.weight_decode

    def forward(self, x: Tensor) -> Tensor:
        phase = _phase_like(x)
        if not torch.is_tensor(phase):
            w = self.weight_prefill if phase else self.weight_decode
            return F.linear(x, w, self.bias)
        return torch.where(phase,
                           F.linear(x, self.weight_prefill, self.bias),
                           F.linear(x, self.weight_decode, self.bias))


def _norm_modules(model: nn.Module):
    """Every RMSNorm-like leaf: has `weight`, no `bias`, and is not a Linear/Embedding."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding, DualParamModule)):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, nn.Parameter) and w.dim() == 1 and getattr(m, "bias", None) is None:
            yield name, m


def _swap(model: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, new)


def apply_full_disag(model: nn.Module) -> dict[str, int]:
    """Replace the embedding, every RMSNorm and the LM head with dual versions."""
    counts = {"embedding": 0, "norm": 0, "lm_head": 0, "tied": 0}

    emb_name = emb_mod = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Embedding):
            emb_name, emb_mod = name, m
            break

    head_name = head_mod = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.rpartition(".")[2] == "lm_head":
            head_name, head_mod = name, m
            break

    tied = (emb_mod is not None and head_mod is not None
            and emb_mod.weight.data_ptr() == head_mod.weight.data_ptr())

    dual_emb = None
    if emb_mod is not None:
        dual_emb = DualEmbedding(emb_mod)
        _swap(model, emb_name, dual_emb)
        counts["embedding"] = 1

    for name, m in list(_norm_modules(model)):
        _swap(model, name, DualRMSNorm(m))
        counts["norm"] += 1

    if head_mod is not None:
        _swap(model, head_name, DualLMHead(head_mod, tied_to=dual_emb if tied else None))
        counts["lm_head"] = 1
        counts["tied"] = int(tied)

    return counts


def dual_lm_head_weights(model: nn.Module):
    """(w_prefill, w_decode) if the LM head is dual, else None."""
    for m in model.modules():
        if isinstance(m, DualLMHead):
            return m.weight_prefill, m.weight_decode
    return None


def is_full_disag(model: nn.Module) -> bool:
    return any(isinstance(m, DualParamModule) for m in model.modules())
