"""NVFP4 prefill trained against a FROZEN, EXTERNAL decode checkpoint.

Every other dual format in this repo trains both halves (shared master or split
masters). This one trains ONLY the prefill half. The decode half is a black box: a
pre-quantized checkpoint someone else produced (the dequantized GSQ-RCO `*-bf16` models
under `models/`), loaded as-is, never updated, never exported.

WHAT "FROZEN" HAS TO MEAN HERE
------------------------------
Freezing the decode LINEARS is not sufficient. The decode pathway also runs through the
norms, the embedding and the LM head; if those keep training, the decode model keeps
moving and the black box is no longer the thing that was measured. So this format also:

  * duplicates every text RMSNorm into a (prefill, decode) pair and freezes the decode
    side, initialised from the external checkpoint;
  * freezes -- and SHARES -- the token embedding and the LM head. Sharing rather than
    duplicating is deliberate: at 248320 x 5120 the embedding is 1.27B parameters and the
    head another 1.27B (this checkpoint is untied), so duplicating them would cost ~2.5B
    parameters plus gradients and optimizer state to express "two tensors that are both
    frozen and both equal".

None of this needs --full-disag, which is deprecated and in any case REFUSES to run on a
model with a vision_config (training/qad.py). Two of the reasons it refuses apply
verbatim here, which is why the norm handling below is a fresh implementation rather
than a reuse:

  * `DualRMSNorm` REIMPLEMENTS the norm as `weight * normed`. Qwen3_5RMSNorm scales by
    `(1.0 + weight)`. Swapping it in does not fail -- it silently computes a different
    model. Delegating to the original module removes that whole class of bug.
  * `Qwen3_5RMSNormGated.forward(hidden_states, gate)` takes an argument the old wrapper
    does not accept. 48 of the 209 text norms are gated.

WHY A BUFFER, NOT A FROZEN PARAMETER
------------------------------------
`requires_grad=False` would already keep the decode weight out of the optimizer -- the
param groups filter on exactly that. The reason it is a NON-PERSISTENT buffer is
state_dict: both `save_training_state` (training/checkpoint.py) and `build_state_dict`
(export/save.py) iterate `model.state_dict()`. As a Parameter, 27B of unchanging decode
weights would be rewritten into every training state save and would land in the export
that is supposed to contain the prefill half only.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocked import BLOCK, qlinear, replace_linears, ste
from . import dual as _dual
from .dual import _DualActMixin, _phase_for
from .full_disag import DualParamModule, _phase_like
from .nvfp4 import NVFP4Linear

__all__ = ["NVFP4FrozenDecodeLinear", "FrozenDecodeNorm", "apply_nvfp4frozendec",
           "load_frozen_decode", "freeze_all_but_prefill", "SKIP_LINEARS"]

# Linears left in bf16, NEVER quantized.
#
# Qwen3.5's linear attention gets its decay/gate terms from in_proj_a and in_proj_b,
# whose weights are (48, 5120) -- 48 being the head count, so each output is a PER-HEAD
# scalar driving the recurrence, not a feature projection. NVFP4 is a 4-bit block format
# built for wide feature dimensions; rounding a per-head decay through it perturbs the
# recurrence itself rather than adding a small weight error, and the damage compounds
# over sequence length instead of averaging out.
#
# Comma-separated and threaded through the registry `defaults` so it is hashed into the
# checkpoint tag: changing what is excluded MUST produce new checkpoint directories
# rather than silently continuing the old ones.
SKIP_LINEARS = "in_proj_a,in_proj_b"

# MULTI-TOKEN PREDICTION IS DELIBERATELY NOT CARRIED THROUGH.
#
# Qwen3.8-27B ships a 1-layer MTP head (15 `mtp.*` tensors, config
# `mtp_num_hidden_layers: 1`). transformers 5.17's Qwen3_5ForConditionalGeneration has no
# `.mtp` attribute at all, so those keys are unexpected on load and silently dropped --
# which means they are frozen by construction, but also absent from anything we export.
#
# That is fine HERE and only here: MTP is a speculative-decoding head, so it accelerates
# the DECODE engine, and this format exports the PREFILL half only. The decode engine is
# the external black-box checkpoint, which carries its own MTP already. Do not "fix" this
# by adding a passthrough -- it would ship a head the prefill engine never calls.
# (It would matter for a format that exported the decode half.)


class NVFP4FrozenDecodeLinear(_DualActMixin, NVFP4Linear):
    """Prefill: NVFP4 W4A4 from the trained master. Decode: a frozen external weight.

    `decode_frozen` starts empty and is filled by `load_frozen_decode`. It stays empty on
    a layer the external checkpoint has no tensor for, which `load_frozen_decode` treats
    as an error rather than letting the decode half silently run on garbage.
    """

    def __init__(self, weight: Tensor, bias, block_size: int = BLOCK):
        super().__init__(weight, bias, block_size=block_size, quantize_act=True)
        # persistent=False: kept out of state_dict, hence out of training states and the
        # export. See the module docstring.
        self.register_buffer("decode_frozen", torch.empty(0, dtype=torch.bfloat16),
                             persistent=False)

    @torch.no_grad()
    def set_decode_frozen(self, w: Tensor) -> None:
        if tuple(w.shape) != tuple(self.weight.shape):
            raise ValueError(f"frozen decode weight {tuple(w.shape)} does not match "
                             f"this layer's {tuple(self.weight.shape)}")
        self.decode_frozen = w.detach().to(device=self.weight.device,
                                           dtype=torch.bfloat16)

    @property
    def has_decode(self) -> bool:
        return self.decode_frozen.numel() > 0

    def _decode_out(self, x: Tensor) -> Tensor:
        if not self.has_decode:
            raise RuntimeError(
                "decode half ran before load_frozen_decode() filled decode_frozen; "
                "the external checkpoint must be loaded before the first forward")
        return F.linear(x, self.decode_frozen.to(x.dtype), self.bias)

    def forward(self, x: Tensor) -> Tensor:
        phase = _phase_for(x)
        w_p = ste(self.weight, self.wq) if self.training else self.wq
        if not torch.is_tensor(phase):
            if phase:
                return qlinear(self._quant_act(x), w_p, self.bias)
            return self._decode_out(x)
        y_p = qlinear(self._quant_act(x, phase), w_p, self.bias)
        y_d = self._decode_out(x)
        return torch.where(phase, y_p, y_d)

    @classmethod
    def export_variants(cls) -> list:
        # Prefill ONLY. The decode half is an external checkpoint that already exists on
        # disk; re-exporting it would write 27B of weights we did not train and did not
        # change, and would invite the two copies to drift.
        return ["prefill"]

    def _variant_quantize_act(self, variant) -> bool:
        return True          # the only variant is prefill, which is W4A4


def _norm_phase(x: Tensor):
    """Phase selector for a norm input of any shape this architecture actually uses.

    Three layouts occur, and only the first two are covered by the existing helpers:

      [B, T, H]                  the ordinary residual-stream norms      -> _phase_like
      [B, T, heads, head_dim]    q_norm / k_norm                         -> _phase_like
      [B*T*heads, head_dim]      Qwen3_5's GATED linear-attention norm   -> here

    The third is flattened before the call and reshaped after:

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    The reshape BACK to (batch_size, seq_len, -1) is what proves the row order is
    row-major over (B, T, heads) -- so each (b, t) contributes `heads` consecutive rows
    and the mask is repeat_interleave'd by that factor. Derived from the source rather
    than guessed: repeating along the wrong axis would mis-route every token's phase
    while still producing correctly shaped output.
    """
    if x.dim() != 2:
        return _phase_like(x)
    m = _dual._PHASE_MASK
    if m is None:
        return True
    rows, bt = x.shape[0], m.numel()
    if rows % bt:
        raise RuntimeError(
            f"norm got a flattened input with {rows} rows, which is not a multiple of the "
            f"[B, T] phase mask's {bt} positions; the phase cannot be aligned")
    return m.reshape(-1).repeat_interleave(rows // bt).unsqueeze(-1)


class FrozenDecodeNorm(DualParamModule):
    """A norm with one scale per phase, the decode scale frozen.

    DELEGATES to the wrapped module instead of reimplementing the norm. That is the whole
    point: `Qwen3_5RMSNorm` scales by `(1.0 + weight)` while the old DualRMSNorm assumed
    `weight * normed`, and `Qwen3_5RMSNormGated` takes a `gate` argument. Calling the
    original forward through `functional_call` with a substituted weight is exact for
    both, passes `gate` through untouched, and keeps working for whatever norm class the
    next architecture introduces.

    Blending a per-position weight is valid because both norms use `weight` purely
    multiplicatively -- `(1+w)*normed` and `w*normed*act(gate)` are each linear in `w` --
    which is the same assumption the per-position path always relied on.
    """

    def __init__(self, norm: nn.Module):
        super().__init__()
        w = norm.weight.data
        self.inner = norm
        self.weight_prefill = nn.Parameter(w.clone())
        # Frozen: the decode pathway must not move while the prefill half trains.
        self.weight_decode = nn.Parameter(w.clone(), requires_grad=False)
        # The wrapped module keeps its own `weight` Parameter, which would be a second,
        # trainable copy of the same thing and would land in the optimizer. Neutralise it.
        self.inner.weight.requires_grad_(False)

    @classmethod
    def export_variants(cls) -> list:
        # MUST match NVFP4FrozenDecodeLinear. export_variants(model) asks whichever
        # quantized layer it happens to find FIRST, so inheriting DualParamModule's
        # ["prefill", "decode"] here makes the answer depend on module iteration order:
        # at 0.6B a linear came first and only prefill/ was written, while at 27B a norm
        # came first and the run wrote a full 27B decode/ checkpoint nobody asked for.
        return ["prefill"]

    def _pair(self):
        return self.weight_prefill, self.weight_decode

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        phase = _norm_phase(x)
        if not torch.is_tensor(phase):
            w = self.weight_prefill if phase else self.weight_decode
        else:
            # phase broadcasts to x's rank; the weight is [dim]. The product is
            # position-dependent, which both norm implementations multiply through
            # unchanged because each is linear in `weight`.
            w = torch.where(phase, self.weight_prefill, self.weight_decode)
        return torch.func.functional_call(self.inner, {"weight": w}, (x, *args), kwargs)


def _norm_modules(root: nn.Module):
    """RMSNorm-like leaves: a 1-D `weight` Parameter, no bias, not Linear/Embedding.

    Scoped to whatever `root` is given -- callers pass the TEXT stack, so the vision
    tower's norms are out of reach rather than merely skipped.
    """
    for name, m in root.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding, DualParamModule)):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, nn.Parameter) and w.dim() == 1 and getattr(m, "bias", None) is None:
            yield name, m


def _swap(root: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, new)


def apply_nvfp4frozendec(model: nn.Module, decode_model: str = "",
                         block_size: int = BLOCK,
                         skip_linears: str = SKIP_LINEARS) -> None:
    """Install the format on a TEXT stack (`text_stack(model).base`).

    `decode_model` is the external checkpoint directory. It is part of the registry
    `defaults`, so it is md5'd into the checkpoint tag: two runs against different decode
    black boxes can never share a checkpoint directory. `skip_linears` is in `defaults`
    for the same reason -- changing what is left unquantized must not silently reuse the
    previous runs' checkpoints.
    """
    skip = tuple(s for s in skip_linears.split(",") if s)
    replace_linears(model, lambda lin: NVFP4FrozenDecodeLinear.from_linear(
        lin, block_size=block_size), skip=("lm_head", *skip))
    # Left in bf16 AND frozen. Unquantized is not enough on its own: these projections
    # are SHARED by both phases, so training them would move the decode pathway, which is
    # the one thing this format exists to hold still -- the same reason the embedding and
    # the head are frozen rather than duplicated.
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.rpartition(".")[2] in skip:
            for p in m.parameters():
                p.requires_grad_(False)
    for name, m in list(_norm_modules(model)):
        _swap(model, name, FrozenDecodeNorm(m))
    # Freeze + share the embedding. Not duplicated: see the module docstring.
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.requires_grad_(False)
    freeze_all_but_prefill(model)


def freeze_all_but_prefill(model: nn.Module) -> list[str]:
    """Freeze every parameter that is not the trained PREFILL half. Returns what it froze.

    A DENY-list of the two trainable things, not an allow-list of things to freeze. The
    allow-list version is what failed: it froze the embedding, the head, the skipped
    linears and the norms' decode side, and silently left behind everything that was
    neither a Linear nor a norm. On Qwen3.5 that is 96 parameters --
    linear_attn.A_log and linear_attn.dt_bias, the state-decay and timestep terms of the
    linear attention, 48 of each.

    Those are SHARED by both phases and they drive the recurrence, so training them moves
    the very dynamics the frozen decode weights were quantized against: the black box
    stops being the model that was measured. Nothing in the earlier freezing touched them
    because they are raw parameters on the attention module rather than a Linear or a
    norm.

    They are FROZEN rather than duplicated per phase, unlike the norms. A norm scale is
    applied pointwise, so a per-position blend is exact. A recurrence is not: the state
    propagates ACROSS positions, so "this position used the prefill decay and that one
    used the decode decay" is not well defined -- there is one state sequence and it was
    produced by one decay. Freezing keeps the dynamics the decode black box assumes.
    """
    keep = set()
    for m in model.modules():
        if isinstance(m, NVFP4FrozenDecodeLinear):
            keep.add(id(m.weight))
            if m.bias is not None:
                keep.add(id(m.bias))
        elif isinstance(m, FrozenDecodeNorm):
            keep.add(id(m.weight_prefill))
    frozen = []
    for name, p in model.named_parameters():
        if id(p) not in keep and p.requires_grad:
            p.requires_grad_(False)
            frozen.append(name)
    return frozen


@torch.no_grad()
def load_frozen_decode(text_base: nn.Module, decode_model: str,
                       prefix: str = "model.language_model.") -> dict[str, int]:
    """Fill every frozen decode tensor from the external checkpoint.

    Reads tensor-by-tensor through safetensors' lazy reader rather than materialising the
    whole 51 GB state dict: under pipeline parallelism each rank holds only its own slice
    of layers, so most of that file is not this rank's business.

    MUST be called AFTER `pipeline.split_stack`, and accounts for its renumbering.
    split_stack reindexes each stage's layers from 0, so on stage 1 `layers.0` is really
    original layer `cut`. Looking the name up verbatim would load layer 0's frozen weights
    into layer 32 -- every tensor found, nothing raised, and a silently wrong model. The
    shift is applied with pipeline's OWN remap_layer_indices so the two can never drift.

    Verifies coverage and RAISES on a miss. A silently unfilled decode weight would leave
    the black box running on the base model's weights -- i.e. measuring the wrong thing
    while looking perfectly healthy -- which is exactly the failure this format cannot
    afford.
    """
    from safetensors import safe_open

    from training.pipeline import remap_layer_indices

    offset = int(getattr(text_base, "_pp_layer_offset", 0))

    def ckpt_key(name: str) -> str:
        """This module's name as it appears in the ORIGINAL, unsplit checkpoint."""
        local = f"{name}.weight"
        if offset:
            local = next(iter(remap_layer_indices({local: None}, offset)))
        return f"{prefix}{local}"

    index = os.path.join(decode_model, "model.safetensors.index.json")
    if os.path.exists(index):
        import json
        weight_map = json.load(open(index))["weight_map"]
    else:
        weight_map = None

    shards: dict[str, object] = {}

    def get(key: str):
        """The tensor for `key`, or None if the checkpoint does not have it."""
        if weight_map is not None:
            shard = weight_map.get(key)
            if shard is None:
                return None
            path = os.path.join(decode_model, shard)
        else:
            path = os.path.join(decode_model, "model.safetensors")
        fh = shards.get(path)
        if fh is None:
            fh = shards[path] = safe_open(path, framework="pt")
        if key not in fh.keys():
            return None
        return fh.get_tensor(key)

    counts = {"linear": 0, "norm": 0, "missing": 0}
    missing: list[str] = []
    try:
        for name, m in text_base.named_modules():
            if isinstance(m, NVFP4FrozenDecodeLinear):
                key = ckpt_key(name)
                t = get(key)
                if t is None:
                    missing.append(key)
                    continue
                m.set_decode_frozen(t)
                counts["linear"] += 1
            elif isinstance(m, FrozenDecodeNorm):
                key = ckpt_key(name)
                t = get(key)
                if t is None:
                    missing.append(key)
                    continue
                m.weight_decode.data.copy_(
                    t.to(device=m.weight_decode.device, dtype=m.weight_decode.dtype))
                counts["norm"] += 1
    finally:
        for fh in shards.values():
            close = getattr(fh, "close", None)
            if close is not None:
                close()

    counts["missing"] = len(missing)
    if missing:
        raise RuntimeError(
            f"{len(missing)} frozen-decode tensors not found in {decode_model} "
            f"(first few: {missing[:5]}). The decode half would have silently run on the "
            f"base model's weights.")
    return counts
