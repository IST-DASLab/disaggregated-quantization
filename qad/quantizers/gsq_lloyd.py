"""GSQ-style learned level assignment on the signed Lloyd 3-bit format.

Same deployable format as `lloyd3bit` (see lloyd.py) — blocks of 16 along the
contraction dim, one FP32 per-tensor global scale shared across the fused groups,
a signed FP8-E4M3 per-block scale, and the 8-level MSE-optimal Lloyd grid — but
optimized the GSQ way instead of with STE:

  lloyd3bit    keeps an FP32 master weight, rounds it to the grid each step, and
               passes gradients straight through the rounding.
  gsqlloyd3bit keeps no master weight at all. Every weight element owns a learnable
               distribution over the 8 grid levels, trained through the
               Gumbel-Softmax relaxation and annealed to a hard argmax. Only the
               block scales remain continuous.

Why the combination is worth trying: STE's gradient is a lie exactly where it
matters — it ignores the rounding, so a weight sitting between two levels gets
the same gradient whether or not it will actually move to the neighbouring level.
GSQ instead optimizes the assignment itself, which is the quantity that survives
into the checkpoint. On a 3-bit grid, where levels are far apart and rounding
error is large, that gap is widest.

Scale parameterisation
----------------------
The block scale is stored as an FP32 log2 multiplier `scale_delta` (init 0) on the
frozen signed init scale, and cast to E4M3 with a straight-through estimator in
the forward:

    eff_block = STE_e4m3(init_block_scale * 2**scale_delta) * global_scale

so training sees exactly the E4M3 grid the format deploys, while the optimizer
updates an unconstrained FP32 master. Working in log2 also makes the update
scale-free: one Adam step of `scale_lr` is the same *fractional* change for every
layer, which a raw scale (magnitudes vary by orders of magnitude across a model)
would not give. The sign lives in the frozen init scale and cannot flip, so the
signed-normalization convention the asymmetric Lloyd grid was fit for is
preserved.

Anneal / saturation
-------------------
The relaxation saturates once `gap * scale_val / temp` is large, where `gap` is a
weight's top-2 logit difference: the softmax goes one-hot, the non-argmax
probability underflows into the gradient sum, and the assignment stops moving for
good. Lion makes this worse rather than better — because its update is
`sign(...) * lr` regardless of gradient magnitude, every logit moves the full `lr`
every step, so the gaps inflate at rate ~`logit_lr` whether or not the data
supports it. Faster reassignment therefore buys an EARLIER freeze.

Measured on Qwen3-0.6B / 100M tokens (2485 steps) with the registry's inherited
schedule (scale_val 100->500, temp 2.0->0.05, i.e. kappa/tau 50 -> 10000):

    logit_lr   median gap   reassigned   saturates at   saturated at end
      3e-6        0.0148        2.8%        step 2325         86%
      1e-5        0.0207        7.7%        step 2245         89%
      3e-5        0.0728       16.4%        step 1687         98%
      1e-4        0.2238       27.5%        step  855        100%
      3e-4        0.2981       37.1%        step  633        100%

At 3e-4 three quarters of training had no gradient on the assignment at all, and
even the gentlest arm ends 86% frozen — the schedule is mis-calibrated regardless
of learning rate. It is inherited from the paper, whose binary GPTQ warm-start
(`l = sigma_init * (eps + alpha * l_GPTQ)`, `l_GPTQ` in {+1,-1}) gives every weight
the SAME gap; the grid-proximity init used here produces gaps proportional to
distance from the decision boundary, ~100x smaller and highly non-uniform, so the
same kappa/tau oversharpens.

To keep the relaxation alive to the end, hold the gain constant and stop the
temperature early — kappa/tau 50 -> 400 puts a typical weight at gap*kappa/tau ~
6-8 at the final step (peaked, but still passing gradient):

    --quantizer-params '{"logit_lr":1e-5,"temp_end":0.25,"scale_end":100.0}'

The registry defaults keep the original schedule so existing checkpoint hashes
stay valid; pass the capped one explicitly.

Learning rates
--------------
Neither the logits (consumed as `logits * scale_val` inside a softmax) nor the
log2 scale deltas live in the same units as weights, so the weight LR (3e-6 in
the current recipe) is meaningless for them — at that rate a logit could move
~1e-2 over a whole run, far less than the ~2e-2 gap that separates competing
levels at init, and nothing would ever be reassigned. `logit_lr` and `scale_lr`
are therefore absolute and part of the hyperparameter hash, so sweeping them
lands in separate checkpoint directories.

Export is pseudo-quantized (dequantized bf16), exactly like lloyd3bit: there is no
3-bit LUT kernel, and the point of the pairing is a like-for-like comparison of
the two optimizers on one format.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .blocked import (BLOCK, GLOBAL_DEN, GroupScaled, SCALE_REF, ste, to_e4m3,
                      replace_linears)
from .grids import grid_spacing
from .gsq import GSQLinearBase, proximity_logits
from .lloyd import GRIDS


class GSQLloydLinear(GroupScaled, GSQLinearBase):
    """Learned 3-bit level assignment + learned signed E4M3 block scales."""

    def __init__(
        self,
        weight: Tensor,
        bias,
        block_size: int = BLOCK,
        grid: str = "lloyd",
        std: float = 0.01,
        strength: float = 6.0,
        noise: float = 0.0,
        logit_lr: float = 1e-4,
        scale_lr: float = 3e-6,
        optim: str = "lion",
        betas: tuple = (0.9, 0.99),
        adam_eps: float = 1e-16,
        logits_dtype: str = "fp32",
        temp_start: float = 2.0,
        temp_end: float = 0.05,
        scale_start: float = 100.0,
        scale_end: float = 500.0,
    ):
        out, in_ = weight.shape
        dev = weight.device
        if in_ % block_size:
            raise ValueError(
                f"in_features={in_} is not a multiple of block_size={block_size}; "
                "the blocked formats assume whole blocks along the contraction dim"
            )
        n_blocks = in_ // block_size
        levels = GRIDS[grid]
        idx = torch.arange(in_, device=dev) // block_size
        # The logits are trainable master parameters, so they are FP32 like every
        # other master here (the student itself is loaded in FP32). The original GSQ
        # used bfloat16 to offset their 8x-per-weight cost, but that caps how far a
        # logit can ever travel: once |logit| reaches ~256*lr its ULP exceeds the
        # update and further steps round away. At lr=1e-4 the ceiling is ~0.026,
        # barely over the ~0.014 median gap between the top two levels, so weights
        # near a tie could never flip.
        #
        # "fp16" halves that cost for models that would not otherwise fit (FP32
        # logits are 32 bytes per quantized weight: 0.6B fits comfortably, 1.7B+ does
        # not), and its 10 mantissa bits suit the logits' small range far better than
        # bfloat16's 7. CAUTION: p.grad inherits the parameter dtype, and these
        # gradients are ~1e-11 — below FP16's smallest subnormal (5.96e-8) — so they
        # flush to zero without gradient scaling. Verify the logits actually move
        # (quant/reassigned) before trusting an fp16 run.
        logits_dtype = {"fp32": torch.float32, "fp16": torch.float16}[logits_dtype]
        super().__init__(out, in_, bias, dtype=weight.dtype, device=dev,
                         grid=levels, idx=idx, logits_dtype=logits_dtype,
                         temp_start=temp_start, temp_end=temp_end,
                         scale_start=scale_start, scale_end=scale_end)

        self.grid_name = grid
        self.block_size = block_size
        # Lion, not AdamW — see training/dist_optim.py. The Gumbel relaxation
        # saturates as it anneals and the logit gradients also carry a factor of the
        # block scale, putting them near 1e-11; AdamW's eps=1e-8 then dominates its
        # normalization and the level assignment freezes outright. Lion's update is
        # sign-based, so its magnitude is exactly `lr` regardless of gradient scale.
        # `eps` is carried anyway so optim="adamw" stays usable for A/B comparison —
        # but at 1e-16, not the 1e-8 default that caused the stall.
        common = {"algo": optim, "betas": tuple(betas), "eps": adam_eps,
                  "weight_decay": 0.0}
        self.logit_optim = {"lr": logit_lr, **common}
        self.scale_optim = {"lr": scale_lr, **common}
        self._std, self._strength, self._noise = std, strength, noise
        self._spacing = grid_spacing(levels)

        # amax of the ORIGINAL weight — `self.weight` is the quantized view, whose
        # amax is slightly smaller (the top Lloyd level is 5.788, not 6.0), and the
        # global scale must be derived from the true magnitude to match lloyd3bit.
        self.register_buffer("_w_amax", weight.detach().float().abs().amax())
        self.register_buffer("_global", torch.ones((), device=dev))
        # Unrounded signed block scale relative to _global; to_e4m3 is applied in
        # the forward, so this stays exact under the rescale in on_group_linked().
        self.register_buffer("_block_init", torch.ones(out, n_blocks, device=dev))

        self.scale_delta = nn.Parameter(torch.zeros(out, n_blocks, device=dev))
        self.quant_logits = nn.Parameter(
            torch.zeros(out * self.n_levels, in_, dtype=logits_dtype, device=dev))
        # round-to-nearest assignment at init, kept for the `reassigned` diagnostic
        self.register_buffer("_init_levels", torch.zeros(out, in_, dtype=torch.uint8,
                                                         device=dev))

        self._w_master = weight.detach().float()   # dropped after group linking
        self._init_quant()

    # --- two-level scaling / fused groups ----------------------------------
    def amax(self) -> Tensor:
        return self._w_amax

    @torch.no_grad()
    def _init_quant(self) -> None:
        """(Re)derive block scales and assignment logits from the master weight.

        Called once at construction with this layer's own global scale, and again
        from on_group_linked() once the fused-group scale is final.
        """
        w = self._w_master
        blocks = w.reshape(self._out, -1, self.block_size)
        amax, arg = blocks.abs().max(dim=-1, keepdim=True)
        # signed block scale: absorbs the sign of the block's max-abs element, so
        # the extreme element normalizes to +SCALE_REF and the grid's asymmetry is
        # oriented consistently.
        block_amax = amax.clamp(min=1e-8) * blocks.take_along_dim(arg, -1).sign()

        self._global.copy_(self.group_global_scale())
        self._block_init.copy_((block_amax.squeeze(-1) / SCALE_REF) / self._global)

        eff = self._effective_scales()
        self.quant_logits.copy_(
            proximity_logits(w / eff[:, self._idx], self._values, self._spacing,
                             self._std, self._strength, self._noise)
            .to(self.logits_dtype).reshape(self._out * self.n_levels, self._in)
        )
        self._init_levels.copy_(self.level_indices().to(torch.uint8))
        self._wq.copy_(self._compute_wq())

    @torch.no_grad()
    def on_group_linked(self) -> None:
        self._init_quant()
        del self._w_master          # the format has no master weight past init

    # --- GSQLinearBase contract --------------------------------------------
    def block_scale(self) -> Tensor:
        """Signed per-block scale as it is stored in the format: on the E4M3 grid,
        with a straight-through gradient to the FP32 log2 master."""
        s = self._block_init * torch.exp2(self.scale_delta)
        return ste(s, to_e4m3(s))

    def _effective_scales(self) -> Tensor:
        return self.block_scale() * self._global

    def scale_parameters(self) -> list[nn.Parameter]:
        return [self.scale_delta]

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "GSQLloydLinear":
        return cls(linear.weight.data, linear.bias, **kwargs)


def apply_gsqlloyd3bit(model: nn.Module, block_size: int = BLOCK, grid: str = "lloyd",
                       **kwargs) -> None:
    """Replace every nn.Linear (except lm_head) with GSQLloydLinear, in-place."""
    replace_linears(
        model,
        lambda lin: GSQLloydLinear.from_linear(lin, block_size=block_size, grid=grid,
                                               **kwargs),
    )


@torch.no_grad()
def assignment_stats(model: nn.Module) -> dict:
    """Diagnostics for how far the learned assignment has moved from init.

    `reassigned` is the fraction of weights whose argmax level differs from the
    round-to-nearest level it started at — if this stays ~0, the logit LR is too
    low and the run is just an expensive no-op. `scale_drift` is the RMS log2
    change of the block scales.
    """
    n = moved = 0
    drift_sq = 0.0
    drift_n = 0
    for m in model.modules():
        if isinstance(m, GSQLloydLinear):
            cur = m.level_indices()
            n += cur.numel()
            moved += int((cur != m._init_levels).sum())
            drift_sq += float((m.scale_delta.float() ** 2).sum())
            drift_n += m.scale_delta.numel()
    if n == 0:
        return {}
    return {
        "quant/reassigned": moved / n,
        "quant/scale_drift_log2_rms": (drift_sq / max(drift_n, 1)) ** 0.5,
    }
