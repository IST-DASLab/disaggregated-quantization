"""Pseudo KV-cache compression: rate-parametrised Gaussian noise, per phase.

Models lossy KV storage without implementing a codec. A token's K/V is degraded ONCE,
at the moment it is written to the paged cache, and every later read sees the degraded
value -- which is what a real compressed cache does.

    prefill engine  ->  kv_bits_prefill
    decode  engine  ->  kv_bits_decode

RATE PARAMETRISATION
--------------------
Shannon's rate-distortion function for a Gaussian source under squared error is
D(R) = sigma^2 * 2^(-2R) = sigma^2 * 4^(-R), i.e. 4^-bits is a ratio of VARIANCES. The
amplitude multiplier is therefore 2^-bits, not 4^-bits -- an error of one square, which
is invisible in the output, so it is stated here rather than inferred:

    sigma_group = 2^(-kv_bits) * RMS(group)          RMS = ||g||_2 / sqrt(16)

Dividing by sqrt(group) makes kv_bits a dimensionless relative error, comparable across
layers, heads, and group sizes; using the raw L2 norm would make the same kv_bits mean
4x more noise at group 16 than at group 4.

Sanity: at 4 bits this is 6.25% amplitude / 1/256 power, against ~1/172 for a real
uniform 4-bit quantizer (SQNR ~ 6.02b - 1.76 dB). The right order, slightly optimistic --
as a rate-distortion BOUND should be.

USABLE RANGE. The cache is bf16, and the noise has to stay above bf16's OWN error or the
run measures the storage format rather than the rate. The relevant floor is the RMS
rounding error, not the machine epsilon: bf16 carries 8 significant bits, so spacing is
eps = 2^-8, the worst-case rounding error is eps/2 = 2^-9, and the RMS over a uniform
distribution in that interval is eps/2/sqrt(3) = 2^-9/sqrt(3) ~ 0.113%.

Against an injected amplitude of 2^-bits that gives:

    kv_bits      6       7       8       9      10
    amplitude  1.56%   0.78%   0.39%   0.20%   0.098%
    vs floor   13.9x    6.9x    3.5x    1.7x    0.9x     <- 10 bits IS the floor

So the ceiling is ~9, not 8. Enforced 2 <= kv_bits <= 8 (>= 3.5x the floor); 9 is
already within 2x and 10 is indistinguishable from no noise. Use >= DISABLED to turn it
off entirely.

WHERE IT IS APPLIED, AND WHY EXACTLY ONCE
-----------------------------------------
`save_kv_layer` fires per attention layer AFTER the layer has written K/V and computed
attention (vllm/model_executor/layers/attention/kv_transfer_utils.py). It receives the
real paged buffer, so mutating in place persists.

The traps this deliberately avoids:

  * `kv_layer` is the WHOLE paged buffer. Noising all of it every call re-noises every
    cached token at every decode step -- a random walk whose variance grows with
    sequence length. Only the slots in `attn_metadata.slot_mapping` (this step's
    freshly-written tokens) are touched.
  * Prefix-cache hits are not recomputed, so their slots never appear in slot_mapping
    and cannot be re-noised. A reused block therefore keeps the noise it was born with,
    which is correct for storage semantics (the sigma is not re-drawn).
  * Blocks transferred prefill->decode over NIXL arrive through the connector LOAD path,
    not through reshape_and_cache, so they never enter the decode engine's slot_mapping.
    Prompt KV therefore keeps kv_bits_prefill and is never re-noised at kv_bits_decode.
  * Preemption/recompute DOES redraw noise. Run with preemption disabled, or treat a run
    that logged preemptions as suspect.

Compose with the real transfer via MultiConnector, this one listed first:

    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer",
      "kv_connector_extra_config":{"connectors":[
        {"kv_connector":"NoisyKVConnector",
         "kv_connector_module_path":"kv_noise_connector",
         "kv_connector_extra_config":{"kv_bits_prefill":4,"kv_bits_decode":2}},
        {"kv_connector":"NixlConnector"}]}}'

Both rates are passed to BOTH engines; each selects its own by kv_role, so one config
string can be shared and the two cannot be swapped by mistake.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

# NOT init_logger(__name__): this module lives outside the "vllm." logger tree, so a
# bare name gets no handler and every line vanishes. That cost a debugging cycle --
# absence of the init line was read as "the connector never constructed".
logger = init_logger(f"vllm.{__name__}")

DISABLED = 32          # kv_bits >= this means "no noise" (and must be exactly inert)
# MIN_BITS=1 (50% amplitude) is deliberately past anything deployable: it exists to ask
# whether a benchmark EVER degrades from an arm, which a floor of 2 could not answer
# for a curve that is flat at 2 bits.
MIN_BITS, MAX_BITS = 1, 8
GROUP = 16


def group_noise(x: torch.Tensor, bits: float, group: int = GROUP,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Return x with N(0, (2^-bits * RMS(group))^2) added per contiguous group.

    x: (..., D) with D % group == 0. Computed in fp32, returned in x's dtype.
    Pure and side-effect free so it can be tested without a model.
    """
    if x.shape[-1] % group:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by group {group}")
    g = x.reshape(-1, group).float()
    rms = g.pow(2).mean(dim=-1, keepdim=True).sqrt()
    sigma = (2.0 ** -float(bits)) * rms
    noise = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=generator)
    return (g + noise * sigma).reshape(x.shape).to(x.dtype)


def noise_slots_(kv_layer: torch.Tensor, slots: torch.Tensor, bits: float,
                 group: int = GROUP, targets: str = "kv",
                 generator: torch.Generator | None = None) -> int:
    """Noise exactly the given slots of a paged KV buffer, IN PLACE. Returns elems moved.

    kv_layer: (2, pages, page_size, heads, head_dim). slots: flat slot ids, possibly
    containing -1 padding.

    Index the PAGED dims (block, offset) separately rather than flattening them. The
    obvious `kv_layer.view(2, pages * page_size, -1)` raises on the real buffer -- "view
    size is not compatible with input tensor's size and stride" -- because the per-layer
    KV slice is not contiguous across those dims. That crash cost a canary run.

    The tempting repair, swapping view for reshape, would be WORSE THAN THE CRASH:
    reshape silently returns a COPY when it cannot alias, so the write-back would mutate
    a temporary and the whole sweep would report clean numbers having injected no noise
    at all. Advanced-indexing assignment is in-place via index_put_ whatever the layout,
    and it keeps head_dim last so each group of 16 sits inside one head.
    """
    slots = slots[slots >= 0]          # -1 padding would wrap and corrupt another slot
    if slots.numel() == 0:
        return 0
    page_size = kv_layer.shape[2]
    blk = torch.div(slots, page_size, rounding_mode="floor")
    off = slots % page_size
    lo, hi = (0, 2) if targets == "kv" else ((0, 1) if targets == "k" else (1, 2))
    sel = kv_layer[lo:hi, blk, off]                          # gather (copy)
    kv_layer[lo:hi, blk, off] = group_noise(sel, bits, group, generator)
    return int(sel.numel())


class _NoiseParams:
    """Parse + validate the noise settings from a vllm_config. Shared by both connectors.

    Kept separate so the standalone connector and the Nixl subclass cannot drift on how a
    rate is selected -- picking the wrong phase's rate is silent and would mislabel every
    number in a sweep.
    """

    def __init__(self, vllm_config):
        ktc = vllm_config.kv_transfer_config
        cfg = dict(getattr(ktc, "kv_connector_extra_config", None) or {})
        kv_role = getattr(ktc, "kv_role", None)
        # Select by ROLE so one config string can be shared by both engines and the two
        # rates cannot be transposed by a launcher bug.
        if "kv_bits" in cfg:
            self.bits, self.phase = float(cfg["kv_bits"]), "explicit"
        elif kv_role == "kv_producer":
            self.bits, self.phase = float(cfg.get("kv_bits_prefill", DISABLED)), "prefill"
        elif kv_role == "kv_consumer":
            self.bits, self.phase = float(cfg.get("kv_bits_decode", DISABLED)), "decode"
        else:
            raise ValueError(
                f"cannot infer the noise phase from kv_role={kv_role!r}; pass kv_bits "
                f"explicitly for a single-engine run.")
        self.group = int(cfg.get("kv_noise_group", GROUP))
        self.targets = str(cfg.get("kv_noise_targets", "kv")).lower()
        self._seed = None if cfg.get("kv_noise_seed") is None else int(cfg["kv_noise_seed"])
        self._gen = None
        self._validate(vllm_config)

    def gen(self, ref: torch.Tensor):
        if self._seed is not None and self._gen is None:
            self._gen = torch.Generator(device=ref.device)
            self._gen.manual_seed(self._seed)
        return self._gen

    def _validate(self, vllm_config) -> None:
        if self.bits >= DISABLED:
            return
        if not (MIN_BITS <= self.bits <= MAX_BITS):
            raise ValueError(
                f"kv_bits={self.bits} outside [{MIN_BITS}, {MAX_BITS}]. Above {MAX_BITS} "
                f"the injected amplitude 2^-bits approaches bf16's own RMS rounding "
                f"error (2^-9/sqrt(3) ~ 0.113%), so the run would measure the storage "
                f"format rather than the rate. Use >= {DISABLED} to disable.")
        if self.targets not in ("k", "v", "kv"):
            raise ValueError(f"kv_noise_targets={self.targets!r} must be k, v or kv")
        dtype = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if dtype not in ("auto", "bfloat16"):
            raise ValueError(
                f"kv_cache_dtype={dtype!r} would quantize the cache as well as noise it, "
                f"conflating two loss sources. Use auto for a noise study.")
        backend = os.environ.get("VLLM_ATTENTION_BACKEND", "")
        if backend and backend != "FLASH_ATTN":
            raise ValueError(
                f"VLLM_ATTENTION_BACKEND={backend!r}: the (2, pages, page_size, ...) "
                f"layout assumed here is the FlashAttention one.")


class NoisyKVConnector(KVConnectorBase_V1):
    """Worker-side only: adds noise to freshly written KV slots. Transfers nothing."""

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        try:
            super().__init__(vllm_config=vllm_config, role=role,
                             kv_cache_config=kv_cache_config)
        except TypeError:                     # older 2-arg signature
            super().__init__(vllm_config=vllm_config, role=role)

        ktc = vllm_config.kv_transfer_config
        cfg = dict(getattr(ktc, "kv_connector_extra_config", None) or {})
        kv_role = getattr(ktc, "kv_role", None)

        # Select this engine's rate by ROLE, so one config string can be shared by both
        # engines and the two rates cannot be transposed by a launcher bug.
        if "kv_bits" in cfg:
            self.bits = float(cfg["kv_bits"])          # explicit override / single-engine
            self.phase = "explicit"
        elif kv_role == "kv_producer":
            self.bits = float(cfg.get("kv_bits_prefill", DISABLED))
            self.phase = "prefill"
        elif kv_role == "kv_consumer":
            self.bits = float(cfg.get("kv_bits_decode", DISABLED))
            self.phase = "decode"
        else:
            raise ValueError(
                f"NoisyKVConnector cannot infer its phase from kv_role={kv_role!r}. "
                f"Pass kv_bits explicitly for a single-engine run.")

        self.group = int(cfg.get("kv_noise_group", GROUP))
        self.targets = str(cfg.get("kv_noise_targets", "kv")).lower()   # k | v | kv
        self._validate(vllm_config)

        seed = cfg.get("kv_noise_seed")
        self._gen: torch.Generator | None = None
        self._seed = None if seed is None else int(seed)
        # Counters are the ONLY evidence the hook actually ran: the wrapper in
        # kv_transfer_utils.py silently returns early when attn_metadata is None or the
        # connector has no metadata, so a run can produce clean numbers having noised
        # nothing. Assert on these, never on "the score moved".
        self.stats = {"calls": 0, "slots": 0, "elems": 0}
        logger.info("NoisyKVConnector[%s] role=%s kv_bits=%s group=%d targets=%s "
                    "(amplitude %.4g of RMS)", self.phase, role, self.bits, self.group,
                    self.targets, 2.0 ** -self.bits if self.bits < DISABLED else 0.0)

    def _validate(self, vllm_config) -> None:
        if self.bits >= DISABLED:
            return
        if not (MIN_BITS <= self.bits <= MAX_BITS):
            raise ValueError(
                f"kv_bits={self.bits} outside [{MIN_BITS}, {MAX_BITS}]. Above {MAX_BITS} "
                f"the injected amplitude 2^-bits approaches bf16's own RMS rounding "
                f"error (2^-9/sqrt(3) ~ 0.113%): 9 bits is only 1.7x it and 10 bits is "
                f"below it, so the run would measure the storage format rather than the "
                f"rate. Use >= {DISABLED} to disable.")
        if self.targets not in ("k", "v", "kv"):
            raise ValueError(f"kv_noise_targets={self.targets!r} must be k, v or kv")
        dtype = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if dtype not in ("auto", "bfloat16"):
            raise ValueError(
                f"kv_cache_dtype={dtype!r} would quantize the cache as well as noise it, "
                f"conflating two loss sources. Use auto for a noise study.")
        backend = os.environ.get("VLLM_ATTENTION_BACKEND", "")
        if backend and backend != "FLASH_ATTN":
            raise ValueError(
                f"VLLM_ATTENTION_BACKEND={backend!r}: the (2, pages, page_size, ...) "
                f"layout assumed here is the FlashAttention one.")

    # ---- worker side --------------------------------------------------------
    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs: Any) -> None:
        if self.bits >= DISABLED:
            return
        slots = getattr(attn_metadata, "slot_mapping", None)
        if slots is None or slots.numel() == 0:
            return
        if kv_layer.dim() < 3 or kv_layer.shape[0] != 2:
            raise RuntimeError(
                f"expected a (2, pages, page_size, ...) KV buffer, got "
                f"{tuple(kv_layer.shape)}. MLA / non-FlashAttention layouts are not "
                f"supported -- noising the wrong axis would be silent.")
        if self._gen is None and self._seed is not None:
            self._gen = torch.Generator(device=kv_layer.device)
            self._gen.manual_seed(self._seed)

        n = noise_slots_(kv_layer, slots, self.bits, self.group, self.targets, self._gen)
        self.stats["calls"] += 1
        self.stats["slots"] += int((slots >= 0).sum())
        self.stats["elems"] += n

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return                                    # loads nothing; Nixl does the transfer

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def wait_for_save(self) -> None:
        return                                    # mutation is synchronous and in place

    # ---- scheduler side (inert: this connector owns no external storage) ----
    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int) -> None:
        return

    def build_connector_meta(self, scheduler_output: "SchedulerOutput"
                             ) -> KVConnectorMetadata:
        return KVConnectorMetadata()


class NoisyNixlConnector(NixlConnector):
    """NixlConnector that degrades freshly-written KV slots. USE THIS FOR DISAGG.

    Why a subclass instead of MultiConnector[NoisyKVConnector, NixlConnector]:

    MultiConnector forwards register_kv_caches and register_cross_layers_kv_cache to its
    children, but NOT set_host_xfer_buffer_ops. That call is made on the TOP-LEVEL
    connector (gpu_model_runner.py:6088) and lands on the base-class no-op, so the wrapped
    NixlConnector never receives its copy operation and NixlConnectorWorker.copy_blocks
    stays None. The decode engine then dies the first time it syncs received KV to the
    device:

        nixl_connector.py:1761 in sync_recved_kv_to_device
            assert self.copy_blocks is not None

    That path is only taken when kv_buffer_device="cpu" -- host-buffer staging -- which is
    exactly what this deployment uses. So MultiConnector is unusable here, and the failure
    is silent until the first KV arrives: both engines start, prefill answers 200, and the
    decode side collapses mid-run.

    Subclassing sidesteps all of it. Every registration hook is inherited, the engine-level
    settings (kv_buffer_device, kv_load_failure_policy) stay on the one top-level config
    where vLLM reads them, and there is no wrapper whose fields can silently revert to
    defaults. Nixl's own save_kv_layer is a documented no-op, so calling super() after
    noising costs nothing and keeps the class honest if that ever changes.
    """

    def __init__(self, vllm_config, role, kv_cache_config=None):
        try:
            super().__init__(vllm_config, role, kv_cache_config)
        except TypeError:
            super().__init__(vllm_config, role)
        self._noise = _NoiseParams(vllm_config)
        self.stats = {"calls": 0, "slots": 0, "elems": 0}
        logger.info("NoisyNixlConnector[%s] role=%s kv_bits=%s group=%d targets=%s "
                    "(amplitude %.4g of RMS)", self._noise.phase, role, self._noise.bits,
                    self._noise.group, self._noise.targets,
                    2.0 ** -self._noise.bits if self._noise.bits < DISABLED else 0.0)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        p = self._noise
        if p.bits < DISABLED:
            slots = getattr(attn_metadata, "slot_mapping", None)
            if slots is not None and slots.numel():
                n = noise_slots_(kv_layer, slots, p.bits, p.group, p.targets, p.gen(kv_layer))
                self.stats["calls"] += 1
                self.stats["slots"] += int((slots >= 0).sum())
                self.stats["elems"] += n
        return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
