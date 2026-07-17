import math
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Tuple

import torch
import torch.nn as nn

from model_utils import (
    InputCollector,
    ForwardInterrupt,
    QuantizedLinear,
    clear_device_cache,
    to,
    _set_submodule,
    maybe_first_element,
)
from quantizer import NVFP_GROUPSIZE, Quantizer, make_quantizer


class QuantizationOrder(str, Enum):
    DEFAULT = "default"
    ACTIVATION = "activation"


@contextmanager
def _column_group_size(*quantizers: Quantizer):
    """Temporarily disable grouping so the quantizers operate one column at a time
    (GPTQ quantizes column-by-column, passing a single group's scale each step)."""
    saved = [(q, q.group_size) for q in quantizers]
    for q, _ in saved:
        q.group_size = None
    try:
        yield
    finally:
        for q, gs in saved:
            q.group_size = gs


class GPTQ:
    """GPTQ for a single-format quantizer: prefill and decode share the same grid, so
    both output weights are identical and the feedback error is ``w - w_q``.

    Subclasses override :meth:`step` to plug in a decode grid and its error term; the
    shared, numerically sensitive machinery lives once in :meth:`_gptq_loop`.
    """

    def __init__(
        self,
        layer: nn.Linear,
        quantizer: Quantizer,
        quantization_order: str = "default",
        block_size: int = 128,
        rel_damp: float = 1e-2,
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = layer.weight.shape
        self.quantizer = quantizer
        self.quantization_order = QuantizationOrder(quantization_order)
        self.block_size = block_size
        self.rel_damp = rel_damp
        # Backup layer properties
        self.W_device = self.W.device
        self.W_dtype = self.W.dtype
        self.W_shape = self.W.shape
        # init hessian
        self.H = None
        self.num_samples = 0

    @staticmethod
    def _validate_layer(layer):
        assert isinstance(layer, nn.Linear), "GPTQ supports only linear layers."

    # preparatory methods
    @torch.no_grad()
    def update(self, input: torch.Tensor) -> None:
        """
        Update the estimate of the Hessian matrix from a batch of layer inputs.

        Args:
            input: batch of layer inputs
        """
        # get batch size
        batch_size = input.shape[0]
        # init hessian
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=input.device, dtype=torch.float32)
        # input reshaping
        input = input.reshape(-1, input.shape[-1])
        # cast input to float32 before addition
        input = input.float()
        # rescale and update matrix
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.mul_(beta)
        input.mul_(math.sqrt(alpha))
        self.H.add_(input.t() @ input)
        self.num_samples += batch_size

    def reset(self) -> None:
        self.W = self.layer.weight
        self.H = None
        self.num_samples = 0
        clear_device_cache()

    @torch.no_grad()
    def quantization_pre_step(self) -> None:
        assert self.H is not None, "One has to process at least one sample of calibration data to run GPTQ."
        # copy weight and convert to float
        self.W = self.W.clone().float()
        # flag pre step as completed
        self.pre_step_completed = True

    @torch.no_grad()
    def _gptq_loop(
        self, quantize_column: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared GPTQ machinery: permute, invert the Hessian, and sweep columns.

        ``quantize_column(w_ci, orig_col, d) -> (w_q, w_d, err)`` returns the prefill and
        decode columns plus the already Hessian-normalised feedback error to propagate.
        Returns ``(w_prefill, w_decode)`` in the layer dtype.
        """
        d_col, block_size, dtype = self.d_col, self.block_size, self.W_dtype
        # Get permutation
        if self.quantization_order == QuantizationOrder.ACTIVATION:
            perm = torch.argsort(self.H.diag(), descending=True)
        else:
            perm = torch.arange(d_col, device=self.W_device)
        perm_inv = torch.argsort(perm)
        # Permute Hessian prior to inversion
        self.H = self.H[perm][:, perm]
        # Get weight (prefill output doubles as the shared running weight; w_dec
        # collects the decode-quantized columns)
        w = self.W[:, perm]
        w_dec = torch.zeros_like(w)
        # Get Hessian inverse
        H_inv_cho = self._get_hessian_inverse(w)
        # Quantize
        for c1 in range(0, d_col, block_size):
            c2 = min(c1 + block_size, d_col)
            ncols = c2 - c1
            w_blk = w[:, c1:c2].clone()
            errs = torch.zeros_like(w_blk)
            H_inv_cho_blk = H_inv_cho[c1:c2, c1:c2]
            # Iterate over block
            for i in range(ncols):
                # Weight column, its Hessian diagonal and original (pre-perm) index
                w_ci = w_blk[:, i]
                d = H_inv_cho_blk[i, i]
                w_q, w_d, err = quantize_column(w_ci, perm[c1 + i], d)
                w[:, c1 + i] = w_q
                w_dec[:, c1 + i] = w_d
                # Update subsequent weights with the rounding error
                w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err
            # Update the weights after block
            w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

        # Invert permutation
        w = w[:, perm_inv].contiguous()
        w_dec = w_dec[:, perm_inv].contiguous()
        self.H = self.H[perm_inv][:, perm_inv]
        return w.to(dtype), w_dec.to(dtype)

    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.quantizer
        group_size = q.group_size or self.d_col
        scales, zeros = q.get_quantization_params(self.W)
        with _column_group_size(q):
            def quantize_column(w_ci, orig_col, d):
                g = orig_col // group_size
                w_q = q.quantize_dequantize(w_ci, scales[:, g], zeros[:, g])
                # Single format: decode == prefill, feedback error is w - w_q.
                return w_q, w_q, (w_ci - w_q) / d

            return self._gptq_loop(quantize_column)

    @torch.no_grad()
    def _get_hessian_inverse(self, w: torch.Tensor):
        # Get columns with all zeros
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H = self.H
        # mask rows and columns with zero input channels
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        # Hessian regularization
        damp = self.rel_damp * torch.diag(self.H).mean()
        self.H[range(self.d_col), range(self.d_col)] += damp
        # invert
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        except Exception:
            H_inv_cho = torch.eye(self.d_col, device=H.device, dtype=torch.float32)
        return H_inv_cho

    def quantize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.quantization_pre_step()
        return self.step()


class PrefillDecodeGPTQ(GPTQ):
    """Independent decode grid: the decode weight comes from ``quantizer.decode_quantizer``
    with its own scales, and both stages' errors feed back (``(w-w_q) + (w-w_d)``)."""

    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q, q_dec = self.quantizer, self.quantizer.decode_quantizer
        group_size_p = q.group_size or self.d_col
        group_size_d = q_dec.group_size or self.d_col
        scales_p, zeros_p = q.get_quantization_params(self.W)
        scales_d, zeros_d = q_dec.get_quantization_params(self.W)
        with _column_group_size(q, q_dec):
            def quantize_column(w_ci, orig_col, d):
                g_p, g_d = orig_col // group_size_p, orig_col // group_size_d
                w_q = q.quantize_dequantize(w_ci, scales_p[:, g_p], zeros_p[:, g_p])
                w_d = q_dec.quantize_dequantize(w_ci, scales_d[:, g_d], zeros_d[:, g_d])
                return w_q, w_d, (1/2 * (w_ci - w_q) + (w_ci - w_d)) / d

            return self._gptq_loop(quantize_column)


class DowncastGPTQ(GPTQ):
    """Downcast decode: the decode weight is a 3-bit LUT view of the stored NVFP4 code,
    so it reuses the prefill NVFP4 scales. The decode LUT index is always the
    stored FP4 index with its low bit dropped (``fp4_idx >> 1``).
    """

    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.quantizer
        group_size = q.group_size or self.d_col
        scales, zeros = q.get_quantization_params(self.W)
        with _column_group_size(q):
            def quantize_column(w_ci, orig_col, d):
                g = orig_col // group_size
                w_q = q.quantize_dequantize(w_ci, scales[:, g], zeros[:, g])
                w_d = q.quantize_dequantize_decode(w_ci, scales[:, g], zeros[:, g])
                return w_q, w_d, (1 / 2 * (w_ci - w_q) + (w_ci - w_d)) / d

            return self._gptq_loop(quantize_column)


# scheme -> (GPTQ variant). The quantizer itself is built by make_quantizer.
GPTQ_SCHEMES = {
    "nvfp": GPTQ,
    "int": GPTQ,
    "independent": PrefillDecodeGPTQ,
    "downcast": DowncastGPTQ,
}


def make_gptq(scheme: str, layer: nn.Linear, wbits: int = 3, **kwargs) -> GPTQ:
    """Pick the GPTQ variant + quantizer for a scheme (mirrors make_quantizer)."""
    if scheme not in GPTQ_SCHEMES:
        raise ValueError(f"Unknown GPTQ scheme: {scheme!r}")
    return GPTQ_SCHEMES[scheme](layer, make_quantizer(scheme, wbits), **kwargs)


def gptq_quantization(
    model,
    calibration_data,
    wbits=3,
    device="cuda",
    act_quant=False,
    block_size=128,
    rel_damp=1e-2,
    quantization_order="default",
    scheme="downcast",
):
    print("Start GPTQ quantization...")

    blocks = model.model.layers
    blocks[0] = InputCollector(blocks[0])

    for sample in calibration_data:
        try:
            model(sample)
        except ForwardInterrupt:
            pass

    input_args = blocks[0].input_args
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    device_type = "cuda"

    for block_idx, block in enumerate(blocks):
        print(f"Quantizing block {block_idx}...")
        block = block.to(device)

        # Target the same linears as RTN (attention + mlp projections).
        target_layers = {
            name: layer
            for name, layer in block.named_modules()
            if ("mlp" in name or "attn" in name) and isinstance(layer, nn.Linear)
        }

        # One GPTQ handle (one Hessian, one quantizer) per layer. The scheme selects the
        # GPTQ variant: "nvfp"/"int" (single format), "independent" (own decode scales),
        # or "downcast" (decode = 3-bit LUT view of the stored NVFP4 code).
        gptq = {}
        for name, layer in target_layers.items():
            gptq[name] = make_gptq(
                scheme,
                layer,
                wbits,
                quantization_order=quantization_order,
                block_size=block_size,
                rel_damp=rel_damp,
            )

        # Accumulate Hessians from a full-precision calibration pass over the block.
        hooks = []
        def make_hook(name):
            def _hook(_module, inp, _out):
                gptq[name].update(inp[0])
            return _hook
        for name, layer in target_layers.items():
            hooks.append(layer.register_forward_hook(make_hook(name)))

        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=True):
                block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
        for h in hooks:
            h.remove()

        # Quantize each layer into prefill (NVFP4) and decode (nested LUT3) weights that share
        # the layer's Hessian, then swap in the dual-weight module.
        for layer_name, layer in target_layers.items():
            with torch.no_grad():
                dqweight_prefill, dqweight_decode = gptq[layer_name].quantize()

                bias = layer.bias.detach() if layer.bias is not None else None
                qlinear = QuantizedLinear(
                    dqweight_prefill,
                    dqweight_decode,
                    bias,
                    act_quantizer=Quantizer(format="nvfp", bits=4, symmetric=True, group_size=NVFP_GROUPSIZE) if act_quant else None,
                )
                _set_submodule(block, layer_name, qlinear)
            # Free the Hessian for this layer once both weights are done.
            gptq[layer_name].H = None

        del gptq

        # Propagate quantized activations to the next block.
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=True):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            out = maybe_first_element(out).to(device)
            if len(inp_args) > 0:
                inp_args[0].data = out
            elif "hidden_states" in inp_kwargs:
                inp_kwargs["hidden_states"] = out
            else:
                raise ValueError("Unsupported block input format.")

        clear_device_cache(garbage_collection=True)
    clear_device_cache(garbage_collection=True)
