import math
from enum import Enum
from typing import Tuple

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
from quantizer import NVFP_GROUPSIZE, Quantizer


class QuantizationOrder(str, Enum):
    DEFAULT = "default"
    ACTIVATION = "activation"

class GPTQ:

    def __init__(
        self,
        layer: nn.Linear,
        quantizer_prefill: Quantizer,
        quantizer_decode: Quantizer,
        quantization_order: str = "default",
        block_size: int = 128,
        rel_damp: float = 1e-2,
    ):
        self._validate_layer(layer)
        self.layer = layer
        self.W = self.layer.weight
        self.d_row, self.d_col = layer.weight.shape
        # Quantization properties (dual grids: one per inference stage).
        self.quantizer_prefill = quantizer_prefill
        self.quantizer_decode = quantizer_decode
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
    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Define constants and chunk
        d_col, block_size, device, dtype = self.d_col, self.block_size, self.W_device, self.W_dtype
        # 2) Get per-quantizer group sizes
        group_size_p = self.quantizer_prefill.group_size or d_col
        group_size_d = self.quantizer_decode.group_size or d_col

        # Get scales and zeros (static, from the original weight)
        scales_p, zeros_p = self.quantizer_prefill.get_quantization_params(self.W)
        scales_d, zeros_d = self.quantizer_decode.get_quantization_params(self.W)
        # Dirty hack for GPTQ quantization: quantize one column at a time
        orig_group_size_p = self.quantizer_prefill.group_size
        orig_group_size_d = self.quantizer_decode.group_size
        self.quantizer_prefill.group_size = None
        self.quantizer_decode.group_size = None

        try:
            # Get permutation
            if self.quantization_order == QuantizationOrder.ACTIVATION:
                perm = torch.argsort(self.H.diag(), descending=True)
            else:
                perm = torch.arange(d_col, device=device)
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
                # 2) Iterate over block
                for i in range(ncols):
                    # Get weight column, corresponding Hessian diagonal and group_id
                    w_ci = w_blk[:, i]
                    d = H_inv_cho_blk[i, i]
                    # Original column index -> its group for each quantizer
                    orig_col = perm[c1 + i]
                    g_p = orig_col // group_size_p
                    g_d = orig_col // group_size_d
                    # Quantize weight column with both grids
                    w_q = self.quantizer_prefill.quantize_dequantize(w_ci, scales_p[:, g_p], zeros_p[:, g_p])
                    w_d = self.quantizer_decode.quantize_dequantize(w_ci, scales_d[:, g_d], zeros_d[:, g_d])
                    w[:, c1 + i] = w_q
                    w_dec[:, c1 + i] = w_d
                    # Update subsequent weights with the combined rounding error
                    err = ((w_ci - w_q) * 1 + (w_ci - w_d) * 1) / d
                    w_blk[:, i:].addr_(err, H_inv_cho_blk[i, i:], alpha=-1)
                    errs[:, i] = err
                # 3) Update the weights after block
                w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)

            # Invert permutation
            w = w[:, perm_inv].contiguous()
            w_dec = w_dec[:, perm_inv].contiguous()
            self.H = self.H[perm_inv][:, perm_inv]
        finally:
            # Restore quantizer group sizes
            self.quantizer_prefill.group_size = orig_group_size_p
            self.quantizer_decode.group_size = orig_group_size_d

        return w.to(dtype), w_dec.to(dtype)

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


def gptq_quantization(
    model,
    calibration_data,
    wbits=3,
    device="cuda",
    act_quant=False,
    block_size=128,
    rel_damp=1e-2,
    quantization_order="default",
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

        # One GPTQ handle (one Hessian, both quantizers) per layer.
        gptq = {}
        for name, layer in target_layers.items():
            quantizer_prefill = Quantizer(format="nvfp", bits=4, symmetric=True, group_size=NVFP_GROUPSIZE)
            quantizer_decode = Quantizer(format="int", bits=wbits, symmetric=True, group_size=NVFP_GROUPSIZE)
            gptq[name] = GPTQ(
                layer,
                quantizer_prefill,
                quantizer_decode,
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

        # Quantize each layer into prefill (NVFP4) and decode (INT) weights that share
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
