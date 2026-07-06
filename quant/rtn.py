import torch
import torch.nn as nn

from model_utils import InputCollector, ForwardInterrupt, QuantizedLinear, clear_device_cache, to, _set_submodule, maybe_first_element
from quantizer import NVFP_GROUPSIZE, Quantizer, make_quantizer


def rtn_quantization(
    model,
    calibration_data,
    wbits=3,
    device="cuda",
    act_quant=False,
    scheme="independent",
):
    assert scheme in ("nvfp", "int", "independent"), "RTN supports only single-format and independent schemes."
    print("Start RTN quantization...")

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

        for layer_name, layer in list(block.named_modules()):
            if not("mlp" in layer_name or "attn" in layer_name):
                continue
            if not isinstance(layer, nn.Linear):
                continue

            with torch.no_grad():
                w = layer.weight  # [out_features, in_features]

                quantizer = make_quantizer(scheme, wbits)
                scales, zeros = quantizer.get_quantization_params(w)
                dqweight_prefill = quantizer.quantize_dequantize(w, scales, zeros)

                # Independent decode derives its own scales from the same weight;
                # single-format schemes reuse the prefill weight for both stages.
                quantizer_decode = getattr(quantizer, "decode_quantizer", None)
                if quantizer_decode is not None:
                    scales_decode, zeros_decode = quantizer_decode.get_quantization_params(w)
                    dqweight_decode = quantizer_decode.quantize_dequantize(w, scales_decode, zeros_decode)
                else:
                    dqweight_decode = dqweight_prefill

                bias = layer.bias.detach() if layer.bias is not None else None
                qlinear = QuantizedLinear(
                    dqweight_prefill,
                    dqweight_decode,
                    bias,
                    act_quantizer=Quantizer("nvfp", bits=4, symmetric=True, group_size=NVFP_GROUPSIZE) if act_quant else None
                )
                _set_submodule(block, layer_name, qlinear)

        # Propagate activations to next block
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
