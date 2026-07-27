"""Isolate WHERE my NVFP4 fake-quant diverges from vLLM's real W4A4, at the level
of a single linear layer.

Hook vLLM's actual layers (a non-fused down_proj and the fused qkv_proj/gate_up_proj)
to capture their real (input, output). Then reconstruct each layer's output with my
own quantization (unpack the checkpoint weight the CT way; quantize the captured
input with nvfp4). Compare via cosine similarity:
  - recon WITH activation quant  vs vLLM  -> should be ~1.0 if my quant matches
  - recon WITHOUT activation quant vs vLLM -> shows how much act-quant matters
  - weight-only-dequant @ x        vs vLLM -> isolates weight vs activation error
A non-fused layer matching but a fused one not ⇒ fused per-projection scale collapse.

Usage: python verify_layer.py <ct_ckpt_dir>
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from quantizers import nvfp4
from quantizers.nvfp4 import nvfp4_quantize
from quantizers.blocked import GLOBAL_DEN as _GLOBAL_DEN

BASE = "Qwen/Qwen3-4B"
_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                     dtype=torch.float32)


def unpack_ct(packed, wscale, wgs):
    O, Kh = packed.shape
    K = Kh * 2
    codes = torch.empty(O, K, dtype=torch.long)
    codes[:, 0::2] = (packed & 0x0F).long()
    codes[:, 1::2] = (packed >> 4).long()
    vals = _E2M1[codes]
    bs = wscale.float().reshape(O, K // 16, 1)
    return (vals.reshape(O, K // 16, 16) * bs / wgs.float()).reshape(O, K)


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    ckpt = sys.argv[1]
    sd = load_file(ckpt + "/model.safetensors")

    from vllm import LLM, SamplingParams
    llm = LLM(model=ckpt, tokenizer=BASE, dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.7, enforce_eager=True, trust_remote_code=True)
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model = runner.model

    # CT checkpoint names -> vLLM fused module names + the source projections
    layers = {
        "model.layers.0.mlp.down_proj":      ["model.layers.0.mlp.down_proj"],           # non-fused
        "model.layers.0.self_attn.qkv_proj": ["model.layers.0.self_attn.q_proj",
                                              "model.layers.0.self_attn.k_proj",
                                              "model.layers.0.self_attn.v_proj"],         # fused
        "model.layers.0.mlp.gate_up_proj":   ["model.layers.0.mlp.gate_proj",
                                              "model.layers.0.mlp.up_proj"],              # fused
    }
    cap = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[name] = (inp[0].detach().float().cpu(), o.detach().float().cpu())
        return hook
    hooked = {}
    for n, m in model.named_modules():
        if n in layers:
            m.register_forward_hook(mk(n))
            hooked[n] = True
    print("hooked:", list(hooked), flush=True)

    llm.generate({"prompt_token_ids":
                  __import__("transformers").AutoTokenizer.from_pretrained(BASE)(
                      "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
                  ).input_ids},
                 SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=None))

    print("\n============ PER-LAYER: my recon vs vLLM ============", flush=True)
    for vname, projs in layers.items():
        if vname not in cap:
            print(f"{vname}: NOT captured"); continue
        x, y_vllm = cap[vname]                       # x:[T,K]  y:[T, sum_out]
        # build fused weight from per-projection unpack (CT convention)
        W = torch.cat([unpack_ct(sd[f"{p}.weight_packed"], sd[f"{p}.weight_scale"],
                                 sd[f"{p}.weight_global_scale"]) for p in projs], dim=0)
        igs = sd[f"{projs[0]}.input_global_scale"].float()      # stored = 2688/act_amax
        gscale = (1.0 / igs)                                     # act global for nvfp4_quantize = act_amax/2688
        xq, _, _ = nvfp4_quantize(x, 16, global_scale=gscale)   # static act quant (my emulation)
        xq_dyn, _, _ = nvfp4_quantize(x, 16)                    # dynamic act quant
        y_wonly = F.linear(x, W)                                # weights quantized, act full-prec
        y_static = F.linear(xq, W)
        y_dyn = F.linear(xq_dyn, W)
        print(f"{vname}  ({'fused' if len(projs)>1 else 'non-fused'})", flush=True)
        print(f"    cos(vLLM, weight-only@x)      = {cos(y_vllm, y_wonly):.4f}")
        print(f"    cos(vLLM, static-act-quant)   = {cos(y_vllm, y_static):.4f}")
        print(f"    cos(vLLM, dynamic-act-quant)  = {cos(y_vllm, y_dyn):.4f}")
    print("====================================================", flush=True)


if __name__ == "__main__":
    main()
