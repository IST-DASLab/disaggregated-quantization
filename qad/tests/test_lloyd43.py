"""Gates for the lloyd43 grid — MSE-optimal with 0.0 and +6.0 pinned.

lloyd43 is the same layer, block scaling and pseudo-quantized export as lloyd3bit; the
only difference is the 8-level look-up table. So the things worth defending are exactly
the two properties the grid was chosen for, plus the guarantee that adding it did not
disturb lloyd3bit:

  +6.0 pinned   the SIGNED normalisation maps each block's max-abs element to exactly
                +6, so that element is the block outlier by construction. lloyd3bit tops
                out at 5.788 and carries a fixed 0.212 error there.
   0.0 pinned   a true flush-to-zero. lloyd3bit straddles zero (-0.339, +0.927) and
                cannot represent a zero weight at all.

    python tests/test_lloyd43.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from quantizers import (REGISTRY, build_quantizer_params, uses_compressed_tensors,
                        variants)
from quantizers.grids import LLOYD43_SIGNED_3BIT, LLOYD_SIGNED_3BIT
from quantizers.lloyd import SignedLloydLinear


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def build():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, False
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).cuda().float()
    params, _ = build_quantizer_params("lloyd43", "")
    REGISTRY["lloyd43"]["apply"](m, **params)
    return m


def test_grid_shape_and_pins():
    g = LLOYD43_SIGNED_3BIT
    check("8 levels (3-bit)", g.numel() == 8)
    check("sorted ascending", bool((g.diff() > 0).all()))
    check("0.0 IS a grid point", bool((g == 0).any()))
    check("+6.0 IS a grid point", bool((g == 6.0).any()),
          "the block extreme is representable exactly")
    # the property lloyd43 exists to fix -- assert the contrast, not just the value
    check("lloyd3bit has neither pin",
          not bool((LLOYD_SIGNED_3BIT == 0).any()) and float(LLOYD_SIGNED_3BIT.max()) < 6.0,
          f"lloyd3bit max={float(LLOYD_SIGNED_3BIT.max())}")
    # Compare the WHOLE grid, at fp32 tolerance -- these are float32 tensors, so
    # float(t[-1]) is 5.788000106811523 and a 1e-9 bound is tighter than the dtype can
    # represent. (The tag hash test below is the independent guard on the same fact.)
    want = torch.tensor([-4.795, -3.062, -1.631, -0.339, 0.927, 2.265, 3.797, 5.788])
    check("lloyd3bit grid UNCHANGED by this addition",
          bool(torch.allclose(LLOYD_SIGNED_3BIT.cpu().float(), want, atol=1e-5)),
          "its checkpoints must stay reproducible")


def test_registry_wiring():
    params, h = build_quantizer_params("lloyd43", "")
    check("registered", "lloyd43" in REGISTRY, f"hash={h}")
    check("uses the lloyd43 grid", params["grid"] == "lloyd43")
    check("single-checkpoint format", variants("lloyd43") == [None])
    check("pseudo-quantized (no 3-bit kernel exists)",
          not uses_compressed_tensors("lloyd43"))
    _, h3 = build_quantizer_params("lloyd3bit", "")
    check("tag hash differs from lloyd3bit", h != h3, f"{h} vs {h3}")
    check("lloyd3bit hash is still 3a7ebd60", h3 == "3a7ebd60",
          "changing it would rename every existing lloyd3bit checkpoint")


def test_rounding_lands_on_the_grid():
    m = build()
    lin = m.model.layers[0].mlp.gate_proj
    check("layer is SignedLloydLinear", isinstance(lin, SignedLloydLinear))
    check("layer carries the lloyd43 grid", lin.grid_name == "lloyd43")

    g = LLOYD43_SIGNED_3BIT.to("cuda")
    x = torch.linspace(-8, 8, 8192, device="cuda")
    r = lin.rounder(x)
    off = float((r.unsqueeze(-1) - g).abs().min(-1).values.max())
    check("every rounded value is ON the grid", off < 1e-4, f"max off-grid {off:.2e}")

    # the flush-to-zero that lloyd3bit cannot do
    z = lin.rounder(torch.tensor([0.0, 0.3, -0.3, 0.6, -0.6], device="cuda"))
    check("near-zero flushes to EXACTLY 0", float(z.abs().max()) == 0.0,
          f"got {z.tolist()}")
    # and the outlier that lloyd3bit cannot represent
    e = lin.rounder(torch.tensor([6.0], device="cuda"))
    check("+6.0 maps to itself exactly", float(e.abs().max() - 6.0) == 0.0)


def test_weights_actually_quantize():
    m = build()
    lin = m.model.layers[0].mlp.gate_proj
    wq = lin._wq.detach().float()
    check("quantized cache is finite", bool(torch.isfinite(wq).all()))
    check("weights really changed", float((wq - lin.weight.detach().float()).abs().max()) > 0)
    rel = float(((wq - lin.weight.detach().float()) ** 2).mean()
                / (lin.weight.detach().float() ** 2).mean())
    # sanity band only: a 3-bit grid on real weights, not a Gaussian benchmark
    check("relative error is in a sane 3-bit band", 1e-4 < rel < 0.2, f"rel MSE {rel:.4f}")


if __name__ == "__main__":
    for fn in [test_grid_shape_and_pins, test_registry_wiring,
               test_rounding_lands_on_the_grid, test_weights_actually_quantize]:
        print(f"\n{fn.__name__}")
        fn()
    print("\nall lloyd43 tests passed")
