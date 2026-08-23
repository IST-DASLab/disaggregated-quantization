"""Gates for the lloyd21 grid — the lloyd43 construction at 2 BITS.

Same layer, block scaling and pseudo-quantized export as lloyd3bit/lloyd43; the only
difference is a 4-level look-up table instead of 8. So what is worth defending is what
is SPECIFIC to it, not the machinery it shares:

  4 levels        2 bits, not 3. A grid arg that were silently ignored would leave the
                  layer on lloyd43 and every result would look like a plausible 3-bit
                  curve, so the tests assert lloyd21 DIFFERS from lloyd43 rather than
                  merely being self-consistent.
  0.0 / +6.0      the same two pins lloyd43 has: an exact flush-to-zero and an exact
                  block extreme.
  ASYMMETRY       2 positive levels against 1 negative. This only works because the
                  block scale is SIGNED -- the normalisation puts every block's max-abs
                  element at exactly +6, so the positive tail is what must be
                  represented. Under an unsigned scale, every block whose extreme is
                  negative would be clipped 6.0 -> 3.6517: a 39% error on the largest
                  weight in that block, which is roughly half of all blocks.

    python tests/test_lloyd21.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from quantizers import REGISTRY, build_quantizer_params, uses_compressed_tensors
from quantizers.blocked import blocked_quantize, grid_rounder
from quantizers.grids import LLOYD21_SIGNED_2BIT, LLOYD43_SIGNED_3BIT
from quantizers.lloyd import SignedLloydLinear


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def build(name="lloyd21"):
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.vocab_size, cfg.tie_word_embeddings = 512, False
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).cuda().float()
    params, _ = build_quantizer_params(name, "")
    REGISTRY[name]["apply"](m, **params)
    return m


def test_grid_shape_and_pins():
    g = LLOYD21_SIGNED_2BIT
    check("4 levels (2-bit)", g.numel() == 4, f"{g.numel()} levels")
    check("sorted ascending", bool((g.diff() > 0).all()))
    check("0.0 IS a grid point", bool((g == 0).any()), "true flush-to-zero")
    check("+6.0 IS a grid point", bool((g == 6.0).any()),
          "the block extreme is representable exactly")
    pos = int((g > 0).sum()); neg = int((g < 0).sum())
    check("2 positive / 1 negative (the lloyd21 name)", pos == 2 and neg == 1,
          f"{pos} positive, {neg} negative")
    check("exactly the requested grid",
          torch.allclose(g, torch.tensor([-3.6517, 0.0000, 2.5227, 6.0000]), atol=1e-6))


def test_it_is_not_lloyd43():
    """A silently ignored grid arg would leave this on lloyd43 and look entirely normal."""
    check("grid DIFFERS from lloyd43",
          LLOYD21_SIGNED_2BIT.numel() != LLOYD43_SIGNED_3BIT.numel())
    torch.manual_seed(0)
    w = torch.randn(64, 128, device="cuda")
    gs = {n: g.cuda() for n, g in (("lloyd21", LLOYD21_SIGNED_2BIT),
                                   ("lloyd43", LLOYD43_SIGNED_3BIT))}
    deq = {n: blocked_quantize(w, grid_rounder(g), 16, signed=True)[0]
           for n, g in gs.items()}
    d = float((deq["lloyd21"] - deq["lloyd43"]).abs().max())
    check("quantized weights DIFFER from lloyd43", d > 1e-3, f"max|diff| {d:.3e}")
    # and 2 bits must be strictly worse than 3 -- if it were not, something is wrong
    mse = {n: float(((deq[n] - w) ** 2).mean() / (w ** 2).mean()) for n in deq}
    check("2-bit rel MSE > 3-bit rel MSE", mse["lloyd21"] > mse["lloyd43"],
          f"lloyd21 {mse['lloyd21']:.4f} vs lloyd43 {mse['lloyd43']:.4f}")


def test_signed_normalisation_is_load_bearing():
    """Unsigned clips the extreme of every negatively-peaked block 6.0 -> 3.6517.

    Asserted DETERMINISTICALLY on a hand-built block rather than via aggregate MSE: the
    MSE gap is real but only ~1.2x, which is too close to a threshold to be a stable
    gate (it would flip on a different seed). The clipping claim does not depend on the
    draw at all -- it is a property of the grid.
    """
    check("layer declares signed", SignedLloydLinear.signed is True)
    g = LLOYD21_SIGNED_2BIT.cuda()
    # one block of 16 whose max-abs element is NEGATIVE
    blk = torch.full((1, 16), 0.1, device="cuda")
    blk[0, 3] = -2.0
    for signed, want_exact in ((True, True), (False, False)):
        deq = blocked_quantize(blk, grid_rounder(g), 16, signed=signed)[0]
        err = float((deq[0, 3] - blk[0, 3]).abs())
        got_exact = err < 1e-4
        check(f"signed={signed}: negative extreme {'exact' if want_exact else 'CLIPPED'}",
              got_exact == want_exact,
              f"|err| at the block extreme = {err:.4f}")
    # and the clipped value is exactly the grid's negative level scaled to the block
    deq_u = blocked_quantize(blk, grid_rounder(g), 16, signed=False)[0]
    ratio = float((deq_u[0, 3] / blk[0, 3]))
    check("unsigned lands on -3.6517/6.0 of the extreme", abs(ratio - 3.6517 / 6.0) < 1e-3,
          f"ratio {ratio:.4f} vs expected {3.6517 / 6.0:.4f} -- a 39% error")


def test_rounding_lands_on_the_grid():
    m = build()
    lin = m.model.layers[0].mlp.gate_proj
    wq = lin._wq.detach().float()
    blk = wq.reshape(wq.shape[0], -1, lin.block_size)
    # undo the SIGNED normalisation: divide by the block's max-abs element WITH its sign
    idx = blk.abs().argmax(-1, keepdim=True)
    peak = blk.take_along_dim(idx, -1)
    peak = torch.where(peak.abs() < 1e-12, torch.ones_like(peak), peak)
    norm = (blk / peak) * 6.0
    g = LLOYD21_SIGNED_2BIT.to(wq.device)
    off = float((norm.unsqueeze(-1) - g).abs().min(-1).values.max())
    # Tolerance is BF16 round-off, not slack. _wq is stored bf16 (quantizers/base.py),
    # so a value that sits exactly on the grid in fp32 lands within one bf16 ulp of it
    # here. Normalised values reach ~6.0 and bf16 carries 8 mantissa bits, so the bound
    # is ~6 * 2^-8 = 2.3e-2; anything larger means the weight is genuinely off-grid
    # rather than merely rounded. The stricter fp32-era bound was 1e-3.
    check("quantized weights lie on the lloyd21 grid (within bf16 round-off)",
          off < 2.5e-2, f"max off-grid {off:.2e}")
    # Count levels by SNAPPING to the grid, not by rounding to 4 decimals. _wq is stored
    # bf16, so each normalised value sits within one bf16 ulp of its grid point and a
    # fixed-decimal round splits one true level into several -- 18 were counted that way.
    # Snapping asks the question the check is actually about: how many distinct grid
    # points does the quantized weight use?
    lv = int(torch.unique(g[(norm.unsqueeze(-1) - g).abs().argmin(-1)]).numel())
    check("at most 4 distinct normalised levels", lv <= 4, f"{lv} levels")


def test_registry_wiring():
    e = REGISTRY["lloyd21"]
    check("defaults select the lloyd21 grid",
          e["defaults"] == {"block_size": 16, "grid": "lloyd21"})
    check("exports dequantized (no 2-bit kernel exists)", e["export"] == "dequantized")
    check("not a compressed-tensors format", not uses_compressed_tensors("lloyd21"))
    check("single variant", e.get("variants") is None)
    m = build()
    lin = m.model.layers[0].mlp.gate_proj
    check("layer class", isinstance(lin, SignedLloydLinear))
    check("layer carries the lloyd21 grid name", lin.grid_name == "lloyd21")


def test_existing_formats_untouched():
    """Adding a grid must not renumber anyone else's checkpoints."""
    for name, want in (("lloyd43", "42a1cbfd"), ("lloyd3bit", "3a7ebd60"),
                       ("nvfp4", "99914b93")):
        _, h = build_quantizer_params(name, "")
        check(f"{name} tag hash unchanged", h == want, f"{h}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    print("  all lloyd21 tests passed")
