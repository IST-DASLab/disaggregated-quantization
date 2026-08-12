"""Gates for the pseudo KV-compression connector.

The connector's whole job is to degrade a tensor exactly once, by a calibrated amount,
in the right place. Each of those three can fail silently, so each gets an assertion:

  AMOUNT   sigma must be 2^-bits * RMS, i.e. 4^-bits of POWER. Getting the square
           backwards (4^-bits of amplitude) is invisible in any output -- it just makes
           every run a different, unlabelled rate. Checked numerically against the
           rate-distortion identity, not against itself.
  ONCE     save_kv_layer receives the WHOLE paged buffer; only this step's slots may
           move. A version that noised everything would pass any "did the tensor change"
           test while accumulating a random walk over decode steps.
  INERT    at kv_bits >= DISABLED the buffer must be BIT-identical, so the baseline arm
           of the sweep is trustworthy.

    python tests/test_kv_noise.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from serving.kv_noise_connector import DISABLED, GROUP, group_noise, noise_slots_


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def test_distortion_matches_the_rate():
    """Measured noise POWER / signal POWER must equal 4^-bits."""
    torch.manual_seed(0)
    x = torch.randn(4096, 128, device="cuda")
    print(f"  {'bits':>5}{'measured D':>14}{'4^-bits':>12}{'ratio':>9}")
    for bits in (2, 3, 4, 5, 6):
        y = group_noise(x, bits)
        # per-group, since sigma is per-group: compare against each group's own power
        d = ((y - x).reshape(-1, GROUP).pow(2).mean(-1)
             / x.reshape(-1, GROUP).pow(2).mean(-1)).mean().item()
        want = 4.0 ** -bits
        print(f"  {bits:>5}{d:>14.3e}{want:>12.3e}{d / want:>9.3f}")
        check(f"bits={bits}: distortion within 5% of 4^-bits", abs(d / want - 1) < 0.05)


def test_amplitude_is_not_the_squared_variant():
    """Guard the specific off-by-a-square: 2^-bits amplitude, NOT 4^-bits."""
    torch.manual_seed(0)
    x = torch.randn(8192, 128, device="cuda")
    bits = 4
    amp = ((group_noise(x, bits) - x).pow(2).mean().sqrt()
           / x.pow(2).mean().sqrt()).item()
    check("amplitude ~ 2^-bits", abs(amp / 2.0 ** -bits - 1) < 0.05,
          f"measured {amp:.4f} vs 2^-4={2.0**-4:.4f} (4^-4={4.0**-4:.4f} would be wrong)")


def test_scale_is_per_group_not_global():
    """A group 1000x larger must get 1000x the noise -- otherwise it is a global scale."""
    torch.manual_seed(0)
    x = torch.randn(2, GROUP, device="cuda")
    x[1] *= 1000.0
    d = (group_noise(x, 4) - x).abs().mean(dim=-1)
    ratio = (d[1] / d[0]).item()
    check("noise tracks per-group RMS", 500 < ratio < 2000, f"ratio {ratio:.0f} (want ~1000)")


def test_group_size_invariance():
    """kv_bits is a relative error: RMS normalisation makes it group-size independent."""
    torch.manual_seed(0)
    x = torch.randn(4096, 128, device="cuda")
    ds = []
    for g in (8, 16, 32):
        y = group_noise(x, 4, group=g)
        ds.append(((y - x).pow(2).mean() / x.pow(2).mean()).item())
    spread = max(ds) / min(ds)
    check("distortion independent of group size", spread < 1.10,
          f"g=8/16/32 -> {['%.3e' % d for d in ds]}, spread {spread:.3f}")


def test_disabled_is_bit_identical():
    torch.manual_seed(0)
    x = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    y = group_noise(x.clone(), DISABLED)
    check("kv_bits=DISABLED leaves the tensor untouched", torch.equal(x, y),
          "the baseline arm must be exactly the no-connector run")


def _moved_slots(kv, ref, page_size):
    d = (kv != ref).any(dim=-1).any(dim=-1)            # (2, pages, page_size)
    idx = d.any(dim=0).nonzero()
    return {int(b) * page_size + int(o) for b, o in idx}


def test_only_the_mapped_slots_move():
    """The paged-buffer contract: everything outside slot_mapping must be untouched."""
    torch.manual_seed(0)
    pages, page_size, heads, dh = 8, 16, 4, 64
    kv = torch.randn(2, pages, page_size, heads, dh, device="cuda", dtype=torch.bfloat16)
    ref = kv.clone()
    slots = torch.tensor([0, 5, 17, 100], device="cuda")

    noise_slots_(kv, slots, 3)

    moved = _moved_slots(kv, ref, page_size)
    check("exactly the mapped slots changed", moved == set(slots.tolist()),
          f"changed {sorted(moved)} vs mapped {sorted(slots.tolist())}")


def test_works_on_a_NON_contiguous_buffer():
    """REGRESSION: the real paged buffer is not contiguous across (2, pages, page_size).

    The first version flattened with .view() and died in production with "view size is
    not compatible with input tensor's size and stride" -- while the unit test passed,
    because the test allocated a fresh contiguous tensor. So the test now uses a buffer
    that is deliberately NOT contiguous.

    It also asserts the ORIGINAL tensor changed. That is the half that matters: swapping
    view for reshape makes the crash disappear by silently mutating a copy, which would
    have produced a full sweep of clean-looking numbers with no noise injected.
    """
    torch.manual_seed(0)
    pages, page_size, heads, dh = 8, 16, 4, 64
    backing = torch.randn(2, pages, page_size + 3, heads, dh, device="cuda",
                          dtype=torch.bfloat16)
    kv = backing[:, :, :page_size]                     # non-contiguous view
    check("the test buffer really is non-contiguous", not kv.is_contiguous())
    try:
        kv.view(2, pages * page_size, -1)
        check("(.view would have worked -- test no longer reproduces the bug)", False)
    except RuntimeError:
        check("plain .view() still raises on this layout", True, "the original crash")

    ref = kv.clone()
    n = noise_slots_(kv, torch.tensor([1, 40, 90], device="cuda"), 3)
    check("noise_slots_ succeeds on a non-contiguous buffer", n > 0, f"{n} elements")
    check("the ORIGINAL buffer changed, not a copy", not torch.equal(kv, ref),
          "a reshape-based fix would silently mutate a temporary and fail here")
    check("the change is visible through the backing tensor",
          not torch.equal(backing[:, :, :page_size], ref))
    check("exactly the mapped slots changed",
          _moved_slots(kv, ref, page_size) == {1, 40, 90})


def test_negative_slot_padding_is_ignored():
    """slot_mapping may carry -1; a negative index would wrap and corrupt another slot."""
    torch.manual_seed(0)
    pages, page_size, heads, dh = 4, 16, 2, 32
    kv = torch.randn(2, pages, page_size, heads, dh, device="cuda", dtype=torch.bfloat16)
    ref = kv.clone()
    noise_slots_(kv, torch.tensor([-1, -1, 3], device="cuda"), 3)
    check("only the real slot moved; padding touched nothing",
          _moved_slots(kv, ref, page_size) == {3})


def test_repeated_application_would_be_detectable():
    """Prove the random-walk failure is real, so the slot restriction is load-bearing.

    Not a test of our code -- a test that the bug we designed around actually bites:
    noising the same tensor N times grows the error like sqrt(N).
    """
    torch.manual_seed(0)
    x = torch.randn(2048, 128, device="cuda")
    y = x.clone()
    for _ in range(16):
        y = group_noise(y, 4)
    once = ((group_noise(x, 4) - x).pow(2).mean() / x.pow(2).mean()).sqrt().item()
    many = ((y - x).pow(2).mean() / x.pow(2).mean()).sqrt().item()
    check("16x application inflates error ~4x (sqrt 16)", 3.0 < many / once < 5.0,
          f"once {once:.4f} -> 16x {many:.4f}, ratio {many/once:.2f}")


def test_bf16_floor_locates_the_max_bits_cap():
    """Where the rate stops being measurable at all.

    The floor is bf16's RMS ROUNDING error (2^-9/sqrt(3) ~ 0.113%), not its machine
    epsilon (2^-8) -- a sqrt(3)/2 factor. Getting that wrong understates the usable
    range by ~2 bits, which is why it is measured here rather than asserted.

    Practically moot: real KV quantization is already lossless well below this, so the
    experiment lives at 2-4 bits. This exists to stop anyone reading a null result at
    high kv_bits as evidence about the model.
    """
    torch.manual_seed(0)
    x = torch.randn(8192, 128, device="cuda")
    store = ((x.to(torch.bfloat16).float() - x).pow(2).mean()
             / x.pow(2).mean()).sqrt().item()
    n = {b: ((group_noise(x, b) - x).pow(2).mean() / x.pow(2).mean()).sqrt().item()
         for b in (4, 6, 8, 10)}
    print(f"  bf16 RMS rounding floor {store:.5f}")
    for b, v in n.items():
        print(f"    noise@{b:>2}bit {v:.5f}  = {v / store:5.2f}x floor")
    check("the experiment's range (4 bit) is far above the floor", n[4] > store * 10)
    check("8 bit is still measurable", n[8] > store * 2)
    check("10 bit is AT the floor -- nothing to measure", n[10] <= store * 1.5)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    print("  all kv-noise tests passed")
