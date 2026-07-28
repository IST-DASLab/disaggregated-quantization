"""Correctness gates for gsqlloyd3bit (GSQ optimization on the signed Lloyd format).

The central claim this file defends: at initialisation gsqlloyd3bit is *exactly*
lloyd3bit. If that holds, any difference in a training curve is attributable to the
optimizer (learned assignment vs STE) and not to a format discrepancy — which is
the whole point of pairing the two.

    python tests/test_gsq_lloyd.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from quantizers import REGISTRY, build_quantizer_params
from quantizers.blocked import GLOBAL_DEN, to_e4m3
from quantizers.grids import LLOYD_SIGNED_3BIT, grid_spacing
from quantizers.gsq import GSQLinear
from quantizers.gsq_lloyd import GSQLloydLinear, assignment_stats
from quantizers.lloyd import SignedLloydLinear


class TinyBlock(nn.Module):
    """Two fused groups with deliberately mismatched magnitudes, so a per-layer
    global scale and a group-shared one differ."""

    def __init__(self, d=64, h=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        with torch.no_grad():                       # q dominates its group, gate its own
            self.q_proj.weight.mul_(7.0)
            self.gate_proj.weight.mul_(5.0)


def _model():
    torch.manual_seed(0)
    m = nn.Module()
    m.layer = TinyBlock()
    return m


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def test_init_matches_lloyd3bit():
    """argmax assignment × E4M3 block scale × global == lloyd3bit's rounded weight."""
    torch.manual_seed(0)
    ref = _model()
    REGISTRY["lloyd3bit"]["apply"](ref)
    torch.manual_seed(0)
    got = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](got)

    worst, n_tot, n_bad = 0.0, 0, 0
    for (n, a), (_, b) in zip(ref.named_modules(), got.named_modules()):
        if not isinstance(a, SignedLloydLinear):
            continue
        worst = max(worst, (a._wq - b._wq).abs().max().item())
        n_tot += a._wq.numel()
        n_bad += int((a._wq != b._wq).sum())
    check("init _wq identical to lloyd3bit", worst == 0.0,
          f"max|Δ|={worst:.3e}  mismatched={n_bad}/{n_tot}")


def test_fused_group_scale_shared():
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m)
    lay = m.layer
    for group in (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")):
        gs = [getattr(lay, g)._global.item() for g in group]
        check(f"{group[0][:4]} group shares one global scale", len(set(gs)) == 1, str(gs))
    # and it is the group-max amax, not the member's own
    amax = max(getattr(lay, g).amax().item() for g in ("q_proj", "k_proj", "v_proj"))
    check("group global == group-max amax / 2688",
          abs(lay.k_proj._global.item() - amax / GLOBAL_DEN) < 1e-9)
    check("o_proj (unfused) uses its own amax",
          abs(lay.o_proj._global.item() - lay.o_proj.amax().item() / GLOBAL_DEN) < 1e-9)


def test_representable_in_format():
    """Every weight must be grid_level × E4M3_scale × global — the deployable form."""
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m)
    lin = m.layer.gate_proj
    with torch.no_grad():                       # perturb both learnables off init
        lin.quant_logits.normal_(0, 0.5)
        lin.scale_delta.normal_(0, 0.3)
        lin.post_update(0, 100)

    bs = lin.block_scale()
    check("block scale is on the E4M3 grid", torch.equal(bs, to_e4m3(bs)))
    check("scale_delta actually moved the block scales",
          not torch.equal(bs, to_e4m3(lin._block_init)))
    eff = lin._effective_scales()
    rebuilt = LLOYD_SIGNED_3BIT.to(eff)[lin.level_indices()] * eff[:, lin._idx]
    check("_wq == grid[argmax] * e4m3_block * global",
          torch.allclose(lin._wq, rebuilt, atol=0, rtol=0))
    n_used = len(torch.unique(lin.level_indices()))
    check("all 8 levels reachable", n_used == 8, f"{n_used}/8 used")


def test_gradients_flow():
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m)
    if not torch.cuda.is_available():
        print("  SKIP  gradient test (Gumbel fn needs CUDA RNG replay)")
        return
    m = m.cuda()
    lin = m.layer.down_proj
    lin.train()
    x = torch.randn(4, lin._in, device="cuda")
    lin(x).square().mean().backward()
    check("logits receive gradient",
          lin.quant_logits.grad is not None and lin.quant_logits.grad.abs().sum() > 0)
    check("scale_delta receives gradient (through the E4M3 STE)",
          lin.scale_delta.grad is not None and lin.scale_delta.grad.abs().sum() > 0)


def test_no_master_weight():
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m)
    names = {n for n, _ in m.named_parameters()}
    check("only logits + scale_delta are trainable",
          all(n.endswith(("quant_logits", "scale_delta")) for n in names),
          str(sorted({n.rsplit('.', 1)[1] for n in names})))
    check("master weight released after group linking",
          not any(hasattr(mod, "_w_master") for mod in m.modules()))


def test_param_groups_use_absolute_lrs():
    m = _model()
    params, h = build_quantizer_params("gsqlloyd3bit", "")
    REGISTRY["gsqlloyd3bit"]["apply"](m, **params)
    groups = REGISTRY["gsqlloyd3bit"]["param_groups"](m, lr=3e-6, lr_scale_ratio=0.5)
    check("logit group uses logit_lr, not the weight LR", groups[0]["lr"] == params["logit_lr"])
    check("scale group uses scale_lr", groups[1]["lr"] == params["scale_lr"])
    check("scale group has no weight decay", groups[1]["weight_decay"] == 0.0)
    check("no un-grouped trainable params left",
          len(groups[2]["params"]) == 0, f"{len(groups[2]['params'])} others")
    print(f"  hash for defaults: {h}")


def test_assignment_stats():
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m)
    s = assignment_stats(m)
    check("reassigned == 0 at init", s["quant/reassigned"] == 0.0)
    lin = m.layer.q_proj
    with torch.no_grad():
        lin.quant_logits.normal_(0, 1.0)
        lin.post_update(0, 100)
    s = assignment_stats(m)
    check("reassigned > 0 after perturbing logits", s["quant/reassigned"] > 0,
          f"{s['quant/reassigned']:.3f}")


def test_lion_survives_vanishing_gradients():
    """AdamW stalls on the tiny gradients the saturating Gumbel relaxation produces;
    Lion must not.

    Logit gradients carry a factor of the effective block scale (~1e-3) and land
    around 1e-11, far below AdamW's eps=1e-8: `exp_avg / (sqrt(exp_avg_sq) + eps)`
    collapses from ~1 to ~1e-3 and AdamW becomes vanishingly small plain SGD. The
    first sweep hit exactly this — over 600 steps the logits moved 5e-4 and a 10x
    change in logit_lr altered them by 4.5e-4 in total. Lion's sign-based update has
    magnitude exactly `lr` at any gradient scale, which is why the method uses it.
    """
    if not torch.cuda.is_available():
        print("  SKIP  needs CUDA (the step kernels are compiled CUDA)")
        return
    from training.dist_optim import _adamw_step, _lion_step

    LR, STEPS = 1e-4, 600
    ideal = LR * STEPS
    t = lambda v: torch.tensor(float(v), device="cpu")

    def drift(algo, grad_mag, eps=1e-8):
        # fp32, the shipped logit dtype — isolates the optimizer effect from the
        # separate low-precision saturation effect checked below.
        p = torch.zeros(1024, device="cuda", dtype=torch.float32)
        ea, ev = torch.zeros_like(p), torch.zeros_like(p)
        g = torch.full_like(p, grad_mag)
        for s in range(1, STEPS + 1):
            if algo == "lion":
                _lion_step(p, g, ea, t(LR), t(0.9), t(0.99), t(0.0))
            else:
                _adamw_step(p, g, ea, ev, t(s), t(LR), t(0.9), t(0.95), t(eps), t(0.0))
        return abs(p[0].item())

    bad = drift("adamw", 1e-11)
    check("AdamW at eps=1e-8 stalls on 1e-11 gradients", bad < 0.01 * ideal,
          f"{bad:.2e} = {bad/ideal:.4f}x ideal")
    for gm in (1e-11, 1e-7, 1e-3):
        d = drift("lion", gm)
        check(f"Lion moves a full lr*steps at grad={gm:.0e}", d > 0.99 * ideal,
              f"{d:.4f} vs ideal {ideal:.4f}")
    # the point of Lion: the update is scale-INVARIANT, so 8 orders of magnitude of
    # gradient scale must give exactly the same drift
    ds = [drift("lion", gm) for gm in (1e-11, 1e-3)]
    check("Lion drift is independent of gradient magnitude", ds[0] == ds[1], str(ds))

    # Why the logits are FP32: a low-precision master caps how far a logit can ever
    # travel, because once |p| reaches ~2^mantissa * lr its ULP exceeds the update and
    # further steps round away. That ceiling must clear the gap between the top two
    # levels (~0.014 median on real weights) or near-tied weights can never flip.
    def ceiling(dtype, lr=1e-4, steps=20000):
        p = torch.zeros(64, device="cuda", dtype=dtype)
        ea = torch.zeros_like(p)
        g = torch.full_like(p, 1e-3)          # large enough to be exact in every dtype
        for _ in range(steps):
            _lion_step(p, g, ea, t(lr), t(0.9), t(0.99), t(0.0))
        return abs(p[0].item())
    c16, c32 = ceiling(torch.bfloat16), ceiling(torch.float32)
    check("bfloat16 logits would saturate near 256*lr", c16 < 0.05,
          f"bf16 ceiling={c16:.4f}, vs a 0.014 median decision gap")
    check("fp32 logits do not saturate", c32 > 20 * c16, f"fp32={c32:.3f} bf16={c16:.4f}")

    params, _ = build_quantizer_params("gsqlloyd3bit", "")
    check("registry keeps logit masters in fp32", params["logits_dtype"] == "fp32")

    params, _ = build_quantizer_params("gsqlloyd3bit", "")
    check("registry selects lion", params["optim"] == "lion")
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m, **params)
    groups = REGISTRY["gsqlloyd3bit"]["param_groups"](m, lr=3e-6, lr_scale_ratio=0.5)
    check("logit group uses lion", groups[0]["algo"] == "lion")
    check("scale group uses lion", groups[1]["algo"] == "lion")
    check("weight group stays on adamw", "algo" not in groups[2])
    check("logits are not weight-decayed toward a uniform distribution",
          groups[0]["weight_decay"] == 0.0)


def test_schedule_preserves_per_group_lrs():
    """The LR schedule must SCALE each group's own LR, not overwrite it.

    qad.py used to assign the weight LR to every param group on every step, which
    silently discarded the per-group logit_lr/scale_lr. A three-arm logit_lr sweep
    then trained all three arms at args.lr and produced identical runs — the logits
    drifted by exactly args.lr * steps in every case.
    """
    from qad import lr_at

    params, _ = build_quantizer_params("gsqlloyd3bit", "")
    m = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](m, **params)
    groups = REGISTRY["gsqlloyd3bit"]["param_groups"](m, lr=3e-6, lr_scale_ratio=0.5)
    for g in groups:
        g.setdefault("lr", 3e-6)            # torch.optim fills group defaults; emulate it
        g.setdefault("initial_lr", g["lr"])

    TOTAL, WARMUP = 2485, 100
    for step, want_frac in ((0, 1 / WARMUP), (WARMUP - 1, 1.0), (700, 1.0)):
        frac = lr_at(step, TOTAL, 1.0, WARMUP, "constant")
        check(f"step {step}: schedule multiplier is {want_frac:g}",
              abs(frac - want_frac) < 1e-9, f"{frac:g}")
        lrs = [g["initial_lr"] * frac for g in groups]
        check(f"step {step}: logit LR stays {params['logit_lr']:g}-based, not 3e-06",
              abs(lrs[0] - params["logit_lr"] * frac) < 1e-18, f"{lrs[0]:.3e}")
        check(f"step {step}: scale LR stays {params['scale_lr']:g}-based",
              abs(lrs[1] - params["scale_lr"] * frac) < 1e-18, f"{lrs[1]:.3e}")
    # the whole point: a logit_lr sweep must actually produce different LRs
    other, _ = build_quantizer_params("gsqlloyd3bit", '{"logit_lr": 3e-4}')
    check("a logit_lr override changes the hash and the LR",
          other["logit_lr"] != params["logit_lr"])


def test_anneal_survives_resume():
    """The Gumbel schedule must depend only on the absolute step, not on how many
    steps this *process* has run — otherwise every restart would silently rewind
    the annealing and the run would never reach a hard assignment.

    `_temp` / `_scale_val` are registered buffers, so they ride along in
    state_dict() and through save_training_state()'s torch.save of it. This pins
    both halves: they survive the round-trip, and stepping on from a resumed state
    lands exactly where an uninterrupted run would.
    """
    import io
    TOTAL = 2485

    ref = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](ref)
    post_update = REGISTRY["gsqlloyd3bit"]["post_update"]

    # uninterrupted run up to step 700
    for s in range(701):
        post_update(ref, s, TOTAL)
    lin = ref.layer.q_proj
    t_700, sv_700 = lin._temp.item(), lin._scale_val.item()
    check("schedule advanced away from its start", t_700 < 2.0 and sv_700 > 100.0,
          f"T={t_700:.4f} scale_val={sv_700:.1f}")

    # save at 700, restore into a fresh model (what a restart actually does)
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in ref.state_dict().items()}, buf)
    buf.seek(0)
    resumed = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](resumed)
    resumed.load_state_dict(torch.load(buf, weights_only=False))
    r_lin = resumed.layer.q_proj
    check("temp/scale_val survive the state_dict round-trip",
          r_lin._temp.item() == t_700 and r_lin._scale_val.item() == sv_700)

    # step both on to 701 — resumed must equal uninterrupted
    for s in (701,):
        post_update(ref, s, TOTAL)
        post_update(resumed, s, TOTAL)
    check("resumed schedule matches uninterrupted at the next step",
          r_lin._temp.item() == lin._temp.item()
          and r_lin._scale_val.item() == lin._scale_val.item(),
          f"T={r_lin._temp.item():.5f} vs {lin._temp.item():.5f}")
    check("_wq identical after resuming and stepping",
          torch.equal(r_lin._wq, lin._wq))

    # and the schedule is a pure function of the absolute step
    fresh = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](fresh)
    post_update(fresh, 701, TOTAL)
    check("schedule depends only on the absolute step",
          fresh.layer.q_proj._temp.item() == lin._temp.item())

    end = _model()
    REGISTRY["gsqlloyd3bit"]["apply"](end)
    post_update(end, TOTAL - 1, TOTAL)
    check("schedule reaches its endpoint at the final step",
          abs(end.layer.q_proj._temp.item() - 0.05) < 1e-6
          and abs(end.layer.q_proj._scale_val.item() - 500.0) < 1e-3,
          f"T={end.layer.q_proj._temp.item():.4f} "
          f"scale_val={end.layer.q_proj._scale_val.item():.1f}")


def test_uniform_gsq_unchanged():
    """The refactor must not move gsq2bit/gsq3bit."""
    torch.manual_seed(0)
    w = torch.randn(32, 128)
    torch.manual_seed(1)
    lin = GSQLinear(w, None, bits=3, groupsize=128)
    check("gsq grid spacing is 1.0 (init behaviour preserved)",
          grid_spacing(lin._values) == 1.0)

    rtn = ((w / lin.scales[:, lin._idx]).round().clamp(-4, 3) + 4).long()
    # With the tiebreak disabled the machinery must reduce to exact round-to-nearest.
    exact = GSQLinear(w, None, bits=3, groupsize=128, noise=0.0)
    check("noise=0 gives exactly round-to-nearest",
          torch.equal(exact.level_indices(), rtn))
    # The gsq default noise=1.0 is COMPARABLE to the gap between the two nearest
    # levels (std=0.01 vs a ~0.015 gap), so ~1 in 5 weights starts off round-to-
    # nearest. That is pre-existing gsq behaviour, recorded here rather than fixed.
    frac = (lin.level_indices() == rtn).float().mean().item()
    check("gsq default init is a noisy round-to-nearest", 0.7 < frac < 0.9,
          f"agree={frac:.4f}")
    check("_wq reconstructs from levels × scales",
          torch.equal(lin._wq,
                      lin._values[lin.level_indices()] * lin.scales[:, lin._idx]))

    for name, want in (("gsq2bit", "6115cd98"), ("gsq3bit", "6115cd98"),
                       ("lloyd3bit", "3a7ebd60"), ("ste3bit", "1a17550c"),
                       ("nvfp4", "99914b93")):
        _, h = build_quantizer_params(name, "")
        check(f"{name} hash unchanged", h == want, f"{h}")


if __name__ == "__main__":
    for fn in (test_init_matches_lloyd3bit, test_fused_group_scale_shared,
               test_representable_in_format, test_gradients_flow,
               test_no_master_weight, test_param_groups_use_absolute_lrs,
               test_assignment_stats, test_lion_survives_vanishing_gradients,
               test_schedule_preserves_per_group_lrs, test_anneal_survives_resume,
               test_uniform_gsq_unchanged):
        print(f"\n{fn.__name__}:")
        fn()
    print("\nPASS: gsqlloyd3bit")
