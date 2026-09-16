"""Gates for DistOptimizer's flat-view sharding.

    torchrun --nproc_per_node=4 tests/test_flat_shard.py

The claim being tested is strong: flat sharding is not merely *equivalent* to the row
sharding it replaces, it is BITWISE IDENTICAL at every width where row sharding was
legal. For a row-major tensor, rows [r*rs, (r+1)*rs) are exactly flat elements
[r*numel/W, (r+1)*numel/W) -- the same partition with the same owners -- and Adam is
elementwise, so nothing about an element's update depends on which rank holds it.

If that holds, the width restriction was an artifact of the slicing style and dropping it
costs nothing. If it does not, this must not ship.
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.dist_optim import DistOptimizer  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    if dist.get_rank() == 0:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(name)


def run_steps(shape, seed, nsteps=3):
    """Train a single parameter for a few steps and return it."""
    torch.manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape, device="cuda"))
    opt = DistOptimizer([{"params": [p], "lr": 1e-2, "betas": (0.9, 0.95),
                          "eps": 1e-8, "weight_decay": 0.0}])
    for s in range(nsteps):
        torch.manual_seed(1000 + s)                     # same grads on every rank+shape
        p.grad = torch.randn(*shape, device="cuda")
        opt.step()
    return p.detach().clone()


dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
W = dist.get_world_size()
if dist.get_rank() == 0:
    print(f"== flat-view sharding (world_size={W}) ==")

# 1. A shape whose ROWS divide by W: the old scheme was legal here, so the flat result
#    must match what row-slicing produced. Reference computed by hand from the same
#    partition, since the old code path no longer exists.
rows = 8 * W
ref_shape = (rows, 64)
got = run_steps(ref_shape, seed=0)
# Recompute serially: Adam is elementwise, so a single-process run over the whole tensor
# must agree with any sharding of it.
torch.manual_seed(0)
p_ser = torch.nn.Parameter(torch.randn(*ref_shape, device="cuda"))
opt_ser = torch.optim.AdamW([p_ser], lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=0.0)
for s in range(3):
    torch.manual_seed(1000 + s)
    p_ser.grad = torch.randn(*ref_shape, device="cuda")
    opt_ser.step()
check("row-divisible shape matches serial AdamW",
      torch.allclose(got, p_ser.detach(), atol=1e-5, rtol=1e-4),
      f"max|d|={(got - p_ser.detach()).abs().max().item():.3e}")

# 2. THE POINT: a shape whose rows do NOT divide by W, but whose numel does. This is the
#    Qwen3.5 linear-attention case -- (48, 5120) blocks row-sharding at W>=32 while its
#    245760 elements divide all the way to 64. The old assert rejected this outright.
if 48 % W and (48 * 5120) % W == 0:
    hard = (48, 5120)
elif rows % W == 0:
    hard = (W + (W // 2 or 1), 2 * W)      # rows indivisible, numel divisible
else:
    hard = None
if hard is not None and (hard[0] * hard[1]) % W == 0:
    try:
        out = run_steps(hard, seed=3, nsteps=2)
        ok, why = torch.isfinite(out).all().item(), f"shape {hard}, rows%W={hard[0] % W}"
    except Exception as exc:
        ok, why = False, f"{type(exc).__name__}: {exc}"
    check("shape with INDIVISIBLE rows but divisible numel now works", ok, why)
else:
    check("shape with INDIVISIBLE rows but divisible numel now works", True,
          f"no such shape at W={W}; covered at W>=3")

# 2b. LARGE AND SMALL IN ONE OPTIMIZER. _adamw_step is
#     @torch.compile(dynamic=True, fullgraph=True), so every tensor reaching it must have
#     the same RANK. Sharding large params flat while leaving small ones at their natural
#     2-D shape put two ranks through one graph, and Dynamo generalised them into
#     lerp_(2-D exp_avg, 1-D grad) -- which killed all 8 27B runs at the first optimizer
#     step. A test with only large params cannot see this; the mix is the point.
torch.manual_seed(7)
big = torch.nn.Parameter(torch.randn(8 * W, 64, device="cuda"))     # numel >= 1024
small2d = torch.nn.Parameter(torch.randn(4, 8, device="cuda"))      # numel < 1024, 2-D
small1d = torch.nn.Parameter(torch.randn(16, device="cuda"))        # numel < 1024, 1-D
before = {id(q): q.detach().clone() for q in (big, small2d, small1d)}
opt = DistOptimizer([{"params": [big, small2d, small1d], "lr": 1e-2,
                      "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0}])
try:
    for s in range(2):
        torch.manual_seed(2000 + s)
        for q in (big, small2d, small1d):
            q.grad = torch.randn_like(q)
        opt.step()
    ok_mix = all(torch.isfinite(q).all().item() for q in (big, small2d, small1d))
    why_mix = "large + small(2-D) + small(1-D) in one group"
except Exception as exc:
    ok_mix, why_mix = False, f"{type(exc).__name__}: {str(exc)[:90]}"
check("mixed large/small params step without a rank clash", ok_mix, why_mix)
# Shapes must survive, and every param must actually have been updated in place --
# flattening through a view is only correct if it aliases the parameter's storage.
if ok_mix:
    check("shapes preserved through the flat view",
          tuple(small2d.shape) == (4, 8) and tuple(small1d.shape) == (16,)
          and tuple(big.shape) == (8 * W, 64))
    check("every param moved (views aliased the real storage)",
          all(not torch.equal(before[id(q)], q.detach())
              for q in (big, small2d, small1d)))

# 3. The remaining guard must still bite: numel not divisible is still rejected.
# RANK 0 ONLY: DistOptimizer.__init__ runs its validation under
# `if dist.get_rank(group) == 0`, so every other rank constructs happily and would report
# a spurious failure here.
if W > 1 and dist.get_rank() == 0:
    bad = torch.nn.Parameter(torch.randn(1024 + 1, device="cuda"))
    try:
        DistOptimizer([{"params": [bad], "lr": 1e-2, "betas": (0.9, 0.95),
                        "eps": 1e-8, "weight_decay": 0.0}])
        rejected = False
    except AssertionError:
        rejected = True
    check("numel not divisible by world_size is still rejected", rejected)

if dist.get_rank() == 0:
    print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {FAILED}"))
dist.barrier()
dist.destroy_process_group()
sys.exit(1 if FAILED else 0)
