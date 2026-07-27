"""Verify restart correctness for the ZeRO-2 sharded optimizer.

The risk this guards: DistAdamW keeps only `p[rank*rsize:(rank+1)*rsize]` of each
moment for large params, so naively saving rank 0's optimizer.state_dict() would
silently drop 7/8 of the state and a "resumed" run would quietly restart Adam.

Run with the same world size the real runs use:
    torchrun --nproc_per_node=8 tests/test_checkpoint.py
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn as nn

from training.checkpoint import (find_latest, gather_optimizer_state,
                                 load_optimizer_state, load_training_state,
                                 save_training_state)
from training.dist_adamw import DistAdamW


class Tiny(nn.Module):
    """Mixes params on both sides of DistAdamW's 1024-element sharding threshold."""

    def __init__(self, ws):
        super().__init__()
        self.big = nn.Linear(256, 8 * ws, bias=False)     # sharded (numel >= 1024)
        self.big2 = nn.Linear(512, 16 * ws, bias=False)   # sharded
        self.small = nn.Parameter(torch.randn(16))        # replicated (numel < 1024)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")
    torch.manual_seed(0)                                  # identical init on all ranks

    model = Tiny(ws).to(device)
    opt = DistAdamW([{"params": list(model.parameters())}], lr=1e-3, weight_decay=0.1)

    # a few steps so the moments are non-trivial
    for i in range(3):
        for p in model.parameters():
            p.grad = torch.randn_like(p) * (0.1 * (i + 1) + 0.01 * rank)
        opt.step()

    ref_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ref_state = {id(p): {k: v.clone() for k, v in opt.state[p].items() if torch.is_tensor(v)}
                 for p in model.parameters()}
    ref_steps = {id(p): opt.state[p]["step"] for p in model.parameters()}

    ckpt_dir = "/tmp/qad_ckpt_test"
    args = argparse.Namespace(ckpt_dir=ckpt_dir, ckpt_tag="unit", lr=1e-3)
    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    dist.barrier()

    save_training_state(model, opt, step=3, args=args, keep_last=2)
    dist.barrier()

    # corrupt everything, then restore
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for p in model.parameters():
            for k in ("exp_avg", "exp_avg_sq"):
                opt.state[p][k].fill_(123.0)
            opt.state[p]["step"] = 0

    latest = find_latest(ckpt_dir, "unit")
    assert latest is not None, "no training state found"
    next_step = load_training_state(latest[0], model, opt, device)
    assert next_step == 4, next_step

    # weights must match bit-for-bit
    for k, v in model.state_dict().items():
        assert torch.equal(v, ref_w[k]), f"weight {k} mismatch after resume"

    # each rank must get ITS OWN shard back, exactly
    for p in model.parameters():
        for k in ("exp_avg", "exp_avg_sq"):
            got, want = opt.state[p][k], ref_state[id(p)][k]
            assert got.shape == want.shape, f"{k}: shape {got.shape} != {want.shape}"
            assert torch.equal(got, want), (
                f"{k} mismatch on rank {rank} (max|Δ|={(got-want).abs().max().item():.3e})")
        assert opt.state[p]["step"] == ref_steps[id(p)]

    # a shard-aware sanity check: with different grads per rank the moments must
    # actually differ across ranks, i.e. we really are comparing distinct shards.
    probe = opt.state[model.big.weight]["exp_avg"].sum()
    gathered = [torch.zeros_like(probe) for _ in range(ws)]
    dist.all_gather(gathered, probe)
    if rank == 0:
        vals = [g.item() for g in gathered]
        assert len(set(round(v, 6) for v in vals)) > 1, (
            f"all ranks hold identical shards ({vals}) — the test is not exercising sharding")
        print(f"  shards genuinely differ across ranks: {[round(v, 4) for v in vals]}")

    # continuing to train from the restored state must not blow up
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters()), "non-finite after resume step"

    if rank == 0:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"  weights + per-rank optimizer shards restored exactly (world_size={ws})")
        print("PASS: restart is exact")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
