# Distributed AdamW optimizer (ZeRO-2 style) from karpathy/nanochat
# github.com/karpathy/nanochat/blob/64a651a/nanochat/adamw.py

import torch
import torch.distributed as dist
from torch import Tensor


@torch.compile(dynamic=False, fullgraph=True)
def _adamw_step(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    b1_t: Tensor,
    b2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - b1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - b2_t)
    bias1 = 1 - b1_t**step_t
    bias2 = 1 - b2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1))


class DistAdamW(torch.optim.Optimizer):
    """ZeRO-2 style: sharded optimizer states + gradient reduction via reduce_scatter."""

    def __init__(
        self,
        param_groups,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        world_size = dist.get_world_size()
        if dist.get_rank() == 0:
            for group in param_groups:
                for p in group["params"]:
                    if p.numel() >= 1024:
                        assert p.shape[0] % world_size == 0, (
                            f"Large param shape {p.shape}: shape[0] must be divisible "
                            f"by world_size={world_size}"
                        )
        super().__init__(param_groups, defaults)
        self._step_t = torch.tensor(0.0, device="cpu")
        self._lr_t = torch.tensor(0.0, device="cpu")
        self._b1_t = torch.tensor(0.0, device="cpu")
        self._b2_t = torch.tensor(0.0, device="cpu")
        self._eps_t = torch.tensor(0.0, device="cpu")
        self._wd_t = torch.tensor(0.0, device="cpu")

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_futs: list = []
        grad_slices: list[Tensor] = []
        is_small: list[bool] = []

        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if p.numel() < 1024:
                    is_small.append(True)
                    reduce_futs.append(
                        dist.all_reduce(g, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    )
                    grad_slices.append(g)
                else:
                    is_small.append(False)
                    rsize = g.shape[0] // world_size
                    g_slice = torch.empty_like(g[:rsize])
                    reduce_futs.append(
                        dist.reduce_scatter_tensor(
                            g_slice, g, op=dist.ReduceOp.AVG, async_op=True
                        ).get_future()
                    )
                    grad_slices.append(g_slice)

        gather_futs: list = []
        idx = 0
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                reduce_futs[idx].wait()
                g_slice = grad_slices[idx]
                small = is_small[idx]
                rsize = p.shape[0] // world_size
                p_slice = p if small else p[rank * rsize : (rank + 1) * rsize]
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_slice)
                    state["exp_avg_sq"] = torch.zeros_like(p_slice)
                state["step"] += 1
                self._step_t.fill_(state["step"])
                self._lr_t.fill_(group["lr"])
                self._b1_t.fill_(b1)
                self._b2_t.fill_(b2)
                self._eps_t.fill_(group["eps"])
                self._wd_t.fill_(group["weight_decay"])
                _adamw_step(
                    p_slice,
                    g_slice,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    self._step_t,
                    self._lr_t,
                    self._b1_t,
                    self._b2_t,
                    self._eps_t,
                    self._wd_t,
                )
                if not small:
                    gather_futs.append(
                        dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                    )
                idx += 1

        if gather_futs:
            torch.futures.collect_all(gather_futs).wait()
