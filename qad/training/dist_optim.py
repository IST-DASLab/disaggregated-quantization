"""ZeRO-2 style distributed optimizer with a per-group update rule (AdamW or Lion).

The sharding shell is from karpathy/nanochat
(github.com/karpathy/nanochat/blob/64a651a/nanochat/adamw.py): parameters with
numel >= 1024 have their gradients reduce_scattered, each rank updates only the
shard `p[rank*rsize:(rank+1)*rsize]`, and the result is all_gathered back. Smaller
parameters are all_reduced and updated redundantly on every rank.

What is added here is a per-group `algo`, because quantization parameters and
weights want genuinely different update rules.

Why Lion for the GSQ parameters
-------------------------------
The Gumbel-Softmax relaxation drifts into a saturated regime as the temperature
anneals and the logit gaps grow, and the logit gradients also carry a factor of
the effective block scale (~1e-3). Together these put the gradients around 1e-11,
which is far below AdamW's eps=1e-8: `exp_avg / (sqrt(exp_avg_sq) + eps)` collapses
from ~1 to ~1e-3 and AdamW degenerates into vanishingly small plain SGD. Measured
on a real run, the logits moved 5e-4 over 600 steps and a 10x change in the learning
rate altered them by 4.5e-4 in total — the level assignment was frozen and training
was a no-op.

Lion sidesteps this entirely: the update is `sign(lerp(momentum, grad, 1-beta1))`,
whose magnitude is exactly `lr` no matter how small the gradient is, so it cannot
stall on vanishing gradients. (The GSQ paper reaches the same conclusion; the
alternative in the literature is hand-tuning AdamW's eps per problem.) A useful
side effect: Lion keeps one momentum buffer instead of two, halving optimizer
memory — which matters when the logits are already 8x the weight tensor.

Lion: Chen et al., "Symbolic Discovery of Optimization Algorithms" (arXiv 2302.06675).
"""

import torch
import torch.distributed as dist
from torch import Tensor


@torch.compile(dynamic=True, fullgraph=True)
def _adamw_step(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor,
    step_t: Tensor, lr_t: Tensor, b1_t: Tensor, b2_t: Tensor,
    eps_t: Tensor, wd_t: Tensor,
) -> None:
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - b1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - b2_t)
    bias1 = 1 - b1_t**step_t
    bias2 = 1 - b2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1))


@torch.compile(dynamic=True, fullgraph=True)
def _lion_step(
    p: Tensor, grad: Tensor, exp_avg: Tensor,
    lr_t: Tensor, b1_t: Tensor, b2_t: Tensor, wd_t: Tensor,
) -> None:
    """Lion: decoupled weight decay, then a step of size exactly `lr` along the sign
    of the beta1-interpolated momentum. Momentum updates with beta2 *after* the step.
    """
    p.mul_(1 - lr_t * wd_t)
    p.sub_(exp_avg.lerp(grad, 1 - b1_t).sign_() * lr_t)
    exp_avg.lerp_(grad, 1 - b2_t)


#: state tensors each algorithm keeps per parameter
STATE_KEYS = {"adamw": ("exp_avg", "exp_avg_sq"), "lion": ("exp_avg",)}


class DistOptimizer(torch.optim.Optimizer):
    """ZeRO-2 style: sharded optimizer states + gradient reduction via reduce_scatter.

    Set `algo` on a param group to pick the update rule ("adamw" default, or "lion").
    Groups may also override lr / betas / eps / weight_decay as usual.
    """

    def __init__(
        self,
        param_groups,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        algo: str = "adamw",
        process_group=None,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, algo=algo)
        # `process_group` scopes EVERY collective below. Default (None) is the whole
        # world, which is the data-parallel group when there is no pipeline. Under
        # pipeline parallelism the stages hold DIFFERENT parameters, so reducing over
        # the world would mix two disjoint models: the caller passes the DP group (the
        # ranks holding the SAME half) and the shard arithmetic follows it.
        self._pg = process_group
        world_size = dist.get_world_size(group=process_group)
        if dist.get_rank(group=process_group) == 0:
            for group in param_groups:
                if group.get("algo", algo) not in STATE_KEYS:
                    raise ValueError(f"unknown algo {group.get('algo')!r}")
                for p in group["params"]:
                    if p.numel() >= 1024:
                        # FLAT sharding: the requirement is on numel, not shape[0].
                        #
                        # Sharding by rows needed shape[0] % world_size == 0 purely so
                        # p[r*rs:(r+1)*rs] was a clean contiguous view. That is a property
                        # of the slicing style, not of the maths: Adam is elementwise, so
                        # which rank owns an element changes nothing about its update.
                        #
                        # And the two are the SAME PARTITION wherever both are legal --
                        # for a row-major tensor, rows [r*rs, (r+1)*rs) are exactly flat
                        # elements [r*numel/W, (r+1)*numel/W) -- so this is bitwise
                        # identical at every width that used to work, not merely
                        # equivalent.
                        #
                        # It matters because Qwen3.5's linear attention has 96 parameters
                        # of shape (48, 5120): 48 rows block DP >= 32, while 245760
                        # elements divide cleanly all the way to 64. Measured across the
                        # 27B text stack: at DP=32 row-sharding fails 96 params and flat
                        # sharding fails none.
                        assert p.is_contiguous(), (
                            f"Large param {tuple(p.shape)} is not contiguous; view(-1) "
                            f"below would not alias its storage")
                        assert p.numel() % world_size == 0, (
                            f"Large param shape {p.shape}: numel {p.numel()} must be "
                            f"divisible by world_size={world_size}"
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
        rank = dist.get_rank(group=self._pg)
        world_size = dist.get_world_size(group=self._pg)
        reduce_futs: list = []
        grad_slices: list[Tensor] = []
        is_small: list[bool] = []

        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if p.numel() < 1024:
                    is_small.append(True)
                    reduce_futs.append(
                        dist.all_reduce(g, op=dist.ReduceOp.AVG, group=self._pg,
                                        async_op=True).get_future()
                    )
                    # Flattened to match the parameter slice below: every tensor reaching
                    # the compiled _adamw_step must have the SAME rank. all_reduce ran on
                    # g itself, so this view is the same storage.
                    grad_slices.append(g.view(-1))
                else:
                    is_small.append(False)
                    # Flat views throughout -- see the assert in __init__. reduce_scatter
                    # wants input numel == world_size * output numel, which a 1-D view
                    # satisfies for any shape whose numel divides.
                    g_flat = g.view(-1)
                    rsize = g_flat.numel() // world_size
                    g_slice = torch.empty_like(g_flat[:rsize])
                    reduce_futs.append(
                        dist.reduce_scatter_tensor(
                            g_slice, g_flat, op=dist.ReduceOp.AVG, group=self._pg,
                            async_op=True
                        ).get_future()
                    )
                    grad_slices.append(g_slice)

        gather_futs: list = []
        idx = 0
        for group in self.param_groups:
            b1, b2 = group["betas"]
            algo = group.get("algo", "adamw")
            for p in group["params"]:
                reduce_futs[idx].wait()
                g_slice = grad_slices[idx]
                small = is_small[idx]
                # FLAT FOR EVERY PARAM, small ones included. p_flat aliases p's storage,
                # so updating the slice updates p in place -- exactly as the row view did.
                #
                # The small path must flatten too, even though it is not sharded.
                # _adamw_step is @torch.compile(dynamic=True, fullgraph=True): passing
                # 1-D slices for large params and 2-D tensors for small ones puts two
                # RANKS through one compiled graph, and Dynamo generalises them into a
                # graph where exp_avg is 2-D while grad is 1-D:
                #
                #   lerp_(FakeTensor(size=(s33, s48)), FakeTensor(size=(s87,)), ...)
                #   RuntimeError: Attempting to broadcast a dimension of length s87 at -1
                #
                # It killed all 8 27B runs ~8 min in, at the first optimizer step. Before
                # flat sharding every call was 2-D, so the mixed-rank case could not
                # arise. One rank for every call keeps it that way.
                p_flat = p.view(-1)
                rsize = p.numel() // world_size
                p_slice = p_flat if small else p_flat[rank * rsize : (rank + 1) * rsize]
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    for key in STATE_KEYS[algo]:
                        state[key] = torch.zeros_like(p_slice)
                state["step"] += 1
                self._step_t.fill_(state["step"])
                self._lr_t.fill_(group["lr"])
                self._b1_t.fill_(b1)
                self._b2_t.fill_(b2)
                self._eps_t.fill_(group["eps"])
                self._wd_t.fill_(group["weight_decay"])
                if algo == "lion":
                    _lion_step(p_slice, g_slice, state["exp_avg"],
                               self._lr_t, self._b1_t, self._b2_t, self._wd_t)
                else:
                    _adamw_step(p_slice, g_slice, state["exp_avg"], state["exp_avg_sq"],
                                self._step_t, self._lr_t, self._b1_t, self._b2_t,
                                self._eps_t, self._wd_t)
                if not small:
                    gather_futs.append(
                        dist.all_gather_into_tensor(p_flat, p_slice, group=self._pg,
                                                    async_op=True).get_future()
                    )
                idx += 1

        if gather_futs:
            torch.futures.collect_all(gather_futs).wait()


# Backwards-compatible name: the default algo is AdamW.
DistAdamW = DistOptimizer
