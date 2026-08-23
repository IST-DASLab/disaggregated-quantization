"""PP=2 equivalence tests. Distributed: launched under torchrun by bin/run_tests.sh.

Pipeline parallelism fails the same way FSDP does -- quietly, with slightly different
numbers -- so nothing here asserts "it ran". Every claim is against a SINGLE-PROCESS
reference that holds the whole model, because the split is only worth having if the
numbers are the ones the unsplit setup would have produced.

METHOD. Every rank builds the same model from the same seed and both stages are fed the
SAME batch, which is exactly the invariant the real loop maintains by sharding data on
dp_rank rather than global rank. The reference is therefore EXACT, not approximate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn as nn

from pathlib import Path

from training import pipeline as pp_mod

FAILED = []


def check(name, cond, detail=""):
    """Record a result. Printing is rank 0's job; FAILING is every rank's.

    The two must not be confused: an earlier version printed only on rank 0 and let each
    rank keep its own FAILED list, so rank 0 announced "ALL PASS" while rank 1 exited
    non-zero -- a green report next to a red runner. main() now reduces the count across
    ranks before deciding.
    """
    if dist.get_rank() == 0:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    elif not cond:
        print(f"  FAIL[rank{dist.get_rank()}]  {name}"
              f"{'  ' + detail if detail else ''}", flush=True)
    if not cond:
        FAILED.append(name)


class TinyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 2 * d, bias=False)
        self.fc2 = nn.Linear(2 * d, d, bias=False)

    def forward(self, h):
        return h + self.fc2(torch.relu(self.fc1(h)))


class Out:
    def __init__(self, h):
        self.last_hidden_state = h


class TinyBase(nn.Module):
    """Mimics a HF decoder stack closely enough for the split: embed_tokens, `layers`,
    a final `norm`, and an `inputs_embeds=` entry point -- which is how stage 1 is fed."""

    def __init__(self, d=32, n=4, vocab=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyLayer(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)

        class _Cfg:
            hidden_size = d
        self.config = _Cfg()

    def forward(self, input_ids=None, inputs_embeds=None):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            h = layer(h)
        return Out(self.norm(h))


class TinyModel(nn.Module):
    def __init__(self, d=32, n=4, vocab=64):
        super().__init__()
        self.base = TinyBase(d, n, vocab)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.base.embed_tokens.weight     # TIED, as Gemma-3 is


class _Text:
    """Stand-in for models.text_stack()'s result."""

    def __init__(self, m):
        self.base, self.head = m.base, m.head


def build(seed=0):
    torch.manual_seed(seed)
    return TinyModel().cuda()


def batch(seed=7):
    g = torch.Generator().manual_seed(seed)          # identical on every rank
    return torch.randint(0, 64, (2, 16), generator=g).cuda()


def reference_loss(m, ids):
    """Whole-model forward + loss, single process, no pipeline anywhere."""
    h = m.base(input_ids=ids).last_hidden_state
    return m.head(h).float().pow(2).mean()


def test_rank_layout_is_intra_node():
    """Pairs must be ADJACENT ranks, or pipeline traffic crosses the node boundary."""
    g = pp_mod.build_groups()
    check("pp pairs are adjacent ranks (2k, 2k+1)",
          g.peer == (dist.get_rank() ^ 1),
          f"rank {dist.get_rank()} <-> peer {g.peer}")
    # The block layout `rank // (world//2)` would pair r with r + world/2. On 8 ranks
    # that is rank 0 with rank 4 -- a different node once the job spans two. Asserting
    # the adjacent form is what keeps the requirement from silently regressing.
    world = dist.get_world_size()
    block_peer = (dist.get_rank() + world // 2) % world
    check("layout is NOT the cross-node block layout",
          world == 2 or g.peer != block_peer,
          f"adjacent peer {g.peer} vs block peer {block_peer}")
    check("dp_size halves the world", g.dp_size == world // 2, f"{g.dp_size}")


def test_split_halves_the_parameters():
    g = pp_mod.build_groups()
    m = build()
    whole = sum(p.numel() for p in m.parameters())
    n_layers = len(m.base.layers)
    pp_mod.split_stack(m.base, m.head, g.pp_rank)
    kept = len(m.base.layers)
    check("each stage keeps half the layers", kept == n_layers // 2,
          f"{kept} of {n_layers}")
    check("stage 0 drops the final norm, stage 1 keeps it",
          isinstance(m.base.norm, nn.Identity) == g.is_first)
    # The embedding is tied and therefore replicated, so the drop is layers-only.
    after = sum(p.numel() for p in m.parameters())
    check("parameter count actually falls", after < whole, f"{after} < {whole}")


def test_forward_matches_single_process():
    """The split forward must reproduce the unsplit hidden states exactly."""
    g = pp_mod.build_groups()
    ids = batch()

    ref = build()
    with torch.no_grad():
        want = ref.base(input_ids=ids).last_hidden_state

    m = build()
    pp_mod.split_stack(m.base, m.head, g.pp_rank)
    with torch.no_grad():
        if g.is_first:
            h = pp_mod.stage_forward(m.base, 0, input_ids=ids)
            pp_mod.send_activations(h, h, g)
            got = None
        else:
            B, T = ids.shape
            h_in, _ = pp_mod.recv_activations((B, T, 32), "cuda", g,
                                             teacher_dtype=torch.float32)
            got = pp_mod.stage_forward(m.base, 1, hidden=h_in)

    ok = True
    detail = ""
    if g.is_last:
        d = (got.float() - want.float()).abs().max().item()
        ok, detail = d < 1e-4, f"max|Δ|={d:.3e}"
    check("split forward == single-process forward", ok, detail)


def test_boundary_gradient_reproduces_single_process():
    """Loss and the gradient of EVERY parameter must match the unsplit model.

    This is the assertion the whole design rests on: the send/recv pair has to be a
    transparent substitute for autograd crossing the layer boundary in one process.
    """
    g = pp_mod.build_groups()
    ids = batch()

    ref = build()
    reference_loss(ref, ids).backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}

    m = build()
    replicas = pp_mod.replicated_params_for_test(m, g)
    pp_mod.split_stack(m.base, m.head, g.pp_rank)

    if g.is_first:
        h = pp_mod.stage_forward(m.base, 0, input_ids=ids)
        pp_mod.send_activations(h, h.detach(), g)
        pp_mod.backward_from_peer(h, g)
        loss_val = 0.0
    else:
        B, T = ids.shape
        h_in, _ = pp_mod.recv_activations((B, T, 32), "cuda", g,
                                         teacher_dtype=torch.float32)
        out = pp_mod.stage_forward(m.base, 1, hidden=h_in)
        loss = m.head(out).float().pow(2).mean()
        loss.backward()
        pp_mod.send_input_grad(h_in, g)
        loss_val = loss.item()

    # The tied weight is split across stages; sum its two contributions as the real loop
    # does, then every parameter should match the reference.
    pp_mod.sync_replicated_grads(replicas, g)

    # split_stack REINDEXES stage 1's layers to 0..k, but they are the reference's
    # layers cut..n. Comparing by raw name therefore checks stage 1's gradients against
    # the WRONG reference tensors -- and since rank 0 does the printing, that mismatch
    # was invisible until the runner exit code disagreed with the output.
    cut = 4 // pp_mod.PP_SIZE          # TinyBase has 4 layers
    def ref_name(n):
        if g.is_last and n.startswith("base.layers."):
            parts = n.split(".")
            parts[2] = str(int(parts[2]) + cut)
            return ".".join(parts)
        return n

    worst, worst_name, compared = 0.0, "", 0
    for n, p in m.named_parameters():
        rn = ref_name(n)
        if p.grad is None or rn not in ref_grads:
            continue
        compared += 1
        d = (p.grad.float() - ref_grads[rn].float()).abs().max().item()
        if d > worst:
            worst, worst_name = d, rn
    check("every parameter was actually compared (no silent name misses)",
          compared == sum(1 for _ in m.parameters()), f"{compared} tensors")
    check("every local parameter gradient matches single-process",
          worst < 1e-4, f"max|Δ|={worst:.3e} at {worst_name}")

    losses = pp_mod.reduce_metrics([loss_val], g, "cuda")
    want = reference_loss(build(), ids).item()
    check("reduced loss == single-process loss",
          abs(losses[0] - want) < 1e-4, f"{losses[0]:.6f} vs {want:.6f}")


def test_norms_match_single_process():
    """grad_norm and weight_norm must be interchangeable with the unsplit run."""
    g = pp_mod.build_groups()
    ids = batch()

    ref = build()
    reference_loss(ref, ids).backward()
    ref_gn = torch.sqrt(sum(p.grad.float().norm() ** 2
                            for p in ref.parameters())).item()
    ref_wn = torch.sqrt(sum(p.float().norm() ** 2 for p in ref.parameters())).item()

    m = build()
    replicas = pp_mod.replicated_params_for_test(m, g)
    pp_mod.split_stack(m.base, m.head, g.pp_rank)
    if g.is_first:
        h = pp_mod.stage_forward(m.base, 0, input_ids=ids)
        pp_mod.send_activations(h, h.detach(), g)
        pp_mod.backward_from_peer(h, g)
    else:
        B, T = ids.shape
        h_in, _ = pp_mod.recv_activations((B, T, 32), "cuda", g,
                                         teacher_dtype=torch.float32)
        out = pp_mod.stage_forward(m.base, 1, hidden=h_in)
        m.head(out).float().pow(2).mean().backward()
        pp_mod.send_input_grad(h_in, g)
    pp_mod.sync_replicated_grads(replicas, g)

    gn = pp_mod.global_grad_norm(m, replicas, g)
    wn = pp_mod.global_weight_norm(m, replicas, g)
    check("pipelined grad_norm == single-process grad_norm",
          abs(gn - ref_gn) < 1e-3 * max(1.0, ref_gn), f"{gn:.6f} vs {ref_gn:.6f}")
    check("pipelined weight_norm == single-process weight_norm",
          abs(wn - ref_wn) < 1e-3 * max(1.0, ref_wn), f"{wn:.6f} vs {ref_wn:.6f}")

    # The trap this guards: the tied weight lives on BOTH stages, so summing the stages
    # without skipping one copy double-counts it. That inflates the norm rather than
    # erroring, and would clip harder than the unsplit run at the same --grad-clip.
    naive = pp_mod.local_sq_skipping_replica(list(m.parameters()), [], grad=False)
    dist.all_reduce(naive, op=dist.ReduceOp.SUM, group=g.pp_group)
    check("counting the replica TWICE inflates the norm (the trap)",
          naive.sqrt().item() > wn + 1e-3,
          f"naive {naive.sqrt().item():.4f} vs correct {wn:.4f}")

    # The opposite trap, and the one that actually shipped: passing `replicas` straight
    # to the norm makes BOTH stages skip the tied weight, so it counts zero times. It
    # reads as an under-count rather than an error -- weight_norm 12.6 against 47.8.
    both = pp_mod.local_sq_skipping_replica(list(m.parameters()), replicas, grad=False)
    dist.all_reduce(both, op=dist.ReduceOp.SUM, group=g.pp_group)
    check("skipping the replica on BOTH stages under-counts it (the trap that shipped)",
          both.sqrt().item() < wn - 1e-3,
          f"skipped-everywhere {both.sqrt().item():.4f} vs correct {wn:.4f}")


def test_untied_model_freezes_the_dead_endpoint():
    """UNTIED models leave each stage an endpoint it never runs.

    split_stack touches only `layers` and `norm`, so stage 0 keeps an lm_head it never
    executes and stage 1 an embed_tokens it never executes (it enters via inputs_embeds).
    Those get no gradient, and DistOptimizer dereferences p.grad unconditionally --
    "AttributeError: 'NoneType' object has no attribute 'shape'" at Qwen3-8B, eight
    minutes into a two-node job.

    Tied models must NOT be touched: there the two names are one tensor both stages use.
    Both cases are asserted, because the whole bug was applying tied logic to an untied
    model.
    """
    g = pp_mod.build_groups()

    # --- untied ---
    m = build()
    m.head = nn.Linear(32, 64, bias=False).cuda()          # break the tie
    pp_mod.split_stack(m.base, m.head, g.pp_rank)
    dead = pp_mod.freeze_unused_endpoints(m.base, m.head, g)
    check("untied: exactly one endpoint frozen", len(dead) == 1, f"{len(dead)}")
    frozen_is_head = dead and dead[0].data_ptr() == m.head.weight.data_ptr()
    check("untied: stage 0 freezes the HEAD, stage 1 the EMBEDDING",
          bool(frozen_is_head) == g.is_first)
    # What the optimizer will actually see -- the default builder filters on this.
    trainable = [q for q in m.parameters() if q.requires_grad]
    check("untied: the dead endpoint is out of the trainable set",
          all(q.data_ptr() != dead[0].data_ptr() for q in trainable))

    # --- tied: must be left alone ---
    m2 = build()                                            # head IS embed_tokens
    pp_mod.split_stack(m2.base, m2.head, g.pp_rank)
    dead2 = pp_mod.freeze_unused_endpoints(m2.base, m2.head, g)
    check("tied: nothing frozen (both stages use the shared tensor)", dead2 == [],
          f"{len(dead2)} frozen")
    check("tied: the shared weight is still trainable",
          m2.base.embed_tokens.weight.requires_grad)


def test_export_merge_reconstructs_the_whole_model():
    """The merged export must contain every layer of the UNSPLIT model, once.

    The failure this guards is silent: split_stack renumbers stage 1's layers to 0..k,
    so a naive merge has stage 1's "layers.0" overwrite stage 0's and the checkpoint
    quietly contains half a model at full size -- it loads fine and scores like noise.
    """
    g = pp_mod.build_groups()
    ref = build()
    ref_keys = {k for k in ref.state_dict()}
    n_layers = len(ref.base.layers)

    m = build()
    pp_mod.split_stack(m.base, m.head, g.pp_rank)
    local = {k: v.detach().cpu() for k, v in m.state_dict().items()}

    check("stage records its layer offset",
          getattr(m.base, "_pp_layer_offset", None) == (0 if g.is_first
                                                        else n_layers // 2),
          f"offset={getattr(m.base, '_pp_layer_offset', None)}")

    # REAL EXPORT DTYPES. A compressed_tensors checkpoint carries float8_e4m3fn block
    # scales and packed uint8 weights, and those are exactly what broke the first
    # implementation: dist.broadcast_object_list pickles through a GPU byte tensor and
    # unpickling fp8 dies in torch.serialization.persistent_load. An all-fp32 payload
    # sails through, which is why the bug reached a 16-rank job. Keep these here.
    if not g.is_first:
        local["base.layers.0.fake_weight_packed"] = torch.zeros(
            8, 4, dtype=torch.uint8)
        local["base.layers.0.fake_weight_scale"] = torch.zeros(
            8, dtype=torch.float8_e4m3fn)

    out_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "_pp_export_tmp"
    merged = pp_mod.merge_export_state(local, m.base, g, out_dir=out_dir)
    if not g.is_first:
        check("stage 1 contributes nothing to the written dict", merged == {})
        return

    # Every layer index of the ORIGINAL stack must be present exactly once.
    check("fp8 block scale survived the handoff",
          merged.get("base.layers.2.fake_weight_scale") is not None
          and merged["base.layers.2.fake_weight_scale"].dtype == torch.float8_e4m3fn,
          str(merged.get("base.layers.2.fake_weight_scale", torch.empty(0)).dtype))
    check("packed uint8 weight survived the handoff",
          merged.get("base.layers.2.fake_weight_packed") is not None
          and merged["base.layers.2.fake_weight_packed"].dtype == torch.uint8)
    check("the handoff file was cleaned up",
          not (out_dir / f"_pp_stage1_dp{g.dp_rank}.safetensors").exists())
    # Every pair runs this concurrently on the same out_dir, which is what caught the
    # shared-filename race at world=8 (one pair's unlink deleted another pair's file).
    check("handoff name is unique per pipeline pair",
          f"dp{g.dp_rank}" in f"_pp_stage1_dp{g.dp_rank}.safetensors")

    got = {int(pp_mod._LAYER_RE.match(k).group(2))
           for k in merged if pp_mod._LAYER_RE.match(k)
           and "fake_" not in k}
    check("merged export covers every original layer index",
          got == set(range(n_layers)), f"{sorted(got)} vs {list(range(n_layers))}")
    missing = ref_keys - set(merged)
    check("merged export has every key the unsplit model has",
          not missing, f"missing {sorted(missing)[:4]}")

    # And the values must be the RIGHT half, not stage 0's layers duplicated.
    ref_sd = ref.state_dict()
    worst, name = 0.0, ""
    for k in ref_keys:
        d = (merged[k].float() - ref_sd[k].detach().cpu().float()).abs().max().item()
        if d > worst:
            worst, name = d, k
    check("merged tensors match the unsplit model exactly", worst == 0.0,
          f"max|Δ|={worst:.3e} at {name}")


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    if dist.get_world_size() % 2:
        if dist.get_rank() == 0:
            print("  SKIP  test_pipeline needs an even world size")
        dist.destroy_process_group()
        return
    for fn in (test_rank_layout_is_intra_node,
               test_split_halves_the_parameters,
               test_forward_matches_single_process,
               test_boundary_gradient_reproduces_single_process,
               test_norms_match_single_process,
               test_untied_model_freezes_the_dead_endpoint,
               test_export_merge_reconstructs_the_whole_model):
        fn()
        dist.barrier()
    # Reduce the failure count so rank 0 cannot report success while a peer failed.
    n_failed = torch.tensor([len(FAILED)], device="cuda")
    dist.all_reduce(n_failed, op=dist.ReduceOp.SUM)
    total = int(n_failed.item())
    if dist.get_rank() == 0:
        print("ALL PASS" if total == 0 else
              f"FAILED: {total} assertion(s) across ranks; local={FAILED}")
    dist.destroy_process_group()
    if total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
