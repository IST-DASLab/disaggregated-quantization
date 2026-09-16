"""Dump per-layer activations from a vLLM-served model, and diff two dumps.

    # one dump per checkpoint, each in its own job (only one big model fits at a time)
    python3 layer_probe.py dump --model <ckpt> --out a.pt [--tp 4]
    python3 layer_probe.py dump --model <ckpt> --out b.pt [--tp 4]

    # then, anywhere
    python3 layer_probe.py diff --a a.pt --b b.pt

WHY THIS EXISTS
---------------
`nemo3.5-w4a4-it1250` serves through vLLM, selects FLASHINFER_TRTLLM, and emits 32k
empty tokens. The weights are fine -- the same weights at W4A16 score normally, and QAD
loss for the W4A4 config is fine -- so the defect is in how it is SERVED. Guessing at
config fields does not localise that; comparing activations does.

Both dumps run through vLLM, so the only difference is the quantization path. The first
layer where the two diverge is the layer whose kernel is wrong. If activations go
NaN/Inf, the layer where that starts is the answer directly.

WHAT IS CAPTURED
----------------
The OUTPUT of every decoder layer, plus the mixer and the routed-expert block inside it,
for a single forward pass of one short prompt. Hidden states are small at this length
(52 layers x ~32 tokens x 2688 hidden x fp32 is ~18 MB), so the real tensors are kept
rather than summary statistics -- a cosine similarity computed from stored means would
hide exactly the kind of structured error being looked for.

enforce_eager=True is REQUIRED. With CUDA graphs the module forwards are replayed from a
captured graph and Python hooks never fire, so every tensor comes back empty and the
dump silently looks fine.
"""
import argparse
import os
import sys

import torch


def _target_modules(model):
    """(name, module) for each decoder layer and the interesting parts inside it."""
    out = []
    for name, mod in model.named_modules():
        # backbone.layers.N  /  ...N.mixer  /  ...N.mixer.experts
        parts = name.split(".")
        if "layers" not in parts:
            continue
        i = parts.index("layers")
        tail = parts[i + 2:]
        if tail in ([], ["mixer"], ["mixer", "experts"], ["mixer", "shared_experts"]):
            out.append((name, mod))
    return out


# Module level, not closures. `apply_model` pickles the callable it is handed, so
# `def _hook_all(...)` inside do_dump raises
#   AttributeError: Can't pickle local object 'do_dump.<locals>._hook_all'
# The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so a module-level dict
# is the same object the hooks write into.
_CAPTURED: dict = {}


def _record(name, t):
    if not isinstance(t, torch.Tensor) or name in _CAPTURED:
        return
    _CAPTURED[name] = t.detach().to(torch.float32).cpu()


def _make_hook(name):
    def hook(_m, _inp, out):
        _record(name, out[0] if isinstance(out, (tuple, list)) else out)
    return hook


def install_hooks(model):
    tg = _target_modules(model)
    for n, m in tg:
        m.register_forward_hook(_make_hook(n))
    return len(tg)


def _apply(llm, fn):
    """Run fn(model) inside the worker.

    `llm.apply_model` is vLLM's supported way to reach the nn.Module and it works
    regardless of executor layout. Walking llm_engine attributes by hand does not:
    `model_executor` is present but the path below it (WorkerWrapperBase -> worker ->
    model_runner -> model) moves between versions and silently dead-ends.
    """
    if hasattr(llm, "apply_model"):
        return llm.apply_model(fn)
    ex = llm.llm_engine.model_executor
    return ex.collective_rpc(lambda w: fn(w.model_runner.model))


def do_dump(args):
    # must precede the vllm import
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,          # see module docstring: hooks vs CUDA graphs
        trust_remote_code=True,
        max_num_seqs=1,
    )

    n_hooked = _apply(llm, install_hooks)
    if isinstance(n_hooked, list):
        n_hooked = n_hooked[0]
    print(f"hooking {n_hooked} modules", flush=True)
    if not n_hooked:
        raise SystemExit("no target modules matched; check _target_modules() naming")

    llm.generate([args.prompt], SamplingParams(max_tokens=1, temperature=0.0))
    captured = _CAPTURED

    meta = {"model": args.model, "prompt": args.prompt}
    bad = {n: (int(torch.isnan(t).sum()), int(torch.isinf(t).sum()))
           for n, t in captured.items()}
    nan_layers = [n for n, (a, b) in bad.items() if a or b]
    print(f"captured {len(captured)} tensors; {len(nan_layers)} contain NaN/Inf")
    if nan_layers:
        print("  first with NaN/Inf:", sorted(nan_layers)[:5])
    torch.save({"meta": meta, "act": captured}, args.out)
    print(f"wrote {args.out}")


def _key(n):
    """Sort decoder modules by layer index, then by depth within the layer."""
    p = n.split(".")
    i = p.index("layers")
    return (int(p[i + 1]), len(p))


def do_diff(args):
    A = torch.load(args.a, map_location="cpu", weights_only=False)
    B = torch.load(args.b, map_location="cpu", weights_only=False)
    print(f"A = {A['meta']['model']}")
    print(f"B = {B['meta']['model']}")
    common = sorted(set(A["act"]) & set(B["act"]), key=_key)
    only = (set(A["act"]) ^ set(B["act"]))
    if only:
        print(f"WARNING: {len(only)} modules present in only one dump, e.g. "
              f"{sorted(only)[:3]}")
    print(f"\n{'module':<48} {'cos':>8} {'relerr':>10} {'maxdiff':>11} "
          f"{'A rms':>10} {'B rms':>10}  flags")
    first_bad = None
    for n in common:
        a, b = A["act"][n].flatten(), B["act"][n].flatten()
        if a.shape != b.shape:
            print(f"{n:<48} SHAPE {tuple(A['act'][n].shape)} vs {tuple(B['act'][n].shape)}")
            continue
        flags = []
        if torch.isnan(b).any(): flags.append("B:NaN")
        if torch.isinf(b).any(): flags.append("B:Inf")
        if torch.isnan(a).any(): flags.append("A:NaN")
        if torch.isinf(a).any(): flags.append("A:Inf")
        af, bf = torch.nan_to_num(a), torch.nan_to_num(b)
        denom = af.norm() * bf.norm()
        cos = float((af @ bf) / denom) if denom > 0 else float("nan")
        rel = float((af - bf).norm() / af.norm()) if af.norm() > 0 else float("nan")
        mx = float((af - bf).abs().max())
        arms, brms = float(af.pow(2).mean().sqrt()), float(bf.pow(2).mean().sqrt())
        if first_bad is None and (flags or (cos == cos and cos < args.cos_threshold)):
            first_bad = n
        print(f"{n:<48} {cos:8.4f} {rel:10.4f} {mx:11.4g} {arms:10.4g} {brms:10.4g}"
              f"  {','.join(flags)}")
    print()
    if first_bad:
        print(f"FIRST DIVERGENCE: {first_bad}")
        print("  Everything above it agrees, so the defect is in this module's kernel or "
              "the weights/scales it was given -- not upstream.")
    else:
        print(f"No module fell below cos {args.cos_threshold} and none went NaN/Inf.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--model", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--tp", type=int, default=4)
    d.add_argument("--max-model-len", type=int, default=4096)
    d.add_argument("--prompt", default="Question: What is the capital of France?\nAnswer:")
    d.set_defaults(fn=do_dump)

    f = sub.add_parser("diff")
    f.add_argument("--a", required=True, help="reference dump (e.g. the W4A16 arm)")
    f.add_argument("--b", required=True, help="suspect dump (e.g. the W4A4 arm)")
    f.add_argument("--cos-threshold", type=float, default=0.98)
    f.set_defaults(fn=do_diff)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
