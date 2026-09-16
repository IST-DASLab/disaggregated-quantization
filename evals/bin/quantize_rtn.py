"""RTN-quantize a model from models.json to NVFP4 (W4A4) or NVFP4A16 (W4A16).

    python3 quantize_rtn.py --model qwen3.8-2.4t --scheme NVFP4
    python3 quantize_rtn.py --model qwen3.8-2.4t --scheme NVFP4A16

Writes `<out-dir>/<checkpoint-name>-<scheme>`, which vLLM loads with no --quantization
flag; compressed-tensors is detected from config.json.

The `quantize` block in models.json drives everything: `ignore` is what stays bf16,
`expect` is a regex the resolved target set must match exactly, and the written
checkpoint is verified against both before the job reports success.
"""
import argparse
import contextlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

OUT_ROOT = os.environ.get(
    "MUSE_QUANT_OUT", "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/models")
MODELS_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models.json")


def load_spec(key):
    with open(MODELS_JSON) as f:
        models = json.load(f)
    if key not in models:
        keys = [k for k in models if not k.startswith("_")]
        raise SystemExit(f"unknown model {key!r}; known: {', '.join(keys)}")
    spec = models[key]
    if "quantize" not in spec:
        raise SystemExit(f"{key!r} has no `quantize` block in models.json")
    return spec


def quantized_module_names(model, ignore):
    # A literal ignore entry matches the FULL module name only, as llm-compressor does.
    # Use `re:.*lm_head$` when the head is nested.
    pats = [p[3:] if p.startswith("re:") else None for p in ignore]
    literal = {p for p in ignore if not p.startswith("re:")}
    import torch.nn as nn
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name in literal:
            continue
        if any(p and re.match(p, name) for p in pats):
            continue
        out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "NVFP4A16"])
    ap.add_argument("--model", default="muse-glimmer", help="key in models.json")
    ap.add_argument("--out-dir", default=OUT_ROOT)
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extra-ignore", nargs="*", default=[],
                    help="appended to the model's ignore list, to match a third-party "
                         "checkpoint's layer selection")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the module lists and exit without quantizing")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer

    spec = load_spec(args.model)
    model_path = spec["path"]
    IGNORE = list(spec["quantize"]["ignore"]) + list(args.extra_ignore)
    EXPECT = re.compile(spec["quantize"]["expect"])
    PER_LAYER = spec["quantize"].get("per_layer")

    dest = Path(args.out_dir) / f"{Path(model_path).name}-{args.scheme}{args.suffix}"
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{dest} exists; pass --overwrite to replace it")

    auto_class = spec["quantize"].get("auto_class", "AutoModelForImageTextToText")
    remote = bool(spec["quantize"].get("trust_remote_code", False))
    attn = spec["quantize"].get("attn_implementation")
    print(f"loading {args.model} from {model_path} via {auto_class}"
          f"(trust_remote_code={remote}, attn={attn or 'default'}) ...", flush=True)
    Auto = getattr(transformers, auto_class)
    kw = {"attn_implementation": attn} if attn else {}

    SEQUENTIAL = bool(spec["quantize"].get("sequential"))
    if args.dry_run:
        device_map = None
    elif SEQUENTIAL:
        device_map = "auto_offload"
    else:
        device_map = "cuda"
    load = dict(dtype=torch.bfloat16, trust_remote_code=remote,
                device_map=device_map, **kw)
    if device_map == "auto_offload":
        # Node-local NVMe, not Lustre: spilled weight storage, written once and read back
        # per layer, a few TB. run_quantize.sh creates it before entering the container.
        offload = os.environ.get("QUANT_OFFLOAD_DIR", "/raid/scratch/quant_offload")
        Path(offload).mkdir(parents=True, exist_ok=True)
        load["offload_folder"] = offload
        print(f"  offload_folder={offload}", flush=True)

    with contextlib.ExitStack() as stack:
        if SEQUENTIAL and not args.dry_run:
            from llmcompressor.utils.dev import load_offloaded_model
            # Reserve host RAM the loader must NOT fill. linearize_moe converts the fused
            # 3D expert banks to 2D per-expert after loading and holds the result in RAM,
            # so without a reserve the loader fills memory with weights and the
            # conversion is OOM-killed part way through.
            gb = spec["quantize"].get("offload_extra_cpu_mem_gb")
            extra = {"extra_cpu_mem": int(gb) * 1024 ** 3} if gb else {}
            stack.enter_context(load_offloaded_model(Auto, **extra))
        if spec["quantize"].get("linearize_moe"):
            from llmcompressor.modeling.moe.linearize import load_quantizable_moe
            stack.enter_context(load_quantizable_moe(Auto))
        model = Auto.from_pretrained(model_path, **load)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=remote)

    names = quantized_module_names(model, IGNORE)
    unexpected = [n for n in names if not EXPECT.match(n)]
    if unexpected:
        raise SystemExit(
            f"ignore list does not isolate the LLM decoder linears.\n"
            f"  {len(unexpected)} unexpected target(s), e.g. {unexpected[:5]}")
    if PER_LAYER:
        n_layers = len(model.config.text_config.layer_types)
        expected = n_layers * PER_LAYER
        if len(names) != expected:
            raise SystemExit(f"expected {expected} quantized linears "
                             f"({n_layers} layers x {PER_LAYER}), resolved {len(names)}")
    if spec["quantize"].get("linearize_moe") and not any(
            re.search(r"experts\.\d+\.", n) for n in names):
        raise SystemExit(
            f"linearize_moe requested but no per-expert Linear resolved ({len(names)} "
            f"modules). The experts are still fused, so this recipe would quantize only "
            f"attention -- a few percent of an MoE -- and report it as a 4-bit arm.")
    print(f"quantizing {len(names)} linears; everything matched by {IGNORE} "
          f"stays bf16", flush=True)
    if args.dry_run:
        for n in names[:4] + ["..."] + names[-2:]:
            print("   ", n)
        return

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=IGNORE)

    kw = {}
    # NVFP4A16 needs no data, but on a sequential model it still gets the dataset: the
    # data-free pipeline onloads modules to the GPU and does not offload them again, so at
    # this scale it fills a 276 GB card and dies in the weight observer. The sequential
    # pipeline onloads one layer at a time and releases it. The calibration set cannot
    # affect an A16 result -- its weight observer is memoryless_minmax, computed from the
    # weight tensor alone -- so this buys the memory behaviour and changes no number.
    if args.scheme == "NVFP4" or SEQUENTIAL:
        from datasets import load_dataset
        # Local parquet: with HF_HUB_OFFLINE=1 the by-name form resolves the repo before
        # it looks at the cache and raises, even though the shard is downloaded.
        hub = os.path.join(os.environ.get(
            "HF_HOME", "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache"), "hub")
        import glob as _glob
        shards = sorted(_glob.glob(os.path.join(
            hub, "datasets--HuggingFaceH4--ultrachat_200k", "snapshots", "*",
            "data", "train_sft-*.parquet")))
        if not shards:
            raise SystemExit("ultrachat_200k not in the hub cache; "
                             "hf download it on a login node first")
        ds = load_dataset("parquet", data_files={"train": shards[:1]},
                          split="train").select(range(args.nsamples))

        def to_ids(row):
            text = " ".join(t["content"] for t in row["messages"])[: args.seqlen * 8]
            enc = tok(text, truncation=True, max_length=args.seqlen)
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}

        ds = ds.map(to_ids, remove_columns=ds.column_names)
        kw = dict(dataset=ds, num_calibration_samples=args.nsamples,
                  max_seq_length=args.seqlen, processor=tok, pipeline="basic")

    if SEQUENTIAL and kw.get("dataset") is not None:
        kw["pipeline"] = "sequential"
        kw["moe_calibrate_all_experts"] = True
        kw["sequential_offload_device"] = spec["quantize"].get(
            "sequential_offload_device", "cpu")
        targets = spec["quantize"].get("sequential_targets")
        if targets:
            kw["sequential_targets"] = targets

    calib_forward = spec["quantize"].get("calib_forward") if kw else None
    original_forward = model.forward
    undo = []
    if calib_forward:
        sub = getattr(model, calib_forward)
        print(f"  calibrating through model.{calib_forward} "
              f"({type(sub).__name__}); the top-level forward requires images",
              flush=True)
        # MethodType, not a bare lambda: llm-compressor reads model.forward.__func__.
        import types
        model.forward = types.MethodType(lambda _self, **batch: sub(**batch), model)

        # The vendor's remote code targets an older transformers signature.
        import sys as _sys
        vendor = _sys.modules.get(type(sub).__module__)
        fn = getattr(vendor, "create_causal_mask", None)
        if fn is not None and "inputs_embeds" not in inspect.signature(fn).parameters:
            raise SystemExit(
                f"create_causal_mask in this transformers "
                f"({transformers.__version__}) does not take `inputs_embeds` either: "
                f"{inspect.signature(fn)}. The rename this shim performs is wrong for "
                f"this version -- recheck before trusting any activation scale.")
        if fn is not None:
            def _compat(*a, _orig=fn, **kwargs):
                if "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                kwargs.pop("cache_position", None)
                return _orig(*a, **kwargs)
            vendor.create_causal_mask = _compat
            undo.append(lambda: setattr(vendor, "create_causal_mask", fn))
            print(f"  patched create_causal_mask in {vendor.__name__} for "
                  f"transformers {transformers.__version__}", flush=True)
    try:
        oneshot(model=model, recipe=recipe, **kw)
    finally:
        model.forward = original_forward
        for f in undo:
            f()
    model.save_pretrained(str(dest), save_compressed=True)
    tok.save_pretrained(str(dest))
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(
            model_path, trust_remote_code=remote).save_pretrained(str(dest))
    except Exception as exc:                                          # noqa: BLE001
        print(f"  WARNING: no processor saved ({type(exc).__name__}: {exc}); "
              f"vLLM will not be able to serve images from this checkpoint")

    # save_pretrained does not emit preprocessor_config.json / merges.txt / vocab.json,
    # and vLLM needs them to serve images.
    import shutil
    carried = []
    for src in Path(model_path).iterdir():
        if src.is_dir() or src.suffix in (".safetensors", ".bin", ".pt"):
            continue
        if src.name.endswith(".index.json") or (dest / src.name).exists():
            continue
        shutil.copy2(src, dest / src.name)
        carried.append(src.name)
    if carried:
        print(f"  carried over from the source: {', '.join(sorted(carried))}")

    from safetensors import safe_open
    shards = sorted(dest.glob("*.safetensors"))
    keys = []
    for sh in shards:
        with safe_open(str(sh), "pt") as f:
            keys.extend(f.keys())
    packed = [k for k in keys if k.endswith(".weight_packed")]
    quantized = {k[: -len(".weight_packed")] for k in packed}
    if len(packed) != len(names):
        raise SystemExit(f"{dest}: {len(packed)} packed weights written, "
                         f"{len(names)} modules were resolved")
    # Normalise the leading prefix before comparing. `names` comes from the live model
    # graph, which transformers 5.16 builds as `model.layers...`, while the SAVED keys
    # carry the checkpoint's own `backbone.layers...`. Comparing them raw makes every
    # correctly-quantized module look stray -- 41072 of them on the Super.
    def _norm(k):
        for pre in ("model.", "backbone."):
            if k.startswith(pre):
                return k[len(pre):]
        return k
    norm_names = {_norm(n) for n in names}
    stray = sorted(k for k in quantized if _norm(k) not in norm_names)
    if stray:
        raise SystemExit(f"{dest}: quantized {len(stray)} module(s) that should have "
                         f"been ignored, e.g. {stray[:5]}")
    n_act = sum(1 for k in keys if k.endswith(".input_global_scale"))
    want_act = len(names) if args.scheme == "NVFP4" else 0
    if n_act != want_act:
        raise SystemExit(f"{dest}: {n_act} activation scales for scheme "
                         f"{args.scheme}, expected {want_act}")
    print(f"wrote {dest}  ({len(packed)} quantized, {n_act} activation scales)")


if __name__ == "__main__":
    sys.exit(main())
