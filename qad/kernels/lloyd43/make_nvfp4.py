"""Quantize a bf16 checkpoint to NVFP4 with llm-compressor, for vLLM's NATIVE path.

    python make_nvfp4.py --model Qwen/Qwen3-8B --scheme NVFP4
    python make_nvfp4.py --model google/gemma-3-4b-it --scheme NVFP4A16
    python make_nvfp4.py --all                       # every model, both schemes

Writes `<out-dir>/<model>-<scheme>`, which vLLM loads with no `--quantization` flag:
compressed-tensors is detected from the checkpoint's config.json. That is the point --
the NVFP4 arms then measure vLLM's own kernels through vLLM's own code path, not a
bespoke integration of ours.

WHY NOT THIRD-PARTY CHECKPOINTS
-------------------------------
`cortecs/Qwen3-8B-NVFP4` and friends exist for exactly one model, and come from another
publisher with different calibration and different opinions about which layers to skip.
That confounds the format with the checkpoint and leaves Gemma 3 unmeasurable. Quantizing
here means every arm runs the SAME bf16 weights with the same ignore list, so the
comparison is about the format and its kernels.

STATIC ABSMAX FOR THE ACTIVATIONS
---------------------------------
NVFP4 (W4A4) carries a per-tensor global activation scale plus per-group(16) scales. The
preset does the right thing already -- `observer='static_minmax'` on the input activations
-- so the global scale is a fixed absmax measured over the calibration set and baked into
the checkpoint, exactly as a served model does it. Only the group scales are computed at
inference; that is inherent to the format, not a dynamic-quantization choice.

    NVFP4     W4A4  -- weights AND activations fp4, CUTLASS fp4 tensor cores
    NVFP4A16  W4A16 -- weights only, bf16 activations, Marlin. No activation scale at all,
                       so `--scheme NVFP4A16` ignores the calibration arguments.

Calibration is deliberately small (`--nsamples`, default 32). The activation global scale
is one absmax per layer; it converges almost immediately, and a long calibration would
mostly buy patience. Raise it if a model looks off.
"""

import argparse
import os
from pathlib import Path

MODELS = [
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
    "google/gemma-3-270m", "google/gemma-3-1b-it", "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
]
SCHEMES = ["NVFP4", "NVFP4A16"]
# lm_head stays bf16, matching quantizers.blocked.replace_linears' skip list and the lloyd
# plugin's, so the arms are comparable layer for layer.
IGNORE = ["lm_head"]
# Multimodal checkpoints need their vision tower and projector left alone as well.
# `targets="Linear"` otherwise sweeps up the SigLIP encoder, and vLLM's Gemma3 then refuses
# the checkpoint outright:
#     ValueError: There is no module or parameter named
#     'encoder.layers.0.mlp.fc1.input_global_scale' in SiglipVisionTransformer
# The weights quantize fine; there is simply no kernel path for a quantized vision tower,
# and quantizing it would not be measuring the language model anyway.
# Leading `.*` matters: these are matched against the FULL module path, and
# Gemma3ForConditionalGeneration nests the tower as `model.vision_tower....`, so a pattern
# anchored at the start never matches.
VISION_IGNORE = ["re:.*vision_tower.*", "re:.*multi_modal_projector.*"]


def out_path(root: str, model: str, scheme: str) -> Path:
    return Path(root) / f"{model.split('/')[-1]}-{scheme}"


def convert(model: str, scheme: str, root: str, nsamples: int, seqlen: int,
            overwrite: bool = False) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    dest = out_path(root, model, scheme)
    if dest.exists() and not overwrite:
        print(f"  {dest} exists, skipping")
        return dest

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16,
                                             device_map="cuda")

    ignore = list(IGNORE)
    # Detect the vision tower from the MODULE TREE, not the config. `m.config` here is the
    # text config for these checkpoints, so `config.vision_config` is absent even when the
    # model plainly has a `vision_tower` -- that check silently did nothing and produced a
    # second batch of unusable checkpoints.
    # Any depth, not just top level: the tower lives at `model.vision_tower` on these
    # checkpoints, so checking only the first path component found nothing and quietly
    # produced a third batch of unusable weights.
    mods = [n for n, _ in m.named_modules()]
    if any("vision_tower" in n or "multi_modal_projector" in n for n in mods):
        ignore += VISION_IGNORE
        print(f"  multimodal: ignoring {VISION_IGNORE}")
    recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)

    kw = {}
    if scheme == "NVFP4":
        # W4A4 only: the activation global scale is an absmax over real activations, so it
        # needs data. W4A16 has no activation quantization and calibrating it would be
        # measuring nothing.
        #
        # PRE-TOKENIZED PLAIN TEXT, on purpose. Three separate failures came from letting
        # llmcompressor derive the inputs itself:
        #   * apply_chat_template dies on base models (gemma-3-270m has no chat_template);
        #   * a dataset makes it construct a "model processor", which fails on gemma-3-1b;
        #   * and on gemma-3-4b/12b -- Gemma3ForConditionalGeneration, a multimodal wrapper
        #     -- the sequential pipeline cannot trace the model at all.
        # Handing it input_ids directly, with the tokenizer as the processor and the basic
        # pipeline, sidesteps all three: none of them are about NVFP4 or about Gemma's
        # weights, they are about how the calibration batch is built.
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{nsamples}]")

        def to_ids(row):
            text = " ".join(t["content"] for t in row["messages"])[: seqlen * 8]
            enc = tok(text, truncation=True, max_length=seqlen)
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

        ds = ds.map(to_ids, remove_columns=ds.column_names)
        kw = dict(dataset=ds, num_calibration_samples=nsamples, max_seq_length=seqlen,
                  processor=tok, pipeline="basic")

    oneshot(model=m, recipe=recipe, **kw)
    m.save_pretrained(str(dest), save_compressed=True)
    tok.save_pretrained(str(dest))
    # Multimodal checkpoints (gemma-3-4b/12b are Gemma3ForConditionalGeneration) also need
    # their processor in the directory: vLLM loads an image processor from the checkpoint
    # and fails with a bare OSError if it is absent. The tokenizer alone is enough only for
    # text-only models, which is why 270m and 1b converted and loaded fine while 4b and 12b
    # quantized fine and then would not serve.
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(model).save_pretrained(str(dest))
    except Exception as exc:
        print(f"  (no processor for {model}: {type(exc).__name__}; tokenizer only)")
    # Assert the ignore list actually landed. Both previous attempts at this fix looked
    # like they worked -- the conversion succeeded and only vLLM, minutes later, revealed
    # that the vision tower had been quantized anyway.
    import json as _json
    cfg = _json.loads((dest / "config.json").read_text())["quantization_config"]
    # llm-compressor EXPANDS `re:` patterns into explicit module names here, so look for
    # the tower by substring rather than for the pattern itself -- checking for the literal
    # pattern reported a failure on a conversion that had in fact worked.
    got = cfg.get("ignore", [])
    if VISION_IGNORE[0] in ignore and not any("vision_tower" in i for i in got):
        raise RuntimeError(f"{dest}: vision tower not ignored; config ignore={got[:4]}")
    n_ign = len(got)
    print(f"  wrote {dest}  ({n_ign} modules ignored)")
    del m
    torch.cuda.empty_cache()
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--scheme", default="NVFP4", choices=SCHEMES)
    ap.add_argument("--all", action="store_true", help="every model x both schemes")
    ap.add_argument("--out-dir", default=str(Path.home() / ".nvfp4_checkpoints"))
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = ([(m, s) for m in MODELS for s in SCHEMES] if args.all
            else [(args.model, args.scheme)])
    if any(m is None for m, _ in jobs):
        raise SystemExit("--model or --all required")

    os.makedirs(args.out_dir, exist_ok=True)
    for model, scheme in jobs:
        print(f"=== {model}  {scheme} ===", flush=True)
        try:
            convert(model, scheme, args.out_dir, args.nsamples, args.seqlen,
                    args.overwrite)
        except Exception as exc:
            # One model failing (an unsupported architecture, an OOM) must not take the
            # rest of the sweep with it.
            print(f"  FAILED {model} {scheme}: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:160]}", flush=True)


if __name__ == "__main__":
    main()
