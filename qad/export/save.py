"""Model-level checkpoint assembly.

The FORMAT is owned by the layers (see QuantizedLinear.export_tensors /
export_config / load_tensors). This module only does the part that is inherently
model-level and cannot belong to any single layer:

  * walk the model and prefix each layer's tensors with its module path — the path
    is what vLLM looks up, so it has to come from named_modules() and not from
    anything the layer knows about itself;
  * carry over the tensors no quantizer owns (embeddings, norms, lm_head, rotary
    buffers);
  * write config.json, asking one quantized layer for the quantization_config.

A `variant` selects which checkpoint is being written for formats that emit more
than one per step (prefill vs decode). It is passed straight through to the layers;
None means "the only checkpoint", which is what every ordinary format returns from
export_variants().

One model-level reshaping happens here and nowhere else: a MULTIMODAL WRAPPER is
exported as the text-only causal LM it was trained as (see `_TEXT_ONLY_ARCH`). Every
other model is written exactly as it is loaded.
"""

import copy
from pathlib import Path

import torch
from torch import Tensor, nn

from quantizers.base import QuantizedLinear
from quantizers.full_disag import DualParamModule


# Wrapper class -> the causal-LM class its TEXT-ONLY export declares itself to be.
#
# WHY the checkpoint is reshaped at all, when training/models.py goes to such lengths
# NOT to reshape the loaded model: vLLM's `ModelConfig.is_mm_prefix_lm`
# (config/model.py:1125) keys on `model_type` ALONE -- "gemma3" or "paligemma" -- not on
# whether a vision tower is present and not on whether any image is ever passed. That
# sets use_mm_prefix, and only flex_attention and triton_attn implement
# supports_mm_prefix(); FLASH_ATTN and FlashInfer both refuse to start. Meanwhile
# gemma-3-270m/1b REQUIRE FLASH_ATTN (FlashInfer asserts on head_dim=256 with
# block_size 16). So `model_type: gemma3` and `model_type: gemma3_text` cannot be served
# by one backend, and the text-only shape is the only one that covers the whole family.
# It is also the only thing we train: QAD quantizes text_stack(model).base and nothing
# else.
#
# Explicit table, for the same reason training/models.py keeps _TEXT_PATHS explicit: a
# guess here produces a checkpoint that loads and then serves garbage.
_TEXT_ONLY_ARCH = {"Gemma3ForConditionalGeneration": "Gemma3ForCausalLM"}


def text_only_arch(model_or_config) -> str | None:
    """The class this checkpoint must be exported AS, or None to export it unchanged.

    None is the answer for every model exported before Gemma-3 4b/12b existed, which is
    what keeps the Qwen path byte-identical.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    for name in [type(model_or_config).__name__,
                 *(getattr(cfg, "architectures", None) or [])]:
        if name in _TEXT_ONLY_ARCH:
            return _TEXT_ONLY_ARCH[name]
    return None


def _quant_layers(model: nn.Module) -> dict:
    """Every module that owns its own export, keyed by module name.

    DualParamModule (--full-disag embeddings, norms, LM head) is included alongside the
    quantized linears: both keep their raw parameters OUT of the checkpoint and decide
    per variant what to ship, which is exactly the contract build_state_dict needs.
    """
    return {name: m for name, m in model.named_modules()
            if isinstance(m, (QuantizedLinear, DualParamModule))}


def _ignore_list_from_state(state: dict) -> list[str]:
    """`quantization_config.ignore`, derived from the TENSORS ABOUT TO BE WRITTEN.

    Must not be derived from the live model under pipeline parallelism. Each stage holds
    only its half of the layers, so a list built from rank 0's modules names only stage
    0's -- layers 0..31 of 64, which is 24 of the 48 GDN blocks, EXACTLY HALF. The
    tensors are fine (merge_export_state stitches both halves before writing); it is only
    the config that was written from one stage's view, so vLLM then built quantized
    layers for every GDN block in the second half whose weights are plain `.weight`.

    The merged state dict has all 64 layers by construction, which is the whole point of
    passing it in. A module is quantized iff it shipped a `weight_packed`; unquantized
    Linears are the 2-D `.weight` tensors that are not packed. 2-D also matches an
    EMBEDDING, so those are excluded by name -- they are not Linears and `targets` is
    ["Linear"].

    Plus the non-Linear PARENT of any MIXED module (holding both a quantized and an
    unquantized Linear) and that parent's norm: at 27B the 48 linear_attn blocks, each
    with a quantized out_proj beside its ignored in_proj_a/b. With those entries this
    reproduces the reference nvfp4 config for this model EXACTLY -- set-equal, 303
    entries, zero difference either way.
    """
    packed = {k[: -len(".weight_packed")] for k in state if k.endswith(".weight_packed")}

    def is_embedding(mod: str) -> bool:
        return "embed" in mod.rpartition(".")[2]

    names = {
        k[: -len(".weight")] for k, v in state.items()
        if k.endswith(".weight") and getattr(v, "ndim", 0) == 2
        and k[: -len(".weight")] not in packed
        and ".inner." not in k and not is_embedding(k[: -len(".weight")])
    }

    def parent(n: str) -> str:
        return n.rpartition(".")[0]

    mixed = {parent(p) for p in packed} & {parent(n) for n in names}
    return sorted(names | mixed | {f"{m}.norm" for m in mixed})


def _owned_by_quant(key: str, quant: dict) -> bool:
    """Is `key` inside a module that decides its own export?

    ANY ancestor, not just the immediate parent. The old test was
    `key.rpartition(".")[0] in quant`, which holds only when the layer keeps its tensors
    as direct children -- true for every layer that existed when it was written.
    FrozenDecodeNorm wraps the original module as `.inner` so it can delegate the norm
    maths instead of reimplementing it, and `...input_layernorm.inner.weight` has parent
    `...input_layernorm.inner`, which is NOT in `quant`. So the fp32 training master
    shipped alongside the bf16 tensor the wrapper exports -- 208 duplicate tensors at
    27B, and vLLM refuses the checkpoint outright:

        ValueError: There is no module or parameter named
          'layers.0.input_layernorm.inner' in Qwen3_5Model

    Walking ancestors fixes it for any nesting depth, not just this one.
    """
    parts = key.split(".")
    for i in range(len(parts) - 1, 0, -1):
        if ".".join(parts[:i]) in quant:
            return True
    return False


def build_state_dict(model: nn.Module, variant=None) -> dict[str, Tensor]:
    """Assemble the full tensor dict for one checkpoint variant."""
    quant = _quant_layers(model)
    out: dict[str, Tensor] = {}
    # Everything the quantizers do NOT own. Quant-internal state (the FP32 master,
    # _wq, act_amax, schedule buffers, logits) is skipped — the layer decides what
    # of itself is worth serializing, and it is never the training-time internals.
    seen: set[int] = set()
    for key, tensor in model.state_dict().items():
        if _owned_by_quant(key, quant):
            continue
        t = tensor.detach().cpu()
        # Tied weights (every Gemma-3 size, and Qwen3 below 8B) are ONE storage under
        # two names, and safetensors refuses to serialize shared storage. Production
        # exports from CUDA, where each .cpu() allocates its own buffer, so this never
        # fires there and the bytes written are unchanged; a CPU-side export would
        # otherwise die in save_file with "some tensors share memory".
        if t.untyped_storage().data_ptr() in seen:
            t = t.clone()
        seen.add(t.untyped_storage().data_ptr())
        out[key] = t
    for name, m in quant.items():
        for leaf, tensor in m.export_tensors(variant).items():
            out[f"{name}.{leaf}"] = tensor
    if text_only_arch(model) is not None:
        out = _text_only_tensors(model, out)
    return out


def _text_prefixes(model: nn.Module) -> tuple[str, str]:
    """(prefix in the live model, prefix in the checkpoint) for the text stack.

    Taken from training/models.py's `_TEXT_PATHS` rather than from a literal here, so
    there is exactly one place that knows where a wrapper's text stack lives.
    """
    from training.models import text_stack

    stack = text_stack(model)
    container = stack.base_path.rpartition(".")[0]      # "model" for the wrapper
    return f"{stack.base_path}.", (f"{container}." if container else "")


def to_model_keys(model: nn.Module, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    """Re-key a checkpoint written by this model back onto its live module tree.

    The inverse of the text-only rename, for callers that load a checkpoint into the
    wrapper it was exported from (eval/eval_transformers.py). Unambiguous because the
    text-only export drops every other tensor under the container prefix.
    """
    if text_only_arch(model) is None:
        return tensors
    old, new = _text_prefixes(model)
    return {(old + k[len(new):] if k.startswith(new) else k): v
            for k, v in tensors.items()}


def _text_only_tensors(model: nn.Module, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    """Drop everything outside the text stack and rename it to the plain layout.

    The wrapper calls its text weights `model.language_model.*`; `Gemma3ForCausalLM`
    (and vLLM's `gemma3.py`, which resolves `model.embed_tokens.weight` and
    `model.layers.N.*` by bare dict lookup) expects `model.*`.
    """
    old, new = _text_prefixes(model)
    container = new.rstrip(".")
    out: dict[str, Tensor] = {}
    dropped = 0
    for key, tensor in tensors.items():
        if key.startswith(old):
            out[new + key[len(old):]] = tensor
        elif container and key.startswith(f"{container}."):
            dropped += 1                                 # vision tower / projector
        else:
            out[key] = tensor                            # lm_head, top-level buffers
    if not dropped:
        raise RuntimeError(
            f"{type(model).__name__}: text-only export dropped nothing — the wrapper "
            f"layout under {container!r} is not what _TEXT_PATHS describes")
    bad = [k for k in out if "vision" in k or "multi_modal" in k or "language_model" in k]
    if bad:
        raise RuntimeError(f"text-only export still carries wrapper tensors: {bad[:5]}")
    return out


def load_into(model: nn.Module, tensors: dict[str, Tensor], variant=None) -> int:
    """Restore quantized layers from a checkpoint dict. Returns layers restored.

    The inverse of build_state_dict for the quantized layers; non-quantized tensors
    are loaded by the caller through the usual load_state_dict path.
    """
    n = 0
    # The checkpoint may have been written under the text-only names (a multimodal
    # wrapper); every other model exports its module names verbatim. Resolved ONCE:
    # _text_prefixes walks the module tree, and per-layer it would be quadratic.
    rename = text_only_arch(model) is not None
    old, new = _text_prefixes(model) if rename else ("", "")
    for name, m in _quant_layers(model).items():
        ckpt = new + name[len(old):] if rename and name.startswith(old) else name
        prefix = f"{ckpt}."
        sub = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        if sub:
            m.load_tensors(sub, variant)
            n += 1
    return n



def export_variants(model: nn.Module) -> list:
    """Variants this model's quantizer emits ([None] for single-checkpoint formats).

    Asks the LAYER CLASS, so a format that emits prefill/ + decode/ is discovered from
    the model itself rather than from its name. quantizers.variants(name) is the
    equivalent lookup for callers that only have the CLI name and no built model.
    """
    quant = _quant_layers(model)
    if not quant:
        return [None]
    return type(next(iter(quant.values()))).export_variants()

from export.config_fix import fix_serving_fields


def save_checkpoint(model: nn.Module, out_dir: Path, variant=None, step: int = 0,
                    state: dict | None = None, write: bool = True) -> int:
    """Assemble one checkpoint and write it. Returns the tensor count.

    `state` overrides the locally-assembled tensors, and `write` gates the file I/O.
    Both exist for PIPELINE PARALLELISM: each stage owns half the layers, so the halves
    are assembled on their own ranks, merged by pipeline.merge_export_state(), and only
    then written -- by rank 0 alone. Without the seam the writer would have to be the
    assembler, and a stage-0-only export writes half a model with no error at all.
    """
    from safetensors.torch import save_file
    import json

    out_dir = Path(out_dir)
    state = build_state_dict(model, variant) if state is None else state
    if not write:
        return len(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "step": str(step)})
    arch = text_only_arch(model)
    if arch is None:
        model.config.save_pretrained(out_dir)
    else:
        text_only_config(model.config, arch).save_pretrained(out_dir)
        _save_tokenizer(model, out_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out_dir)

    quant = _quant_layers(model)
    # First module that actually HAS a config, not simply the first module. Once
    # --full-disag put DualEmbedding/DualRMSNorm into this set, the first entry in
    # named_modules() order became the embedding, whose export_config is None -- so
    # quantization_config silently vanished from config.json and vLLM loaded the
    # checkpoint as plain bf16, then died on the extra input_global_scale tensors.
    qcfg = next((c for c in (m.export_config(variant) for m in quant.values())
                 if c is not None), None)
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if qcfg is not None:
        # From `state`, never from `model`: under PP the local model is half the layers.
        qcfg = {**qcfg, "ignore": _ignore_list_from_state(state)}
        cfg["quantization_config"] = qcfg
    fix_serving_fields(cfg)
    if arch is not None:
        _fix_text_only_fields(cfg, model.config, arch)
    elif _has_per_layer_rope(cfg):
        # PLAIN Gemma-3 (270m, 1b) is not a wrapper, so it never reaches
        # _fix_text_only_fields and its rope was left in the transformers-5 per-layer-type
        # form. vLLM rejects that outright, before loading a single weight:
        #   pydantic ValidationError: rope_parameters should have a 'rope_type' key
        # which is how the first six Gemma eval jobs died. Normalise it to the flat 4.x
        # fields the stock Gemma repos ship. Qwen is deliberately NOT touched: its
        # rope_parameters is flat and already carries rope_type at the top level, so it
        # validates fine and its exports stay byte-identical.
        _fix_rope_fields(cfg, getattr(model.config, "text_config", model.config))
    # The one incident this export path has already caused: quantization_config went
    # missing and vLLM served the checkpoint as plain bf16 until it choked on the extra
    # tensors, wasting a whole eval sweep. Cheap to assert, expensive to miss.
    if qcfg is not None and "quantization_config" not in cfg:
        raise RuntimeError(f"{out_dir}: quantization_config was dropped from config.json")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return len(state)


def text_only_config(config, arch: str):
    """The config a wrapper's text-only checkpoint carries: its OWN text_config, plus
    the two fields `text_config` does not serialize for itself.

    `model_type` is already "gemma3_text" on the sub-config -- that field is what makes
    FLASH_ATTN legal again -- but `architectures` is None on it, handled here at export time.
    """
    text = copy.deepcopy(config.text_config)
    text.architectures = [arch]
    # Token ids live on the WRAPPER, not on its text sub-config (gemma-3-4b-it declares
    # eos_token_id [1, 106] at top level and nothing in text_config); the stock 1b repo
    # carries all three explicitly, and this is the shape we are matching.
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, key, None)
        if value is not None:
            setattr(text, key, value)
    return text


def _fix_text_only_fields(cfg: dict, config, arch: str) -> None:
    """Post-serialization repair of the text-only config.json.

    `save_pretrained` writes a DIFF against the class defaults, so any field that
    happens to equal a default is absent -- and `fix_serving_fields` then defaults some
    of them to QWEN's values. rope_scaling is the dangerous one: gemma-3-4b/12b specify
    a linear factor-8 scaling, and serving without it silently changes every position
    beyond the original context.
    """
    text = config.text_config
    cfg["architectures"] = [arch]
    cfg["model_type"] = getattr(text, "model_type", cfg.get("model_type"))
    for key in ("vision_config", "text_config"):
        cfg.pop(key, None)
    for key in ("sliding_window", "vocab_size", "tie_word_embeddings",
                "max_position_embeddings", "query_pre_attn_scalar", "layer_types"):
        value = getattr(text, key, None)
        if value is not None:
            cfg[key] = value
    _fix_rope_fields(cfg, text)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, key, None)
        if value is None:
            value = getattr(text, key, None)
        if value is not None:
            cfg[key] = value


def _has_per_layer_rope(cfg: dict) -> bool:
    """True if `rope_parameters` is keyed BY LAYER TYPE rather than flat.

    Two shapes reach config.json under transformers 5:
      * flat (Qwen3):   {"rope_theta": 1e6, "rope_type": "default"}       -- vLLM accepts
      * per-layer (G3): {"full_attention": {...}, "sliding_attention": {...}} -- rejected,
        because vLLM validates for a top-level "rope_type" that a per-layer dict has no
        place for.
    Only the second needs rewriting, and distinguishing them is what keeps the Qwen export
    byte-identical.
    """
    rope = cfg.get("rope_parameters")
    if (isinstance(rope, dict) and "rope_type" not in rope
            and any(isinstance(v, dict) for v in rope.values())):
        return True
    # Second trigger, for the SILENT variant of the same defect. `to_diff_dict` omits any
    # field equal to a class default, so a config can arrive with NO `rope_parameters` at
    # all while `fix_serving_fields` has already injected Qwen's flat `rope_theta` and
    # `rope_scaling: null`. Gemma-3 needs a SECOND theta (`rope_local_base_freq`, 1e4) for
    # its sliding layers; without it they silently run at the full-attention 1e6 and the
    # model is quietly wrong rather than rejected. Keyed on model_type so Qwen, which has
    # no such field, is never touched.
    return (cfg.get("model_type") == "gemma3_text"
            and "rope_local_base_freq" not in cfg)


def _fix_rope_fields(cfg: dict, text) -> None:
    """Write rope in the ONE form the stock Gemma-3 repos use, and only that form.

    Gemma-3 has two ropes: 1e6 with linear factor-8 scaling on the full-attention
    layers, 1e4 unscaled on the sliding ones. transformers 5 represents that as
    `rope_parameters` keyed BY LAYER TYPE, while every stock config.json (including
    gemma-3-4b-it's own text_config) writes the flat 4.x fields and lets transformers
    convert. Two reasons to emit the flat form:
      * shipping BOTH forms is what vLLM REJECTS. Measured, and worth stating precisely
        because the obvious guess is wrong: on re-read transformers lets the NESTED form
        win, so the factor-8 scaling is NOT silently dropped. The real consequence is a
        hard failure at engine init, before any weight loads —
            pydantic ValidationError: rope_parameters should have a 'rope_type' key
        — because vLLM validates for a top-level `rope_type` that a per-layer dict has
        nowhere to put. A crash, not silent corruption. (Six eval jobs died on exactly
        this before the plain-model path was wired up.)
      * it is byte-for-byte the shape the stock repos ship, i.e. the shape both
        transformers and vLLM (gemma3.py:168-179 handles either) are known to read.

    The genuinely SILENT failure is the neighbouring one: emitting neither form's local
    theta, so the sliding layers fall back to the full-attention 1e6. See
    `_has_per_layer_rope` for the guard against that.
    """
    rope = dict(getattr(text, "rope_parameters", None) or {})
    full = dict(rope.get("full_attention", rope))
    local = dict(rope.get("sliding_attention", {}))
    cfg.pop("rope_parameters", None)
    if full.get("rope_theta") is not None:
        cfg["rope_theta"] = full["rope_theta"]
    local_theta = local.get("rope_theta", getattr(text, "rope_local_base_freq", None))
    if local_theta is not None:
        cfg["rope_local_base_freq"] = local_theta
    scaling = {k: v for k, v in full.items() if k != "rope_theta"}
    cfg["rope_scaling"] = scaling if scaling.get("rope_type", "default") != "default" \
        else None


def _save_tokenizer(model: nn.Module, out_dir: Path) -> None:
    """Copy the tokenizer into a text-only export, best effort.

    Only this export path does it: every other checkpoint is served with
    `--tokenizer <base repo>` (serving/run_nixl_server.sh, eval_vllm.py), so adding
    ~35 MB of tokenizer to each of them would change nothing except the size of the
    existing tree. A text-only export is the one checkpoint whose config no longer
    matches the repo it came from, so it is worth making self-contained.

    `save_pretrained` is used rather than a file copy on purpose: it re-emits the chat
    template in whatever form this transformers version uses, so there is no pattern
    list to get wrong (gemma-3-270m-it ships a standalone chat_template.jinja, 4b/12b a
    chat_template.json).
    """
    repo = getattr(model.config, "_name_or_path", None)
    if not repo:
        return
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(repo).save_pretrained(out_dir)
    except Exception as e:                                        # noqa: BLE001
        # Never fail an export over a sidecar: the checkpoint is servable without it.
        print(f"[export] tokenizer not copied ({type(e).__name__}: {e})", flush=True)
