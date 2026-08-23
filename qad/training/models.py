"""Addressing the TEXT stack of a causal LM, whatever wrapper the checkpoint ships in.

Qwen3 and the small Gemma-3 sizes are plain causal LMs in the layout QAD assumes
everywhere: `.model` is the decoder stack, `.lm_head` is the output projection.
Gemma-3 at 4b and 12b ships ONLY as `Gemma3ForConditionalGeneration` — a SigLIP vision
tower, a multimodal projector and the language model under one roof — so `.model` is
NOT the decoder stack, it is the multimodal container, and `.model.language_model` is
what QAD actually wants to train.

APPROACH: load the checkpoint as the class it declares itself to be, change NOTHING
about the resulting module tree or config, and resolve the text stack through an
explicit per-architecture mapping (`_TEXT_PATHS`). The vision tower stays loaded and
unused — ~0.42B params at 4b — which is the price of never touching the architecture
or the checkpoint format.

WHY NOT REHOME THE LANGUAGE MODEL INTO A `Gemma3ForCausalLM` SHELL. It does hand QAD
the `.model`/`.lm_head` layout for free, but it pays for that by rewriting both the
module tree and the config, and the CHECKPOINT is what breaks: `config.text_config`
serializes with `architectures: None`, so the export has to be hand-patched to stay
servable, and the tied head then has to be hand-untied to be writable at all. Keeping
the model exactly as transformers built it keeps `save_pretrained`/vLLM semantics the
stock ones, and moves all the model-specific knowledge into one table.

DEAD ENDS — do not regress into these (all checked on transformers 5.3.0):
  * NOT `Gemma3ForCausalLM.from_pretrained(<4b repo>)`. That call SUCCEEDS and returns a
    half-random model: `Gemma3ForCausalLM._checkpoint_conversion_mapping` is `{}`, so
    every `language_model.*` tensor is an unexpected key and every text tensor is a
    missing one, silently re-initialized. A weight-magnitude check does NOT catch this,
    because random init is not zero. `verify_text_load` does.
  * NOT `from_pretrained(repo, state_dict=...)`. transformers 5 rejects that combination
    outright (`modeling_utils.py`: "`state_dict` cannot be passed together with a model
    name").
  * NOT `Gemma3ForCausalLM(text_cfg)` then `load_state_dict`. Constructing the class
    random-initializes ~4B parameters (~16 GB, minutes of CPU) purely to overwrite them.

CONSUMERS MUST SCOPE THEMSELVES TO `text_stack(model).base`. Walking the whole model is
now wrong in ways that do not raise: the first `nn.Embedding` in `named_modules()` order
is the SigLIP position embedding, not `embed_tokens`, and 163 of the 401 `nn.Linear`s
belong to the vision tower.
"""

import glob
import os
import re
from typing import NamedTuple

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

__all__ = ["load_model", "text_stack", "TextStack", "is_multimodal_wrapper",
           "verify_text_load"]


# Model class name -> (dotted path of the decoder stack, dotted path of the output head).
# Plain causal LMs use `_PLAIN`; only wrappers that BURY the text stack need an entry, and
# an unrecognised wrapper is an error rather than a guess -- guessing is precisely the
# failure mode this module exists to prevent.
_TEXT_PATHS: dict[str, tuple[str, str]] = {
    "Gemma3ForConditionalGeneration": ("model.language_model", "lm_head"),
}
_PLAIN = ("model", "lm_head")


class TextStack(NamedTuple):
    """Where the text-only parts of `model` actually live.

    `base` is the decoder stack: call it as `base(input_ids=...).last_hidden_state`.
    `base_path`/`head_path` are the dotted names, for callers that need to swap a module
    in place (full-disag) rather than just read it.
    """
    model: nn.Module        # the loaded model, exactly as transformers built it
    base: nn.Module         # decoder stack -- the ONLY subtree quantizers may rewrite
    head: nn.Module         # output projection (nn.Linear until a quantizer replaces it)
    embed: nn.Module        # input embedding
    base_path: str
    head_path: str


def is_multimodal_wrapper(model_id: str) -> bool:
    """True if this repo ships a vision tower alongside the language model."""
    cfg = AutoConfig.from_pretrained(model_id)
    return hasattr(cfg, "text_config") and hasattr(cfg, "vision_config")


def load_model(model_id: str, dtype: torch.dtype,
               attn_implementation: str = "flash_attention_2") -> nn.Module:
    """Load `model_id` as the class it declares, with no surgery of any kind.

    The declared class is the one whose `_checkpoint_conversion_mapping` matches the
    tensor names on disk, which is what makes the load correct. Resolving it from
    `config.architectures` rather than from an Auto class matters for Gemma-3: the
    `gemma3` config maps to `Gemma3ForCausalLM` under `AutoModelForCausalLM`, and that
    class loads the 4b repo into a half-random model without complaining.
    """
    cfg = AutoConfig.from_pretrained(model_id)
    archs = list(getattr(cfg, "architectures", None) or [])
    cls = getattr(transformers, archs[0], None) if archs else None
    if cls is None:
        # No usable declaration: fall back to the Auto class, which is right for every
        # plain causal LM. A wrapper with no `architectures` would land in text_stack's
        # "unknown wrapper" error below rather than silently mis-loading.
        cls = AutoModelForCausalLM
    return cls.from_pretrained(model_id, dtype=dtype,
                               attn_implementation=attn_implementation)


def text_stack(model: nn.Module) -> TextStack:
    """Resolve the text-only parts of `model` through `_TEXT_PATHS`.

    Raises if the model is a wrapper this module has never been taught to address, and
    verifies the two invariants every consumer relies on: the head is not inside the
    base (so quantizers scoped to `base` cannot touch it), and no vision module is.
    """
    cls = type(model).__name__
    if cls in _TEXT_PATHS:
        base_path, head_path = _TEXT_PATHS[cls]
    elif _buries_the_text_stack(model):
        raise RuntimeError(
            f"{cls} looks like a multimodal wrapper but has no _TEXT_PATHS entry; add "
            f"one (children: {sorted(n for n, _ in model.named_children())}) rather "
            "than letting the caller walk the whole model")
    else:
        base_path, head_path = _PLAIN

    try:
        base = model.get_submodule(base_path)
        head = model.get_submodule(head_path)
    except AttributeError as e:
        raise RuntimeError(f"{cls}: _TEXT_PATHS says {base_path!r}/{head_path!r} but "
                           f"that path does not exist ({e})") from e

    inside = {n for n, _ in base.named_modules()}
    if any("vision" in n or "multi_modal" in n for n in inside):
        raise RuntimeError(f"{cls}: {base_path!r} still contains vision modules; the "
                           "mapping points at the container, not the decoder stack")
    if head is base or any(m is head for m in base.modules()):
        raise RuntimeError(f"{cls}: {head_path!r} lives inside {base_path!r}; quantizers "
                           "scoped to the base would rewrite the head")

    embed = model.get_input_embeddings()
    return TextStack(model=model, base=base, head=head, embed=embed,
                     base_path=base_path, head_path=head_path)


def _buries_the_text_stack(model: nn.Module) -> bool:
    """True if `model.config` carries both a text and a vision sub-config."""
    cfg = getattr(model, "config", None)
    return hasattr(cfg, "text_config") and hasattr(cfg, "vision_config")


def _to_module_name(key: str, mapping: dict[str, str]) -> str:
    """Translate a checkpoint tensor name into the module's parameter name.

    Uses the model class's own `_checkpoint_conversion_mapping` -- the same table
    transformers applies when loading -- so this stays correct if the renaming changes.
    """
    for pattern, replacement in mapping.items():
        renamed = re.sub(pattern, replacement, key)
        if renamed != key:
            return renamed
    return key


@torch.no_grad()
def verify_text_load(model: nn.Module, model_id: str, n: int = 6) -> int:
    """Assert the TEXT weights ARE the checkpoint's. Returns the number compared.

    Compares against the raw safetensors rather than trusting the loader, because the
    failure mode this module exists to prevent is a load that reports success and hands
    back re-initialized tensors. Only tensors that land inside the text stack are
    checked -- the vision tower is loaded but is not what QAD trains.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    stack = text_stack(model)
    mapping = getattr(type(model), "_checkpoint_conversion_mapping", None) or {}
    prefixes = tuple(p for p in (f"{stack.base_path}.", f"{stack.head_path}.") if p != ".")
    have = {name: p for name, p in model.named_parameters() if name.startswith(prefixes)}

    # "*.jinja" is NOT optional: gemma-3-270m-it ships its chat template as a standalone
    # chat_template.jinja (1b keeps it in tokenizer_config.json, 4b/12b in
    # chat_template.json). A *.json-only fetch populates the shared HF cache with a
    # template-less snapshot, and every eval then dies with "Cannot use chat template
    # functions because tokenizer.chat_template is not set".
    path = snapshot_download(model_id,
                             allow_patterns=["*.json", "*.jinja", "*.safetensors"])
    checked = 0
    # Stream shard by shard and stop early: slurping every shard first costs 8 GB at 4b
    # (24 GB at 12b) to compare a handful of tensors.
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        for key, ref in load_file(shard).items():
            name = _to_module_name(key, mapping)
            if name not in have:
                continue
            got = have[name].detach().float().cpu()
            if not torch.allclose(got, ref.float(), atol=1e-3, rtol=1e-3):
                raise RuntimeError(f"{model_id}: {name} does not match the checkpoint "
                                   f"(max|d|={(got - ref.float()).abs().max():.3e})")
            checked += 1
            if checked >= n:
                return checked
    if checked == 0:
        raise RuntimeError(f"{model_id}: verified nothing — no checkpoint tensor mapped "
                           f"onto the text stack of a {type(model).__name__}")
    return checked
