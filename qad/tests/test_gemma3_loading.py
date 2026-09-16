"""The text stack of a Gemma-3 checkpoint is ADDRESSED, never rehomed.

training/models.py loads every checkpoint as the class it declares and resolves the
text-only parts through a mapping table. These tests pin the two things that buys:
the weights really are the checkpoint's (not a silent re-init), and the mapping points
at the decoder stack rather than at the multimodal container.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache")

import torch

from training.models import (is_multimodal_wrapper, load_model, text_stack,
                             verify_text_load)

SMALL = "google/gemma-3-270m-it"      # Gemma3ForCausalLM, no vision
MM = "google/gemma-3-4b-it"           # Gemma3ForConditionalGeneration
QWEN = "Qwen/Qwen3-0.6B"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    assert cond, name


def test_detects_the_multimodal_wrapper():
    check("4b is a multimodal wrapper", is_multimodal_wrapper(MM))
    check("270m is not", not is_multimodal_wrapper(SMALL))
    check("qwen is not", not is_multimodal_wrapper(QWEN))


def test_plain_causal_lm_passthrough():
    """The default `.model`/`.lm_head` branch, on both families QAD trains."""
    for repo, cls in [(SMALL, "Gemma3ForCausalLM"), (QWEN, "Qwen3ForCausalLM")]:
        m = load_model(repo, torch.bfloat16)
        check(f"{repo} -> {cls}", type(m).__name__ == cls, type(m).__name__)
        s = text_stack(m)
        check("base is .model", s.base is m.model)
        check("head is .lm_head", s.head is m.lm_head)
        check("weights match the checkpoint", verify_text_load(m, repo) >= 4)


def test_multimodal_is_loaded_whole_and_addressed_text_only():
    m = load_model(MM, torch.bfloat16)
    # the wrapper is kept EXACTLY as transformers built it -- vision tower and all
    check("4b -> Gemma3ForConditionalGeneration",
          type(m).__name__ == "Gemma3ForConditionalGeneration")
    check("vision tower still present", hasattr(m.model, "vision_tower"))
    check("config untouched", m.config.architectures == ["Gemma3ForConditionalGeneration"],
          str(m.config.architectures))

    s = text_stack(m)
    check("base is model.language_model", s.base is m.model.language_model)
    check("head is lm_head", s.head is m.lm_head)
    check("34 layers", len(s.base.layers) == 34, str(len(s.base.layers)))
    # the point of the mapping: these are the counts of the TEXT stack, not the wrapper
    lin = [n for n, x in s.base.named_modules() if isinstance(x, torch.nn.Linear)]
    check("34*7 linears in the base", len(lin) == 34 * 7, str(len(lin)))
    check("no vision module inside the base",
          not any("vision" in n or "multi_modal" in n for n, _ in s.base.named_modules()))
    check("head tied to embedding",
          s.embed.weight.data_ptr() == s.head.weight.data_ptr())
    # the point of the whole module: these are the CHECKPOINT's weights, not fresh init
    check("weights match the checkpoint", verify_text_load(m, MM) >= 4)


def test_mapping_avoids_the_whole_model_traps():
    """Walking the wrapper instead of the base picks the WRONG modules, silently.

    Both of these are what consumers (quantizer apply, full-disag) would hit if they
    took the model rather than text_stack(model).base.
    """
    m = load_model(MM, torch.bfloat16)
    s = text_stack(m)

    first_whole = next(n for n, x in m.named_modules() if isinstance(x, torch.nn.Embedding))
    first_base = next(n for n, x in s.base.named_modules() if isinstance(x, torch.nn.Embedding))
    check("first embedding of the wrapper is the vision one",
          "vision" in first_whole, first_whole)
    check("first embedding of the base is embed_tokens",
          first_base == "embed_tokens", first_base)

    n_whole = sum(1 for _, x in m.named_modules() if isinstance(x, torch.nn.Linear))
    n_base = sum(1 for _, x in s.base.named_modules() if isinstance(x, torch.nn.Linear))
    check("wrapper carries vision linears the base does not",
          n_whole - n_base == 163, f"{n_whole} vs {n_base}")


def test_naive_direct_load_would_have_been_wrong():
    """Pins the trap: the obvious one-liner returns a half-random model."""
    from transformers import Gemma3ForCausalLM
    m = Gemma3ForCausalLM.from_pretrained(MM, dtype=torch.bfloat16)
    try:
        verify_text_load(m, MM)
        check("naive load is detectably wrong", False, "it matched — trap is gone?")
    except RuntimeError:
        check("naive load is detectably wrong (verify_text_load catches it)", True)


def test_unknown_wrapper_is_refused_not_guessed():
    """A wrapper with no mapping entry must raise, not fall back to `.model`."""
    from training import models
    m = load_model(MM, torch.bfloat16)
    saved = models._TEXT_PATHS.pop("Gemma3ForConditionalGeneration")
    try:
        text_stack(m)
        check("unknown wrapper refused", False, "it guessed a layout")
    except RuntimeError as e:
        check("unknown wrapper refused", "no _TEXT_PATHS entry" in str(e), str(e)[:60])
    finally:
        models._TEXT_PATHS["Gemma3ForConditionalGeneration"] = saved


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"  {fn.__name__}")
        fn()
    print("  all gemma3 loading tests passed")
