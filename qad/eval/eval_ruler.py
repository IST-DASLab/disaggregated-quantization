"""Evaluate a checkpoint on RULER under REAL disaggregated vLLM serving.

Same serving path as eval_disagg.py (NixlConnector, one proxy, real KV transfer) --
this file only differs in what it asks lm-eval to run. RULER's tasks are governed by
a `metadata={"max_seq_lengths": [...]}` TaskManager option rather than by --tasks
alone: each task generates `num_samples` (500, hardcoded in lm-eval's RULER
implementation -- not overridable from here) fresh docs PER requested length, and
scores each length as its own metric within the one task result. So one job, given
several seqlens, covers all of them -- there is no need to submit one job per length.

    python eval_ruler.py --quantizer nvfp4 --run-name qad3x-Qwen-Qwen3-4B --iter 2450 \\
        --tokenizer Qwen/Qwen3-4B --seqlens 8192,16384,32768 --think

RULER's own per-task generation budget (128 tokens, set in every task's YAML) is left
alone here, unlike run_lm_eval's gsm8k/mmlu_pro path which overrides it -- overriding
would fight the benchmark's own spec rather than fix a genuine cross-task inconsistency.

COST, READ BEFORE SCALING THIS UP: num_samples=500 per requested length is fixed by
lm-eval's RULER tasks and not a flag here. 13 tasks x 500 docs x however many lengths
you pass is the real unit of work; --limit takes the first N docs of whatever the
combined (length-concatenated) dataset happens to be, NOT N per length, so it is not a
safe way to get balanced per-length coverage on a budget -- use it only for a
same-length smoke test.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_QAD = Path(__file__).resolve().parent.parent   # eval/ -> qad
sys.path.insert(0, str(_QAD))

from quantizers import REGISTRY as _REGISTRY, build_quantizer_params as _build_quant_params
from quantizers.full_disag import full_disag_hash as _full_disag_hash

from eval_disagg import Stack, build_client, free_port_base, resolve_pair, log

# The 13 tasks the RULER paper reports; NOT lm-eval's "ruler" group -- that group's
# aggregate_metric_list only tracks the "4096" metric (its hardcoded default length),
# which is meaningless for the custom lengths this sweeps. The per-task results below
# still carry a correct value per requested length; only the group-level rollup would
# have been junk.
RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_vt", "ruler_cwe", "ruler_fwe",
    "ruler_qa_squad", "ruler_qa_hotpot",
]

# RULER's own per-task YAML: generation_kwargs.max_gen_toks. Not a CLI knob upstream;
# duplicated here only so this file can size --max-model-len without importing YAML.
RULER_GEN_TOKS = 128


def _model_max_context(model_id: str) -> int | None:
    """The model's own ceiling (max_position_embeddings), or None if unresolvable.

    Needed to CLAMP the auto --max-model-len: run_eval_ruler.sh's seqlen sweep goes
    up to and including the model's own max context (by design, that is the point of
    RULER), so max(seqlens) + margin routinely lands PAST what vLLM will serve --
    vLLM hard-refuses (ModelConfig ValidationError: "User-specified max_model_len is
    greater than the derived max_model_len") rather than silently extrapolating RoPE
    positions, which is the right call on vLLM's part (it can produce NaN). Confirmed
    against a live run: gemma-3-270m/1b-it (ceiling 32768) and 4b/12b-it (ceiling
    131072) all crashed here; Qwen3 did not only because its sweep top (32768) sits
    well under its own ceiling (40960), not because the underlying math is fine.
    """
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception:
        return None
    tc = getattr(cfg, "text_config", cfg)
    # NOT getattr(tc, 'x', getattr(cfg, 'x')) -- see run_eval_ruler.sh's identical
    # fix: that inner getattr is an ordinary argument, evaluated eagerly even when
    # tc already has the attribute, and raises for any wrapper whose OWN config
    # lacks it (Gemma-3 4b/12b's Gemma3Config: max_position_embeddings lives only
    # on .text_config).
    mpe = getattr(tc, "max_position_embeddings", None)
    if mpe is None:
        mpe = getattr(cfg, "max_position_embeddings", None)
    return int(mpe) if mpe else None


def _patch_datasets_fingerprint() -> None:
    """Work around a dill/pyarrow incompatibility hit only by RULER's synthetic
    tasks: niah_utils.download_dataset() builds an in-memory Dataset via
    Dataset.from_list(), and unlike loading a pre-existing HF dataset (which
    ships its own fingerprint), that always calls generate_fingerprint() at
    construction time, which dill-hashes the whole Dataset, pyarrow Table
    included. On this stack's numpy/pyarrow versions that hash crashes even
    on a trivial 1-row dataset:
        PicklingError: Can't pickle <class 'MonthDayNano'>: it's not found as
        builtins.MonthDayNano
    -- pyarrow's compiled MonthDayNano reports __module__="builtins" (wrong,
    and the class is immutable so that attribute can't just be corrected),
    which breaks dill's save_global lookup and is unrelated to anything
    RULER-specific or to this dataset's actual content.

    generate_fingerprint has no fallback for this. update_fingerprint (used by
    .map()/.filter() instead of __init__) already falls back to
    generate_random_fingerprint() on exactly this class of failure --
    datasets/fingerprint.py's own comment there: "various errors might raise
    here from pickle or dill". Mirror that here: random is fine, since RULER
    regenerates its synthetic dataset fresh every run, so there is no
    cross-run cache identity to lose.
    """
    import datasets.arrow_dataset as _ad
    import datasets.fingerprint as _fp

    _orig = _fp.generate_fingerprint

    def _safe_generate_fingerprint(dataset):
        try:
            return _orig(dataset)
        except Exception:
            return _fp.generate_random_fingerprint()

    _fp.generate_fingerprint = _safe_generate_fingerprint
    _ad.generate_fingerprint = _safe_generate_fingerprint  # `from .fingerprint import
    # generate_fingerprint` bound its own name at import time; patching the
    # source module alone would not reach arrow_dataset's call site.


def _patch_ruler_vt_generate_chains() -> None:
    """Fix a real bug in lm-eval's vendored RULER 'variable tracking' task
    (lm_eval/tasks/ruler/vt_utils.py:generate_chains). Its padding loop grows
    vars_all one string at a time until there are enough UNIQUE names, but never
    checks that len(vars_all) stays a multiple of (num_hops + 1) -- the very next
    line chops it into fixed-size (num_hops + 1) chunks, so whenever padding was
    needed the LAST chunk comes up short and this_vars[j + 1] indexes past the end:
        IndexError: list index out of range
    Confirmed live: ruler_vt at seqlen 65536 (num_chains/num_hops scale with the
    requested length, raising the birthday-paradox collision odds enough to
    trigger padding) crashed gemma-3-4b/12b-it's entire RULER run with zero
    results written for ANY task -- TaskManager.load() builds every task's
    dataset up front, so one task's crash takes the whole job down for any task
    list that includes ruler_vt (lm-eval's own default). The shorter seqlens
    already evaluated apparently never needed padding, hence never hit this.

    Fix: generate directly into a set until it holds exactly the needed count of
    unique names, so the chunking below is always exact. Same output distribution
    (uniform random k-letter strings, rejecting duplicates) as the original.

    WHY A PLAIN ATTRIBUTE PATCH IS NOT ENOUGH. RULER's YAML tasks reference this
    function through a `custom_dataset: !function vt_utils.get_vt_dataset` tag, which
    lm-eval resolves via its own loader
    (lm_eval/tasks/_yaml_loader.py:_load_module_with_cache), NOT a plain `import`. That
    loader computes the SAME dotted name this patch uses ("lm_eval.tasks.ruler.vt_utils"),
    finds it already in sys.modules (from the `import` two lines down), but only trusts
    that cached copy if it carries a `__mtime__` attribute matching the file's current
    mtime -- a marker ONLY that loader ever sets, never a plain import. Without it, the
    loader re-execs vt_utils.py straight from disk and overwrites sys.modules with a
    fresh, UNPATCHED module right before calling generate_chains -- silently, no error,
    no trace of the patch ever having run. Confirmed live: two full job re-runs crashed
    with the exact original traceback despite this patch executing without error.
    Stamping __mtime__ here makes the loader's own freshness check pass, so it reuses
    THIS (patched) module instead of reloading.
    """
    import os
    import random
    import string

    import numpy as np
    import lm_eval.tasks.ruler.vt_utils as vt_utils

    def _fixed_generate_chains(num_chains, num_hops, is_icl=False):
        k = 5 if not is_icl else 3
        num_hops = num_hops if not is_icl else min(10, num_hops)
        need = num_chains * (num_hops + 1)
        pool = set()
        while len(pool) < need:
            pool.add("".join(random.choices(string.ascii_uppercase, k=k)).upper())
        vars_all = list(pool)

        vars_ret, chains_ret = [], []
        for i in range(0, len(vars_all), num_hops + 1):
            this_vars = vars_all[i : i + num_hops + 1]
            vars_ret.append(this_vars)
            this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]
            for j in range(num_hops):
                this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
            chains_ret.append(this_chain)
        return vars_ret, chains_ret

    vt_utils.generate_chains = _fixed_generate_chains
    vt_utils.__mtime__ = os.stat(vt_utils.__file__).st_mtime_ns


def run_ruler_eval(port: int, args) -> dict:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    _patch_datasets_fingerprint()
    _patch_ruler_vt_generate_chains()
    lm = build_client(port, args.tokenizer, args.concurrency, RULER_GEN_TOKS,
                      args.max_model_len, args.think)
    # "tokenizer" here is unrelated to build_client's -- RULER's synthetic tasks
    # generate their own haystacks and need a tokenizer to size them to each
    # target length (lm_eval/tasks/ruler/common_utils.py:resolve_tokenizer_name),
    # independent of whatever tokenizer the served client itself uses.
    # REAL lengths here -- these are the prompts RULER actually builds. The nominal
    # labels are put back on the metric keys at write time (see seqlen_alias).
    tm = TaskManager(metadata={"max_seq_lengths": args.seqlens_real,
                                "tokenizer": args.tokenizer})
    log(f"ruler: tasks={args.tasks} seqlens={args.seqlens_real} "
        f"(reported as {args.seqlens}) "
        f"concurrency={args.concurrency} max_model_len={args.max_model_len}")
    return evaluator.simple_evaluate(
        model=lm, tasks=args.tasks, task_manager=tm, limit=args.limit,
        apply_chat_template=True, num_fewshot=args.num_fewshot,
        log_samples=args.log_samples or bool(args.limit),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--unquantized", action="store_true",
                     help="serve the base BF16 model on both engines (no checkpoint)")
    src.add_argument("--quantizer", choices=list(_REGISTRY))
    src.add_argument("--step-dir", help="a dual checkpoint step dir holding prefill/ and decode/")
    src.add_argument("--prefill-model", help="explicit prefill model dir (with --decode-model)")
    p.add_argument("--decode-model")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=None, help="base HF model id; defaults to --tokenizer")
    p.add_argument("--quantizer-params", default="")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ckpt-dir", default=str(_QAD / "checkpoints"))
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--tasks", nargs="+", default=RULER_TASKS)
    p.add_argument("--seqlens", required=True,
                   help="comma-separated context lengths, e.g. 8192,16384,32768")
    p.add_argument("--log-samples", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="lm-eval's own doc cap -- see the module docstring's COST "
                        "note before using this for anything but a smoke test")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--port-base", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="served context window. Default: max(--seqlens) + generation "
                        "budget + margin, which is what RULER's own docs are sized "
                        "to; override only if you know the model needs less headroom "
                        "or the server needs more (e.g. chat-template overhead)")
    p.add_argument("--full-disag", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--output-dir", default=None,
                   help="default: results/ruler/think/ or results/ruler/nothink/")
    p.add_argument("--log-dir", default=None)
    args = p.parse_args()

    args.seqlens = sorted(int(s) for s in args.seqlens.split(","))

    # NOMINAL vs REAL prompt length.
    #
    # A requested length equal to the model's own ceiling cannot be served: vLLM needs
    # room above the prompt for the 128 generated tokens and the chat template, so
    # gemma-3-270m/1b (max_position_embeddings=32768 exactly) could never be evaluated
    # at "32K" at all -- run_eval_ruler.sh's sweep drops such a top value and those two
    # models stopped at 16384.
    #
    # So ask RULER for the largest 1024-aligned length that DOES fit -- 31744 for a
    # 32768 ceiling -- and report it as the nominal 32768. The alias is applied to the
    # metric keys and the seqlens stamp just before writing, so every consumer
    # (submit_missing_evals' coverage test, plots.ipynb, filenames) sees a clean 32768
    # and the 32K column lines up with the other models' real 32768.
    #
    # This is a 3% shorter prompt than the label claims. It is recorded in the result
    # JSON as ruler_seqlen_real so the approximation is never invisible to whoever reads
    # the numbers later.
    cap = _model_max_context(args.model or args.tokenizer)
    budget = RULER_GEN_TOKS + 512
    args.seqlen_alias = {}                      # real -> nominal
    reals = []
    for nominal in args.seqlens:
        if not cap or nominal + budget <= cap:
            reals.append(nominal)
            continue
        real = ((cap - budget) // 1024) * 1024
        if real <= 0 or real in reals:
            log(f"cannot fit seqlen {nominal} under ceiling {cap}; dropping it")
            continue
        args.seqlen_alias[real] = nominal
        reals.append(real)
        log(f"seqlen {nominal} exceeds what {args.model or args.tokenizer} can serve "
            f"(ceiling {cap}, needs {budget} for generation + template); "
            f"running at {real} and reporting it as {nominal}.")
    args.seqlens_real = sorted(reals)

    if args.max_model_len is None:
        # +margin: RULER targets max_seq_length for the PROMPT; the 128-token
        # generation and chat-template wrapping both add a little on top, and
        # coming up short truncates the needle out of the haystack rather than
        # erroring, which would silently corrupt scores instead of failing loudly.
        # from seqlens_REAL: the nominal 32768 of an aliased run would re-introduce the
        # very overflow the alias exists to avoid.
        args.max_model_len = max(args.seqlens_real) + RULER_GEN_TOKS + 512
        if cap and args.max_model_len > cap:
            # vLLM will not serve past the model's own ceiling at all (hard error,
            # not a truncation) -- clamping is strictly better than refusing to run.
            # It does eat into the margin above, though: with zero slack left, a
            # prompt that lands within ~RULER_GEN_TOKS of `cap` (chat-template
            # overhead on top of a max(seqlens) that IS `cap`) can still get
            # front-truncated by lm-eval's own client-side guard in build_client.
            # That is a real, small residual risk for whichever docs place the
            # needle in roughly the first ~1% of the top-length bucket -- logged
            # here rather than hidden, since silence is exactly what this margin
            # exists to avoid.
            log(f"max_model_len {args.max_model_len} exceeds {args.model or args.tokenizer}'s "
                f"own max_position_embeddings={cap}; clamping to {cap}. The top seqlen "
                f"({max(args.seqlens)}) now has little to no margin for chat-template "
                f"overhead -- pass --max-model-len explicitly to trade this off differently.")
            args.max_model_len = cap

    step_key = args.iter or 0
    if args.unquantized:
        base = args.model or args.tokenizer
        prefill = decode = Path(base)
        tag = args.tag or f"{base.replace('/', '-')}-unquantized"
        step_key = 0
    elif args.quantizer:
        if args.iter is None:
            p.error("--iter is required with --quantizer")
        base = args.model or args.tokenizer
        _, quant_hash = _build_quant_params(args.quantizer, args.quantizer_params)
        if args.full_disag:
            quant_hash = _full_disag_hash(quant_hash)
        run_name = args.run_name or f"qad-{base.replace('/', '-')}"
        tag_name = f"{run_name}-{args.quantizer}-{quant_hash}"
        prefill, decode = resolve_pair(Path(args.ckpt_dir), tag_name, args.iter)
        tag = args.tag or tag_name
    elif args.step_dir:
        step = Path(args.step_dir)
        prefill, decode = step / "prefill", step / "decode"
        for d in (prefill, decode):
            if not (d / "model.safetensors").exists():
                p.error(f"missing {d}/model.safetensors — is this a dual checkpoint?")
        tag = args.tag or f"{step.parent.parent.name}-{step.name}"
    else:
        if not args.decode_model:
            p.error("--decode-model is required with --prefill-model")
        prefill, decode = Path(args.prefill_model), Path(args.decode_model)
        tag = args.tag or f"{prefill.name}__{decode.name}"

    if args.port_base is None:
        args.port_base = free_port_base()
    log_dir = Path(args.log_dir) if args.log_dir else Path("/tmp") / f"ruler_{args.port_base}"

    with Stack(prefill, decode, args.tokenizer, args.port_base,
               args.max_model_len, log_dir) as stack:
        results = run_ruler_eval(stack.port, args)

    for task, metrics in results["results"].items():
        # RULER's per-length metrics report -1 for any length that was NOT
        # requested (see common_utils.aggregate_metrics) -- drop those here so the
        # printed line shows only what this run actually measured, not every
        # length RULER's YAML happens to declare a metric name for.
        shown = {k: v for k, v in metrics.items()
                if not k.endswith("_stderr") and v != -1}
        log(f"  {task}: {shown}")

    # A DIFFERENT tree from results/disagg/: RULER's result shape (per-length
    # metrics inside each task, not one score per task) is not what the
    # gsm8k/mmlu_pro plotting path expects, and submit_missing_evals.py's gap
    # detection only reads results/disagg/ -- landing here means it is never
    # mistaken for one of those gaps.
    default_root = _QAD / "results" / "ruler" / ("think" if args.think else "nothink")
    out_dir = Path(args.output_dir or default_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = results.pop("samples", None)
    if samples:
        for task, recs in samples.items():
            sp = out_dir / f"step_{step_key:07d}_samples_{task}.jsonl"
            with open(sp, "w") as f:
                for r in recs:
                    f.write(json.dumps({
                        "doc_id": r.get("doc_id"), "target": r.get("target"),
                        "resps": r.get("resps"), "filtered_resps": r.get("filtered_resps"),
                        "arguments": r.get("arguments"),
                    }, default=str) + "\n")
            log(f"samples -> {sp}  ({len(recs)} docs)")

    if getattr(args, "limit", None):
        results["limit"] = args.limit
    # Put the nominal labels back on: lm-eval keyed every metric by the REAL length it
    # generated ("31744,none"), and every consumer downstream expects the nominal one
    # ("32768,none"). Done here, on the way out, so the alias touches nothing else.
    if args.seqlen_alias:
        for entry in results.get("results", {}).values():
            if not isinstance(entry, dict):
                continue
            for real, nominal in args.seqlen_alias.items():
                for suffix in (",none", "_stderr,none"):
                    src = f"{real}{suffix}"
                    if src in entry:
                        entry[f"{nominal}{suffix}"] = entry.pop(src)
        # Keep the approximation visible to whoever reads the JSON later, rather than
        # letting "32768" quietly mean something 3% shorter with no record of it.
        results["ruler_seqlen_real"] = {str(v): k for k, v in args.seqlen_alias.items()}
    results["seqlens"] = args.seqlens
    out_path = out_dir / f"step_{step_key:07d}.json"
    if out_path.exists():
        # Merge, not clobber: a later job adding a new seqlen (or task) to an
        # already-evaluated step must not erase what an earlier job wrote -- same
        # reasoning as eval_disagg.py's merge, plus "seqlens" here.
        merged = json.loads(out_path.read_text())
        for key in ("results", "configs", "versions", "n-shot", "n-samples"):
            if key not in results:
                continue
            if not isinstance(merged.get(key), dict):
                merged[key] = results[key]
            elif key == "results":
                # PER-LENGTH merge, not per-task. lm-eval emits one column per SUPPORTED
                # length on every task -- {"4096,none": .., "8192,none": ..} -- and marks
                # the lengths this run did not request with -1. A plain dict.update()
                # here replaces the whole task entry, so a 4k-only backfill would
                # overwrite real 8k/16k/32k scores with those -1 placeholders and
                # silently destroy the results it was meant to extend.
                for task, entry in results["results"].items():
                    tgt = merged[key].get(task)
                    if not isinstance(tgt, dict) or not isinstance(entry, dict):
                        merged[key][task] = entry
                        continue
                    for mk, mv in entry.items():
                        # Never let a not-measured marker displace a real measurement.
                        # "N/A" is the stderr counterpart of -1.
                        if mv == -1 and tgt.get(mk, -1) != -1:
                            continue
                        if mv == "N/A" and tgt.get(mk, "N/A") != "N/A":
                            continue
                        tgt[mk] = mv
            else:
                merged[key].update(results[key])
        merged["seqlens"] = sorted(set(merged.get("seqlens", [])) | set(args.seqlens))
        # Carry the nominal->real note across too. The loop above copies a WHITELIST of
        # keys, so anything set on `results` outside it is dropped by `results = merged`
        # -- which is exactly what happened to the first 282 32K fills: the relabelled
        # 32768 scores landed correctly but the record of them really being 31744 did
        # not, leaving the approximation invisible in precisely the files that have it.
        if results.get("ruler_seqlen_real"):
            merged.setdefault("ruler_seqlen_real", {}).update(results["ruler_seqlen_real"])
        results = merged
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"results -> {out_path}")


if __name__ == "__main__":
    main()
