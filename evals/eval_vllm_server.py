"""
Evaluate a model already served over an OpenAI-compatible endpoint (`vllm serve`,
the prefill/decode proxy, ...) with lm-eval's `local-chat-completions` backend.

    python eval_vllm_server.py eval_server_config.yaml [--groups think instruct]

Tasks are organised into groups; each group is one `simple_evaluate` call, which
is what lets `apply_chat_template` / `gen_kwargs` differ per group (both are
global to a call). Top-level config keys are the defaults and each group
overrides them — nested dicts (`gen_kwargs`, `model_args`) merge key-wise.

Every config key is a `simple_evaluate` kwarg except `output_dir` / `run_name`;
results land in <output_dir>/<run_name>/<group>.json plus a combined
summary.json. JSON configs work too — JSON is valid YAML.
"""

import argparse
import copy
import inspect
import json
import traceback
from pathlib import Path

import yaml
from lm_eval import evaluator

DEFAULTS = {
    "model_args": {
        # full chat-completions route, not just the host
        "base_url": "http://127.0.0.1:8000/v1/chat/completions",
        "tokenized_requests": False,
        "num_concurrent": 32,
        "max_retries": 3,
        "timeout": 3600,
    },
    "gen_kwargs": {"max_gen_toks": 4096, "temperature": 0.0},
    "tasks": [],
    "apply_chat_template": True,
    "log_samples": False,
}
VALID = set(inspect.signature(evaluator.simple_evaluate).parameters) | {"name"}
SAMPLE_KEYS = ("doc_id", "target", "resps", "filtered_resps", "exact_match", "arguments")


def merge(base: dict, over: dict) -> dict:
    """Shallow merge, one level deep for nested dicts (gen_kwargs, model_args)."""
    out = dict(base)
    for k, v in over.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path) -> tuple[Path, list[dict]]:
    """Return (output dir, groups) — each group is a simple_evaluate kwarg dict."""
    cfg = yaml.safe_load(path.read_text())
    out_dir = Path(cfg.pop("output_dir", "eval_results_server"))
    run_name = cfg.pop("run_name", None)
    groups = cfg.pop("groups", None) or [{"name": "default"}]
    base = merge(DEFAULTS, cfg)
    assert base["model_args"].get("model"), "model_args.model (the served name) is required"

    resolved = []
    for i, group in enumerate(groups):
        # deepcopy: lm-eval pops from gen_kwargs/model_args, so groups must never
        # share a dict with the base config
        g = copy.deepcopy(merge(base, group))
        g.setdefault("name", f"group{i}")
        bad = sorted(set(g) - VALID)
        assert not bad, f"group {g['name']}: {bad} are not simple_evaluate kwargs (generation settings go under gen_kwargs)"
        assert g["tasks"], f"group {g['name']}: no tasks"
        resolved.append(g)
    return out_dir / (run_name or base["model_args"]["model"].replace("/", "-")), resolved


def main() -> None:
    p = argparse.ArgumentParser(description="lm-eval against an OpenAI-compatible server")
    p.add_argument("config", type=Path, help="YAML/JSON eval config")
    p.add_argument("--groups", nargs="+", help="run only these groups")
    args = p.parse_args()

    out_dir, groups = load_config(args.config)
    if args.groups:
        groups = [g for g in groups if g["name"] in args.groups]
        assert groups, f"no group matched {args.groups}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, failed = {}, []
    for group in groups:
        name = group.pop("name")
        print(f"\n=== {name}: {group['tasks']} gen_kwargs={group['gen_kwargs']}", flush=True)
        try:
            results = evaluator.simple_evaluate(model="local-chat-completions", **group)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue

        summary[name] = {t: {k: v for k, v in m.items() if not k.endswith("_stderr")} for t, m in results["results"].items()}
        print(json.dumps(summary[name], indent=2), flush=True)

        for task, recs in (results.pop("samples", None) or {}).items():
            (out_dir / f"{name}_samples_{task}.jsonl").write_text("".join(json.dumps({k: r.get(k) for k in SAMPLE_KEYS}, default=str) + "\n" for r in recs))
        (out_dir / f"{name}.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"  → {out_dir / f'{name}.json'}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary → {out_dir / 'summary.json'}", flush=True)
    if failed: raise SystemExit(f"failed groups: {failed}")


if __name__ == "__main__":
    main()
