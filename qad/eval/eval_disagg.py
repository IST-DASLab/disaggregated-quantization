"""Evaluate a prefill/decode checkpoint pair under REAL disaggregated vLLM serving.

Two vLLM servers (prefill = W4A4, decode = W4A16) exchange KV over NixlConnector
behind nixl_proxy.py, and lm-eval drives the pair through one OpenAI endpoint. Unlike
eval_transformers.py --dual, which simulates the split inside one process, this is
the deployment itself: separate engines, separate weights, a real KV transfer.

    # real eval of a dual-trained checkpoint
    python eval_disagg.py --step-dir .../weights/step_0002450 \\
        --tokenizer Qwen/Qwen3-0.6B --tasks gsm8k

    # homogeneous baseline through the SAME stack (one model on both sides)
    python eval_disagg.py --prefill-model DIR --decode-model DIR ...

    # KV-transfer verification (see --probe and verify_kv_transfer.py)
    python eval_disagg.py --prefill-model A --decode-model B --probe out.json

THE FAILURE THAT MATTERS
------------------------
If the KV never crosses, the decode server silently recomputes the prompt with its
own weights and returns a fluent answer, so the run looks healthy while measuring
homogeneous-decode. Nothing here can detect that from a single number — use
verify_kv_transfer.py, which compares a disaggregated pair against the homogeneous
decode-only stack and requires them to DIFFER.

Two non-default settings are required on this cluster and are applied by
run_nixl_server.sh; see DISAGG.md for the evidence behind each:
  * kv_buffer_device=cpu   — the container's UCX is built without CUDA support, so
    registering VRAM fails at engine init on every node.
  * enforce_handshake_compat=false — Nixl's compatibility hash includes the model
    PATH, so a W4A4/W4A16 pair is rejected even though every factor that determines
    KV geometry is identical. The runtime TP/block-size/layout checks still apply.

Numbers from this harness are NOT comparable to single-server vLLM numbers: prefill
runs on a separate engine, and greedy decoding flips on near-ties, so text differs
even for an identical model. Compare disaggregated against disaggregated.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_QAD = Path(__file__).resolve().parent.parent   # eval/ -> qad
sys.path.insert(0, str(_QAD))

# Same registry/hash helpers eval_vllm.py uses, so a checkpoint tag means exactly the
# same thing on both backends and results line up tag-for-tag.
from quantizers import REGISTRY as _REGISTRY, build_quantizer_params as _build_quant_params

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


PROBE_PROMPTS = [
    "Natalia sold clips to 48 friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "What is 17 multiplied by 23? Answer with just the number.",
    "A train travels 60 miles in 1.5 hours. What is its average speed in mph?",
    "If a shirt costs $25 and is discounted by 20%, what is the sale price?",
]


def free_ports(n: int) -> list:
    """Bind n ephemeral ports, then release them and hand back the numbers.

    Arithmetic port schemes collide on a shared node, and a collision on a KV port
    surfaces deep inside the engine as `ZMQError: Address already in use`, which
    looks like a vLLM fault rather than a scheduling one. Asking the OS is the only
    reliable way. There is still a race between release and rebind, but it is
    seconds wide and vastly narrower than picking fixed numbers.
    """
    import socket
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 0.0.0.0, not loopback: the KV ports get bound on the node's interface
        # address, and a port free on 127.0.0.1 can still be taken there.
        s.bind(("0.0.0.0", 0))
        ports.append(s.getsockname()[1])
        socks.append(s)
    for s in socks:
        s.close()
    return ports


def resolve_pair(ckpt_dir: Path, ckpt_tag: str, step: int):
    """Map a checkpoint tag + step onto (prefill_dir, decode_dir).

    Dual-format checkpoints export `prefill/` and `decode/` side by side; every other
    method exports one directory that both engines serve. Returning a pair for BOTH
    cases is what lets one eval path cover the whole sweep -- a homogeneous checkpoint
    is just the degenerate case where the two halves are the same model.
    """
    root = ckpt_dir / ckpt_tag / "weights" / f"step_{step:07d}"
    pre, dec = root / "prefill", root / "decode"
    if (pre / "model.safetensors").exists() and (dec / "model.safetensors").exists():
        return pre, dec
    if (root / "model.safetensors").exists():
        return root, root
    raise FileNotFoundError(
        f"No checkpoint at {root} (looked for model.safetensors, or prefill/+decode/)")


class Stack:
    """Runs run_nixl_server.sh and tears the whole process group down after."""

    def __init__(self, prefill: Path, decode: Path, tokenizer: str, port_base: int,
                 max_model_len: int, log_dir: Path):
        self.prefill, self.decode, self.tokenizer = prefill, decode, tokenizer
        self.port_base, self.max_model_len = port_base, max_model_len
        self.log_dir = log_dir
        self.proc = None
        self.port = None

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ready = self.log_dir / "ready"
        ready.unlink(missing_ok=True)
        # Keep the lm-eval overlay OFF the servers' PYTHONPATH. It ships its own
        # huggingface-hub (1.24.0), which shadows the container's and makes vLLM
        # refuse to start: "huggingface-hub>=0.34.0,<1.0 is required ... found
        # 1.24.0". Only this process needs lm-eval; the engines must see the
        # container's environment exactly as built.
        clean_path = os.pathsep.join(
            d for d in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if d and "lm_eval_overlay" not in d)
        pf, dc, px, psc, dsc = free_ports(5)
        env = {**os.environ, "LOG_DIR": str(self.log_dir), "PYTHONPATH": clean_path,
               "PREFILL_PORT": str(pf), "DECODE_PORT": str(dc), "PROXY_PORT": str(px),
               # Nixl's handshake side channel, one unique port per worker.
               "PREFILL_SIDE_PORT": str(psc), "DECODE_SIDE_PORT": str(dsc)}
        # Nixl's compatibility hash includes the model PATH, so any two different
        # checkpoint dirs are rejected even when every factor that determines KV
        # geometry matches. Open the connector's documented hatch only for a genuinely
        # mixed pair, so an accidental mismatch in a homogeneous run still fails loudly.
        if self.prefill != self.decode:
            env["NIXL_ENFORCE_COMPAT"] = "0"
            log("heterogeneous pair -> handshake compat check disabled")
        log(f"server PYTHONPATH={clean_path or '(empty)'}")
        log(f"ports: prefill={pf} decode={dc} proxy={px} side={psc}/{dsc}")
        cmd = [str(_QAD / "serving" / "run_nixl_server.sh"),
               "--prefill-model", str(self.prefill),
               "--decode-model", str(self.decode),
               "--tokenizer", self.tokenizer,
               "--port-base", str(self.port_base),
               "--max-model-len", str(self.max_model_len),
               "--ready-file", str(ready)]
        log(f"starting stack: prefill={self.prefill.name} decode={self.decode.name}")
        # New process group: vllm spawns children, and killing only the shell would
        # leave two servers holding GPUs for the rest of the allocation.
        self.proc = subprocess.Popen(cmd, env=env, start_new_session=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        deadline = time.time() + 1800
        while time.time() < deadline:
            if ready.exists():
                self.port = int(ready.read_text().strip())
                log(f"stack ready on proxy port {self.port}")
                return self
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"server stack exited early:\n{out[-3000:]}")
            time.sleep(2)
        raise TimeoutError("server stack did not become ready within 1800s")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            log("tearing down stack")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=120)
            except Exception:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        return False


def probe(port: int, tokenizer_id: str, out_path: Path, max_tokens: int = 64,
          limit: int | None = None) -> list:
    """Greedy-complete a few fixed prompts; used to compare two stacks token for token."""
    import requests
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    rows = []
    prompts = PROBE_PROMPTS[:limit] if limit else PROBE_PROMPTS
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        r = requests.post(f"http://127.0.0.1:{port}/v1/completions",
                          json={"model": "model", "prompt": text,
                                "max_tokens": max_tokens, "temperature": 0, "seed": 0},
                          timeout=420)
        r.raise_for_status()
        rows.append({"prompt": p, "completion": r.json()["choices"][0]["text"]})
        log(f"  probe ok ({len(rows)}/{len(prompts)})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    log(f"probe written -> {out_path}")
    return rows


# Client request timeout, in seconds. lm-eval defaults this to 300 and it is an
# aiohttp ClientTimeout(total=...) covering the WHOLE generation, not just connecting.
# At concurrency 512 a queued request passes 300s routinely, so the client abandons
# healthy work, retries, and feeds the queue it was already stuck behind. That is not
# merely slow: once the retries are exhausted lm-eval tears down the session and the
# run dies with "RuntimeError: Session is closed". It killed 21 mmlu_pro think jobs at
# ~80% completion, each writing nothing after nearly two hours.
# A timeout is here to notice a DEAD server; the 4h wall and reap_stalled.sh already
# bound a hung one.
API_TIMEOUT = 3600


def run_lm_eval(port: int, args) -> dict:
    from lm_eval import evaluator
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    # lm-eval's max_length must cover prompt + generation, and it defaults to 2048
    # (then minus 1 internally). The generate path computes
    #     max_context_len = self.max_length - max_gen_toks
    #     encodings_list  = [x[-max_context_len:] for x in encodings_list]
    # so once max_gen_toks exceeds max_length the value goes NEGATIVE and
    # `x[-negative:]` slices from the FRONT, silently sending an empty prompt. vLLM
    # then answers "please provide at least one prompt" with no hint of the cause.
    # Tie it to the served context window instead, and refuse the impossible case up
    # front rather than emitting empty prompts.
    # NOTE: this guards the CLI value; per-task YAML budgets are neutralised by the
    # gen_kwargs override below, which is what actually keeps max_context_len positive.
    if args.max_gen_toks >= args.max_model_len:
        raise SystemExit(
            f"--max-gen-toks {args.max_gen_toks} must be < --max-model-len "
            f"{args.max_model_len}; otherwise lm-eval truncates every prompt to empty "
            f"and the servers reject the request.")
    lm = LocalCompletionsAPI(
        base_url=f"http://127.0.0.1:{port}/v1/completions",
        model="model",
        tokenizer=args.tokenizer,
        tokenizer_backend="huggingface",
        num_concurrent=args.concurrency,
        max_gen_toks=args.max_gen_toks,
        max_length=args.max_model_len,
        # MUST be True. lm-eval only renders the chat template to text when
        # `tokenizer_backend == "huggingface" AND tokenized_requests` (see
        # TemplateAPI.apply_chat_template); otherwise it returns the raw message list
        # as a JsonChatStr, which LocalCompletionsAPI puts straight into `prompt` and
        # vLLM rejects with
        #   400 {'loc': ('body','prompt','list[int]',0), 'msg': 'Input should be a
        #        valid integer', 'input': {'role': 'user', ...}}
        # Keeping the template client-side also keeps --no-think under our control:
        # the patch below wraps this tokenizer, whereas server-side templating would
        # need chat_template_kwargs the API model does not send.
        tokenized_requests=True,
        max_retries=3,
        # lm-eval's default is 300s, and it is an aiohttp ClientTimeout(total=...) --
        # it covers the whole generation, not just connecting. At concurrency 512 a
        # request sitting behind 511 others exceeds it routinely, so the client times
        # out on a perfectly healthy server, retries, and adds its work back to the
        # very queue that was already too long. Measured on mmlu_pro: 1000-3000
        # timeouts per job, ~20-35% of requests, every one of them a completed
        # generation thrown away and redone.
        # The timeout exists to notice a DEAD server; the wall clock and reap_stalled.sh
        # already bound a hung one, so make it long enough not to fire on a busy queue.
        timeout=API_TIMEOUT,
    )
    # A kwarg accepted is not a kwarg applied: if lm-eval renames or drops `timeout`,
    # we silently inherit its 300s default and jobs start dying at ~80% again with no
    # signal. Same reasoning as the --no-think probe below -- check the built object.
    if getattr(lm, "timeout", None) != API_TIMEOUT:
        raise SystemExit(
            f"lm-eval client timeout is {getattr(lm, 'timeout', None)!r}, expected "
            f"{API_TIMEOUT}. The `timeout` kwarg was not honoured; at the 300s default "
            f"long generations exhaust the retries and the run dies with "
            f"'RuntimeError: Session is closed' after hours of work.")
    log(f"lm-eval client: timeout={lm.timeout}s concurrency={args.concurrency} "
        f"max_retries=3 max_gen_toks={args.max_gen_toks}")

    if not args.think:
        # Same patch as eval_transformers.py: Qwen3 suppresses thinking through a
        # chat-template kwarg, not generation config, so it must be injected where
        # lm-eval renders the template.
        orig = lm.tokenizer.apply_chat_template

        def _no_think(conversation, **kw):
            # FORCE, do not setdefault: the setdefault form silently failed in
            # eval_vllm.py and mislabelled a whole results tree as --no-think while it
            # was thinking-enabled throughout.
            kw["enable_thinking"] = False
            return orig(conversation, **kw)

        lm.tokenizer.apply_chat_template = _no_think
        # Verify on a real render rather than trusting the assignment: suppression must
        # put an empty <think></think> block in the prompt.
        probe = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
        if "<think>" not in probe:
            raise SystemExit("--no-think requested but the rendered prompt carries no "
                             "<think></think> block; thinking is NOT suppressed.")
        log("thinking suppressed (verified in rendered prompt)")
    else:
        log("thinking ENABLED (default)")

    log(f"lm_eval: tasks={args.tasks} concurrency={args.concurrency} "
        f"max_gen_toks={args.max_gen_toks}")
    return evaluator.simple_evaluate(
        model=lm, tasks=args.tasks, limit=args.limit,
        apply_chat_template=True, num_fewshot=args.num_fewshot,
        # MUST pass this, exactly as eval_vllm.py does. Task YAMLs carry their own
        # max_gen_toks -- aime25 asks for 32768 -- and lm-eval then computes
        #     max_context_len = max_length - max_gen_toks
        # which goes NEGATIVE, turning `x[-max_context_len:]` into `x[+n:]` and
        # truncating every prompt to EMPTY. The server answers 400 "please provide at
        # least one prompt", lm-eval raises, and the whole run dies AFTER generating --
        # losing every completed task. Overriding here pins all tasks to one budget and
        # keeps the numbers comparable to the single-server backend.
        gen_kwargs=f"max_gen_toks={args.max_gen_toks}",
        # On a small debug run keep the raw generations: a score of 0.0 is ambiguous
        # between "model got them wrong" and "the harness produced empty/garbled text",
        # and only the samples distinguish those.
        log_samples=args.log_samples or bool(args.limit),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--unquantized", action="store_true",
                     help="serve the base BF16 model on both engines (no checkpoint); "
                          "tagged <model>-unquantized at step 0, matching eval_vllm.py")
    src.add_argument("--quantizer", choices=list(_REGISTRY),
                     help="resolve the checkpoint the same way eval_vllm.py does, from "
                          "--run-name/--quantizer/--iter. This is the first-class path.")
    src.add_argument("--step-dir", help="a dual checkpoint step dir holding prefill/ and decode/")
    src.add_argument("--prefill-model", help="explicit prefill model dir (with --decode-model)")
    p.add_argument("--decode-model", help="explicit decode model dir")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=None,
                   help="base HF model id; defaults to --tokenizer (used to build the tag)")
    p.add_argument("--quantizer-params", default="")
    p.add_argument("--run-name", default=None)
    p.add_argument("--ckpt-dir", default=str(_QAD / "checkpoints"))
    p.add_argument("--iter", type=int, default=None)
    # Same default task list as eval_vllm.py so the two backends stay comparable.
    p.add_argument("--tasks", nargs="+",
                   default=["gsm8k", "minerva_math500", "aime25"])
    p.add_argument("--log-samples", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--max-gen-toks", type=int, default=512,
                   help="generation cap; part of the measurement, so keep it fixed "
                        "across configurations being compared")
    p.add_argument("--concurrency", type=int, default=32,
                   help="in-flight requests to the proxy; this is the throughput knob "
                        "here, since batching happens server-side")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=True,
                   help="Qwen3 thinking mode. ON by default, matching "
                        "results/vllm/think/. Pass --no-think to suppress it; the "
                        "same model differs by ~20 points on GSM8K between the modes.")
    p.add_argument("--port-base", type=int, default=None,
                   help="default derives from SLURM_ARRAY_TASK_ID so tasks sharing a "
                        "node do not collide on ports")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="served context window, and lm-eval's max_length. Must exceed "
                        "--max-gen-toks with room for the prompt")
    p.add_argument("--probe", help="write greedy completions for fixed prompts here "
                                   "instead of running lm-eval (KV verification)")
    p.add_argument("--probe-limit", type=int, default=None,
                   help="use only the first N probe prompts; 1 keeps DISAGG_DEBUG "
                        "traces small enough to read")
    p.add_argument("--tag", default=None, help="results subdirectory name")
    p.add_argument("--output-dir", default=None,
                   help="default: results/disagg/think/ or results/disagg/nothink/ per --think")
    p.add_argument("--log-dir", default=None)
    args = p.parse_args()

    step_key = args.iter or 0
    if args.unquantized:
        # Both engines serve the same base model straight from the hub id. The pair is
        # identical, so the Nixl compat hash stays ENABLED here -- a mismatch would be a
        # real bug rather than the expected weight difference of a dual checkpoint.
        base = args.model or args.tokenizer
        prefill = decode = Path(base)
        tag = args.tag or f"{base.replace('/', '-')}-unquantized"
        step_key = 0
    elif args.quantizer:
        if args.iter is None:
            p.error("--iter is required with --quantizer")
        base = args.model or args.tokenizer
        _, quant_hash = _build_quant_params(args.quantizer, args.quantizer_params)
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
        args.port_base = 8500 + 10 * (int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) % 200)
    log_dir = Path(args.log_dir) if args.log_dir else Path("/tmp") / f"disagg_{args.port_base}"

    with Stack(prefill, decode, args.tokenizer, args.port_base,
               args.max_model_len, log_dir) as stack:
        if args.probe:
            probe(stack.port, args.tokenizer, Path(args.probe), limit=args.probe_limit)
            return
        results = run_lm_eval(stack.port, args)

    for task, metrics in results["results"].items():
        log(f"  {task}: { {k: v for k, v in metrics.items() if not k.endswith('_stderr')} }")

    # Same layout as eval_vllm.py -- <root>/<tag>/step_%07d.json -- so the plotting
    # notebook reads disaggregated results with no special cases, and thinking mode
    # picks the root so the two can never be mixed.
    default_root = _QAD / "results" / "disagg" / ("think" if args.think else "nothink")
    out_dir = Path(args.output_dir or default_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = results.pop("samples", None)
    if samples:
        for task, recs in samples.items():
            sp = out_dir / f"step_{step_key:07d}_samples_{task}.jsonl"
            with open(sp, "w") as f:
                for r in recs:
                    f.write(json.dumps({
                        "doc_id": r.get("doc_id"),
                        "target": r.get("target"),
                        "resps": r.get("resps"),
                        "filtered_resps": r.get("filtered_resps"),
                        "exact_match": r.get("exact_match"),
                        "arguments": r.get("arguments"),
                    }, default=str) + "\n")
            log(f"samples -> {sp}  ({len(recs)} docs)")
            for r in recs[:2]:
                log(f"  resp={str(r.get('resps'))[:200]!r}")

    # Stamp the limit into the artifact. n-samples alone is easy to misread; an
    # explicit field means a subset run can never be mistaken for a full sweep.
    if getattr(args, "limit", None):
        results["limit"] = args.limit
    out_path = out_dir / f"step_{step_key:07d}.json"
    # Merge rather than clobber: one step is evaluated across several task groups in
    # separate jobs, exactly as eval_vllm.py does it.
    if out_path.exists():
        merged = json.loads(out_path.read_text())
        # n-samples MUST be merged too. It was omitted, so a merged file kept the
        # PREVIOUS run's counts: a --limit 25 calibration run (350 docs) landed in the
        # results tree carrying n-samples that said 1819 docs from an older gsm8k pass,
        # making a limited run indistinguishable from a full one. That is a silent
        # correctness trap for anything that reads coverage off a result file.
        for key in ("results", "configs", "versions", "n-shot", "n-samples"):
            if key in results:
                if isinstance(merged.get(key), dict):
                    merged[key].update(results[key])
                else:
                    merged[key] = results[key]
        results = merged
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"results -> {out_path}")


if __name__ == "__main__":
    main()
