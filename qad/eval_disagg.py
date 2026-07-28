"""Evaluate a prefill/decode checkpoint pair under REAL disaggregated vLLM serving.

Two vLLM servers (prefill = W4A4, decode = W4A16) exchange KV over P2pNcclConnector
behind p2p_proxy.py, and lm-eval drives the pair through one OpenAI endpoint. Unlike
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
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_QAD = Path(__file__).resolve().parent
sys.path.insert(0, str(_QAD))

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


class Stack:
    """Runs run_disagg_server.sh and tears the whole process group down after."""

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
        pf, dc, px, pkv, dkv = free_ports(5)
        env = {**os.environ, "LOG_DIR": str(self.log_dir), "PYTHONPATH": clean_path,
               "PREFILL_PORT": str(pf), "DECODE_PORT": str(dc), "PROXY_PORT": str(px),
               "PREFILL_KV_PORT": str(pkv), "DECODE_KV_PORT": str(dkv)}
        log(f"server PYTHONPATH={clean_path or '(empty)'}")
        log(f"ports: prefill={pf} decode={dc} proxy={px} kv={pkv}/{dkv}")
        cmd = [str(_QAD / "run_disagg_server.sh"),
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


def run_lm_eval(port: int, args) -> dict:
    from lm_eval import evaluator
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    lm = LocalCompletionsAPI(
        base_url=f"http://127.0.0.1:{port}/v1/completions",
        model="model",
        tokenizer=args.tokenizer,
        tokenizer_backend="huggingface",
        num_concurrent=args.concurrency,
        max_gen_toks=args.max_gen_toks,
        tokenized_requests=False,      # send text; the chat template is applied here
        max_retries=3,
    )
    if args.no_think:
        # Same patch as eval_transformers.py: Qwen3 suppresses thinking through a
        # chat-template kwarg, not generation config, so it must be injected where
        # lm-eval renders the template.
        orig = lm.tokenizer.apply_chat_template

        def _no_think(conversation, **kw):
            kw.setdefault("enable_thinking", False)
            return orig(conversation, **kw)

        lm.tokenizer.apply_chat_template = _no_think
        log("thinking suppressed via chat-template patch")

    log(f"lm_eval: tasks={args.tasks} concurrency={args.concurrency} "
        f"max_gen_toks={args.max_gen_toks}")
    return evaluator.simple_evaluate(
        model=lm, tasks=args.tasks, limit=args.limit,
        apply_chat_template=True, num_fewshot=args.num_fewshot,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--step-dir", help="a dual checkpoint step dir holding prefill/ and decode/")
    src.add_argument("--prefill-model", help="explicit prefill model dir (with --decode-model)")
    p.add_argument("--decode-model", help="explicit decode model dir")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--tasks", nargs="+", default=["gsm8k"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--max-gen-toks", type=int, default=512,
                   help="generation cap; part of the measurement, so keep it fixed "
                        "across configurations being compared")
    p.add_argument("--concurrency", type=int, default=32,
                   help="in-flight requests to the proxy; this is the throughput knob "
                        "here, since batching happens server-side")
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--port-base", type=int, default=None,
                   help="default derives from SLURM_ARRAY_TASK_ID so tasks sharing a "
                        "node do not collide on ports")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--probe", help="write greedy completions for fixed prompts here "
                                   "instead of running lm-eval (KV verification)")
    p.add_argument("--probe-limit", type=int, default=None,
                   help="use only the first N probe prompts; 1 keeps DISAGG_DEBUG "
                        "traces small enough to read")
    p.add_argument("--tag", default=None, help="results subdirectory name")
    p.add_argument("--output-dir", default=str(_QAD / "eval_results_disagg"))
    p.add_argument("--log-dir", default=None)
    args = p.parse_args()

    if args.step_dir:
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

    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results.pop("samples", None)
    out = out_dir / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    log(f"results -> {out}")


if __name__ == "__main__":
    main()
