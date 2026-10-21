"""Run matched single-decoder and real NVFP4-prefill/IQ1_S-decode evaluations.

Uses the existing benchmark drivers and NIXL proxy, without Slurm or containers.
One MMLU-Pro pass and four MMMU-Pro vision passes per arm by default.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from huggingface_hub.constants import HF_HOME
import requests

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evals"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--prefiller", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--run-id", default="iq1s-qad980-20261016")
    parser.add_argument("--port-base", type=int, default=8870)
    parser.add_argument("--gpus", default="0,1,2", help="baseline,prefill,decode")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--mmmu-repeats", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    model = json.loads((EVAL / "models.json").read_text())[args.model]
    decoder, prefiller = args.decoder.resolve(), args.prefiller.resolve()
    config = json.loads((prefiller / "config.json").read_text())
    contract = config.get("text_config", config)
    if contract.get("prefiller_handoff_boundary", "prompt-end") != "last-prompt-decode":
        raise ValueError("Prefiller was trained with a different handoff boundary: "
                         "hybrid NIXL evaluates the last prompt token on the decoder. "
                         "The legacy checkpoint must not be presented as equivalent.")
    gpus = args.gpus.split(",")
    assert len(gpus) == len(set(gpus)) == 3
    log = ROOT / "logs/evals" / ("local-" + args.run_id)
    log.mkdir(parents=True, exist_ok=True)
    ports = [args.port_base + n for n in (0, 1, 2, 3, 101, 102)]
    checks = []
    for port in ports:
        sock = socket.socket()
        sock.bind(("127.0.0.1", port))
        checks.append(sock)
    for sock in checks:
        sock.close()
    env = os.environ.copy()
    env.update(HF_HOME=HF_HOME, MMMU_HARNESS=str(EVAL / ".cache/MMMU/mmmu-pro"),
               VLLM_HOST_IP="127.0.0.1", VLLM_SSM_CONV_STATE_LAYOUT="DS",
               UCX_TLS="cuda_ipc,cuda_copy,sm,self,tcp", UCX_CUDA_IPC_ENABLE_MNNVL="n",
               OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", EVAL_MODEL=model["served_name"])
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    plugin_hash = hashlib.sha256(
        (EVAL / "serving/serve_trained_prefiller.py").read_bytes() +
        (ROOT / "transformers/serve_prefiller.py").read_bytes()).hexdigest()[:16]
    env["VLLM_CACHE_ROOT"] = str(EVAL / ".cache/vllm-trained" / plugin_hash)
    env["VLLM_USE_AOT_COMPILE"] = "0"
    # Resolve the venv site-packages rather than the base interpreter symlink.
    site = Path(importlib.metadata.distribution("vllm").locate_file(""))
    libs = sorted(str(p) for p in (site / "nvidia").glob("*/lib"))
    env["LD_LIBRARY_PATH"] = ":".join([*libs, env.get("LD_LIBRARY_PATH", "")])
    manifest = {"arguments": vars(args) | {"decoder": str(decoder), "prefiller": str(prefiller)},
                "scale_preserving_serving_sha256": plugin_hash,
                "model": model, "packages": {p: importlib.metadata.version(p)
                    for p in ("vllm", "torch", "transformers", "nixl", "datasets")}}
    (log / "manifest.json").write_text(json.dumps(manifest, indent=2))
    processes, handles = [], []
    stopping, process_lock = threading.Event(), threading.Lock()

    def start(command, output, overrides=None):
        with process_lock:
            if stopping.is_set():
                raise RuntimeError("Evaluation is stopping")
            process = subprocess.Popen(command, cwd=ROOT, env=env | (overrides or {}),
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
        return process

    def launch(command, name, overrides=None):
        handle = (log / name).open("a")
        handles.append(handle)
        return start(command, handle, overrides)

    def cleanup(*_):
        stopping.set()
        with process_lock:
            running = list(processes)
        for process in reversed(running):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 25
        for process in running:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()

    def interrupted(*_):
        # Stop client subprocesses before ThreadPoolExecutor waits for them.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    vllm = [sys.executable, str(EVAL / "serving/serve_trained_prefiller.py")]
    common = ["--served-model-name", model["served_name"], "--dtype", "bfloat16",
              "--tensor-parallel-size", "1", "--max-model-len", str(model["max_model_len"]),
              "--max-num-seqs", "128", "--gpu-memory-utilization", "0.90",
              "--attention-backend", "FLASH_ATTN", "--generation-config", "auto",
              "--mamba-ssm-cache-dtype", "float32",
              "--enable-request-id-headers", "--no-async-scheduling", "--host", "127.0.0.1",
              *model["serve_flags"]]

    def wait_ready(port):
        for _ in range(1800):
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Server {process.pid} exited {process.returncode}; see {log}")
            try:
                if requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"Server {port} not ready")

    def driver(bench, arm, port, repeat=0, smoke=False):
        tag = args.run_id + ("-decoder" if arm == "decoder" else "-nvfp4-prefill")
        if repeat:
            tag += f"-r{repeat}"
        if smoke:
            tag += "-smoke"
        out = EVAL / "results" / args.model / ("mmmu" if bench == "mmmu_pro" else bench) / tag
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(EVAL / "drivers" / f"{bench}_infer.py"),
                   "--out", str(out), "--base-url", f"http://127.0.0.1:{port}/v1",
                   "--model", model["served_name"], "--max-tokens", str(model["max_tokens"]),
                   "--workers", str(args.workers)]
        if bench == "mmmu_pro":
            command += ["--setting", "vision", "--mode", "cot"]
        if smoke:
            command += ["--limit", "14" if bench == "mmlu_pro" else "2"]
        with (log / f"{bench}-{arm}-r{repeat}{'-smoke' if smoke else ''}.log").open("a") as output:
            process = start(command, output)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
            if bench == "mmmu_pro":
                scoring = [sys.executable, str(EVAL / "drivers/mmmu_pro_score.py"), str(out)]
                process = start(scoring, output)
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, scoring)
        raw = out / "raw.jsonl"
        n = sum(1 for _ in raw.open()) if raw.exists() else 0
        expected = (14 if bench == "mmlu_pro" else 2) if smoke else (12032 if bench == "mmlu_pro" else 1730)
        if n != expected:
            raise RuntimeError(f"Incomplete {bench}/{tag}: {n}/{expected}; rerun to resume")
        print(f"Completed {bench} {tag}: {n} rows", flush=True)
        return out

    try:
        launch([*vllm, "serve", str(decoder), *common, "--port", str(ports[0])],
               "baseline.log", {"CUDA_VISIBLE_DEVICES": gpus[0]})
        for index, (weights, role, name) in enumerate(((prefiller, "kv_producer", "prefill"),
                                                      (decoder, "kv_consumer", "decode")), 1):
            config = {"kv_connector": "NixlConnector", "kv_role": role,
                      "kv_buffer_device": "cuda", "kv_load_failure_policy": "fail",
                      "kv_connector_extra_config": {"enforce_handshake_compat": False,
                                                    "kv_recompute_threshold": 0}}
            launch([*vllm, "serve", str(weights), *common, "--port", str(ports[index]),
                    "--kv-transfer-config", json.dumps(config)], name + ".log",
                   {"CUDA_VISIBLE_DEVICES": gpus[index],
                    "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": str(ports[3 + index])})
        for port in ports[:3]:
            wait_ready(port)
        launch([sys.executable, str(EVAL / "serving/nixl_proxy.py"), "--workers", "4",
                "--port", str(ports[3]), "--prefill-host", "127.0.0.1", "--prefill-port", str(ports[1]),
                "--decode-host", "127.0.0.1", "--decode-port", str(ports[2])], "proxy.log")
        wait_ready(ports[3])
        for name in ("baseline", "prefill", "decode"):
            text = (log / (name + ".log")).read_text()
            if "generation config" not in text.lower() and "generation_config" not in text:
                raise RuntimeError(f"No generation-config evidence in {name}.log")
        print("All engines ready; testing text, images and cross-engine state transfer", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(driver, bench, arm, port, smoke=True)
                     for bench in ("mmlu_pro", "mmmu_pro")
                     for arm, port in (("decoder", ports[0]), ("combo", ports[3]))]
            try:
                for task in tasks:
                    task.result()
            except BaseException:
                cleanup()
                raise
        time.sleep(15)
        subprocess.run([sys.executable, str(EVAL / "drivers/verify_disagg.py"), str(log),
                        "--out", str(log)], cwd=ROOT, env=env, check=True)
        print("Smoke and transfer verification passed", flush=True)
        if not args.smoke_only:
            def arm_bench(bench, arm, port):
                for repeat in range(args.mmmu_repeats if bench == "mmmu_pro" else 1):
                    driver(bench, arm, port, repeat)
            with ThreadPoolExecutor(max_workers=4) as pool:
                tasks = [pool.submit(arm_bench, bench, arm, port)
                         for bench in ("mmlu_pro", "mmmu_pro")
                         for arm, port in (("decoder", ports[0]), ("combo", ports[3]))]
                try:
                    for task in tasks:
                        task.result()
                except BaseException:
                    cleanup()
                    raise
            subprocess.run([sys.executable, str(EVAL / "drivers/verify_disagg.py"), str(log),
                            "--out", str(log)], cwd=ROOT, env=env, check=True)
        (log / "completed.json").write_text(json.dumps({"success": True, "smoke_only": args.smoke_only}))
    finally:
        cleanup()


if __name__ == "__main__":
    main()
