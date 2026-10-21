"""Load a local export with stock vLLM and exercise real HTTP generation."""
import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = json.loads((args.model / "config.json").read_text())
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(args.model.resolve()),
               "--served-model-name", "prefiller-check", "--host", "127.0.0.1", "--port", str(port),
               "--dtype", "bfloat16", "--max-model-len", "4096", "--max-num-seqs", "2",
               "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.50",
               "--attention-backend", "FLASH_ATTN",
               "--mamba-ssm-cache-dtype", "float32", "--enforce-eager"]
    env = os.environ | {"CUDA_VISIBLE_DEVICES": args.gpu, "OMP_NUM_THREADS": "4",
                       "VLLM_USE_AOT_COMPILE": "0", "VLLM_HOST_IP": "127.0.0.1"}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    import importlib.metadata
    site = Path(importlib.metadata.distribution("vllm").locate_file(""))
    env["LD_LIBRARY_PATH"] = ":".join([*(str(p) for p in (site / "nvidia").glob("*/lib")),
                                         env.get("LD_LIBRARY_PATH", "")])
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        def stop(*_):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        def interrupted(*_):
            stop()
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Stock vLLM exited {process.returncode}; see {args.output / 'server.log'}")
                try:
                    if requests.get(base + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Stock vLLM did not become ready within 20 minutes")
            requests_to_test = {"text": "Reply with the number 4."}
            if "vision_config" in config:
                from PIL import Image
                data = BytesIO()
                Image.new("RGB", (224, 224), "red").save(data, format="PNG")
                url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
                requests_to_test["image"] = [{"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "What is the solid color? Reply with one word."}]
            results = {}
            for name, content in requests_to_test.items():
                body = {"model": "prefiller-check", "messages": [{"role": "user", "content": content}],
                        "max_tokens": 32, "temperature": 0,
                        "chat_template_kwargs": {"enable_thinking": False}}
                response = requests.post(base + "/v1/chat/completions", json=body, timeout=180)
                if not response.ok:
                    raise RuntimeError(f"{name} generation failed: HTTP {response.status_code}: {response.text[:2000]}")
                result = response.json()
                if not result.get("choices") or result["usage"]["completion_tokens"] < 1:
                    raise ValueError(f"Empty generation: {name}")
                results[name] = result
            text = (args.output / "server.log").read_text()
            scales_preserved = "global scale is different" not in text
            report = {"stock_vllm": importlib.metadata.version("vllm"), "command": command,
                      "scope": "HTTP generation, native NVFP4 and global-scale preservation; not full-model numerical parity",
                      "model": str(args.model.resolve()), "responses": results,
                      "global_scales_preserved": scales_preserved,
                      "native_nvfp4_kernel_logged": "NvFp4LinearKernel" in text,
                      "faithfully_servable_as_is": scales_preserved and "NvFp4LinearKernel" in text}
            (args.output / "serving-check.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({key: value for key, value in report.items() if key != "responses"}, indent=2), flush=True)
            if not report["faithfully_servable_as_is"]:
                raise SystemExit("Stock vLLM did not preserve native NVFP4 scales; serving validation failed")
        finally:
            stop()


if __name__ == "__main__":
    main()
