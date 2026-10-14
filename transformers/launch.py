"""Bare-metal torchrun launcher with a host RAM limit for this job's processes."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--host-memory-limit-gib", type=float, default=768)
    args, training_args = parser.parse_known_args()
    # This is a single-host launcher. Avoid hostname/DNS selecting an
    # unreachable address for the worker bootstrap store.
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--local-addr=127.0.0.1",
               f"--nproc_per_node={args.nproc_per_node}", str(Path(__file__).with_name("train.py")),
               *training_args]
    # Prefer this environment's CUDA libraries over paths inherited from other
    # virtualenvs, especially cuDNN's dynamically loaded sublibraries.
    env = os.environ.copy()
    nvidia = Path(sysconfig.get_path("purelib")) / "nvidia"
    libraries = sorted(str(path) for path in nvidia.glob("*/lib") if path.is_dir())
    if libraries:
        env["LD_LIBRARY_PATH"] = ":".join([*libraries, env.get("LD_LIBRARY_PATH", "")])
    process = subprocess.Popen(command, start_new_session=True, env=env)
    root = psutil.Process(process.pid)
    peak = 0.0

    def stop(signum=signal.SIGTERM):
        # Elastic workers may create their own process groups. Snapshot them
        # before stopping torchrun so a forced stop cannot leave orphan workers.
        try:
            workers = root.children(recursive=True)
        except psutil.NoSuchProcess:
            workers = []
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        for worker in workers:
            try:
                worker.send_signal(signum)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(workers, timeout=10)
        for worker in alive:
            try:
                worker.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def interrupted(signum, frame):
        stop(signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    while process.poll() is None:
        rss = 0
        try:
            for child in [root, *root.children(recursive=True)]:
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            break
        gib = rss / 1024**3
        peak = max(peak, gib)
        available = psutil.virtual_memory().available / 1024**3
        reserve = min(256, psutil.virtual_memory().total / 1024**3 / 8)
        if gib > args.host_memory_limit_gib or available < reserve:
            print(f"Stopping this training job: RSS={gib:.1f} GiB, available={available:.1f} GiB; "
                  f"limit={args.host_memory_limit_gib:.1f} GiB, reserve={reserve:.1f} GiB", file=sys.stderr, flush=True)
            stop()
            return 1
        time.sleep(1)
    result = process.wait()
    print(f"Training process peak aggregate RSS: {peak:.1f} GiB", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
