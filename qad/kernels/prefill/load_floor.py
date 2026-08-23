"""Measured load-only floor: what it costs to stream each model in, with no compute at all.

    python load_floor.py                  # every (model, quant) in offload_prefill.csv
    python load_floor.py --reps 5 --threads 16

Why this exists. The floor drawn on the prefill figures used to be
`n_blocks x block_bytes / 5.7 GB/s`, where 5.7 GB/s came from block_stream.py. That rate
was measured on Qwen3-8B's **386 MB** blocks, and a drive does not deliver the same
throughput on a 30 MiB read: queue depth per read is far lower and the fixed per-read cost
is a much bigger share. Applying one rate to every model made the floor honest for 8B and
progressively optimistic as blocks shrink -- 0.6B's curve sat a long way above a floor that
was never reachable at that block size.

So the floor is now measured per (model, quant), at that configuration's actual block size,
through the same pipeline the real forward uses:

    SSD -> pinned host staging buffer -> GPU slot

double buffered on a copy stream, 16 parallel preads per block, page cache dropped before
every pass. The only thing removed is the compute. That makes it a true lower bound for
offload_forward.py's `ssd` mode: no forward can beat the time it takes to get the weights
across, and this measures exactly that, including the H2D leg and the per-block overheads
a closed-form estimate ignores.

Block files are the ones offload_forward.py already wrote, at the same paths, so the sizes
match the benchmark by construction rather than by recalculation. Missing files are created
with real bytes -- never `truncate`, since a sparse file costs no disk blocks and reads back
at memory speed, which would look like a spectacularly fast drive.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

CHUNK = 8 << 20
DEV = torch.device("cuda")


def drop_cache(paths):
    """Evict these files' clean pages. Per-file and unprivileged."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def ensure_blocks(directory: Path, n_blocks: int, nbytes: int):
    """The block files offload_forward.py uses, created with real bytes if absent."""
    directory.mkdir(parents=True, exist_ok=True)
    paths, filler = [], os.urandom(CHUNK)
    for i in range(n_blocks):
        path = directory / f"block_{i:03d}.bin"
        if not (path.exists() and path.stat().st_size == nbytes):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            remaining = nbytes
            while remaining:
                remaining -= os.write(fd, filler[:min(CHUNK, remaining)])
            os.fsync(fd)
            os.close(fd)
        paths.append(str(path))
    return paths


def stream(paths, nbytes, pool, threads, host, gpu, copy_stream, events):
    """One full pass: every block SSD -> pinned host -> GPU, double buffered, no compute.

    Mirrors SSDOffloadRunner.run with the compute removed, so the number it produces is
    directly the floor of that runner rather than an independent approximation of it.
    """
    span = -(-nbytes // threads)
    for i, path in enumerate(paths):
        s = i % len(gpu)
        events[s].synchronize()              # previous H2D out of this buffer is done
        view = memoryview(host[s].numpy())
        fd = os.open(path, os.O_RDONLY)
        try:
            def worker(lo):
                hi = min(lo + span, nbytes)
                while lo < hi:
                    got = os.preadv(fd, [view[lo:hi]], lo)
                    if not got:
                        raise EOFError(f"short read on {path} at {lo}")
                    lo += got
            list(pool.map(worker, range(0, nbytes, span)))
        finally:
            os.close(fd)
        with torch.cuda.stream(copy_stream):
            gpu[s].copy_(host[s], non_blocking=True)
        events[s].record(copy_stream)
    torch.cuda.synchronize()


def measure(directory: Path, n_blocks: int, nbytes: int, threads: int, reps: int,
            slots: int = 2):
    paths = ensure_blocks(directory, n_blocks, nbytes)
    host = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(slots)]
    gpu = [torch.empty(nbytes, dtype=torch.uint8, device=DEV) for _ in range(slots)]
    copy_stream = torch.cuda.Stream()
    events = [torch.cuda.Event() for _ in range(slots)]
    for e in events:
        e.record()

    ts = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(reps):
            drop_cache(paths)                # or we time memcpy, not the drive
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            stream(paths, nbytes, pool, threads, host, gpu, copy_stream, events)
            ts.append(time.perf_counter() - t0)
    del host, gpu
    torch.cuda.empty_cache()
    ts.sort()
    return ts[len(ts) // 2]


def params_b(model: str) -> float:
    return float(model.rsplit("-", 1)[1].removesuffix("B"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="offload_prefill.csv",
                    help="where the (model, quant, n_blocks, block_mb) shapes come from")
    ap.add_argument("--out", default="load_floor.csv")
    ap.add_argument("--threads", type=int, default=16,
                    help="parallel preads per block; matches offload_forward.py's default")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--block-dir", default=str(Path.home() / ".bench_offload_blocks"))
    args = ap.parse_args()

    here = Path(__file__).parent
    df = pd.read_csv(here / args.source)
    shapes = (df[["model", "quant", "n_blocks", "block_mb"]]
              .drop_duplicates()
              .sort_values(["quant", "model"], key=lambda c: c.map(
                  lambda v: params_b(v) if str(v).startswith("Qwen") else v)))

    print(f"{torch.cuda.get_device_name(0)}   {args.threads} threads, {args.reps} reps, "
          f"cold page cache\n")
    print(f"{'model':<12}{'quant':<7}{'blocks':>7}{'block MiB':>11}{'total GiB':>11}"
          f"{'load ms':>10}{'GB/s':>8}")

    rows = []
    for _, r in shapes.iterrows():
        model, quant = r["model"], r["quant"]
        n_blocks, block_mib = int(r["n_blocks"]), float(r["block_mb"])
        nbytes = int(round(block_mib * 2**20))
        directory = Path(args.block_dir) / model.replace("/", "_") / quant
        secs = measure(directory, n_blocks, nbytes, args.threads, args.reps)
        total_bytes = n_blocks * nbytes
        gbs = total_bytes / 1e9 / secs
        rows.append(dict(model=model, quant=quant, n_blocks=n_blocks,
                         block_mb=round(block_mib, 1),
                         total_gib=round(total_bytes / 2**30, 3),
                         load_ms=round(secs * 1e3, 3), gb_s=round(gbs, 2),
                         threads=args.threads))
        print(f"{model.split('/')[-1]:<12}{quant:<7}{n_blocks:>7}{block_mib:>11.0f}"
              f"{total_bytes/2**30:>11.2f}{secs*1e3:>10.1f}{gbs:>8.2f}", flush=True)

    out = here / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    print("the plots should read load_ms from this file rather than deriving a floor from "
          "one throughput figure")


if __name__ == "__main__":
    main()
