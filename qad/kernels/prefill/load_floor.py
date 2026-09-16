"""Measured load-only floors: what it costs to stream each model in, with no compute at all.

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
through the same pipeline the real forward uses. TWO pipelines, TWO floors, matching
offload_forward.py's two runners:

    ssd:  SSD -> pinned host staging buffer -> GPU slot, double buffered, 16 parallel
          preads per block, page cache dropped before every pass. True lower bound for
          `ssd` mode: no forward can beat the time it takes to get the weights across the
          drive.

    ram:  every block already resident in pinned host RAM (as RamOffloadRunner keeps it)
          -> GPU slot, double buffered. No disk anywhere in the timed loop -- this is the
          H2D-only floor, i.e. the interconnect ceiling `ram` mode is measuring, not
          storage I/O. There is no page cache to drop and no cold-vs-warm distinction; the
          only knob is how fast the copy stream can move bytes.

Both are the same thing minus compute -- the only thing removed from each real runner's
loop is the layer_fn call. That makes each a true lower bound for its corresponding
offload_forward.py mode, including the H2D leg and the per-block overheads a closed-form
estimate ignores.

SSD block files are the ones offload_forward.py already wrote, at the same paths, so the
sizes match the benchmark by construction rather than by recalculation. Missing files are
created with real bytes -- never `truncate`, since a sparse file costs no disk blocks and
reads back at memory speed, which would look like a spectacularly fast drive. RAM mode
needs no files at all -- pinned buffers are allocated fresh and their content is irrelevant
to a memcpy's timing.
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


def stream_ssd(paths, nbytes, pool, threads, host, gpu, copy_stream, events):
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


def measure_ssd(directory: Path, n_blocks: int, nbytes: int, threads: int, reps: int,
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
            stream_ssd(paths, nbytes, pool, threads, host, gpu, copy_stream, events)
            ts.append(time.perf_counter() - t0)
    del host, gpu
    torch.cuda.empty_cache()
    ts.sort()
    return ts[len(ts) // 2]


def stream_ram(host, gpu, copy_stream, events):
    """Every block, already in pinned host RAM, -> GPU slot, double buffered, no compute.

    Mirrors RamOffloadRunner.run with the compute removed -- including HOW it waits: the
    runner never blocks the host, it makes the COPY STREAM wait on a device event, so this
    does the same.

    That symmetry is the only reason for the device-side wait; it is NOT faster. The
    previous version blocked the host (`events[s].synchronize()`) before each enqueue, and
    the two were measured against each other across five block sizes from 3 MiB to 368 MiB:
    identical to three digits, 1.00x everywhere, 55-59 GB/s either way. The argument for
    expecting a difference -- that the host stalls until the copy two blocks back retires
    and then pays submit latency inside the timed region -- does not survive contact with
    double buffering, because the copy ONE block back is still in flight during that submit
    and a few microseconds of launch hide under even a 3 MiB copy's ~50 us. Both forms are
    bandwidth-bound at every size that occurs here.

    Note the asymmetry with stream_ssd: there `host` is `slots` reusable staging buffers
    because the drive is the thing being read from repeatedly; here `host` is `n_blocks`
    buffers, one per block, because on this box host RAM already holds the whole model --
    there is nothing to stage from, only to copy out of. Only the GPU side is double
    buffered, and with no compute to wait for, a slot is free again as soon as the previous
    copy into it has landed.
    """
    for i in range(len(host)):
        s = i % len(gpu)
        copy_stream.wait_event(events[s])    # device-side, as RamOffloadRunner._issue does
        with torch.cuda.stream(copy_stream):
            gpu[s].copy_(host[i], non_blocking=True)
        events[s].record(copy_stream)
    torch.cuda.synchronize()


def measure_ram(n_blocks: int, nbytes: int, reps: int, slots: int = 2):
    """No disk, no page cache to drop -- content is irrelevant to a memcpy's timing."""
    host = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) for _ in range(n_blocks)]
    gpu = [torch.empty(nbytes, dtype=torch.uint8, device=DEV) for _ in range(slots)]
    copy_stream = torch.cuda.Stream()
    events = [torch.cuda.Event() for _ in range(slots)]
    for e in events:
        e.record()

    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        stream_ram(host, gpu, copy_stream, events)
        ts.append(time.perf_counter() - t0)
    del host, gpu
    torch.cuda.empty_cache()
    ts.sort()
    return ts[len(ts) // 2]


def params_b(model: str) -> float:
    return float(model.rsplit("-", 1)[1].removesuffix("B"))


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _carveout_bytes(here, model: str, quant: str) -> int:
    """C for this (model, quant), from odp_prefill.csv. 0 if the mode never ran."""
    f = here / "odp_prefill.csv"
    if not f.exists():
        return 0
    import csv as _csv
    with open(f, newline="") as fh:
        for r in _csv.DictReader(fh):
            if r["model"] == model and r["quant"] == quant:
                return int(r["slot_bytes"])
    return 0


def measure_span(path, nbytes: int, threads: int, reps: int) -> float:
    """Seconds to stream `nbytes` from one file through the same staged path as a block."""
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    view = memoryview(host.numpy())
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    span = -(-nbytes // threads)
    ts = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(reps):
            drop_cache([str(path)])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fd = os.open(str(path), os.O_RDONLY)

            def worker(lo):
                hi = min(lo + span, nbytes)
                while lo < hi:
                    got = os.preadv(fd, [view[lo:hi]], lo)
                    if not got:
                        raise EOFError(f"short read on {path} at {lo}")
                    lo += got

            try:
                list(pool.map(worker, range(0, nbytes, span)))
            finally:
                os.close(fd)
            gpu.copy_(host, non_blocking=True)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    del host, gpu
    torch.cuda.empty_cache()
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="offload_prefill.csv",
                    help="where the (model, quant, n_blocks, block_mb) shapes come from")
    ap.add_argument("--out-ssd", default="load_floor_ssd.csv")
    ap.add_argument("--out-zero", default="load_floor_zero_ssd.csv",
                    help="floor for the zero-ssd protocol: P + C, not P")
    ap.add_argument("--out-ram", default="load_floor_ram.csv")
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

    print(f"{torch.cuda.get_device_name(0)}   {args.threads} threads, {args.reps} reps\n")
    print(f"{'model':<12}{'quant':<7}{'blocks':>7}{'block MiB':>11}{'total GiB':>11}"
          f"{'ssd ms':>9}{'ssd GB/s':>10}{'ram ms':>9}{'ram GB/s':>10}")

    ssd_rows, ram_rows, zero_rows = [], [], []
    for _, r in shapes.iterrows():
        model, quant = r["model"], r["quant"]
        n_blocks, block_mib = int(r["n_blocks"]), float(r["block_mb"])
        nbytes = int(round(block_mib * 2**20))
        total_bytes = n_blocks * nbytes

        directory = Path(args.block_dir) / model.replace("/", "_") / quant
        ssd_secs = measure_ssd(directory, n_blocks, nbytes, args.threads, args.reps)
        # zero-ssd is the default offload mode now, and it reads P + C per request: the
        # prefill checkpoint PLUS the decode carve-out restored into the same arena. Its
        # floor is therefore not the `ssd` floor -- quoting that one would understate the
        # drive-bound region by exactly the carve-out. C comes from the odp accounting
        # rather than being re-derived, so the floor and the benchmark cannot disagree
        # about how many bytes the protocol moves (a hybrid model borrows 4 slots, not 2).
        carve = _carveout_bytes(here, model, quant)
        zero_secs = None
        if carve:
            payload = directory / f"decode_carveout_{carve}.bin"
            if payload.exists():
                zero_secs = ssd_secs + measure_span(payload, carve, args.threads,
                                                    args.reps)
        ram_secs = measure_ram(n_blocks, nbytes, args.reps)

        ssd_gbs = total_bytes / 1e9 / ssd_secs
        ram_gbs = total_bytes / 1e9 / ram_secs

        if zero_secs is not None:
            zero_rows.append(dict(model=model, quant=quant, n_blocks=n_blocks,
                                  block_mb=round(block_mib, 1),
                                  total_gib=round((total_bytes + carve) / 2**30, 3),
                                  load_ms=round(zero_secs * 1e3, 3),
                                  gb_s=round((total_bytes + carve) / 1e9 / zero_secs, 2),
                                  threads=args.threads))
        ssd_rows.append(dict(model=model, quant=quant, n_blocks=n_blocks,
                             block_mb=round(block_mib, 1),
                             total_gib=round(total_bytes / 2**30, 3),
                             load_ms=round(ssd_secs * 1e3, 3), gb_s=round(ssd_gbs, 2),
                             threads=args.threads))
        ram_rows.append(dict(model=model, quant=quant, n_blocks=n_blocks,
                             block_mb=round(block_mib, 1),
                             total_gib=round(total_bytes / 2**30, 3),
                             load_ms=round(ram_secs * 1e3, 3), gb_s=round(ram_gbs, 2)))

        print(f"{model.split('/')[-1]:<12}{quant:<7}{n_blocks:>7}{block_mib:>11.0f}"
              f"{total_bytes/2**30:>11.2f}{ssd_secs*1e3:>9.1f}{ssd_gbs:>10.2f}"
              f"{ram_secs*1e3:>9.1f}{ram_gbs:>10.2f}", flush=True)

    write_csv(here / args.out_ssd, ssd_rows)
    if zero_rows:
        write_csv(here / args.out_zero, zero_rows)
        print(f"wrote {here / args.out_zero}")
    write_csv(here / args.out_ram, ram_rows)
    print(f"\nwrote {here / args.out_ssd}")
    print(f"wrote {here / args.out_ram}")
    print("the prefill plots should read load_ms from these files rather than deriving a "
          "floor from one throughput figure -- load_floor_ssd.csv for the `ssd` curve, "
          "load_floor_ram.csv for the `ram` curve")


if __name__ == "__main__":
    main()
