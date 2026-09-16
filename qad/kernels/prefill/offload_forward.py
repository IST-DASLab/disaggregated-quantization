"""Interleaved offloaded prefill: stream each block's weights while computing the previous.

The question this answers. If a model does not fit in GPU memory, you can keep it elsewhere
and pull one decoder block across at a time, double-buffered so the fetch of block i+1
overlaps the compute of block i. Then

    per-block latency = max(fetch_time, compute_time)

so the total is FLAT in sequence length while fetching dominates, and only becomes LINEAR
once compute overtakes it. The break-even is the interesting number, and it moves with the
weight format: NVFP4 moves 3.56x fewer bytes per block than bf16.

This is the real thing, not a model of it: actual weights, actual reads, actual compute,
synchronized with events and checked bitwise against the resident path.

    python offload_forward.py --model Qwen/Qwen3-8B --modes resident ssd
    python offload_forward.py --model google/gemma-3-4b-it --modes resident ssd
    python offload_forward.py --modes resident ssd ram --quant bf16 nvfp4 --out o.csv

Two model families are supported. Qwen3 runs every layer full-causal through SDPA; Gemma 3
alternates five sliding-window layers with one global one and needs a different attention
kernel and a different RoPE table for each -- see gemma3_block.py, and note that getting
this wrong inflates Gemma 3 prefill by ~3.2x at 32k while still producing sane-looking
numbers.

TWO OFFLOAD SOURCES, AND ON THIS BOX ONLY ONE OF THEM IS REAL
-------------------------------------------------------------
`--modes ssd`  blocks live on the NVMe and are read during the forward. **This is the
               meaningful one on GB10** and the headline result. Pipeline is
               SSD -> pinned host staging buffer -> GPU slot -> compute; the drive
               dominates by an order of magnitude (~68 ms for a 368 MiB bf16 block at
               5.7 GB/s, against ~7 ms for the same bytes over H2D).

`--modes ram`  blocks live in pinned host RAM. On GB10 this is NOT an offload: host and
               device memory are the same physical LPDDR5X joined by NVLink-C2C, so the
               copy is a memcpy inside one pool and measures the interconnect. Kept
               anyway, for two reasons -- it is the ceiling SSD streaming would approach
               with an infinitely fast drive, and on a discrete GPU (a 5090, say) host RAM
               really is across PCIe and this mode becomes the realistic one.

The 55 GiB/s pinned H2D that `ram` mode achieves is not a caching artefact -- 4 GiB of
never-reused pinned buffers gives 54.4 GiB/s against 55.1 for one buffer copied eight
times. It is that fast because no bus is involved.

HOW THE WEIGHTS ARE REBOUND
---------------------------
All 36 blocks are structurally identical, so ONE decoder-layer module is instantiated and
its parameters are rebound to views into a GPU slot buffer before each block runs. That is
what keeps this to a single torch.compile instead of 36 -- dynamo guards on shape, dtype
and device, none of which change, so swapping the underlying storage is free.

The consequence is that this runs WITHOUT a KV cache (`use_cache=False`): a cache would
make each block's `layer_idx` differ, and dynamo guards on that integer, which would put us
back at 36 compilations. Prefill latency is dominated by the projections and attention, and
both are measured here in full; only the cache write is missing. The resident baseline goes
through the same code path, so the comparison is internally consistent -- do not compare
these numbers against qwen3_latency.py's, which does use a static cache.

WHY PINNED MEMORY IS NOT OPTIONAL
---------------------------------
Pageable host memory cannot be DMA'd asynchronously; the driver stages it through an
internal bounce buffer, which serializes the copy against compute and destroys the overlap
this whole file exists to demonstrate. Both modes stage through pinned buffers.
"""

import argparse
import copy
import csv
import os
import queue
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

DTYPE = torch.bfloat16
DEV = torch.device("cuda")
ALIGN = 256


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bmBM])(?![a-zA-Z])")


def _params_b(model: str) -> float:
    """Parameter count in billions, from the name, so CSV rows sort by model size.

    'Qwen/Qwen3-1.7B' -> 1.7, 'google/gemma-3-12b-it' -> 12.0, 'google/gemma-3-270m' ->
    0.27. Two traps, both of which a rsplit('-') falls into:

      * the size is not always the last token -- 'gemma-3-1b-it' ends in the
        instruction-tuned suffix, which parses as nothing and sorts the model to the end;
      * '270m' is in millions, so read as a bare number it sorts as the LARGEST model in
        the file rather than the smallest.
    """
    hits = _SIZE_RE.findall(model)
    if not hits:
        return float("inf")
    n, unit = hits[-1]
    return float(n) / (1000.0 if unit in "mM" else 1.0)


def _sort_key(row: dict):
    """(family, size, name, quant, mode, seq_len) -- the CSV's row order.

    Family first so the two model families stay in separate blocks, then size, and then the
    NAME. That last one is not cosmetic: gemma-3-4b and Qwen3-4B are both 4.0B, so with
    size as the final model-level key their rows interleave and neither model's block is
    contiguous.
    """
    model = row["model"]
    family = "gemma3" if "gemma" in model.lower() else "qwen3"
    return (family, _params_b(model), model, row["quant"], row["mode"],
            int(row["seq_len"]))


def _tensors(module: nn.Module):
    """Every parameter and buffer, in a stable order, with its byte size."""
    out = []
    for name, p in module.named_parameters():
        out.append((name, p, p.numel() * p.element_size()))
    for name, b in module.named_buffers():
        out.append((name, b, b.numel() * b.element_size()))
    return out


def plan(module: nn.Module):
    """Byte layout of one block: [(name, shape, dtype, offset, nbytes)], total."""
    spec, off = [], 0
    for name, t, nbytes in _tensors(module):
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        spec.append((name, tuple(t.shape), t.dtype, off, nbytes))
        off += nbytes
    return spec, (off + ALIGN - 1) // ALIGN * ALIGN


def pack_pinned(module: nn.Module, spec, total: int) -> torch.Tensor:
    """One block's weights as a single pinned host buffer -- one copy, not one per tensor."""
    flat = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    named = dict(_tensors_by_name(module))
    for name, shape, dtype, off, nbytes in spec:
        src = named[name].detach().to(dtype).contiguous().cpu()
        # reshape(-1) first: a 0-dim tensor (NVFP4's scalar scale) cannot be
        # byte-viewed directly.
        flat[off:off + nbytes].copy_(src.reshape(-1).view(torch.uint8))
    return flat


def _tensors_by_name(module: nn.Module):
    for name, t, _ in _tensors(module):
        yield name, t


def bind(module: nn.Module, spec, gpu_bytes: torch.Tensor) -> None:
    """Point the module's parameters and buffers at views into `gpu_bytes`."""
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    for name, shape, dtype, off, nbytes in spec:
        view = gpu_bytes[off:off + nbytes].view(dtype).reshape(shape)
        if name in params:
            params[name].data = view
        else:
            owner, _, leaf = name.rpartition(".")
            (module.get_submodule(owner) if owner else module)._buffers[leaf] = view


def make_template(blocks: list[nn.Module]) -> nn.Module:
    """A private copy of block 0 to execute with.

    NOT block 0 itself. Binding rewrites the template's `.data` pointers, so using a live
    block as the template silently overwrites that block's weights -- after one forward,
    layer 0 holds layer 35's weights and every subsequent pack reads the wrong tensor.
    Caught only because the offload path was diffed against the resident one; the timings
    looked perfectly healthy throughout.
    """
    return copy.deepcopy(blocks[0])


def make_templates(blocks: list[nn.Module], kind_of: list[int]) -> list[nn.Module]:
    """One template per BLOCK KIND -- a kind being a distinct parameter layout.

    Qwen3 and Gemma 3 have one kind: every block holds the same tensors in the same byte
    layout, so one template, one plan() and one pair of slots cover the model. Gemma's
    sliding/global split is a difference in attention KWARGS, not in parameters.

    Qwen3.5 has two kinds. Three layers in four are gated delta nets (`in_proj`, `conv_w`,
    `A_log`, `dt_bias`, gated norm) and every fourth is attention (`qkv`, `o`); the two
    have different tensors and different byte counts, so they cannot share a slot spec.
    Each kind gets its own template, its own plan(), and its own slots.
    """
    first = {}
    for blk, k in zip(blocks, kind_of):
        first.setdefault(k, blk)
    return [copy.deepcopy(first[k]) for k in sorted(first)]


class RamOffloadRunner:
    """Streams from pinned host RAM. NOT a real offload on GB10 -- kept as the ceiling.

    Host and device memory here are the same LPDDR5X, so this measures the interconnect
    rather than an offload. Useful only as the bound SSD streaming would approach if the
    drive were infinitely fast.
    """

    def __init__(self, templates, blocks: list[nn.Module], slots: int = 2, kind_of=None):
        self.templates = templates
        self.kind_of = [0] * len(blocks) if kind_of is None else list(kind_of)
        self.plans = [plan(t) for t in templates]
        self.host = [pack_pinned(b, self.plans[k][0], self.plans[k][1])
                     for b, k in zip(blocks, self.kind_of)]
        # Slots per kind, and each block's slot is its position AMONG BLOCKS OF ITS KIND,
        # so the double buffer alternates within a kind rather than across kinds.
        kinds = sorted(set(self.kind_of))
        self.slots = slots
        self.gpu = {k: [torch.empty(self.plans[k][1], dtype=torch.uint8, device=DEV)
                        for _ in range(slots)] for k in kinds}
        self.slot_of, seen = [], {k: 0 for k in kinds}
        for k in self.kind_of:
            self.slot_of.append(seen[k] % slots)
            seen[k] += 1
        self.copy_stream = torch.cuda.Stream()
        self.ready = {k: [torch.cuda.Event() for _ in range(slots)] for k in kinds}
        self.free = {k: [torch.cuda.Event() for _ in range(slots)] for k in kinds}
        for k in kinds:
            for e in self.free[k]:
                e.record()

    @property
    def n(self) -> int:
        return len(self.host)

    @property
    def block_bytes(self) -> int:
        return sum(h.numel() for h in self.host) // max(len(self.host), 1)

    def buffer_of(self, kind: int, slot: int):
        return self.gpu[kind][slot]

    def graph_pairs(self, variant_of):
        out = []
        for k in sorted(self.gpu):
            vs = sorted({variant_of[i] for i in range(self.n) if self.kind_of[i] == k})
            out += [(k, s, v) for s in range(self.slots) for v in vs]
        return out

    def _issue(self, i: int) -> None:
        k, s = self.kind_of[i], self.slot_of[i]
        self.copy_stream.wait_event(self.free[k][s])       # slot no longer in use
        with torch.cuda.stream(self.copy_stream):
            self.gpu[k][s].copy_(self.host[i], non_blocking=True)
        self.ready[k][s].record(self.copy_stream)

    def run(self, hidden, layer_fns, graphs=None, variants=None, variant_of=None):
        cur = torch.cuda.current_stream()
        self._issue(0)
        for i in range(self.n):
            k, s, v = self.kind_of[i], self.slot_of[i], variant_of[i]
            if i + 1 < self.n:
                self._issue(i + 1)                          # overlap: next block's copy
            cur.wait_event(self.ready[k][s])                 # this block's weights landed
            if graphs is not None:
                hidden = graphs.replay(k, s, v, hidden)
            else:
                bind(self.templates[k], self.plans[k][0], self.gpu[k][s])
                hidden = layer_fns[k](hidden, **variants[v])
            self.free[k][s].record(cur)                      # slot reusable after compute
        return hidden


class SSDOffloadRunner:
    """The real offload: blocks live on the SSD and are streamed in during the forward.

    This is the only regime on GB10 where "offload" means anything. Host RAM and device
    memory here are the same physical LPDDR5X over NVLink-C2C, so streaming from pinned
    host memory is a memcpy inside one pool -- it measures the interconnect, not an
    offload. The drive is a genuinely separate device.

    Three-stage pipeline per block: SSD -> pinned host staging buffer -> GPU slot ->
    compute. The disk read runs on a thread pool (the GIL is released inside preadv, so
    the reads really do overlap), the H2D runs on a copy stream, and both proceed while
    the GPU computes the previous block. Two host buffers and two GPU slots, so no more
    than two blocks exist anywhere at once.

    The disk read dominates by an order of magnitude -- ~68 ms for a 368 MiB bf16 block at
    5.7 GB/s against ~7 ms for the same bytes over H2D -- so the pipeline is drive-bound
    and the H2D stage is essentially free.

    THE PAGE CACHE MUST BE DROPPED BEFORE EVERY TIMED PASS. This box has 128 GB of RAM, so
    after one pass the entire model is cached and every subsequent read is served from
    memory at ~20x the speed. That measures memcpy, not the drive, and it is the single
    easiest way to get a spectacular and completely fake result.
    """

    def __init__(self, templates, blocks: list[nn.Module], directory,
                 threads: int = 16, slots: int = 2, kind_of=None):
        import os
        from concurrent.futures import ThreadPoolExecutor

        self.templates = templates
        self.kind_of = [0] * len(blocks) if kind_of is None else list(kind_of)
        self.plans = [plan(t) for t in templates]
        self.threads = threads
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

        # One file per block, holding exactly the bytes that block's KIND expects. With two
        # kinds the files are two different sizes, which is why the size check below is
        # against this block's own spec rather than one model-wide constant.
        self.paths, self.nbytes_of = [], []
        for i, (b, k) in enumerate(zip(blocks, self.kind_of)):
            spec, total = self.plans[k]
            path = self.dir / f"block_{i:03d}.bin"
            if not (path.exists() and path.stat().st_size == total):
                buf = pack_pinned(b, spec, total)
                with open(path, "wb") as f:
                    f.write(memoryview(buf.numpy()))
                    f.flush()
                    os.fsync(f.fileno())
            self.paths.append(str(path))
            self.nbytes_of.append(total)

        # SLOTS ARE PER KIND. A linear-attention block and an attention block hold
        # different tensors in different byte counts, so one arena cannot serve both
        # without re-planning; two independent double buffers is the same pipeline twice.
        # The residency budget being simulated becomes 2 blocks PER KIND rather than 2.
        kinds = sorted(set(self.kind_of))
        self.host, self.views, self.gpu = {}, {}, {}
        self.ready, self.h2d_done, self.compute_done = {}, {}, {}
        for k in kinds:
            total = self.plans[k][1]
            self.host[k] = [torch.empty(total, dtype=torch.uint8, pin_memory=True)
                            for _ in range(slots)]
            self.views[k] = [memoryview(h.numpy()) for h in self.host[k]]
            self.gpu[k] = [torch.empty(total, dtype=torch.uint8, device=DEV)
                           for _ in range(slots)]
            self.ready[k] = [torch.cuda.Event() for _ in range(slots)]
            self.h2d_done[k] = [torch.cuda.Event() for _ in range(slots)]
            self.compute_done[k] = [torch.cuda.Event() for _ in range(slots)]
            for e in self.h2d_done[k] + self.compute_done[k]:
                e.record()
        self.copy_stream = torch.cuda.Stream()
        self.slots = slots
        self.pool = ThreadPoolExecutor(max_workers=threads)
        # ONE long-lived fetch thread for the runner's whole life, not one per forward.
        # Creating and joining a thread costs ~50-100 us, which is nothing against a 10 s
        # forward and quite a lot against an 80 ms one, and it would be paid on every rep
        # of every point. submit() onto a warm worker is ~10 us.
        self.fetch_pool = ThreadPoolExecutor(max_workers=1)

    @property
    def n(self) -> int:
        return len(self.paths)

    @property
    def block_bytes(self) -> int:
        """Mean bytes per block. One kind: the block size. Two kinds: total/n, so that
        block_bytes * n is still the model's streamed size."""
        return sum(self.nbytes_of) // max(len(self.nbytes_of), 1)

    def buffer_of(self, kind: int, slot: int):
        return self.gpu[kind][slot]

    def graph_pairs(self, variant_of):
        """Every (kind, slot, variant) reachable. A slot of a given kind cycles through
        every block of that kind, so it needs every variant those blocks use."""
        out = []
        for k in sorted(self.gpu):
            vs = sorted({variant_of[i] for i in range(self.n) if self.kind_of[i] == k})
            out += [(k, s, v) for s in range(self.slots) for v in vs]
        return out

    def pre_rep(self) -> None:
        """Evict the block files from the page cache. Unprivileged, per-file."""
        import os
        for path in self.paths:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

    def _read(self, i: int, kind: int, slot: int):
        """Read block i off the drive into host[kind][slot], `threads` parallel preads."""
        import os
        view, nbytes = self.views[kind][slot], self.nbytes_of[i]
        span = -(-nbytes // self.threads)
        fd = os.open(self.paths[i], os.O_RDONLY)

        def worker(lo):
            hi = min(lo + span, nbytes)
            while lo < hi:
                got = os.preadv(fd, [view[lo:hi]], lo)
                if not got:
                    raise EOFError(f"short read on {self.paths[i]} at {lo}")
                lo += got

        try:
            list(self.pool.map(worker, range(0, nbytes, span)))
        finally:
            os.close(fd)

    def _fetch(self, i: int, kind: int, slot: int):
        """Disk -> host -> device for block i. Returns once the H2D is *issued*.

        Three hazards, and all three have to be closed or the pipeline is quietly wrong:

          h2d_done[slot]      the previous H2D out of this host buffer must have finished
                              before the drive overwrites it. CPU-side wait, because the
                              read is performed by CPU threads.
          compute_done[slot]  the compute that was READING this GPU slot must have finished
                              before the copy overwrites it. Stream-side wait, so it costs
                              nothing on the CPU.
          ready[slot]         the consumer waits on this before using the weights.

        The middle one is easy to miss and was missing. Without it the H2D for block i+2
        can land on top of block i's weights while block i is still being computed. It
        never fires at short sequences -- a 68 ms read dwarfs a 0.2 ms compute -- so a
        correctness check at seq=128 passes happily. At 32k, compute is ~260 ms per block
        against that same 68 ms read and the window is wide open.
        """
        self.h2d_done[kind][slot].synchronize()
        self._read(i, kind, slot)
        self.copy_stream.wait_event(self.compute_done[kind][slot])
        with torch.cuda.stream(self.copy_stream):
            self.gpu[kind][slot].copy_(self.host[kind][slot], non_blocking=True)
        self.ready[kind][slot].record(self.copy_stream)
        self.h2d_done[kind][slot].record(self.copy_stream)

    def preload_first(self) -> None:
        """Put block 0 in its slot BEFORE the timed region, and leave it there.

        The ring buffers are circular and a served model is not idle before a request: the
        slot that will hold block 0 is filled while the PREVIOUS forward is still computing
        its last layers, so a real system never pays that first read on the critical path.
        Timing it inside the measured region charges offloading for a transfer that
        overlaps in deployment, and it is the one read in the whole pass with nothing to
        hide behind -- every later block loads under the compute of the block before it.

        Only block 0 is pre-loaded. Block 1 onwards must still be fetched inside the timed
        region, because those genuinely do compete with compute.

        This models the steady state rather than implementing it; nothing here fills the
        slot during a previous forward, it is simply filled before the clock starts.
        """
        k = self.kind_of[0]
        self._fetch(0, k, 0)
        self._preloaded = (k, 0)

    def run(self, hidden, layer_fns, graphs=None, variants=None, variant_of=None):
        """Consume blocks in order while a background thread fetches the next ones.

        The fetch runs on its own thread rather than inline. Inline, the read for block i+1
        could only start once this thread had finished *launching* block i, and whatever
        Python the launch does is dead time the drive could have spent working. With a
        producer thread the drive is busy from the moment a slot frees.

        `free` and `filled` are the handshake: a slot goes free -> filled -> free, which
        with two slots is exactly the classic double buffer -- one block being computed
        while the other is read off the drive.
        """
        # One free/filled pair PER KIND: a linear block can only go into a linear slot, so
        # the handshake is per kind. Order within a kind is preserved, and the consumer
        # knows each block's kind, so the two queues stay in step with the block order.
        kinds = sorted(self.gpu)
        free = {k: queue.Queue() for k in kinds}
        filled = {k: queue.Queue() for k in kinds}
        pre = getattr(self, "_preloaded", None)
        for k in kinds:
            for s in range(self.slots):
                # The pre-loaded slot is already FULL: it must not be handed out as free,
                # or the producer would overwrite block 0 while the consumer waits for it.
                if pre is not None and (k, s) == pre:
                    continue
                free[k].put(s)
        start = 0
        if pre is not None:
            filled[pre[0]].put(pre[1])     # consumer takes it first, in block order
            start = 1
            self._preloaded = None         # one-shot: the slot is recycled below

        def producer():
            for i in range(start, self.n):
                k = self.kind_of[i]
                s = free[k].get()
                self._fetch(i, k, s)
                filled[k].put(s)

        fut = self.fetch_pool.submit(producer)
        cur = torch.cuda.current_stream()
        try:
            for i in range(self.n):
                k, v = self.kind_of[i], variant_of[i]
                s = filled[k].get()
                cur.wait_event(self.ready[k][s])
                if graphs is not None:
                    # The graph already points at gpu[k][s]; the DMA just refilled it.
                    hidden = graphs.replay(k, s, v, hidden)
                else:
                    bind(self.templates[k], self.plans[k][0], self.gpu[k][s])
                    hidden = layer_fns[k](hidden, **variants[v])
                self.compute_done[k][s].record(cur)   # slot reusable after this compute
                free[k].put(s)
        finally:
            # Re-raises anything the producer hit (a short read, a missing file) here,
            # where it is attributable, rather than losing it inside the worker.
            fut.result()
        return hidden

    def close(self):
        self.fetch_pool.shutdown(wait=True)
        self.pool.shutdown(wait=True)


class ZeroSSDRunner(SSDOffloadRunner):
    """Offloaded prefill that costs NO extra device weight residency.

    The `ssd` runner allocates its streaming slots ON TOP of whatever else is resident, so
    it answers "what does streaming cost?" while quietly assuming room for the slots. This
    one answers the deployment question instead: during prefill the slots occupy memory
    BORROWED from decode-only weights, and before generation can resume those weights have
    to come back from their own checkpoint. Both ends of that trade are inside the clock.

    THE TIMED BOUNDARY, which is the whole point:

        start:  decode weights resident, NO prefill weights loaded
          1. hand the arena to prefill
          2. read block 0 from SSD -- COLD, inside the timer, with nothing to overlap it
          3. run the stack, overlapping later reads with compute as usual
          4. restore the decode carve-out from SSD into that same arena
          5. drain every outstanding read and copy
        stop:   prefill done AND decode weights back on device

    So this is strictly more expensive than `ssd` by two terms the other protocol never
    charges: the cold first block (`ssd` pre-loads it outside the clock, on the argument
    that a served model fills that slot during the previous forward -- true for a steady
    stream of requests, false for the transition this mode measures) and the carve-out
    restoration.

    BYTES. With P the serialized prefill-block bytes and C the device allocation borrowed:

        read per request = P + C          extra over streaming the checkpoint once = C

    C is the ACTUAL slot allocation, not a theoretical carve-out: a hybrid model allocates
    `slots` buffers PER KIND, so the 27B borrows four buffers, and charging it for two
    would be describing a scheme this code does not implement.

    THE PAYLOAD IS SYNTHETIC. It is a real, non-sparse file of exactly C bytes read through
    the same O_DIRECT-less pread path and staged through the same pinned buffers, so the
    traffic is real -- but it is not a decode checkpoint, and this measures restoration
    TRAFFIC rather than decode-model integration. Any result from this mode has to say so.
    """

    PROTOCOL = "odp_carveout_cold_v1"

    def __init__(self, templates, blocks, directory, threads: int = 16, slots: int = 2,
                 kind_of=None):
        super().__init__(templates, blocks, directory, threads=threads, slots=slots,
                         kind_of=kind_of)
        self.prefill_bytes = int(sum(self.nbytes_of))
        # The arena as ALLOCATED: slots x every kind. Not a per-layer estimate.
        self.slot_spans = [(k, s, self.plans[k][1])
                           for k in sorted(self.gpu) for s in range(self.slots)]
        self.carveout_bytes = int(sum(n for _, _, n in self.slot_spans))
        self.n_slots = len(self.slot_spans)
        self.synthetic_payload = True

        # One real, non-sparse file of exactly C bytes. Written once, outside any timing.
        self.payload_path = str(self.dir / f"decode_carveout_{self.carveout_bytes}.bin")
        pp = Path(self.payload_path)
        if not (pp.exists() and pp.stat().st_size == self.carveout_bytes):
            chunk = 1 << 26
            g = torch.Generator().manual_seed(0)
            with open(pp, "wb") as f:
                left = self.carveout_bytes
                while left:
                    n = min(chunk, left)
                    buf = torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)
                    f.write(memoryview(buf.numpy()))
                    left -= n
                f.flush()
                os.fsync(f.fileno())
        # The cycle must START from decode-resident, so seed the arena before the first
        # measured repetition; every completed cycle then leaves it that way for the next.
        self._restore_carveout()
        torch.cuda.synchronize()

    def preload_first(self) -> None:
        """Deliberately a no-op: block 0 must be read INSIDE the timer here."""
        self._preloaded = None

    def pre_rep(self) -> None:
        """Evict the prefill blocks AND the restoration payload. Both are read cold."""
        super().pre_rep()
        fd = os.open(self.payload_path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)

    def _read_span(self, file_off: int, nbytes: int, view) -> None:
        """`threads` parallel preads of [file_off, file_off+nbytes) into view[:nbytes]."""
        span = -(-nbytes // self.threads)
        fd = os.open(self.payload_path, os.O_RDONLY)

        def worker(lo):
            hi = min(lo + span, nbytes)
            while lo < hi:
                got = os.preadv(fd, [view[lo:hi]], file_off + lo)
                if not got:
                    raise EOFError(f"short read on {self.payload_path} at {file_off + lo}")
                lo += got

        try:
            list(self.pool.map(worker, range(0, nbytes, span)))
        finally:
            os.close(fd)

    def _restore_carveout(self) -> None:
        """SSD -> pinned host -> the SAME device arena the prefill slots used.

        No extra device destination: the bytes land back in gpu[kind][slot], which is what
        makes this a carve-out rather than a second allocation.
        """
        off = 0
        for k, sl, n in self.slot_spans:
            self._read_span(off, n, self.views[k][sl])
            with torch.cuda.stream(self.copy_stream):
                self.gpu[k][sl].copy_(self.host[k][sl], non_blocking=True)
            self.h2d_done[k][sl].record(self.copy_stream)
            off += n
        self.copy_stream.synchronize()

    def run(self, hidden, layer_fns, graphs=None, variants=None, variant_of=None):
        """Prefill from block 0, then restore -- OVERLAPPED with the tail of compute.

        super().run() returns once the last block has been LAUNCHED, not finished, so the
        GPU is still working through the final layers when this returns. The restoration
        splits into two halves that can start at different times:

          the SSD read (drive -> pinned host) touches no device memory at all, so it runs
          immediately, against the tail of compute;

          the H2D (host -> arena) must not land before the compute that is still READING
          that slot, so it is ordered behind compute_done[k][slot] on the copy stream --
          a stream-side wait, not a host-side one, so the CPU never blocks and the copies
          begin the instant each slot's last block retires.

        That is the proof the conservative schedule wanted: the destination bytes are free
        exactly when compute_done fires, and the wait is expressed to the GPU rather than
        assumed. The read can only clobber a staging buffer whose H2D is still in flight,
        so each host buffer waits on its own h2d_done first -- already signalled by this
        point in every observed run, but correctness cannot rest on that.
        """
        self._preloaded = None          # never start from a pre-filled slot
        hidden = super().run(hidden, layer_fns, graphs=graphs, variants=variants,
                             variant_of=variant_of)
        off = 0
        for k, sl, n in self.slot_spans:
            self.h2d_done[k][sl].synchronize()          # staging buffer free to overwrite
            self._read_span(off, n, self.views[k][sl])  # drive -> host, overlaps compute
            self.copy_stream.wait_event(self.compute_done[k][sl])   # arena free, on-stream
            with torch.cuda.stream(self.copy_stream):
                self.gpu[k][sl].copy_(self.host[k][sl], non_blocking=True)
            self.h2d_done[k][sl].record(self.copy_stream)
            off += n
        self.copy_stream.synchronize()   # the timer stops only after restoration lands
        return hidden

    def verify_restored(self) -> bool:
        """Untimed: the arena really holds the payload's bytes after a cycle."""
        off = 0
        for k, sl, n in self.slot_spans:
            with open(self.payload_path, "rb") as f:
                f.seek(off)
                want = torch.frombuffer(bytearray(f.read(n)), dtype=torch.uint8)
            if not torch.equal(self.gpu[k][sl].cpu(), want):
                return False
            off += n
        return True


class ResidentRunner:
    """Same loop, every block already on the GPU. The control for the offload numbers."""

    def __init__(self, templates, blocks, kind_of=None, consume=False):
        """`consume` frees each block's parameters once it has been staged.

        Staging copies every block into a flat device buffer, and from then on the template
        is bound to THAT buffer -- the nn.Module the copy came from is dead weight. Holding
        both is a second copy of the whole model: measured on Qwen3.8-27B bf16, 45.4 GiB of
        blocks plus 45.4 GiB of staged buffers peaks at 99 GiB of the box's 121, which is
        why 16K would not run. The peak was flat in sequence length -- 99 GiB at S=2048,
        4096 and 8192 alike -- which is what ruled out activations as the cause.

        Off by default because `blocks` belongs to the caller and other modes reuse it;
        main passes consume=True only for the last mode of the sweep.
        """
        self.templates = templates
        self.kind_of = [0] * len(blocks) if kind_of is None else list(kind_of)
        self.plans = [plan(t) for t in templates]
        self.gpu = []
        for b, k in zip(blocks, self.kind_of):
            # Straight from the live GPU tensors -- no pinned-host round trip, which would
            # allocate and fill another copy of the whole model for nothing.
            spec, total = self.plans[k]
            named = dict(_tensors_by_name(b))
            buf = torch.empty(total, dtype=torch.uint8, device=DEV)
            for name, shape, dtype, off, nbytes in spec:
                src = named[name].detach().to(dtype).contiguous()
                buf[off:off + nbytes].copy_(src.reshape(-1).view(torch.uint8))
            self.gpu.append(buf)
            if consume:
                # Release the source block NOW, not after the loop: peak matters, and the
                # whole point is never to hold two copies of the model at once.
                for mod in b.modules():
                    for pname, prm in list(mod.named_parameters(recurse=False)):
                        setattr(mod, pname, None)
                        mod._parameters.pop(pname, None)
                    for bname, bt in list(mod.named_buffers(recurse=False)):
                        mod._buffers[bname] = None
        if consume:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    @property
    def n(self):
        return len(self.gpu)

    @property
    def block_bytes(self):
        """Mean bytes per block. With one kind this is exactly the block size; with two it
        is total/n, so block_bytes * n stays the model's streamed size -- which is what the
        floor and the CSV's block_mb column are for."""
        return sum(b.numel() for b in self.gpu) // max(len(self.gpu), 1)

    def graph_pairs(self, variant_of):
        """(kind, slot, kwargs-variant) triples that can occur. Resident runs block i out
        of buffer i, so each buffer needs only its own layer's kind and variant."""
        return [(self.kind_of[i], i, variant_of[i]) for i in range(self.n)]

    def buffer_of(self, kind: int, slot: int):
        return self.gpu[slot]          # resident indexes slots by BLOCK

    def run(self, hidden, layer_fns, graphs=None, variants=None, variant_of=None):
        for i, buf in enumerate(self.gpu):
            k, v = self.kind_of[i], variant_of[i]
            if graphs is not None:
                hidden = graphs.replay(k, i, v, hidden)
            else:
                bind(self.templates[k], self.plans[k][0], buf)
                hidden = layer_fns[k](hidden, **variants[v])
        return hidden


class BlockGraphs:
    """One CUDA graph per (weight buffer, attention variant), replaying against a shared input.

    The trick that makes this work for offloading: a captured graph refers to its weight
    tensors by ADDRESS. The slot buffers never move -- only their contents do, rewritten by
    the DMA -- so two graphs captured against the two slots stay valid for the whole model.
    Block i replays the graph for (its slot, its layer type) once its weights have landed.

    WHY A VARIANT AXIS. Qwen3 runs every layer the same way, so there is one variant and
    this is one graph per buffer, as before. Gemma 3 does not: five layers in six attend
    over a sliding window and the sixth attends globally, and the two differ in the
    attention kernel AND in the RoPE table. That is two kwarg sets, so two compilations and
    two graphs per buffer -- still a fixed, tiny number, rather than one per layer.

    `pairs` says which (buffer, variant) combinations are actually reachable, so nothing is
    captured that will never replay: resident needs exactly one variant per block (the one
    its layer type uses), while a 2-slot offload run needs both variants on both slots.

    What it buys: at replay there is no Python at all -- no bind(), no dynamo guards, no
    per-kernel launches. That fixed per-block CPU cost is what caps the drive-bound regime,
    where the drive is only kept busy if the consumer gets out of its way. It is worth
    nothing in the compute-bound regime, where launches are already hidden.

    What it costs: one activation copy per block. Each graph writes its output at a fixed
    address from the capture pool, so the result must be copied into the shared input
    before the next graph replays. That is ~1% at 8B/32k (a 268 MiB D2D copy against a
    9.4 s forward) and negligible at short sequences, where activations are tiny -- i.e.
    it costs least exactly where the launch saving is worth most.
    """

    def __init__(self, layer_fns, buffer_of, bind_to, hidden, variants, pairs):
        """`pairs` are (kind, slot, kwargs-variant). `buffers[kind][slot]` is the arena that
        kind's template binds to, and `layer_fns[kind]` is its compiled callable -- with one
        kind this is exactly the old one-template behaviour."""
        self.inp = hidden.clone()
        self.graphs, self.outs = {}, {}
        pool = None
        side = torch.cuda.Stream()
        # Grouped by (kind, buffer) so each one is bound once, not once per variant:
        # binding is cheap but capture is not, and rebinding between captures of the same
        # buffer would be pure noise.
        for kind, buf_idx in sorted({(k, b) for k, b, _ in pairs}):
            bind_to(kind, buffer_of(kind, buf_idx))
            layer_fn = layer_fns[kind]
            for var_idx in sorted({v for k, b, v in pairs if (k, b) == (kind, buf_idx)}):
                kw = variants[var_idx]
                # Warm up this binding on a side stream first: cuBLAS workspace allocation
                # and any lazy init must not happen inside the capture.
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        layer_fn(self.inp, **kw)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    out = layer_fn(self.inp, **kw)
                pool = g.pool()
                self.graphs[(kind, buf_idx, var_idx)] = g
                self.outs[(kind, buf_idx, var_idx)] = out[0] if isinstance(out, tuple) else out

    def replay(self, kind: int, buf_idx: int, var_idx: int,
               hidden: torch.Tensor) -> torch.Tensor:
        self.inp.copy_(hidden)
        self.graphs[(kind, buf_idx, var_idx)].replay()
        return self.outs[(kind, buf_idx, var_idx)]


def _trim_host_heap() -> None:
    """Return freed host memory to the OS, not just to glibc's arenas.

    Dropping a transformers layer makes Python free it, but glibc keeps the arena, so RSS
    does not fall. That is invisible on a discrete-GPU box and fatal here: host and device
    share one 121 GiB pool, so a resident host copy is 1:1 competition with the weights
    being built on the device.

    Measured on Qwen3.8-27B bf16: without this the build peaks at 110 GiB -- the 52 GiB
    checkpoint that Python has already freed, plus 47.6 GiB of blocks -- and trips the
    memory watchdog. The NVFP4 arm never showed it, because there each bf16 layer really
    does die as soon as its fp4 block exists, so its peak is one model, not two.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:                            # noqa: BLE001 - best effort, never fatal
        pass


def build(model_name: str, quant: str):
    """Load the checkpoint and return (fused blocks, config, attention_kwargs_fn).

    The blocks are our own fused ones, not transformers' layer -- see qwen3_block.py for
    why. The transformers model is loaded only to source the weights and is dropped
    immediately, so nothing downstream depends on it.

    The third return value is what the caller uses to get per-sequence-length attention
    kwargs: `fn(cfg, seq_len) -> (variants, variant_of)`. Qwen3 has one variant for every
    layer; Gemma 3 has two, because its layers alternate sliding-window and global
    attention and the two need different kernels AND different RoPE tables.

    The fourth is `kind_of`: which PARAMETER LAYOUT each block has. Qwen3 and Gemma 3 have
    one kind -- every block holds the same tensors, so one template and one pair of slots
    serve the model. Qwen3.5 has two (gated delta net / attention), which is a different
    axis from the kwargs variant: Gemma's two variants share a layout, Qwen3.5's two kinds
    do not.
    """
    import transformers
    from transformers import AutoModel
    transformers.logging.set_verbosity_error()

    hf = AutoModel.from_pretrained(model_name, dtype=DTYPE,
                                   attn_implementation="sdpa").eval()
    # Gemma 3 4b/12b are multimodal wrappers; the decoder stack is the language model, and
    # the vision tower is not part of what this benchmark measures.
    hf = getattr(hf, "language_model", hf)
    cfg = getattr(hf.config, "text_config", hf.config)

    # Detected by what the layers ARE, not by a model_type string: the wrapper config
    # says "qwen3_5" while the text config says "qwen3_5_text", and a renamed release
    # would silently fall through to Qwen3Block and fail on the first layer.
    if "linear_attention" in tuple(getattr(cfg, "layer_types", ()) or ()):
        from qwen35_block import block_for, rope_tables as q35_rope

        def kwargs_fn(cfg, seq_len, dtype=DTYPE, device=DEV):
            cos, sin = q35_rope(cfg, seq_len, dtype=dtype, device=device)
            return [dict(cos=cos, sin=sin)], [0] * cfg.num_hidden_layers

        # MEMORY-SAFE, AND ON A 27B IT HAS TO BE. Host and device share one 121 GB pool on
        # this box, so the obvious loop -- build every block, then quantize them all --
        # holds the whole model twice: 52 GiB of transformers layers on the host plus up to
        # 49 GiB of blocks on the device, before activations.
        #
        # Instead each layer is converted, quantized and then DROPPED from the transformers
        # model in the same iteration. Every layer that arrives on the device leaves the
        # host, so the total stays flat at roughly one model rather than two, and for NVFP4
        # it falls as it goes -- the bf16 layer is released as soon as its fp4 block exists.
        import gc

        from nvfp4_linear import convert
        blocks, kind_of = [], []
        # The benchmark times DECODER BLOCKS. The embedding table is 248320 x 5120 = 2.5 GiB
        # that is never read here, and on a shared pool that is 2.5 GiB taken from the
        # weights being built. Dropped before the loop rather than left to the GC.
        for _dead in ("embed_tokens", "rotary_emb"):
            if hasattr(hf, _dead):
                setattr(hf, _dead, None)
        gc.collect()
        _trim_host_heap()
        layers = hf.layers
        for i in range(len(layers)):
            layer = layers[i]
            blk = block_for(layer, cfg, dtype=DTYPE, device=DEV)
            if quant == "nvfp4":
                convert(blk)
            blocks.append(blk)
            kind_of.append(0 if layer.block_type == "linear_attention" else 1)
            layers._modules[str(i)] = None      # release this layer's host memory now
            # Every layer, not every eighth: a bf16 layer of this model is 726 MiB, so
            # batching the trim leaves up to 5.8 GiB of dead host memory competing with the
            # device for the same pool.
            gc.collect()
            _trim_host_heap()
        del hf, layers
        gc.collect()
        torch.cuda.empty_cache()
        return blocks, cfg, kwargs_fn, kind_of

    if "gemma-3" in model_name.lower():
        from gemma3_block import Gemma3Block, attention_kwargs
        blocks = [Gemma3Block.from_hf(layer, cfg, dtype=DTYPE, device=DEV)
                  for layer in hf.layers]
        kwargs_fn = attention_kwargs
    else:
        from qwen3_block import Qwen3Block, rope_tables

        def kwargs_fn(cfg, seq_len, dtype=DTYPE, device=DEV):
            cos, sin = rope_tables(cfg, seq_len, dtype=dtype, device=device)
            return [dict(cos=cos, sin=sin)], [0] * cfg.num_hidden_layers

        blocks = [Qwen3Block.from_hf(layer, cfg, dtype=DTYPE, device=DEV)
                  for layer in hf.layers]
    del hf
    torch.cuda.empty_cache()

    if quant == "nvfp4":
        from nvfp4_linear import convert
        for b in blocks:
            convert(b)          # qkv / o / gate_up / down are plain nn.Linear
    return blocks, cfg, kwargs_fn, [0] * len(blocks)


@torch.no_grad()
def measure(cfg, runner, layer_fns, seq_len, kwargs_fn, warmup=2, reps=5,
            use_graphs=True):
    """One point. `runner` and the compiled `layer_fn` are built once, by the caller.

    THE TIMED REGION IS THE DECODER BLOCKS, NOTHING ELSE. Embeddings, the RoPE tables and
    the final norm are evaluated once beforehand and the stack is fed a fixed hidden state.
    None of them has anything to do with where the weights live, all three run outside the
    compiled block, and together they would add uncompiled eager noise to both arms while
    diluting exactly the quantity being compared. Making the RoPE tables fixed tensors is
    also what lets the blocks be captured into CUDA graphs at all.

    Building the runner per point is what made the first version take ten minutes: a fresh
    SSD runner rewrites the entire model to disk and a RAM one allocates 13 GiB of pinned
    memory.
    """
    hidden = torch.randn(1, seq_len, cfg.hidden_size, dtype=DTYPE, device=DEV)
    variants, variant_of = kwargs_fn(cfg, seq_len, dtype=DTYPE, device=DEV)

    # Which (buffer, variant) pairs can actually occur. Resident runs block i out of buffer
    # i, so each buffer needs only its own layer's variant; an offload runner cycles every
    # block through every slot, so each slot needs every variant that any block uses.
    pairs = runner.graph_pairs(variant_of)

    graphs = None
    if use_graphs:
        try:
            graphs = BlockGraphs(
                layer_fns, runner.buffer_of,
                lambda k, buf: bind(runner.templates[k], runner.plans[k][0], buf),
                hidden, variants, pairs)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            # Resident captures one graph per block, so capture can run out of memory at
            # long sequences. Fall back loudly rather than silently reporting a different
            # execution path as though it were the same one.
            print(f"  [graph capture failed at seq={seq_len}: "
                  f"{str(exc).splitlines()[0][:70]}; eager]", flush=True)
            graphs = None
            torch.cuda.empty_cache()

    def fwd():
        return runner.run(hidden, layer_fns, graphs=graphs, variants=variants,
                          variant_of=variant_of)

    pre_rep = getattr(runner, "pre_rep", lambda: None)
    # Outside the clock on purpose -- see SSDOffloadRunner.preload_first.
    preload = getattr(runner, "preload_first", lambda: None)
    for _ in range(warmup):
        pre_rep()
        preload()
        fwd()
    torch.cuda.synchronize()

    ts = []
    for _ in range(reps):
        pre_rep()                    # drop the page cache; see SSDOffloadRunner
        preload()                    # block 0 lands in its slot BEFORE the clock starts
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    del graphs
    torch.cuda.empty_cache()
    return ts[len(ts) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seq-lens", nargs="+", type=int,
                    default=[128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--quant", nargs="+", default=["bf16", "nvfp4"])
    ap.add_argument("--modes", nargs="+", default=["resident", "ssd"],
                    choices=["resident", "ssd", "zero-ssd", "ram"],
                    help="ssd = the real offload; ram = pinned host RAM (ceiling only)")
    ap.add_argument("--block-dir", default=str(Path.home() / ".bench_offload_blocks"),
                    help="where the per-block files live")
    ap.add_argument("--threads", type=int, default=16,
                    help="parallel preads per block; 16 is the conservative point")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-graphs", action="store_true",
                    help="skip CUDA graph capture; launch each block from Python")
    ap.add_argument("--model-name", default=None,
                    help="label written to the CSV's model column. Defaults to --model, "
                         "which is wrong when the checkpoint is a local path: a 52 GiB "
                         "model lives on disk rather than in the HF cache, and the rows "
                         "would key on a machine-specific path instead of the hub id.")
    ap.add_argument("--out", default="offload_prefill.csv")
    ap.add_argument("--out-odp", default="odp_prefill.csv",
                    help="zero-ssd results go here, NOT into --out: the timed boundary is "
                         "a different protocol and the (model, quant, mode, seq) key "
                         "cannot distinguish them, so mixing would silently compare a "
                         "cold-start carve-out latency against a preloaded streaming one")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    # One compile per sequence length is intended (dynamic=False keeps shapes static and
    # the kernels specialized). Dynamo's default recompile_limit is 8, so a sweep of 9+
    # lengths trips "Dynamo recompile limit exceeded" on the last one and takes the whole
    # run with it. Raise it to cover any sweep we would realistically ask for.
    for name, value in (("recompile_limit", 64), ("cache_size_limit", 64),
                        ("accumulated_recompile_limit", 512)):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, value)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"no KV cache (see module docstring)\n")

    grid = [(q, m, s) for q in args.quant for m in args.modes for s in args.seq_lens]
    print(f"{len(grid)} points: {len(args.quant)} quant x {len(args.modes)} modes x "
          f"{len(args.seq_lens)} lengths, {args.reps} reps + 2 warmup each")
    print("expect several minutes: one torch.compile per sequence length, and every SSD "
          "pass re-reads the whole model from a cold page cache\n")

    # Merge, keyed on (model, quant, mode, seq_len), so a run covering only the new
    # sequence lengths updates exactly those points and leaves the rest in place. Rewritten
    # and flushed after every point: a crash or a kill costs the current point, not the
    # whole sweep -- which it did, once.
    fields = ["model", "quant", "mode", "seq_len", "latency_ms", "block_mb", "n_blocks"]
    merged = {}
    if args.out and Path(args.out).exists():
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                merged[(r["model"], r["quant"], r["mode"], int(r["seq_len"]))] = r
        print(f"merging into {args.out} ({len(merged)} existing points)")

    def flush_csv():
        if not args.out:
            return
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            # Sorted on (family, model, dtype, mode, seq_len), with model ordered by
            # parameter count rather than lexicographically so 1.7B comes before 4B and 8B
            # before 14B. See _sort_key.
            w.writerows(sorted(merged.values(), key=_sort_key))

    # zero-ssd lives in its own file with its own columns. Merged on a key that INCLUDES
    # the protocol, so a future protocol lands beside this one instead of overwriting it.
    ODP_FIELDS = ["protocol", "model", "quant", "mode", "seq_len", "latency_ms", "reps",
                  "n_blocks", "prefill_bytes", "slot_bytes", "restore_bytes", "n_slots",
                  "ssd_bytes_per_request", "synthetic_payload"]

    def write_rows(path, new, key):
        merged_odp = {}
        if path.exists():
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    merged_odp[tuple(str(r[k]) for k in key)] = r
        for r in new:
            merged_odp[tuple(str(r[k]) for k in key)] = r
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ODP_FIELDS)
            w.writeheader()
            w.writerows(sorted(merged_odp.values(),
                               key=lambda r: (str(r["protocol"]), str(r["model"]),
                                              str(r["quant"]), int(r["seq_len"]))))

    rows, odp_rows = [], []
    bar = tqdm(total=len(grid), unit="pt", dynamic_ncols=True)
    for quant in args.quant:
        bar.set_description(f"{quant}: loading model")
        blocks, cfg, kwargs_fn, kind_of = build(args.model, quant)
        if quant == args.quant[0]:
            # Say out loud what the attention actually is. On Gemma 3 the sliding/global
            # split is the single biggest lever on these numbers, and a silent harness
            # would let a wrong one pass for a real result.
            _, vof = kwargs_fn(cfg, 4096, dtype=DTYPE, device=DEV)
            n_var = len(set(vof))
            bar.write(f"{args.model}: {len(blocks)} blocks, {n_var} attention variant"
                      f"{'s' if n_var > 1 else ''}"
                      + (f" ({sum(1 for v in vof if v) } sliding / "
                         f"{sum(1 for v in vof if not v)} global, window "
                         f"{getattr(cfg, 'sliding_window', None)})" if n_var > 1 else
                         " (full-causal)"))
        templates = make_templates(blocks, kind_of)
        layer_fns = [t if args.no_compile else
                     torch.compile(t, fullgraph=True, dynamic=False) for t in templates]
        if len(templates) > 1:
            bar.write(f"{args.model}: {len(templates)} block kinds "
                      f"({', '.join(str(kind_of.count(k)) + 'x kind' + str(k) for k in sorted(set(kind_of)))})"
                      f" -- one template, plan and slot pair each")
        for mode in args.modes:
            bar.set_description(f"{quant}/{mode}: staging blocks")
            # built ONCE and reused for every seq_len
            if mode == "zero-ssd":
                runner = ZeroSSDRunner(
                    templates, blocks,
                    Path(args.block_dir) / (args.model_name or args.model).replace("/", "_")
                    / quant,
                    threads=args.threads, kind_of=kind_of)
            elif mode == "ssd":
                # Model name in the path: without it, two models with different block
                # sizes churn the same files, rewriting them on every switch.
                runner = SSDOffloadRunner(
                    templates, blocks,
                    Path(args.block_dir) / (args.model_name or args.model).replace("/", "_")
                    / quant,
                    threads=args.threads, kind_of=kind_of)
            elif mode == "ram":
                runner = RamOffloadRunner(templates, blocks, kind_of=kind_of)
            else:
                runner = ResidentRunner(templates, blocks, kind_of=kind_of,
                                        consume=(mode == args.modes[-1]))
            for seq in args.seq_lens:
                bar.set_description(f"{quant}/{mode}/seq={seq}")
                try:
                    ms = measure(cfg, runner, layer_fns, seq, kwargs_fn, reps=args.reps,
                                 use_graphs=not args.no_graphs)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bar.write(f"{quant:<6}{mode:<9}{seq:>7}   OOM")
                    bar.update(1)
                    continue
                if mode == "zero-ssd":
                    # Into the MAIN csv too: `mode` is literally "zero-ssd", so the
                    # (model, quant, mode, seq_len) key already separates it from `ssd`
                    # and the two sit side by side for comparison. The odp file keeps the
                    # byte accounting and protocol tag the main columns cannot hold.
                    row = dict(model=(args.model_name or args.model), quant=quant,
                               mode=mode, seq_len=seq, latency_ms=round(ms, 3),
                               block_mb=round(runner.block_bytes / 2**20, 1),
                               n_blocks=runner.n)
                    rows.append(row)
                    merged[(row["model"], row["quant"], row["mode"], row["seq_len"])] = row
                    flush_csv()
                    odp_rows.append(dict(
                        protocol=runner.PROTOCOL,
                        model=(args.model_name or args.model), quant=quant, mode=mode,
                        seq_len=seq, latency_ms=round(ms, 3), reps=args.reps,
                        n_blocks=runner.n,
                        prefill_bytes=runner.prefill_bytes,
                        slot_bytes=runner.carveout_bytes,
                        restore_bytes=runner.carveout_bytes,
                        n_slots=runner.n_slots,
                        ssd_bytes_per_request=runner.prefill_bytes + runner.carveout_bytes,
                        synthetic_payload=runner.synthetic_payload))
                    write_rows(Path(__file__).parent / args.out_odp, odp_rows,
                               key=("protocol", "model", "quant", "mode", "seq_len"))
                    bar.write(f"{quant:<6}{mode:<9}{seq:>7}   {ms:9.2f} ms"
                              f"   (P={runner.prefill_bytes/2**30:.2f} GiB + "
                              f"C={runner.carveout_bytes/2**30:.2f} GiB, "
                              f"{runner.n_slots} slots)")
                    bar.update(1)
                    continue
                row = dict(model=(args.model_name or args.model), quant=quant,
                           mode=mode, seq_len=seq,
                           latency_ms=round(ms, 3),
                           block_mb=round(runner.block_bytes / 2**20, 1),
                           n_blocks=runner.n)
                rows.append(row)
                merged[(row["model"], row["quant"], row["mode"], row["seq_len"])] = row
                flush_csv()
                bar.write(f"{quant:<6}{mode:<9}{seq:>7}   {ms:9.2f} ms"
                          f"   ({runner.n} x {runner.block_bytes/2**20:.0f} MiB blocks)")
                bar.update(1)
            if hasattr(runner, "close"):
                runner.close()
            del runner
            torch.cuda.empty_cache()
        del blocks, templates, layer_fns
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    bar.close()
    if args.out:
        flush_csv()
        print(f"\nwrote {args.out} ({len(rows)} new, {len(merged)} total)")


if __name__ == "__main__":
    main()
