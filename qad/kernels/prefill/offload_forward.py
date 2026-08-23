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


class RamOffloadRunner:
    """Streams from pinned host RAM. NOT a real offload on GB10 -- kept as the ceiling.

    Host and device memory here are the same LPDDR5X, so this measures the interconnect
    rather than an offload. Useful only as the bound SSD streaming would approach if the
    drive were infinitely fast.
    """

    def __init__(self, template: nn.Module, blocks: list[nn.Module], slots: int = 2):
        self.template = template
        self.spec, self.block_bytes = plan(template)
        self.host = [pack_pinned(b, self.spec, self.block_bytes) for b in blocks]
        self.slots = [torch.empty(self.block_bytes, dtype=torch.uint8, device=DEV)
                      for _ in range(slots)]
        self.copy_stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in self.slots]
        self.free = [torch.cuda.Event() for _ in self.slots]
        for e in self.free:
            e.record()

    @property
    def n(self) -> int:
        return len(self.host)

    def _issue(self, i: int) -> None:
        s = i % len(self.slots)
        self.copy_stream.wait_event(self.free[s])          # slot no longer in use
        with torch.cuda.stream(self.copy_stream):
            self.slots[s].copy_(self.host[i], non_blocking=True)
        self.ready[s].record(self.copy_stream)

    def run(self, hidden, layer_fn, graphs=None, variants=None, variant_of=None):
        cur = torch.cuda.current_stream()
        self._issue(0)
        for i in range(self.n):
            s = i % len(self.slots)
            v = variant_of[i]
            if i + 1 < self.n:
                self._issue(i + 1)                          # overlap: next block's copy
            cur.wait_event(self.ready[s])                    # this block's weights landed
            if graphs is not None:
                hidden = graphs.replay(s, v, hidden)
            else:
                bind(self.template, self.spec, self.slots[s])
                hidden = layer_fn(hidden, **variants[v])
            self.free[s].record(cur)                         # slot reusable after compute
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

    def __init__(self, template: nn.Module, blocks: list[nn.Module], directory,
                 threads: int = 16, slots: int = 2):
        import os
        from concurrent.futures import ThreadPoolExecutor

        self.template = template
        self.spec, self.block_bytes = plan(template)
        self.threads = threads
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

        # One file per block, holding exactly the bytes the GPU slot expects.
        self.paths = []
        for i, b in enumerate(blocks):
            path = self.dir / f"block_{i:03d}.bin"
            if not (path.exists() and path.stat().st_size == self.block_bytes):
                buf = pack_pinned(b, self.spec, self.block_bytes)
                with open(path, "wb") as f:
                    f.write(memoryview(buf.numpy()))
                    f.flush()
                    os.fsync(f.fileno())
            self.paths.append(str(path))

        self.host = [torch.empty(self.block_bytes, dtype=torch.uint8, pin_memory=True)
                     for _ in range(slots)]
        self.views = [memoryview(h.numpy()) for h in self.host]
        self.gpu = [torch.empty(self.block_bytes, dtype=torch.uint8, device=DEV)
                    for _ in range(slots)]
        self.copy_stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in range(slots)]         # weights landed
        self.h2d_done = [torch.cuda.Event() for _ in range(slots)]      # host buf reusable
        self.compute_done = [torch.cuda.Event() for _ in range(slots)]  # gpu slot reusable
        for e in self.h2d_done + self.compute_done:
            e.record()
        self.pool = ThreadPoolExecutor(max_workers=threads)
        # ONE long-lived fetch thread for the runner's whole life, not one per forward.
        # Creating and joining a thread costs ~50-100 us, which is nothing against a 10 s
        # forward and quite a lot against an 80 ms one, and it would be paid on every rep
        # of every point. submit() onto a warm worker is ~10 us.
        self.fetch_pool = ThreadPoolExecutor(max_workers=1)

    @property
    def n(self) -> int:
        return len(self.paths)

    def pre_rep(self) -> None:
        """Evict the block files from the page cache. Unprivileged, per-file."""
        import os
        for path in self.paths:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)

    def _read(self, i: int, slot: int):
        """Read block i off the drive into host[slot], with `threads` parallel preads."""
        import os
        view, nbytes = self.views[slot], self.block_bytes
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

    def _fetch(self, i: int, slot: int):
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
        self.h2d_done[slot].synchronize()
        self._read(i, slot)
        self.copy_stream.wait_event(self.compute_done[slot])
        with torch.cuda.stream(self.copy_stream):
            self.gpu[slot].copy_(self.host[slot], non_blocking=True)
        self.ready[slot].record(self.copy_stream)
        self.h2d_done[slot].record(self.copy_stream)

    def run(self, hidden, layer_fn, graphs=None, variants=None, variant_of=None):
        """Consume blocks in order while a background thread fetches the next ones.

        The fetch runs on its own thread rather than inline. Inline, the read for block i+1
        could only start once this thread had finished *launching* block i, and whatever
        Python the launch does is dead time the drive could have spent working. With a
        producer thread the drive is busy from the moment a slot frees.

        `free` and `filled` are the handshake: a slot goes free -> filled -> free, which
        with two slots is exactly the classic double buffer -- one block being computed
        while the other is read off the drive.
        """
        nslots = len(self.gpu)
        free: queue.Queue = queue.Queue()
        filled: queue.Queue = queue.Queue()
        for s in range(nslots):
            free.put(s)

        def producer():
            for i in range(self.n):
                s = free.get()
                self._fetch(i, s)
                filled.put(s)

        fut = self.fetch_pool.submit(producer)
        cur = torch.cuda.current_stream()
        try:
            for i in range(self.n):
                s = filled.get()
                v = variant_of[i]
                cur.wait_event(self.ready[s])
                if graphs is not None:
                    # The graph already points at gpu[s]; the DMA just refilled it.
                    hidden = graphs.replay(s, v, hidden)
                else:
                    bind(self.template, self.spec, self.gpu[s])
                    hidden = layer_fn(hidden, **variants[v])
                self.compute_done[s].record(cur)   # slot reusable after this compute
                free.put(s)
        finally:
            # Re-raises anything the producer hit (a short read, a missing file) here,
            # where it is attributable, rather than losing it inside the worker.
            fut.result()
        return hidden

    def close(self):
        self.fetch_pool.shutdown(wait=True)
        self.pool.shutdown(wait=True)


class ResidentRunner:
    """Same loop, every block already on the GPU. The control for the offload numbers."""

    def __init__(self, template, blocks):
        self.template = template
        self.spec, self.block_bytes = plan(template)
        self.gpu = []
        for b in blocks:
            # Straight from the live GPU tensors -- no pinned-host round trip, which would
            # allocate and fill another copy of the whole model for nothing.
            named = dict(_tensors_by_name(b))
            buf = torch.empty(self.block_bytes, dtype=torch.uint8, device=DEV)
            for name, shape, dtype, off, nbytes in self.spec:
                src = named[name].detach().to(dtype).contiguous()
                buf[off:off + nbytes].copy_(src.reshape(-1).view(torch.uint8))
            self.gpu.append(buf)

    @property
    def n(self):
        return len(self.gpu)

    def run(self, hidden, layer_fn, graphs=None, variants=None, variant_of=None):
        for i, buf in enumerate(self.gpu):
            v = variant_of[i]
            if graphs is not None:
                hidden = graphs.replay(i, v, hidden)
            else:
                bind(self.template, self.spec, buf)
                hidden = layer_fn(hidden, **variants[v])
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

    def __init__(self, layer_fn, buffers, bind_to, hidden, variants, pairs):
        self.inp = hidden.clone()
        self.graphs, self.outs = {}, {}
        pool = None
        side = torch.cuda.Stream()
        # Grouped by buffer so each one is bound once, not once per variant: binding is
        # cheap but capture is not, and rebinding between captures of the same buffer would
        # be pure noise.
        for buf_idx in sorted({b for b, _ in pairs}):
            bind_to(buffers[buf_idx])
            for var_idx in sorted({v for b, v in pairs if b == buf_idx}):
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
                self.graphs[(buf_idx, var_idx)] = g
                self.outs[(buf_idx, var_idx)] = out[0] if isinstance(out, tuple) else out

    def replay(self, buf_idx: int, var_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        self.inp.copy_(hidden)
        self.graphs[(buf_idx, var_idx)].replay()
        return self.outs[(buf_idx, var_idx)]


def build(model_name: str, quant: str):
    """Load the checkpoint and return (fused blocks, config, attention_kwargs_fn).

    The blocks are our own fused ones, not transformers' layer -- see qwen3_block.py for
    why. The transformers model is loaded only to source the weights and is dropped
    immediately, so nothing downstream depends on it.

    The third return value is what the caller uses to get per-sequence-length attention
    kwargs: `fn(cfg, seq_len) -> (variants, variant_of)`. Qwen3 has one variant for every
    layer; Gemma 3 has two, because its layers alternate sliding-window and global
    attention and the two need different kernels AND different RoPE tables.
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
    return blocks, cfg, kwargs_fn


@torch.no_grad()
def measure(cfg, runner, layer_fn, seq_len, kwargs_fn, warmup=2, reps=5, use_graphs=True):
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
    if len(runner.gpu) == len(variant_of):
        pairs = [(i, v) for i, v in enumerate(variant_of)]
    else:
        pairs = [(s, v) for s in range(len(runner.gpu)) for v in sorted(set(variant_of))]

    graphs = None
    if use_graphs:
        try:
            graphs = BlockGraphs(layer_fn, runner.gpu,
                                 lambda buf: bind(runner.template, runner.spec, buf),
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
        return runner.run(hidden, layer_fn, graphs=graphs, variants=variants,
                          variant_of=variant_of)

    pre_rep = getattr(runner, "pre_rep", lambda: None)
    for _ in range(warmup):
        pre_rep()
        fwd()
    torch.cuda.synchronize()

    ts = []
    for _ in range(reps):
        pre_rep()                    # drop the page cache; see SSDOffloadRunner
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
                    choices=["resident", "ssd", "ram"],
                    help="ssd = the real offload; ram = pinned host RAM (ceiling only)")
    ap.add_argument("--block-dir", default=str(Path.home() / ".bench_offload_blocks"),
                    help="where the per-block files live")
    ap.add_argument("--threads", type=int, default=16,
                    help="parallel preads per block; 16 is the conservative point")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-graphs", action="store_true",
                    help="skip CUDA graph capture; launch each block from Python")
    ap.add_argument("--out", default="offload_prefill.csv")
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

    rows = []
    bar = tqdm(total=len(grid), unit="pt", dynamic_ncols=True)
    for quant in args.quant:
        bar.set_description(f"{quant}: loading model")
        blocks, cfg, kwargs_fn = build(args.model, quant)
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
        template = make_template(blocks)
        layer_fn = template if args.no_compile else torch.compile(
            template, fullgraph=True, dynamic=False)
        for mode in args.modes:
            bar.set_description(f"{quant}/{mode}: staging blocks")
            # built ONCE and reused for every seq_len
            if mode == "ssd":
                # Model name in the path: without it, two models with different block
                # sizes churn the same files, rewriting them on every switch.
                runner = SSDOffloadRunner(
                    template, blocks,
                    Path(args.block_dir) / args.model.replace("/", "_") / quant,
                    threads=args.threads)
            elif mode == "ram":
                runner = RamOffloadRunner(template, blocks)
            else:
                runner = ResidentRunner(template, blocks)
            for seq in args.seq_lens:
                bar.set_description(f"{quant}/{mode}/seq={seq}")
                try:
                    ms = measure(cfg, runner, layer_fn, seq, kwargs_fn, reps=args.reps,
                                 use_graphs=not args.no_graphs)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bar.write(f"{quant:<6}{mode:<9}{seq:>7}   OOM")
                    bar.update(1)
                    continue
                row = dict(model=args.model, quant=quant, mode=mode, seq_len=seq,
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
        del blocks, template, layer_fn
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    bar.close()
    if args.out:
        flush_csv()
        print(f"\nwrote {args.out} ({len(rows)} new, {len(merged)} total)")


if __name__ == "__main__":
    main()
