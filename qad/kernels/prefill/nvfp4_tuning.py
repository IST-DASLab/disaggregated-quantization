"""Per-shape NVFP4 GEMM plans, MEASURED on the box that will run them.

    python tune_nvfp4.py            # re-measure and rewrite the table below
    python tune_nvfp4.py --show     # print the current table without measuring

WHY A TABLE AND NOT A RULE
--------------------------
There are two fp4 GEMM kernels reachable from here -- vLLM's `cutlass_scaled_fp4_mm` and
FlashInfer's `mm_fp4` -- and neither wins everywhere on GB10:

    gemma4b  gate_up (20480x2560)  M=16384   vllm 18.2 ms   flashinfer  9.5 ms
    gemma12b down    (3840x15360)  M=16384   vllm  8.6 ms   flashinfer 12.9 ms

Both kernels have a collapse on tall outputs, at different places: vLLM's starts between
M=10240 and M=12288, FlashInfer's only at M=32768 on the larger `gate_up` shapes. Chunking
the M axis rescues whichever one is in its bad region, so the choice is three-way --
kernel, chunk, or both -- and it is a property of (shape, rows, DEVICE), not of any of them
alone. Any closed-form rule that fits GB10 is a guess on a 5090, which is exactly the port
this table exists to survive: re-run the tuner there and the numbers below are replaced
with that box's, with no code change.

The entries are BITWISE-VERIFIED against the unchunked vLLM call before being timed, so a
plan can only ever be a speed choice and never an accuracy one.

An untuned (device, shape) returns None and `NVFP4Linear` falls back to its own L2
heuristic -- i.e. a fresh box behaves exactly as it did before this file existed.
"""

# --- BEGIN GENERATED TABLE (tune_nvfp4.py rewrites everything between these markers) ---
# device -> {(out_features, in_features): ((max_rows, backend, chunk), ...)}
# chunk == 0 means "one call, no chunking". A lookup takes the first rule whose max_rows
# is >= the row count being run, so the rules must stay sorted by max_rows.
DEVICE_TABLES = {
    'NVIDIA GB10': {
        (96, 5120): ((32768, 'flashinfer', 0),),
        (640, 1024): ((32768, 'vllm', 0),),
        (640, 2048): ((32768, 'vllm', 0),),
        (1024, 2048): ((2048, 'flashinfer', 0), (32768, 'vllm', 0),),
        (1024, 3072): ((2048, 'flashinfer', 0), (32768, 'vllm', 0),),
        (1152, 1024): ((2048, 'flashinfer', 0), (32768, 'vllm', 0),),
        (1152, 6912): ((2048, 'flashinfer', 0), (32768, 'vllm', 0),),
        (1536, 640): ((2048, 'flashinfer', 0), (32768, 'vllm', 0),),
        (1536, 1152): ((32768, 'vllm', 0),),
        (2048, 2048): ((32768, 'vllm', 0),),
        (2048, 6144): ((32768, 'vllm', 0),),
        (2560, 2048): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (2560, 4096): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (2560, 9728): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (2560, 10240): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (3840, 4096): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (3840, 15360): ((2048, 'vllm', 0), (32768, 'vllm', 2048),),
        (4096, 640): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (4096, 1024): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (4096, 2048): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (4096, 2560): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (4096, 4096): ((32768, 'vllm', 0),),
        (4096, 12288): ((2048, 'vllm', 0), (4096, 'vllm', 2048), (16384, 'vllm', 1536), (32768, 'vllm', 2048),),
        (5120, 6144): ((4096, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'vllm', 0),),
        (5120, 17408): ((2048, 'vllm', 0), (8192, 'flashinfer', 1024), (16384, 'flashinfer', 1280), (32768, 'vllm', 2048),),
        (6144, 1024): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (6144, 2560): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (6144, 4096): ((4096, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'vllm', 0),),
        (8192, 3840): ((2048, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'vllm', 0),),
        (12288, 2048): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (8192, 'vllm', 0), (16384, 'flashinfer', 0), (32768, 'vllm', 0),),
        (13824, 1152): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (32768, 'vllm', 0),),
        (14336, 5120): ((4096, 'vllm', 0), (8192, 'flashinfer', 0), (16384, 'vllm', 1280), (32768, 'flashinfer', 2688),),
        (16384, 5120): ((4096, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'flashinfer', 2688),),
        (16480, 5120): ((4096, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'flashinfer', 2688),),
        (19456, 2560): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (8192, 'vllm', 0), (32768, 'flashinfer', 0),),
        (20480, 2560): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (8192, 'vllm', 0), (32768, 'flashinfer', 0),),
        (24576, 4096): ((2048, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'flashinfer', 2688),),
        (30720, 3840): ((2048, 'vllm', 0), (4096, 'flashinfer', 0), (8192, 'vllm', 0), (32768, 'flashinfer', 2688),),
        (34816, 5120): ((4096, 'vllm', 0), (8192, 'flashinfer', 0), (32768, 'flashinfer', 2688),),
    },
}
# --- END GENERATED TABLE ---


def rules(device: str, out_features: int, in_features: int):
    """The measured rules for this shape on this device, or () if it was never tuned.

    Callers resolve this ONCE, outside any traced region -- `NVFP4Linear` does it in
    __init__ -- because the device lookup is a torch.* call and the consumer runs under
    torch.compile(fullgraph=True).
    """
    return DEVICE_TABLES.get(device, {}).get((out_features, in_features), ())


def plan(device: str, out_features: int, in_features: int, rows: int):
    """(backend, chunk_rows) for this shape at this height, or None if never tuned.

    Pure integer/string work on values that are static at trace time, so this survives
    `torch.compile(fullgraph=True)` -- which the block is compiled with, and which rules
    out anything that has to look at a tensor to decide.
    """
    rules = DEVICE_TABLES.get(device, {}).get((out_features, in_features))
    if not rules:
        return None
    for max_rows, backend, chunk in rules:
        if rows <= max_rows:
            return backend, (chunk or rows)
    # Taller than anything measured: the last rule is the closest evidence there is.
    _, backend, chunk = rules[-1]
    return backend, (chunk or rows)
