"""Dump the P2pNcclConnector consumer path, to explain why the decode leg blocks.

Run inside the container. The decode engine establishes NCCL with the producer and
then schedules nothing, so the question is what its load path waits on and under
which conditions that wait can never be satisfied.
"""

import inspect
import re

import vllm
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)

print("vllm", vllm.__version__)
print("connector file", inspect.getfile(P2pNcclConnector))

for cls, names in [
    # The producer only sends for requests present in its connector metadata, and the
    # consumer only waits for those in its own. If the two disagree about which
    # requests (or which layers) are in play, the consumer blocks forever. So the
    # metadata builder and the id parser are the remaining unknowns.
    (P2pNcclConnector, ["__init__", "parse_request_id", "build_connector_meta",
                        "get_finished", "start_load_kv", "save_kv_layer",
                        "wait_for_save", "get_num_new_matched_tokens",
                        "update_state_after_alloc", "request_finished"]),
    (P2pNcclEngine, ["send_tensor", "recv_tensor", "send_sync", "wait_for_sent",
                     "_listen_for_requests", "_send_async"]),
]:
    print("\n" + "#" * 78)
    print("#", cls.__name__)
    for n in names:
        fn = getattr(cls, n, None)
        if fn is None:
            print(f"\n--- {n}: ABSENT")
            continue
        try:
            print(f"\n--- {n} ---\n{inspect.getsource(fn)}")
        except OSError as e:
            print(f"\n--- {n}: no source ({e})")

# Does this build refuse to combine async scheduling with a KV connector, or does it
# silently allow the combination? Both servers logged "Asynchronous scheduling is
# enabled" while holding a connector, which would be a deadlock candidate.
print("\n" + "#" * 78)
print("# async scheduling vs kv connector")
import vllm.config as vc
for mod in (vc, getattr(vc, "vllm", None)):
    if mod is None:
        continue
    try:
        src = inspect.getsource(mod)
    except OSError:
        continue
    for m in re.finditer(r"[^\n]*async_scheduling[^\n]*", src):
        line = m.group(0).strip()
        if "kv" in line.lower() or "connector" in line.lower() or "disable" in line.lower():
            print("  ", line)

print("\n# CLI flags mentioning async scheduling")
from vllm.engine.arg_utils import EngineArgs
for f in dir(EngineArgs):
    if "async" in f:
        print("  ", f, "=", getattr(EngineArgs, f, None))

# What request_id does the OpenAI completions layer hand the engine? The connector
# keys KV by it, so prefill and decode must produce byte-identical strings from the
# same X-Request-Id header. Anything that wraps or suffixes it differently per server
# breaks the match and hangs the consumer.
print("\n" + "#" * 78)
print("# request_id construction in the completions path")
import vllm.entrypoints.openai.serving_completion as sc
import vllm.entrypoints.openai.serving_engine as se
for mod in (sc, se):
    try:
        src = inspect.getsource(mod)
    except OSError:
        continue
    for m in re.finditer(r"[^\n]*(request_id\s*=|_base_request_id)[^\n]*", src):
        print(f"  [{mod.__name__.split('.')[-1]}] {m.group(0).strip()}")
try:
    print("\n--- _base_request_id ---\n" + inspect.getsource(se.OpenAIServing._base_request_id))
except Exception as e:
    print("  _base_request_id:", e)
