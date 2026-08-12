"""P2pNcclConnector that makes producer and consumer agree on the KV key.

THE BUG THIS FIXES
------------------
P2pNcclConnector keys every transferred tensor `<request_id>#<layer_name>`, and the
documented xPyD scheme assumes both servers derive an identical request_id from the
`X-Request-Id` the proxy sets. In this vLLM build they do NOT: the engine appends
`-<index>-<8 hex salt>` and the salt is drawn independently on each server.

    producer: cmpl-___prefill_addr_...___decode_addr_..._<uuid>-0-a9a2cbf5#...layers.0...
    consumer: cmpl-___prefill_addr_...___decode_addr_..._<uuid>-0-910280a4#...layers.0...
                                                              ^^^^^^^^ differs

Observed directly: the consumer's store held all 28 layers (store_size=28) while it
waited on a key that never arrives. Because recv_tensor()'s PUT_ASYNC path is

    while tensor_id not in self.recv_store:
        self.recv_store_cv.wait()          # no timeout, no error, no log

the decode engine then blocks forever in complete silence. No proxy can fix this from
outside the server: the salt is added inside the engine.

THE FIX
-------
Strip that trailing salt from the key on both sides, in the one place both sides pass
through. The proxy already puts a fresh uuid4 in every request id, so the normalized
key stays unique per request; the `-<index>` is preserved so multi-prompt requests
keep distinct keys.

Wired in via the supported extension point, so no vLLM files are touched:

    --kv-transfer-config '{"kv_connector":"P2pNcclConnectorStableId",
                           "kv_connector_module_path":"p2p_trace", ...}'

Set DISAGG_TRACE_VERBOSE=1 to also print every key (grep TRACE_KV).

The decode engine hangs inside P2pNcclEngine.recv_tensor(), whose PUT_ASYNC path is

    while tensor_id not in self.recv_store:
        self.recv_store_cv.wait()          # no timeout, no error, no log

so a producer/consumer disagreement about the key `<request_id>#<layer_name>` stalls
the engine forever in total silence. Reading the source says the two sides build that
key identically, and yet a homogeneous pair (same checkpoint, same flags) still hangs
-- so the source reading is missing something. This prints what each side ACTUALLY
uses instead of what it ought to.

Wired in via the supported extension point, so no vLLM files are touched:

    --kv-transfer-config '{"kv_connector":"P2pNcclConnectorTraced",
                           "kv_connector_module_path":"p2p_trace", ...}'

Every line is prefixed TRACE_KV so it survives grep through very noisy server logs.

A tracer must not change what it measures. Two rules follow, both learned the hard
way: never put a value in boolean context (block_ids is a torch.Tensor, and
`x or []` raises "Boolean value of Tensor with more than one value is ambiguous",
which killed both engines before a single transfer), and never let a formatting
error escape into the engine.
"""

import os
import re
import threading

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)

_LOCK = threading.Lock()
_VERBOSE = bool(os.environ.get("DISAGG_TRACE_VERBOSE"))

# Trailing "-<index>-<salt>" appended per-server by the engine. Anchored, and the salt
# must be hex, so a layer name or a caller-supplied id is never mangled. If a build
# does not add the suffix this simply does not match and the key passes through.
_SALT_RE = re.compile(r"-(\d+)-[0-9a-f]{6,16}$")


def canonical_key(tensor_id: str) -> str:
    """`<request_id>#<layer>` with the per-server salt removed from the request part."""
    req, sep, layer = tensor_id.partition("#")
    return _SALT_RE.sub(r"-\1", req) + sep + layer


def _emit(role: str, event: str, **kw) -> None:
    try:
        if not _VERBOSE:
            return
        fields = " ".join(f"{k}={v}" for k, v in kw.items())
        with _LOCK:
            print(f"TRACE_KV role={role} event={event} {fields}", flush=True)
    except Exception:                                     # noqa: BLE001
        pass          # tracing must never be the reason a request fails


def _fingerprint(t) -> dict:
    """Shape + a cheap content hash, so the two ends can be compared directly.

    A matching fingerprint means the transport is faithful and any corruption is in
    WHERE the data lands; a differing one means the wire itself is wrong. Guarded
    because this runs inside the engine and must never raise. float64 accumulation
    keeps the sum from saturating in bf16.
    """
    if t is None:
        return {}
    try:
        return {"shape": tuple(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                "absum": f"{t.detach().float().abs().sum().item():.6e}"}
    except Exception:                                     # noqa: BLE001
        return {"shape": "?"}


def _count(x) -> object:
    """len() that never evaluates truthiness (see module docstring)."""
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return "?"


class P2pNcclConnectorStableId(P2pNcclConnector):
    """Logs every key the producer sends and every key the consumer waits for.

    The engine's send_tensor/recv_tensor are wrapped rather than the connector's
    save_kv_layer/start_load_kv, because the wrap then sees the exact string that
    reaches the store -- the connector methods would only show what we think they
    pass. recv_tensor is logged BEFORE the call, since the hanging call never
    returns and a post-hoc log would never print.

    vLLM builds one connector per role (scheduler-side has no engine, worker-side
    does), so both instances announce themselves and only the engine-bearing one
    is wrapped.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        role = self._trace_role()
        engine = getattr(self, "p2p_nccl_engine", None)
        _emit(role, "init", has_engine=engine is not None,
              rank=getattr(self, "_rank", "?"))
        if engine is None or getattr(engine, "_traced", False):
            return
        engine._traced = True

        real_send, real_recv = engine.send_tensor, engine.recv_tensor

        def send_tensor(tensor_id, tensor, remote_address=None):
            key = canonical_key(tensor_id)
            _emit(role, "send", tensor_id=key, to=remote_address, **_fingerprint(tensor))
            return real_send(key, tensor, remote_address)

        def recv_tensor(tensor_id, remote_address=None):
            key = canonical_key(tensor_id)
            store = getattr(engine, "recv_store", {})
            _emit(role, "recv_enter", tensor_id=key, raw=tensor_id, frm=remote_address,
                  already_present=key in store, store_size=_count(store))
            out = real_recv(key, remote_address)
            _emit(role, "recv_done", tensor_id=key, got=out is not None, **_fingerprint(out))
            return out

        engine.send_tensor = send_tensor
        engine.recv_tensor = recv_tensor

    def _trace_role(self) -> str:
        # NOT `_role`: the base class stores a KVConnectorRole enum on that attribute,
        # and an instance attribute shadows a method, giving
        # "TypeError: 'KVConnectorRole' object is not callable" at construction.
        return "producer" if self.is_producer else "consumer"

    def start_load_kv(self, forward_context, **kwargs):
        # Called on BOTH servers; the base returns immediately when is_producer.
        # n_requests==0 on the consumer would mean the hang is upstream of the keys.
        meta = self._get_connector_metadata()
        reqs = getattr(meta, "requests", None)
        _emit(self._trace_role(), "start_load_kv", n_requests=_count(reqs),
              n_layers=_count(getattr(forward_context, "no_compile_layers", None)),
              attn_meta=getattr(forward_context, "attn_metadata", None) is not None)
        for r in (reqs or ()):
            _emit(self._trace_role(), "load_req",
                  request_id=getattr(r, "request_id", "?"),
                  n_blocks=_count(getattr(r, "block_ids", None)))
        return super().start_load_kv(forward_context, **kwargs)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        meta = self._get_connector_metadata()
        reqs = getattr(meta, "requests", None)
        _emit(self._trace_role(), "save_kv_layer", layer_name=layer_name,
              n_requests=_count(reqs))
        for r in (reqs or ()):
            _emit(self._trace_role(), "save_req", request_id=getattr(r, "request_id", "?"))
        return super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
