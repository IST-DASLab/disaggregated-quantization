"""1P1D disaggregated-serving proxy for vLLM's P2pNcclConnector.

Fronts a prefill server and a decode server with one OpenAI-compatible endpoint, so
lm-eval (or anything else) can talk to a disaggregated pair as if it were a single
model. The prefill worker serves the W4A4 checkpoint and the decode worker the W4A16
one, which is the deployment our dual-format checkpoints are built for.

Why not toy_proxy_server.py
---------------------------
That proxy implements vLLM's *request-level* KV handshake: it seeds
`kv_transfer_params` with `do_remote_decode`, reads the connector's params back out
of the prefill response, and forwards them to decode. NixlConnector fills those in.
**P2pNcclConnector does not** — its `request_finished()` returns `(False, None)` and
its source never mentions `kv_transfer_params`. Driving P2P NCCL with that proxy
fails SILENTLY: the decode server receives no transfer parameters, recomputes the
prompt with its own weights, and returns a fluent, plausible answer. The eval would
report a disaggregated number that is really homogeneous-decode.

How P2P NCCL actually routes
----------------------------
The connector recovers both peer addresses from the request id itself:

    r"___prefill_addr_(.*):(\\d+)___"      # producer's ZMQ address
    r"___decode_addr_(.*):(\\d+)"          # consumer's ZMQ address

so the id must look like

    ___prefill_addr_<host>:<kv_port>___decode_addr_<host>:<kv_port>_<uuid>

and both servers must see the SAME id. vLLM's OpenAI server adopts a client-supplied
`X-Request-Id` (serving_engine._base_request_id) and wraps it as `cmpl-<id>`; the
connector matches with `re.search`, so that prefix is harmless.

Sequence per request:
  1. POST to prefill with max_tokens=1 and the crafted X-Request-Id. The producer
     computes the KV for the prompt and pushes it to the consumer's ZMQ address.
  2. POST the ORIGINAL request to decode with the same X-Request-Id. The consumer
     finds the transferred KV under that id and generates from it, so the prompt is
     never recomputed with the decode weights.

Because the addresses travel in the id, the engines need no registration ping — the
servers are started WITHOUT `proxy_ip`/`proxy_port`, which also means `http_port` is
not required.

    python p2p_proxy.py --port 8595 \
        --prefill-host 127.0.0.1 --prefill-port 8500 --prefill-kv-port 21001 \
        --decode-host  127.0.0.1 --decode-port  8600 --decode-kv-port  22001
"""

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("p2p_proxy")
ARGS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=100_000, max_keepalive_connections=100_000)
    timeout = httpx.Timeout(6 * 60 * 60.0, connect=60.0)
    # Short leash on the prefill leg. It runs max_tokens=1 over one prompt, so it is
    # a seconds-scale call; anything longer means the KV push is stuck (typically a
    # wrong peer address -- see make_request_id). Under the 6h decode timeout that
    # stall is indistinguishable from a slow eval, so bound it separately and let the
    # request fail loudly.
    app.state.prefill = httpx.AsyncClient(
        base_url=f"http://{ARGS.prefill_host}:{ARGS.prefill_port}/v1",
        limits=limits, timeout=httpx.Timeout(ARGS.prefill_timeout, connect=60.0))
    app.state.decode = httpx.AsyncClient(
        base_url=f"http://{ARGS.decode_host}:{ARGS.decode_port}/v1",
        limits=limits, timeout=timeout)
    app.state.counter = itertools.count()
    logger.info("proxy ready: prefill=%s:%s(kv %s) decode=%s:%s(kv %s)",
                ARGS.prefill_host, ARGS.prefill_port, ARGS.prefill_kv_port,
                ARGS.decode_host, ARGS.decode_port, ARGS.decode_kv_port)
    yield
    await app.state.prefill.aclose()
    await app.state.decode.aclose()


app = FastAPI(lifespan=lifespan)


def make_request_id() -> str:
    """Encode both ZMQ addresses so the connector can find its peer (see module doc).

    Two traps here, both of which hang rather than error:

    * The ports are the connectors' `kv_port`, NOT the HTTP ports.
    * The host must be the address the engines actually BOUND to. P2pNcclEngine does
      `hostname = get_ip()` and binds its router socket to `tcp://<that ip>:<kv_port>`
      -- a real interface address, never loopback. A ZMQ socket bound to a specific
      interface is unreachable through 127.0.0.1 even on the same machine, so
      advertising loopback makes the producer connect to nothing and block forever
      with no error on either side. run_disagg_server.sh pins VLLM_HOST_IP and passes
      the same value here so the two cannot drift.
    """
    return (f"___prefill_addr_{ARGS.kv_host}:{ARGS.prefill_kv_port}"
            f"___decode_addr_{ARGS.kv_host}:{ARGS.decode_kv_port}"
            f"_{uuid.uuid4().hex}")


def _headers(request_id: str) -> dict:
    h = {"X-Request-Id": request_id}
    if os.environ.get("OPENAI_API_KEY"):
        h["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    return h


async def _handle(api: str, request: Request):
    body = await request.json()
    request_id = make_request_id()

    # 1) prefill: one token is enough to populate and publish the KV cache.
    pre = dict(body)
    pre["stream"] = False
    pre["max_tokens"] = 1
    if "max_completion_tokens" in pre:
        pre["max_completion_tokens"] = 1
    pre.pop("stream_options", None)
    try:
        r = await app.state.prefill.post(api, json=pre, headers=_headers(request_id))
        r.raise_for_status()
        await r.aread()
    except Exception as e:                       # noqa: BLE001 - surfaced to caller
        logger.exception("prefill failed")
        return JSONResponse(status_code=502,
                            content={"error": f"prefill: {type(e).__name__}: {e}"})

    # 2) decode: the ORIGINAL request, same id, so the consumer picks up that KV.
    if body.get("stream"):
        async def gen():
            async with app.state.decode.stream(
                    "POST", api, json=body, headers=_headers(request_id)) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        r = await app.state.decode.post(api, json=body, headers=_headers(request_id))
        r.raise_for_status()
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:                       # noqa: BLE001
        logger.exception("decode failed")
        return JSONResponse(status_code=502,
                            content={"error": f"decode: {type(e).__name__}: {e}"})


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.get("/v1/models")
async def models(request: Request):
    """lm-eval and OpenAI clients probe this; forward the decode server's answer."""
    r = await app.state.decode.get("/models")
    return JSONResponse(status_code=r.status_code, content=r.json())


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok",
            "prefill": f"{ARGS.prefill_host}:{ARGS.prefill_port}",
            "decode": f"{ARGS.decode_host}:{ARGS.decode_port}"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8595)
    p.add_argument("--prefill-host", default="127.0.0.1")
    p.add_argument("--prefill-port", type=int, required=True)
    p.add_argument("--prefill-kv-port", type=int, required=True,
                   help="the prefill engine's kv_port (ZMQ), not its HTTP port")
    p.add_argument("--decode-host", default="127.0.0.1")
    p.add_argument("--decode-port", type=int, required=True)
    p.add_argument("--decode-kv-port", type=int, required=True,
                   help="the decode engine's kv_port (ZMQ), not its HTTP port")
    p.add_argument("--prefill-timeout", type=float, default=300.0,
                   help="seconds to wait for the prefill leg before failing the "
                        "request; generous for a max_tokens=1 call, tight enough to "
                        "expose a stuck KV push")
    p.add_argument("--kv-host", default=None,
                   help="host part of the ZMQ addresses put in the request id. MUST "
                        "equal what the engines bound to, i.e. vllm's get_ip() / "
                        "VLLM_HOST_IP -- see make_request_id(). Defaults to "
                        "--prefill-host, which is only correct if the engines were "
                        "also told to bind there.")
    a = p.parse_args()
    if a.kv_host is None:
        a.kv_host = a.prefill_host
    return a


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ARGS = parse_args()
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
