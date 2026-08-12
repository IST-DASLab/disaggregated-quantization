"""1P1D disaggregated-serving proxy for vLLM's NixlConnector.

Fronts a prefill server and a decode server with one OpenAI-compatible endpoint so
lm-eval (or anything else) can drive a disaggregated pair as if it were one model.

Why this proxy has to be STATEFUL
---------------------------------
NixlConnector routes at the *request* level, not through the request id. The
handshake is carried in the `kv_transfer_params` field of the OpenAI request/response
bodies:

  1. POST the prompt to PREFILL with `max_tokens=1` and
     `kv_transfer_params={"do_remote_decode": true}`. The prefill engine computes the
     KV, keeps the blocks pinned, and returns `kv_transfer_params` in its response
     body describing where they live:
        {"do_remote_prefill": true, "remote_engine_id": ..., "remote_block_ids":
         [...], "remote_host": ..., "remote_port": ...}
  2. POST the ORIGINAL request to DECODE with those returned params attached. The
     decode engine pulls the blocks over NIXL/UCX from the prefill engine's side
     channel and generates without recomputing the prompt.

This is the same protocol as vLLM's
`examples/online_serving/disaggregated_serving/disagg_proxy_demo.py`, kept minimal
for 1P1D.  Contrast with p2p_proxy.py: P2pNcclConnector ignores
`kv_transfer_params` entirely and encodes peer addresses in the request id instead,
so the two proxies are NOT interchangeable in either direction.

THE SILENT FAILURE THIS GUARDS AGAINST
--------------------------------------
If the prefill response comes back with no `kv_transfer_params`, the decode engine
gets an ordinary request and simply recomputes the prompt with its OWN weights. The
answer is fluent and plausible and nothing looks wrong — but the "disaggregated"
result is really homogeneous-decode. So by default this proxy REFUSES to forward a
request whose prefill leg produced no transfer params (HTTP 502), rather than
degrading quietly. `--allow-missing-kv-params` turns that back into a warning if you
explicitly want to see the fallback behaviour.

The servers should additionally be started with
`"kv_load_failure_policy":"fail"` so that a transfer which is *attempted* and fails
raises instead of falling back to local recompute.

    python nixl_proxy.py --port 8595 \
        --prefill-host 127.0.0.1 --prefill-port 8500 \
        --decode-host  127.0.0.1 --decode-port  8600
"""

import argparse
import itertools
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Configure logging at MODULE level, not in __main__. With --workers>1 uvicorn forks
# workers that re-import this module, so a basicConfig() call guarded by __main__ never
# runs in them and every request log vanishes -- including the "prefill ok" line the
# sweep monitors count for progress. Silence then looks like a stalled run.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s [pid %(process)d] %(message)s")
logger = logging.getLogger("nixl_proxy")
ARGS = None

# Counters, exposed on /healthcheck. `kv_ok` vs `kv_missing` is the cheap online
# answer to "is KV actually crossing?" — a run with kv_missing>0 is not
# disaggregated regardless of how good its completions look.
STATS = {"requests": 0, "kv_ok": 0, "kv_missing": 0, "prefill_err": 0, "decode_err": 0,
         "prefill_complete": 0}


def _finished_in_prefill(pre_json: dict) -> str | None:
    """finish_reason if the model STOPPED on its own during the prefill leg, else None.

    The prefill leg is sent with max_tokens=1, so a request that still needs decoding
    comes back with finish_reason="length" (it hit that cap) plus kv_transfer_params to
    hand off. A model that emits EOS or a stop string as its very first token is instead
    already DONE: there is nothing to hand off, so vLLM correctly returns no
    kv_transfer_params.

    That is not an error, but it is indistinguishable from a misconfigured kv_producer
    unless you look at finish_reason -- which is why this exists. A misconfigured
    producer returns "length" (generation was cut off and should have continued) with no
    KV params; an early stop returns "stop". Only the latter is safe to serve directly.
    """
    for ch in (pre_json.get("choices") or []):
        fr = ch.get("finish_reason")
        if fr and fr != "length":
            return fr
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=100_000, max_keepalive_connections=100_000)
    # Short leash on the prefill leg: it is a max_tokens=1 call, so seconds. A long
    # stall there means the NIXL side channel handshake is stuck, and under the
    # decode timeout that is indistinguishable from a slow eval.
    app.state.prefill = httpx.AsyncClient(
        base_url=f"http://{ARGS.prefill_host}:{ARGS.prefill_port}/v1",
        limits=limits, timeout=httpx.Timeout(ARGS.prefill_timeout, connect=60.0))
    app.state.decode = httpx.AsyncClient(
        base_url=f"http://{ARGS.decode_host}:{ARGS.decode_port}/v1",
        limits=limits, timeout=httpx.Timeout(6 * 60 * 60.0, connect=60.0))
    app.state.counter = itertools.count()
    logger.info("proxy ready: prefill=%s:%s decode=%s:%s",
                ARGS.prefill_host, ARGS.prefill_port,
                ARGS.decode_host, ARGS.decode_port)
    yield
    await app.state.prefill.aclose()
    await app.state.decode.aclose()


app = FastAPI(lifespan=lifespan)


def _headers(request_id: str) -> dict:
    h = {"X-Request-Id": request_id}
    if os.environ.get("OPENAI_API_KEY"):
        h["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    return h


def _summarise(params: dict) -> str:
    """One-line, log-safe view of kv_transfer_params (block id lists are long)."""
    if not params:
        return "<none>"
    blocks = params.get("remote_block_ids")
    short = {k: v for k, v in params.items() if k != "remote_block_ids"}
    if blocks is not None:
        short["remote_block_ids"] = f"[{len(blocks)} blocks]"
    return json.dumps(short, default=str)


async def _handle(api: str, request: Request):
    body = await request.json()
    request_id = f"nixl-{next(app.state.counter)}-{uuid.uuid4().hex[:12]}"
    STATS["requests"] += 1

    # ---- 1) prefill leg -----------------------------------------------------
    pre = dict(body)
    pre["stream"] = False
    pre["max_tokens"] = 1
    if "max_completion_tokens" in pre:
        pre["max_completion_tokens"] = 1
    pre.pop("stream_options", None)
    # Seed the handshake. The full dict (rather than just do_remote_decode) matches
    # what vLLM's own example sends; the connector reads the missing keys as None.
    pre["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    try:
        r = await app.state.prefill.post(api, json=pre, headers=_headers(request_id))
        r.raise_for_status()
        await r.aread()
        pre_json = r.json()
    except Exception as e:                       # noqa: BLE001 - surfaced to caller
        STATS["prefill_err"] += 1
        # vLLM puts the ACTUAL reason (bad field, context-length overflow, ...) in the
        # response body. Without it a 400 reaches the caller as an opaque
        # "Client error '400 Bad Request'", which says nothing about what to fix.
        detail = ""
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                detail = resp.text[:600]
            except Exception:                    # noqa: BLE001
                pass
        # Also describe the `prompt` we sent: its type and size are what distinguish
        # "wrong element type" from "empty", and neither is visible in sent_keys.
        pr = pre.get("prompt")
        shape = f"{type(pr).__name__}"
        try:
            shape += f"(len={len(pr)})"
            if isinstance(pr, list) and pr:
                shape += f" first={type(pr[0]).__name__}"
                if isinstance(pr[0], list):
                    shape += f"(len={len(pr[0])})"
        except TypeError:
            pass
        logger.error("prefill leg failed (%s): %s: %s | body=%s | sent_keys=%s "
                     "| prompt=%s | preview=%.160r",
                     request_id, type(e).__name__, e, detail, sorted(pre), shape, pr)
        return JSONResponse(status_code=502,
                            content={"error": f"prefill: {type(e).__name__}: {e}",
                                     "prefill_body": detail})

    kv = pre_json.get("kv_transfer_params") or {}
    logger.info("%s prefill ok; kv_transfer_params=%s", request_id, _summarise(kv))

    if not kv:
        # The request may simply be FINISHED: a model whose first token is EOS/a stop
        # string has nothing left to decode, so vLLM returns no KV params by design.
        # Serving the prefill response is then correct and complete -- there is no
        # decode leg to run. Failing it instead makes every badly-degraded checkpoint
        # unevaluable, which is exactly when the measurement matters: step-0 (PTQ)
        # 2-bit checkpoints emit EOS immediately on ~6% of prompts, and 502-ing those
        # dropped the client connection and killed the whole job.
        #
        # NOT streaming-safe, and it does not need to be: this path returns a complete
        # non-streamed body, and the eval client never sets stream=true (verified in the
        # proxy logs). A streaming client falls through to the error below rather than
        # being handed a body it cannot parse.
        fr = _finished_in_prefill(pre_json) if not body.get("stream") else None
        if fr:
            STATS["prefill_complete"] += 1
            logger.info("%s finished during prefill (finish_reason=%s); returning the "
                        "prefill response, no decode leg needed", request_id, fr)
            return JSONResponse(status_code=200, content=pre_json)

        STATS["kv_missing"] += 1
        msg = (f"prefill returned no kv_transfer_params for {request_id} and did not "
               f"finish on its own (finish_reason=length); the decode server would "
               f"recompute the prompt with its own weights (silent fallback). Check "
               f"that the prefill server was started with kv_role=kv_producer and a "
               f"NixlConnector kv-transfer-config.")
        if not ARGS.allow_missing_kv_params:
            logger.error(msg)
            return JSONResponse(status_code=502, content={"error": msg})
        logger.warning("%s (continuing because --allow-missing-kv-params)", msg)
    else:
        STATS["kv_ok"] += 1

    # ---- 2) decode leg ------------------------------------------------------
    dec = dict(body)
    if kv:
        dec["kv_transfer_params"] = kv

    if body.get("stream"):
        async def gen():
            async with app.state.decode.stream(
                    "POST", api, json=dec, headers=_headers(request_id)) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    # Stream the decode body straight through. The proxy has no reason to understand
    # it -- only the PREFILL response is parsed, for kv_transfer_params. Previously this
    # did r.json() then re-serialised via JSONResponse, i.e. two full JSON transits of a
    # 4096-token completion per request, in one event loop. That is CPU work the GPUs
    # then wait on.
    try:
        req = app.state.decode.build_request("POST", api, json=dec,
                                             headers=_headers(request_id))
        resp = await app.state.decode.send(req, stream=True)
        if resp.status_code >= 400:
            body = await resp.aread()
            await resp.aclose()
            STATS["decode_err"] += 1
            logger.error("decode leg failed (%s): HTTP %s: %.400s",
                         request_id, resp.status_code, body.decode("utf-8", "replace"))
            return Response(content=body, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type", "application/json"))

        async def passthrough():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(
            passthrough(), status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"))
    except Exception as e:                       # noqa: BLE001
        STATS["decode_err"] += 1
        logger.exception("decode leg failed (%s)", request_id)
        return JSONResponse(status_code=502,
                            content={"error": f"decode: {type(e).__name__}: {e}"})


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.get("/v1/models")
async def models():
    """lm-eval and OpenAI clients probe this; forward the decode server's answer."""
    r = await app.state.decode.get("/models")
    return JSONResponse(status_code=r.status_code, content=r.json())


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok",
            "prefill": f"{ARGS.prefill_host}:{ARGS.prefill_port}",
            "decode": f"{ARGS.decode_host}:{ARGS.decode_port}",
            "stats": STATS}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=int(os.environ.get("NIXL_PROXY_WORKERS", 1)),
                   help="uvicorn worker processes. The proxy is CPU-bound on JSON and "
                        "HTTP framing, so one event loop caps throughput below what the "
                        "engines can serve. >1 requires the settings to travel through "
                        "the environment, since each worker re-imports this module.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8595)
    p.add_argument("--prefill-host", default="127.0.0.1")
    p.add_argument("--prefill-port", type=int, required=True)
    p.add_argument("--decode-host", default="127.0.0.1")
    p.add_argument("--decode-port", type=int, required=True)
    p.add_argument("--prefill-timeout", type=float, default=300.0)
    p.add_argument("--allow-missing-kv-params", action="store_true",
                   help="downgrade 'prefill returned no kv_transfer_params' from a "
                        "502 to a warning. Only for deliberately observing the "
                        "silent-fallback path.")
    return p.parse_args()


def _args_from_env():
    """Rebuild ARGS inside a uvicorn worker, which re-imports this module fresh."""
    blob = os.environ.get("NIXL_PROXY_ARGS")
    if not blob:
        return None
    import types
    return types.SimpleNamespace(**json.loads(blob))


# Workers re-import the module rather than inheriting __main__ state, so ARGS must come
# back from the environment.
if ARGS is None:
    ARGS = _args_from_env()


if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.workers > 1:
        os.environ["NIXL_PROXY_ARGS"] = json.dumps(vars(ARGS))
        # uvicorn needs an import string (not the app object) to fork workers
        uvicorn.run("nixl_proxy:app", host=ARGS.host, port=ARGS.port,
                    log_level="warning", workers=ARGS.workers)
    else:
        uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
