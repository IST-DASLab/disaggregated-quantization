"""Shared client for driving a local vLLM endpoint from an eval harness.

Every benchmark here talks to the same OpenAI-compatible server, and every one of them
needs the same three properties, so they live in one place rather than three.

1. PER-ITEM DURABILITY. Results are appended and fsynced as they land. The 4h wall
   clock is shorter than a full MMMU-Pro `vision` pass against a 30B reasoning model,
   so a run WILL be killed mid-flight; a restart must keep what it already paid for.
   Harness-native "resume" is usually all-or-nothing per file (MMMU-Pro) or rewrites
   the whole file every N items (IFBench), neither of which survives a SIGKILL.

2. FAILURES ARE NOT RECORDED. A transient 500 or a read timeout leaves the item absent
   from the output file, so the next run retries it. IFBench's own generator writes an
   empty response on error instead, which is indistinguishable from a model that
   answered with nothing -- a network blip becomes a permanent zero on that item.

3. NO CLIENT-SIDE SAMPLING. The payload carries `max_tokens` and nothing else, so the
   server applies the model's own generation_config (temperature 1.0 / top_p 0.95 /
   top_k 64 for Muse-Glimmer) via --generation-config auto. Sending explicit values --
   which both upstream harnesses do, IFBench at temperature 0.6 and a fixed seed --
   silently substitutes the harness author's settings for the vendor's and makes the
   result incomparable with the published number we are trying to reproduce.

REASONING AND TRUNCATION. This model emits channel-scoped messages: the chain lands in
`reasoning_content` and the answer in `content`. Only `content` is scored. If the token
budget runs out mid-chain the request returns finish_reason="length" with an EMPTY
content and no error, which scores as a wrong answer -- so an under-sized budget is
indistinguishable from a weak model. `finish_reason` and `completion_tokens` are
recorded on every row precisely so that difference is measurable after the fact.
"""
import json
import os
import threading
import time

import requests
from tqdm import tqdm

DEFAULT_BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000/v1")
# The name the SERVER answers to (--served-model-name), not the checkpoint. The launcher
# exports it from models.json; a wrong value here is a 404 from the endpoint, not a
# quietly different model.
DEFAULT_MODEL = os.environ.get("EVAL_MODEL", "muse-glimmer")
DEFAULT_MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "32768"))
# Model-specific NON-SAMPLING request fields, as JSON, from models.json via run_eval.sh.
# This does not weaken point 3 above -- it is not for temperature or top_p, which still
# come from the server, and the launcher's guard still fails a job whose server did not
# apply the vendor's generation_config.
#
# It exists because some models put a REQUIRED stop condition in the request body rather
# than in generation_config, where no server-side setting can supply it. Nemotron is one:
# thinking mode has no forced end without `thinking_token_budget`, so 14% of MMMU items
# ran to the 32768-token cap and came back with EMPTY content -- 989 of 6920, against 2
# for Gemma on the same benchmark. That is not a weak model, it is an unconfigured one,
# and it read as a 14-point deficit.
DEFAULT_EXTRA_BODY = json.loads(os.environ.get("EVAL_EXTRA_BODY", "") or "{}")
DEFAULT_TIMEOUT = int(os.environ.get("EVAL_TIMEOUT", "1800"))
DEFAULT_WORKERS = int(os.environ.get("EVAL_WORKERS", "32"))
DEFAULT_RETRIES = int(os.environ.get("EVAL_RETRIES", "3"))


# ONE pooled Session per thread, not requests.post per call. The module-level
# requests.post builds a fresh Session -- and so a fresh TCP connection -- for every
# request and discards it. At WORKERS=32 that was invisible; at 512 it is 512 threads
# each opening and closing a connection per item, which burns ephemeral ports into
# TIME_WAIT and adds a handshake to the latency of every single request.
#
# Thread-local rather than one shared Session: urllib3's connection pool defaults to
# pool_maxsize=10 per host, so a single shared Session would funnel all 512 threads
# through 10 connections and serialise them -- the opposite of the intent. One Session
# per thread keeps its own keep-alive connection.
_tls = threading.local()


def _session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=4, max_retries=0))
        _tls.s = s
    return s


def chat(messages, max_tokens=None, base_url=None, model=None, timeout=None,
         retries=None):
    """One chat completion. Returns (content, reasoning, finish_reason, n_tokens)."""
    base_url = base_url or DEFAULT_BASE_URL
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    timeout = timeout or DEFAULT_TIMEOUT
    retries = DEFAULT_RETRIES if retries is None else retries
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    # Merged, never overriding: model/messages/max_tokens are the harness's to set, and
    # a models.json typo must not be able to silently redirect a run to another model or
    # quietly resize the budget the truncation columns are reported against.
    for k, v in DEFAULT_EXTRA_BODY.items():
        payload.setdefault(k, v)
    last = None
    for attempt in range(retries + 1):
        try:
            r = _session().post(f"{base_url.rstrip('/')}/chat/completions",
                              headers={"Content-Type": "application/json"},
                              json=payload, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            choice = d["choices"][0]
            msg = choice["message"]
            # HTTP 200 IS NOT SUCCESS. When the engine aborts a request -- which is
            # what happens to everything in flight when a job is cancelled, or when the
            # server is torn down under it -- vLLM answers 200 with an empty message
            # and finish_reason "abort". Recording that row marks the item done, and
            # since resume keys on presence in the file, it is never retried: a
            # cancelled job leaves ~n_workers items permanently scored as wrong.
            #
            # Measured, and it is not small: cancelling and resuming four Qwen BF16
            # passes left 29-31 aborted rows each, ~1.8 points, while the arms that
            # were never cancelled had zero. That is the same size as the effect these
            # evals exist to measure, and it lands on one arm only.
            #
            # This is exactly the failure this module's docstring criticises IFBench's
            # generator for -- writing an empty response on error, which its own resume
            # then treats as done. Raising here puts an aborted item back in the retry
            # path, where a transport failure already was.
            if choice.get("finish_reason") == "abort":
                raise RuntimeError("engine aborted the request (finish_reason=abort)")
            # vLLM 0.28 renamed this field to `reasoning`; `reasoning_content` survives
            # only as a deprecated REQUEST-side alias that protocol.py pops off inbound
            # messages, so reading it here always yielded "" and every chain of thought
            # was silently discarded. Scores are unaffected -- only content is scored --
            # but the field is evidence, so read the new name first and keep the old one
            # for any older server.
            return (msg.get("content") or "",
                    msg.get("reasoning") or msg.get("reasoning_content") or "",
                    choice.get("finish_reason"),
                    d.get("usage", {}).get("completion_tokens"))
        except Exception as e:                      # noqa: BLE001 - retry anything
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise last


def load_done(out_path, key_field="uid"):
    """Ids already present in the output file. Tolerates a torn final line."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)[key_field])
            except Exception:                       # noqa: BLE001
                continue        # a hard kill can leave a half-written last line
    return done


def run_resumable(items, uid_of, request_of, out_path, workers=None, desc="",
                  max_tokens=None, base_url=None, model=None):
    """Generate for every item not already in out_path, appending as results land.

    `uid_of(item) -> str` and `request_of(item) -> messages`. Rows carry the item's own
    fields plus content/reasoning/finish_reason/completion_tokens.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = workers or DEFAULT_WORKERS
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = load_done(out_path)
    todo = [it for it in items if uid_of(it) not in done]
    print(f"{len(items)} total | {len(done)} done | {len(todo)} to generate "
          f"| {workers} workers -> {out_path}", flush=True)
    if not todo:
        return 0

    lock = threading.Lock()
    n_err = 0
    with open(out_path, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(chat, request_of(it), max_tokens=max_tokens,
                          base_url=base_url, model=model): it for it in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc or "generate"):
            it = futs[fut]
            try:
                content, reasoning, finish, ntok = fut.result()
            except Exception as e:                  # noqa: BLE001
                n_err += 1
                print(f"  {uid_of(it)}: {type(e).__name__}: {e}", flush=True)
                continue        # unrecorded on purpose -- a rerun retries this item
            row = dict(it)
            row.update(uid=uid_of(it), content=content, reasoning=reasoning,
                       finish_reason=finish, completion_tokens=ntok)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
    if n_err:
        print(f"WARNING: {n_err} items failed and were NOT recorded; rerun to retry",
              flush=True)
    return n_err


def truncation_report(rows):
    """How many rows hit the token ceiling. A non-zero count invalidates the score."""
    n = len(rows)
    trunc = [r for r in rows if r.get("finish_reason") == "length"]
    empty = [r for r in rows if not (r.get("content") or "").strip()]
    toks = sorted(r["completion_tokens"] for r in rows if r.get("completion_tokens"))
    out = {"n": n, "truncated": len(trunc), "empty_content": len(empty)}
    if toks:
        out["tokens_p50"] = toks[len(toks) // 2]
        out["tokens_p95"] = toks[int(len(toks) * 0.95)]
        out["tokens_max"] = toks[-1]
    return out
