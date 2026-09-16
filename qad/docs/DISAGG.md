# Disaggregated prefill/decode serving in vLLM — findings

Goal: serve a dual-format checkpoint the way it is meant to be deployed — a **W4A4
prefill** engine and a **W4A16 decode** engine, KV transferred between them — so
GSM8K measures deployment accuracy rather than an in-process simulation.

**Outcome: working, via NixlConnector.** P2pNcclConnector was tried first and
abandoned. Everything below was established on vLLM `0.14.2.dev0+gd7de043d5`
(container `nemo:26.02.nemotron_3_super_luts_v2`), 1P1D on one node, 2x B200,
Qwen3-0.6B checkpoints at step 2450.

---

## The failure mode that governs all of this

Both connectors can fail by **silently recomputing the prompt on the decode engine
with its own weights**. The request succeeds, the text is fluent, and the reported
number is really homogeneous-decode. No log line says so.

So a returned completion is *not* evidence. Two independent checks are required, and
`verify_kv_transfer.py` implements the second:

1. **Sanity** — compare against a single server with no connector at all. Our P2P
   NCCL stack passed every internal check and still emitted
   `'Assistantivy铿作出girl.\n.\n.\n.\n美女 girl十一...'`, while the same checkpoint on
   one server produced `'17 × 23 = **401**.'`
2. **Attribution** — run three stacks on identical prompts, greedy:
   `A = a4->a16`, `B = a16->a16`, `C = a4->a4`. **A must differ from both.**
   `A == B` means the KV never crossed; `A == C` means generation is not running on
   the decode weights.

---

## NixlConnector — the working path

`run_nixl_server.sh` + `nixl_proxy.py`, driven by `eval_disagg.py`.

Two **non-default** settings are required here, both deliberate:

| setting | why |
|---|---|
| `kv_buffer_device=cpu` | This container's UCX is built without CUDA: `"8 NVIDIA GPU(s) were detected, but UCX CUDA support was not found!"` and `"VRAM memory is detected as host by UCX. VRAM registration cannot proceed."` With `cuda` every engine dies at init in `register_kv_caches` with `nixlBackendError: NIXL_ERR_BACKEND`, on every node. Host buffer is slower but is still a real cross-engine transfer. |
| `enforce_handshake_compat=false` | Only needed for a mixed pair — see below. |

`kv_load_failure_policy: "fail"` is set so a *failed* load raises instead of falling
back to local recompute.

> **2026-09-09 update, b300/GB300 cluster.** The `kv_buffer_device=cpu` row above is
> a finding from the b200 cluster's `nemo:26.02.nemotron_3_super_luts_v2` container
> specifically, not a NIXL/UCX property in general. On b300, with a plain
> `nvcr.io/nvidia/nemo:26.02` pulled fresh from NGC, this container's UCX *does* have
> CUDA support: `kv_buffer_device=cuda` registers cleanly on every worker
> (`use_host_buffer: False`, no `NIXL_ERR_BACKEND`), and a disaggregated gsm8k canary
> completed end to end on it. `run_nixl_server.sh`'s default is now `cuda` on this
> cluster. `cpu`-staging turned out to be actively worse here too, not just slower:
> under `--tensor-parallel-size 2` each TP worker mirrors its own
> `--gpu-memory-utilization`-sized KV budget into host RAM, and four workers' worth
> exceeded the node's available memory (SLURM cgroup OOM). Re-check against the
> engine log (`NIXL_ERR_BACKEND` at `register_kv_caches` means `cpu` is still
> required) before assuming either default on a container/cluster combination this
> hasn't been measured on.

### Why the compatibility hash rejects our pair, and why overriding it is sound

A heterogeneous pair is refused at handshake:

```
RuntimeError: NIXL compatibility hash mismatch.
  Local: dbe017c9d7eafeb5263a51b7ec4893260f6ad3a883eef42cc418ed8ee45d027a
  Remote: f2f65e37ef08a8...
```

`compute_nixl_compatibility_hash` covers: vllm version, connector version, **model**,
dtype, num_kv_heads, head_size, num_hidden_layers, attn_backend_name, cache_dtype.

For our two checkpoints **every one of those is identical except `model`, which is the
checkpoint path string**. Quantization here is weight-side; the KV cache is bf16 in
both. So the rejection is on a path, not a real incompatibility.

The override is the connector's own documented option, and the hash's docstring notes
that TP size, block size and KV layout are *deliberately excluded* because they are
validated at runtime in `_validate_remote_agent_handshake` — those checks still fire.
`eval_disagg.py` opens the hatch **only when the two model dirs differ**, so an
accidental mismatch in a homogeneous run still fails loudly.

### Verification result (jobs 476985 / 476986 / 476987)

```
A==B (KV never crossed)                    : 0/2
A==C (not running on decode weights)       : 0/2
VERDICT: PASS — A differs from BOTH homogeneous stacks
```

The clearest evidence is prompt 1, where A is a genuine hybrid — it inherits the W4A4
prefill's clips/friends confusion but renders it in the W4A16 decode's phrasing:

```
A het a4->a16 : 'Natalia sold **48 clips** to **48 friends** in **April**...'
B homo a16    : 'Natalia sold **48 clips** in **April**...'
C homo a4     : 'Natalia sold **48 friends in April**...'
```

### Comparability caveat

Disaggregated output is **not** bit-identical to single-server output for the same
model: prefill runs on a different engine and device, and greedy decoding flips on
near-ties (`B == ctrl_a16` on one prompt but not the other). Compare disaggregated
against disaggregated only.

---

## P2pNcclConnector — abandoned (see `../attic/p2p_nccl/`)

Two real bugs found and fixed, after which KV transferred **byte-perfectly**
(28/28 layers, identical shape/dtype/checksum on both ends) and the generated text
was still garbage in all four configurations tried (homogeneous/heterogeneous x CUDA
graphs/eager). The remaining fault is in how vLLM's consumer *uses* the injected KV;
not pursued further.

Worth keeping regardless:

1. **The engine binds ZMQ to the node IP, never loopback.** `P2pNcclEngine` does
   `hostname = get_ip()`. A proxy advertising `127.0.0.1:<kv_port>` in the request id
   points at nothing and the producer blocks forever with no error. Pin
   `VLLM_HOST_IP` and hand the same string to the proxy.
2. **vLLM salts the engine request id per server.**
   `v1/engine/input_processor.py`:
   `request.request_id = f"{request.external_req_id}-{random_uuid():.8}"`.
   P2pNcclConnector keys KV as `<request_id>#<layer_name>` and assumes both servers
   derive the same id from `X-Request-Id`, so the two ends name the same tensor
   differently. vLLM keeps the unsalted value as `external_req_id` — which is what the
   connector should correlate on. The documented request-id format ends `-0` with no
   salt, i.e. the docs predate this.
3. **It fails as an unbounded silent hang.** `recv_tensor` under PUT_ASYNC is
   `while tensor_id not in self.recv_store: self.recv_store_cv.wait()` — no timeout.
   The decode engine blocks inside forward while the API server keeps logging
   `Running: 0 reqs`, which reads like an idle server. Always bound the prefill leg
   with its own short HTTP timeout.

---

## Files

| file | role |
|---|---|
| `run_nixl_server.sh` | brings up prefill + decode + proxy in one allocation; writes `--ready-file` |
| `nixl_proxy.py` | stateful proxy: forwards `kv_transfer_params` from prefill to decode |
| `run_nixl_1p1d.sh` | smoke test; `--mode homo\|hetero\|homo_a4`, `--kv-buffer-device` |
| `run_nixl_check.sh`, `run_nixl_ucx_diag.sh` | install / UCX diagnostics |
| `eval_disagg.py` | drives lm-eval through the stack; `--probe` for raw completions |
| `verify_kv_transfer.py` | the A/B/C gate — run this before trusting any number |
| `../attic/p2p_nccl/` | the abandoned P2P NCCL attempt, kept for the findings above |
