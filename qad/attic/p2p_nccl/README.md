# P2pNcclConnector — dead end, kept for the record

This directory holds the P2P NCCL disaggregated-serving attempt. It is NOT used by
anything; the working path is NixlConnector (`../../serving/run_nixl_server.sh`). Moved rather
than deleted because two of the bugs found here are real and were expensive to find,
and this tree is kept for the record.

See `../../docs/DISAGG.md` for the full write-up.

Outcome: KV transfers correctly (verified: 28/28 layers, byte-identical checksums on
both ends) but generated text is garbage in all four configurations tried
(homogeneous/heterogeneous x CUDA graphs/eager), against a clean single-server
control. The corruption is in how vLLM's consumer *uses* the injected KV, which was
not worth pursuing further.

Still-valid findings preserved here:
* `run_disagg_server.sh` — pins VLLM_HOST_IP because P2pNcclEngine binds its ZMQ
  router to `get_ip()`, never loopback; advertising 127.0.0.1 hangs the producer.
* `p2p_trace.py` — `P2pNcclConnectorStableId`, which strips the per-server random
  salt vLLM appends to the engine request id
  (`request.request_id = f"{request.external_req_id}-{random_uuid():.8}"`).
  Without it, producer and consumer key the same KV differently and the decode
  engine blocks forever in an unbounded `recv_tensor` wait. Injected through vLLM's
  `kv_connector_module_path` hook, so no vLLM file is patched.
* `dump_p2p_source.py` — dumps connector/engine internals from inside the container;
  useful for any future connector spelunking.
