# prefill-decode-shenanigans

# lmeval

1. Start the prefill/decode pair + proxy (configured via env vars, proxy on 8595):

       PREFILL_GPU=0 DECODE_GPU=1 ./run_server.sh

2. Point the config at it. `model_args` says which server and served model name to
   hit; `groups` splits the tasks into separate runs so each can have its own
   `gen_kwargs` (thinking on/off, token budget, temperature). See
   `evals/configs/eval_server_config.yaml`.

3. Run the eval — results go to `<output_dir>/<run_name>/<group>.json` plus a
   `summary.json`:

       python evals/eval_vllm_server.py evals/configs/eval_server_config.yaml
       python evals/eval_vllm_server.py evals/configs/eval_server_config.yaml --groups think
