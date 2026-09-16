"""MMMU-Pro inference against a local vLLM endpoint.

    python3 mmmu_pro_infer.py --setting vision --out results/mmmu/vision_cot/bf16

Prompt construction is the MMMU-Pro repo's own, imported from its `infer/infer_gpt.py`
rather than reimplemented, so the text and image layout the model sees are the
benchmark's. Everything around that -- transport, resume, output location -- is ours.

WHY NOT infer_gpt.py DIRECTLY
-----------------------------
1. Its base_url is hardcoded to api.openai.com inside run_and_save(), so it cannot be
   pointed at a local server without editing the file.

2. Its "resume" is all-or-nothing per split, which is not resume at all here:

       if os.path.exists(output_path):   # -> reuse the whole file, generate nothing
       else:                             # -> generate ALL items, then save ONCE

   The single save happens only after every future completes. `vision` is 1730 items
   against a 30B reasoning model; a kill at item 1700 discards 1700 finished
   generations. common.run_resumable appends and fsyncs per item instead.

3. It sends its own sampling parameters, which would override the vendor generation
   config the server applies.

SETTINGS. MMMU-Pro is three subsets and they are not interchangeable: `standard4` is
the original 4-option form, `standard10` adds six distractors, and `vision` renders the
whole question into a screenshot so nothing is readable from text.

WHICH ONE THE PUBLISHED FIGURE REFERS TO IS NOT STATED. The report describes "1730
multiple choice questions ... the answer choice space has been significantly expanded",
which does not disambiguate -- `vision` and `standard (10 options)` are each 1730 items
and both derive from the 10-option construction -- and it sources the number from
Artificial Analysis rather than measuring it. Measurement is the only evidence
available, and it favours `vision` (73.4 vs standard10's 72.0 against a published 74.0).
Both are run regardless: the comparison this exists for is single-engine vs
disaggregated, and for that a second independent subset is worth more than picking one.
"""
import argparse
import base64
import glob
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_resumable, truncation_report  # noqa: E402

HARNESS = os.environ.get(
    "MMMU_HARNESS",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/MMMU/mmmu-pro")

SETTINGS = {"vision": "vision",
            "standard4": "standard (4 options)",
            "standard10": "standard (10 options)"}

HF_HUB = os.path.join(os.environ.get(
    "HF_HOME", "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/hf_cache"), "hub")


def _parquet_files(config):
    """The split's parquet shards, straight out of the hub cache.

    NOT load_dataset("MMMU/MMMU_Pro", config): with HF_HUB_OFFLINE=1 -- which every
    compute node needs, having no network -- that call raises

        ConnectionError: Couldn't reach 'MMMU/MMMU_Pro' on the Hub (OfflineModeIsEnabled)

    even though the data is fully downloaded, because it wants to resolve the repo
    before it will look at the cache. Reading the shards directly makes the offline
    path the ONLY path, so it cannot work on a login node and fail in the queue.
    """
    pat = os.path.join(HF_HUB, "datasets--MMMU--MMMU_Pro", "snapshots", "*",
                       config, "*.parquet")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no parquet shards for {config!r} under {pat}\n"
                         f"run bin/setup_harnesses.sh on a login node first")
    return files


def _b64(img):
    # Tolerates both shapes an image column can arrive in: a PIL Image when the parquet
    # metadata restores the Image feature, or a {"bytes":..., "path":...} struct when it
    # does not. Which one you get depends on how the shard was written, and a driver
    # that only handles one of them fails at item 1 after the model is already loaded.
    if isinstance(img, dict):
        img = Image.open(io.BytesIO(img["bytes"]))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", default="vision", choices=sorted(SETTINGS))
    ap.add_argument("--mode", default="cot", help="key in the harness prompts.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None, help="served model name; "
                    "defaults to $EVAL_MODEL")
    args = ap.parse_args()

    args.out = os.path.abspath(args.out)

    import yaml
    from datasets import load_dataset
    global Image
    from PIL import Image

    # chdir, not just sys.path: infer_gpt.py does `open("prompts.yaml")` at MODULE
    # level, so importing it from anywhere else raises FileNotFoundError before a
    # single line of ours runs. Doing it here rather than in the launcher keeps the
    # driver runnable by hand. --out is resolved absolute first, so the move is safe.
    os.chdir(HARNESS)
    sys.path.insert(0, os.path.join(HARNESS, "infer"))
    from infer_gpt import (  # noqa: E402 - the benchmark's own prompt construction
        mmmu_doc_to_text, origin_mmmu_doc_to_visual, vision_mmmu_doc_to_visual)

    with open(os.path.join(HARNESS, "prompts.yaml")) as f:
        prompt_config = yaml.safe_load(f)[args.mode]

    ds = load_dataset("parquet",
                      data_files={"test": _parquet_files(SETTINGS[args.setting])},
                      split="test")
    rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]

    # Items carry metadata ONLY. `run_resumable` serialises the item into the output
    # row, and a PIL image is not JSON -- so the images stay in `rows` and are fetched
    # by index when the request is built.
    # Subdomain comes from the ID, not a column: the shards carry `subject` for
    # `vision` and nothing for `standard`, so reading a "subdomain" field lands every
    # item in one bucket and the per-subject breakdown says nothing. `test_History_1`
    # -> `History` is the same decomposition the harness's own extract_subset_name does.
    def subdomain_of(r):
        m = re.match(r"^[^_]+_(.+?)_\d+$", r["id"])
        return m.group(1) if m else (r.get("subject") or "unknown")

    items = [{"id": r["id"], "idx": i,
              "options": str(r.get("options", "")),
              "answer": r.get("answer"),
              "subdomain": subdomain_of(r)}
             for i, r in enumerate(rows)]

    def request_of(it):
        data = rows[it["idx"]]
        if args.setting.startswith("standard"):
            prompt, order = mmmu_doc_to_text(data)
            images = origin_mmmu_doc_to_visual(data, order)
        else:
            prompt = prompt_config["vision"]
            images = vision_mmmu_doc_to_visual(data)
        parts = [{"type": "text", "text": prompt}]
        for im in images:
            if im is not None:
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{_b64(im)}"}})
        return [{"role": "user", "content": parts}]

    out_path = os.path.join(args.out, "raw.jsonl")
    run_resumable(items, uid_of=lambda x: x["id"], request_of=request_of,
                  out_path=out_path, workers=args.workers,
                  desc=f"mmmu/{args.setting}", max_tokens=args.max_tokens,
                  base_url=args.base_url, model=args.model)

    import json
    got = [json.loads(l) for l in open(out_path)]
    print(f"generation diagnostics: {truncation_report(got)}")
    if len(got) < len(items):
        print(f"WARNING: {len(items) - len(got)} items missing -- rerun to retry")


if __name__ == "__main__":
    main()
