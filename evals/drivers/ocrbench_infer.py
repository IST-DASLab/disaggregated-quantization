"""OCRBench v2 (English) inference against a local vLLM endpoint.

    python3 ocrbench_infer.py --out results/<model>/ocrbench/<tag>

7,400 English items over 21 task types, which the benchmark's own aggregation folds into
8 scoring categories. The Chinese half is not run: it doubles the cost, neither model
here is documented as targeting Chinese OCR, and a floored category still contributes
variance to an aggregate that is an UNWEIGHTED mean of category means. `--lang cn` and
`--lang both` exist if that changes.

WHY THIS BENCHMARK IS WORTH THE GPU TIME
----------------------------------------
Every other instrument here is multiple choice (GPQA 4-way, MMMU-Pro 10-way) or
constraint checking (IFBench). Those have a guessing floor and an option set that snaps a
slightly-wrong logit back onto the right answer. OCRBench scores LONG EXACT STRINGS with
continuous metrics -- edit distance, TEDS, IoU -- so per-token weight noise accumulates
instead of being rounded away. If 4-bit weights hurt anywhere, long-form transcription is
where it should show, and nothing else in this repo would see it.

THE STANDING RISK, recorded so a flat result is read correctly: only the language model
is quantized and the vision tower stays bf16 in every arm. OCR accuracy is substantially
encoder-bound, so the arms may simply converge. That is a finding -- "quantizing the
decoder alone leaves OCR intact" -- not a failed experiment, but it is what to expect if
the per-category table comes back flat.

PROMPTS ARE THE BENCHMARK'S OWN, VERBATIM. Its `question` field already carries the full
instruction including the required output format ("Output the normalized coordinates of
the left-top and right-bottom corners..."). Adding a system prompt or an output hint here
would make the score incomparable with every published OCRBench number, and the
coordinate formats in particular are exactly what the scorer parses.

EVERY SCORING FIELD IS CARRIED THROUGH. eval.py dispatches on `type` and `eval`, and
needs `answers`, `bbox`, `bbox_list` and `content` for the grounding, spotting and
page-OCR metrics. They are copied into each output row rather than re-joined later, so
the generated file is self-contained and scoring never has to re-read the parquet.

DATASET: the ORIGINAL `OCRBench_v2.json` plus its image folders, as distributed by the
authors, not a HuggingFace parquet mirror.

That choice is the whole reason this driver is short. The parquet mirrors
(`ling99/OCRBench_v2`, `lmms-lab/OCRBench-v2`) are third-party re-uploads, and parquet
needs a fixed column schema where the source JSON is heterogeneous. Two lossy
conversions follow, both of which corrupt scoring silently rather than loudly:

  * a key the JSON OMITS becomes the literal string "None", and eval.py branches on key
    PRESENCE (`if "eval" in data_item.keys()`), so every ordinary VQA item is routed
    into a branch it should never enter and falls off the end of it;
  * `answers` is typed list<string>, so the dict answers that `chart parsing en` and
    `key information extraction en` carry arrive as strings, and the scorer dies in
    dict_to_html a couple of thousand items into the run.

Both were worked around against the mirror before switching. Reading the original file
removes the class of problem rather than the two instances of it, which matters because
`bbox_list` and `content` are also structured and had not been reached yet.
"""
import argparse
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_resumable, truncation_report  # noqa: E402

DATA_ROOT = os.environ.get(
    "OCRBENCH_DATA",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/"
    "ocrbench_v2_data/OCRBench_v2")


def _b64(path):
    """PNG base64 of an image on disk."""
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_items(lang):
    """The benchmark's own records, verbatim, for one language.

    Every field is kept exactly as the authors wrote it -- including the ABSENCE of
    `eval`, `bbox`, `bbox_list` and `content` on the tasks that do not use them, which
    is what eval.py's key-presence branching reads.
    """
    path = os.path.join(DATA_ROOT, "OCRBench_v2.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found.\n"
            f"Download the dataset from the Google Drive link in the OCRBench v2 README "
            f"and extract it so that {DATA_ROOT}/OCRBench_v2.json exists.")
    with open(path) as f:
        data = json.load(f)
    suffix = {"en": (" en",), "cn": (" cn",), "both": (" en", " cn")}[lang]
    items = [d for d in data if d["type"].endswith(suffix)]

    # NAME COLLISION. run_resumable stores the model's output under `content`, and the
    # benchmark's spotting records ALREADY have a `content` field -- the ground-truth
    # word list that spotting_evaluation zips against `bbox`. Left alone, the generation
    # overwrites the ground truth in the output row, and the failure is a KeyError in
    # the scorer if you are lucky and a silently zeroed `text spotting` category if you
    # are not. Renamed here and restored by ocrbench_score before eval.py sees it.
    for d in items:
        if "content" in d:
            d["gt_content"] = d.pop("content")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    # 8192, not the 32768 the multiple-choice benchmarks use. Full-page OCR emits an
    # entire page and nothing here needs a 30k-token chain; a smaller ceiling bounds the
    # slow tail without truncating real answers. finish_reason is recorded either way,
    # so if truncation shows up in the diagnostics this is the number to raise.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--lang", default="en", choices=["en", "cn", "both"])
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)

    items = load_items(args.lang)
    if args.limit:
        # Stratified, never a head slice: the file is ordered by task, so the first N
        # records are 2-3 task types out of 21 and the category table would come back
        # mostly empty -- a smoke test that exercises almost none of the scorers.
        by_type = {}
        for it in items:
            by_type.setdefault(it["type"], []).append(it)
        per = max(1, args.limit // len(by_type))
        items = [it for v in by_type.values() for it in v[:per]]
    print(f"{len(items)} items over {len({it['type'] for it in items})} task types "
          f"(lang={args.lang})", flush=True)

    def request_of(it):
        return [{"role": "user", "content": [
            {"type": "text", "text": it["question"]},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,"
                       + _b64(os.path.join(DATA_ROOT, it["image_path"]))}}]}]

    out_path = os.path.join(args.out, "raw.jsonl")
    run_resumable(items, uid_of=lambda x: str(x["id"]), request_of=request_of,
                  out_path=out_path, workers=args.workers, desc="ocrbench",
                  max_tokens=args.max_tokens, base_url=args.base_url,
                  model=args.model)

    got = [json.loads(l) for l in open(out_path)]
    print(f"generation diagnostics: {truncation_report(got)}")
    if len(got) < len(items):
        print(f"WARNING: {len(items) - len(got)} items missing -- rerun to retry")
    else:
        # A completeness marker, written only when every item is present. Scoring is
        # 15-20 minutes and skips directories that already have a summary, so scoring a
        # HALF-FINISHED arm is worse than useless: it burns the CPU and then blocks the
        # real scoring, because the partial summary makes the directory look done.
        # A row count is not enough on its own -- a resumed run passes through every
        # intermediate count -- so completeness is recorded explicitly.
        with open(os.path.join(args.out, "generation_complete"), "w") as f:
            f.write(f"{len(got)}\n")


if __name__ == "__main__":
    main()
