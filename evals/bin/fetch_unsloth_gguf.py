"""Fetch the 1/2/3-bit unsloth Qwen3.8-27B GGUFs plus the BF16 mmproj.

    python3 bin/fetch_unsloth_gguf.py

Only the eight Q1/Q2/Q3 quants are pulled. The repo also ships Q4 through Q8, a BF16
backbone and an imatrix; together those are several hundred GB and are not part of this
sweep. ~75 GB comes down.

RUN THIS ON THE LOGIN NODE. Eval and quantize jobs all set HF_HUB_OFFLINE=1 -- everything
they need must already be on disk, and a compute node has no route to huggingface.co.

mmproj-BF16.gguf is the vision tower and is NOT optional: Qwen3.8-27B is multimodal, and
the backbone GGUFs carry no vision weights. It is BF16 in this repo, as it was for
GSQ-RCO, so the vision path is identical across every arm and MMMU differences reflect
the language model alone.
"""
import os

from huggingface_hub import snapshot_download

P = ("/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/"
     "prefill_decode")
REPO = "unsloth/Qwen3.8-27B-GGUF"
OUT = f"{P}/models/Qwen3.8-27B-unsloth-GGUF"

# 1-bit: IQ1_S, IQ1_M | 2-bit: IQ2_XXS, IQ2_S, Q2_K_XL | 3-bit: IQ3_XXS, IQ3_S, Q3_K_XL
WANT = ["UD-IQ1_S", "UD-IQ1_M",
        "UD-IQ2_XXS", "UD-IQ2_S", "UD-Q2_K_XL",
        "UD-IQ3_XXS", "UD-IQ3_S", "UD-Q3_K_XL"]


def main():
    os.environ.setdefault("HF_HOME", f"{P}/hf_cache")
    patterns = [f"Qwen3.8-27B-{w}.gguf" for w in WANT] + ["mmproj-BF16.gguf"]
    path = snapshot_download(REPO, local_dir=OUT, allow_patterns=patterns,
                             max_workers=8)
    print(f"done -> {path}", flush=True)
    for f in sorted(os.listdir(path)):
        if f.endswith(".gguf"):
            print(f"  {f:38} {os.path.getsize(os.path.join(path, f)) / 1e9:6.1f} GB")


if __name__ == "__main__":
    main()
