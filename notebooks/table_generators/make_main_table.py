"""Generate the main cost/accuracy tabular from the real measurement files.

    python notebooks/table_generators/make_main_table.py  # prints ONLY the latex tabular

Nothing here is transcribed by hand. Every cell is derived:

  prefill   qad/kernels/prefill/offload_prefill.csv   end-to-end at S=16384, bf16/nvfp4
  decode    qad/kernels/lloyd43/benchmarks/vllm_decode.csv   end-to-end vLLM, batch 1
  memory    HF configs (analytic linear + embedding parameter counts);
            ODP slots reuse decode-weight memory and add no weight residency
  accuracy  plots.ipynb's decode-heavy and RULER loaders, averaged over TAIL_STEPS
            and the same fixed model cohorts as the family bar figures

The point of deriving rather than transcribing is that a re-measurement propagates. The
prefill CSV was found to hold one bad row (Qwen3-8B bf16 resident at 16k read 11% high,
which its SSD counterpart undercut -- impossible, since offloading cannot be free); fixing
the CSV moved the published 1.51x to 1.36x, and a hand-copied table would not have noticed.

WHICH ARM EACH ROW MEASURES. The ladder rows differ in how disaggregated the format is,
and the decode column has to follow that or the table flatters the un-disaggregated rows:

  NVFP4                 W4A4: pays to quantize the activation      -> nvfp4
  NVFP4 +Format disagg. decode skips it                            -> nvfp4a16
  3-bit weight-only     W3A16 everywhere                           -> lloyd43
  LUT3                  LUT3 weights, quantized activations        -> lloyd43aq
  LUT3 +Format disagg.  decode skips the quantization              -> lloyd43

The *aq arms run the identical kernel and weights plus one discarded fp4 activation
quantization per linear -- the cost of NOT being format-disaggregated, which this weight-
only GEMV cannot use. It is an upper bound on that tax, not a deployable configuration.

Speedups and memory are for the largest model of each family. Both accuracy columns
average the four Qwen sizes and the three Gemma sizes above 270M. DH averages the
reasoning benchmarks (Qwen think and no-think, Gemma think only); PH averages RULER's
13 tasks equally, then the notebook's common 4K/8K/16K/32K context-length cohort.
ODP inherits the corresponding full-disaggregation accuracy in both columns.
"""

import contextlib
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent
ROOT = NOTEBOOKS.parent
PREFILL_CSV = ROOT / "qad/kernels/prefill/offload_prefill.csv"
DECODE_CSV = ROOT / "qad/kernels/lloyd43/benchmarks/vllm_decode.csv"
SEQ = 16384
# The reporting window is FIXED, not "the last five exports". A run that has only reached
# 1750 would otherwise be averaged over an earlier window than the finished runs beside it
# in the same column, and the column would silently mix two different measurements.
TAIL_STEPS = (1250, 1500, 1750, 2000, 2250)

QWEN_BIG, GEMMA_BIG = "Qwen/Qwen3-8B", "google/gemma-3-12b-it"
QWEN = ["Qwen-Qwen3-0.6B", "Qwen-Qwen3-1.7B", "Qwen-Qwen3-4B", "Qwen-Qwen3-8B"]
# Gemma-3-270M scores near chance on decode-heavy tasks; exclude it from both workload
# family means to keep the cohorts consistent. Cost columns are GEMMA_BIG anyway.
GEMMA = ["google-gemma-3-1b-it", "google-gemma-3-4b-it", "google-gemma-3-12b-it"]

BF16, NVFP4_B, LUT3_B, LUT2_B = 2.0, 0.5625, 0.4375, 0.3125

# (label, accuracy method, prefill mode, decode quant, weight bytes, extra memory)
#   prefill mode : "none" = format is not used at prefill (1.00x) | "nvfp4" | "odp"
#   extra memory : "full" = also store the NVFP4 copy | None = decode weights only
# ODP buffers replace unused decode weights during prefill, then restore them.
ROWS = [
    ("BF16",              None,                     "none",  "none",      BF16,    None),
    ("NVFP4A16",          "nvfp4a16",               "none",  "nvfp4a16",  NVFP4_B, None),
    ("NVFP4",             "nvfp4",                  "nvfp4", "nvfp4",     NVFP4_B, None),
    ("+Format disagg.",   "nvfp4pdshared",          "nvfp4", "nvfp4a16",  NVFP4_B, None),
    ("3-bit weight-only", "lloyd43",                "none",  "lloyd43",   LUT3_B,  None),
    ("LUT3",              "nvfp4lloyd43upcastboth", "nvfp4", "lloyd43aq", LUT3_B,  None),
    ("+Format disagg.",   "nvfp4lloyd43upcast",     "nvfp4", "lloyd43",   LUT3_B,  None),
    ("+Full disagg.",     "nvfp4lloyd43split",      "nvfp4", "lloyd43",   LUT3_B,  "full"),
    ("+ODP",              "nvfp4lloyd43split",      "odp",   "lloyd43",   LUT3_B,  None),
    ("2-bit weight-only", "lloyd21",                "none",  "lloyd21",   LUT2_B,  None),
    ("LUT2",              "nvfp4lloyd21upcastboth", "nvfp4", "lloyd21aq", LUT2_B,  None),
    ("+Format disagg.",   "nvfp4lloyd21upcast",     "nvfp4", "lloyd21",   LUT2_B,  None),
    ("+Full disagg.",     "nvfp4lloyd21split",      "nvfp4", "lloyd21",   LUT2_B,  "full"),
    ("+ODP",              "nvfp4lloyd21split",      "odp",   "lloyd21",   LUT2_B,  None),
]
RULE_AFTER = {0, 3, 8}


def prefill_speedups():
    """(resident, odp) speedup per model. ODP is offloaded NVFP4 against resident BF16.

    The ODP arm is `zero-ssd`, not `ssd`: the slots are carved out of decode weights and
    restored from the drive, and the first block is read on the clock. `ssd` excludes
    these operations and is not the protocol reported in the paper.
    """
    d = {}
    with open(PREFILL_CSV, newline="") as stream:
        for r in csv.DictReader(stream):
            if int(r["seq_len"]) != SEQ:
                continue
            key = (r["model"], r["quant"], r["mode"])
            value = float(r["latency_ms"])
            if key in d or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Duplicate or invalid prefill measurement: {key}")
            d[key] = value
    out = {}
    for m in (QWEN_BIG, GEMMA_BIG):
        b = d[(m, "bf16", "resident")]
        out[m] = (b / d[(m, "nvfp4", "resident")], b / d[(m, "nvfp4", "zero-ssd")])
    return out


def decode_speedups():
    tok = {}
    for r in csv.DictReader(open(DECODE_CSV)):
        tok[(r["model"], r["quant"])] = float(r["decode_tok_s"])
    return {m: {q: tok[(m, q)] / tok[(m, "none")] for (mm, q) in tok if mm == m}
            for m in (QWEN_BIG, GEMMA_BIG)}


def param_counts(model):
    """(linear params, embedding params) analytically, from the config alone."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model)
    cfg = getattr(cfg, "text_config", cfg)
    h, L = cfg.hidden_size, cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", h // cfg.num_attention_heads)
    q, kv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    inter = cfg.intermediate_size
    per_layer = h * (q + 2 * kv) + q * h + 2 * h * inter + inter * h
    emb = cfg.vocab_size * h
    # Gemma 3 ties the head to the embedding; Qwen 3 does not, so it stores both.
    return per_layer * L, emb * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)


def ruler_accuracy(ns, model, method, incomplete):
    """One model's PH mean, requiring every length and every reporting checkpoint."""
    lengths = ns["RULER_LENGTHS"]
    if not lengths:
        incomplete.append((model, method, "prefill-heavy: no context lengths configured"))
        return None
    if method is None:
        baseline = ns["ruler_baselines"].get(model, {})
        missing = [str(length) for length in lengths
                   if length not in baseline or not np.isfinite(baseline[length])]
        if missing:
            incomplete.append((model, "BF16", "prefill-heavy: missing/invalid length(s) "
                               + ", ".join(missing)))
            return None
        return float(np.mean([baseline[length] for length in lengths]))

    curves = ns["ruler_models"].get(model, {}).get(method, {})
    gaps = []
    for length in lengths:
        curve = curves.get(length, {})
        missing = [str(step) for step in TAIL_STEPS
                   if step not in curve or not np.isfinite(curve[step])]
        if missing:
            gaps.append(f"{length // 1024}K step(s) " + ", ".join(missing))
    if gaps:
        incomplete.append((model, method, "prefill-heavy: missing/invalid " + "; ".join(gaps)))
        return None
    curve = ns["ruler_avg_curve"](model, curves)
    return float(np.mean([curve[step] for step in TAIL_STEPS]))


def accuracies():
    """Execute notebook data loaders (not plots), then average over TAIL_STEPS.

    Returns (mean_over_sizes, incomplete). The callable accepts workload="decode"
    or "prefill"; an incomplete model makes its entire family cell a TODO.
    """
    incomplete = []
    nb = json.loads((NOTEBOOKS / "plots.ipynb").read_text())
    ns = {"__name__": "nb"}
    # Cell 0 discovers results through paths relative to the NOTEBOOK ("../qad/results/..."),
    # so it has to be executed from there. Run from the repo root instead and every path
    # misses, which used to surface as a silent table of "-" with a nan BF16 row rather than
    # as an error -- exactly the quiet-hole failure the decode check below exists to prevent.
    with contextlib.chdir(NOTEBOOKS), contextlib.redirect_stdout(io.StringIO()):
        exec("".join(nb["cells"][0]["source"]), ns)
        # Locate the loader by definition so moving plotting cells cannot change it.
        ruler_cells = ["".join(cell.get("source", [])) for cell in nb["cells"]
                       if cell.get("cell_type") == "code"
                       and "def load_ruler():" in "".join(cell.get("source", []))]
        if len(ruler_cells) != 1:
            raise ValueError("Expected exactly one RULER loader cell in plots.ipynb")
        exec(ruler_cells[0], ns)
    if not ns["TASKS"] or not any(ns["models"].values()):
        raise SystemExit(f"plots.ipynb cell 0 found no results under {ROOT}/qad/results")

    def one(model, method):
        """Tail mean, or None if the arm is not finished. None is never a small number."""
        want = [t for t in ns["TASKS"]
                if t.rsplit("@", 1)[1] in ns["FAMILY_MODES"][ns["family_of"](model)]]
        if not want:
            return None
        if method is None:
            bl = ns["baselines"].get(model, {})
            v = [bl.get(t) for t in want]
            return float(np.mean(v)) if all(x is not None for x in v) else None
        c = ns["models"].get(model, {}).get(method)
        if not c:
            incomplete.append((model, method, "no results"))
            return None
        if any(t not in c for t in want):
            incomplete.append((model, method,
                               "missing " + ", ".join(t for t in want if t not in c)))
            return None
        gaps = sorted({s for t in want for s in TAIL_STEPS if s not in c[t]})
        if gaps:
            incomplete.append((model, method,
                               "missing step(s) " + ", ".join(str(s) for s in gaps)))
            return None
        return float(np.mean([sum(c[t][s] for t in want) / len(want) for s in TAIL_STEPS]))

    def mean_over_sizes(models, method, workload="decode"):
        if workload not in ("decode", "prefill"):
            raise ValueError(f"Unknown workload: {workload}")
        v = [ruler_accuracy(ns, m, method, incomplete) if workload == "prefill"
             else one(m, method) for m in models]
        return None if any(x is None for x in v) else float(np.mean(v))

    return mean_over_sizes, incomplete


def main():
    pre, dec = prefill_speedups(), decode_speedups()
    acc, incomplete = accuracies()
    lin, emb = {}, {}
    for m in (QWEN_BIG, GEMMA_BIG):
        lin[m], emb[m] = param_counts(m)

    def mem(m, bytes_per_w, extra):
        g = (lin[m] * bytes_per_w + emb[m] * BF16) / 1e9
        if extra == "full":
            g += lin[m] * NVFP4_B / 1e9
        return g

    # Fail loudly and specifically on a missing arm. A paper table must not quietly print
    # "-" for a measurement that simply has not been run yet.
    missing = sorted({(m, dq) for _, _, _, dq, _, _ in ROWS
                      for m in (QWEN_BIG, GEMMA_BIG) if dq not in dec[m]})
    if missing:
        cmds = "\n".join(
            f"  python vllm_serve.py --model {m} --quant none {q} "
            f"--out benchmarks/vllm_decode.csv" for m, q in missing)
        raise SystemExit(f"missing decode measurements for "
                         f"{', '.join(f'{m}:{q}' for m, q in missing)}\nrun:\n{cmds}")

    TODO = r"\textcolor{red}{TODO}"
    rows = []
    for i, (label, method, pmode, dq, bpw, extra) in enumerate(ROWS):
        cells = [label]
        for big, sizes in ((QWEN_BIG, QWEN), (GEMMA_BIG, GEMMA)):
            p = 1.0 if pmode == "none" else pre[big][0 if pmode == "nvfp4" else 1]
            a = acc(sizes, method)
            ph = acc(sizes, method, workload="prefill")
            cells += [f"{p:.2f}x", f"{dec[big][dq]:.2f}x", f"{mem(big, bpw, extra):.2f}",
                      TODO if a is None else f"{a:.1f}",
                      TODO if ph is None else f"{ph:.1f}"]
        rows.append(" & ".join(cells) + (r" \\\midrule" if i in RULE_AFTER else r" \\"))

    print(r"\begin{tabular}{l|ccccc|ccccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{Format} & \multicolumn{5}{c|}{Qwen 3} & "
          r"\multicolumn{5}{c}{Gemma 3} \\")
    print(r"  & \makecell{Prefill\\speedup} & \makecell{Decode\\speedup} & "
          r"\makecell{Device\\GB} & \makecell{Acc.\\DH} & \makecell{Acc.\\PH} & "
          r"\makecell{Prefill\\speedup} & \makecell{Decode\\speedup} & "
          r"\makecell{Device\\GB} & \makecell{Acc.\\DH} & \makecell{Acc.\\PH} \\ \midrule")
    for r in rows:
        print(r)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    for model, method, why in incomplete:
        print(f"note: {model} {method} not reported -- {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
