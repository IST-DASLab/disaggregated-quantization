"""Compare two serving arms with a paired bootstrap over items and repeats.

    python3 compare_arms.py                          # every model, MMMU-Pro
    python3 compare_arms.py --model qwen3.8-27b      # one model
    python3 compare_arms.py --benches gpqa ifbench mmmu

WHY NOT JUST SUBTRACT THE TWO SCORES AND ADD THE ERROR BARS IN QUADRATURE
------------------------------------------------------------------------
Because the arms are not independent samples. Every arm answers THE SAME 198 GPQA
questions, THE SAME 300 IFBench prompts and THE SAME 1730 MMMU items. Item difficulty is
therefore common-mode: if question 47 is hard, it is hard for both arms, and that
variation contributes nothing to the difference between them. Treating the two scores as
independent throws that away and inflates the uncertainty on the comparison -- which is
the only quantity anyone actually wants.

The size of the mistake is not subtle. GPQA's binomial standard error is 2.7 points at
n=198; its measured spread over 4 repeats is 0.5. The 2.7 is almost entirely the item
sample, and the item sample is held fixed.

THE TWO NOISE SOURCES, AND HOW EACH IS HANDLED
-----------------------------------------------
  * ITEMS -- resampled with replacement, and the SAME resampled index set is used for
    both arms in a draw. That is what pairs them, so the per-item difficulty cancels
    exactly as it does in the real comparison.
  * REPEATS -- resampled with replacement INDEPENDENTLY per arm, because the arms really
    were generated in separate runs. This is the generation noise that the repeat-SEM in
    report_all.py measures; here it enters the same interval as the item term instead of
    being reported next to it.

A benchmark with a single repeat still contributes: resampling one repeat with
replacement is a no-op, so its interval reflects item variation only, and the result is
an UNDERSTATED uncertainty for that benchmark. Every such benchmark is flagged in the
output. Run repeats rather than trusting the flag.

THE DEFAULT INTERVAL IS t OVER REPEATS, NOT A BOOTSTRAP
--------------------------------------------------------
    delta = mean(arm repeats) - mean(baseline repeats)
    SE    = sqrt(SE_arm^2 + SE_base^2),  SE_x = stdev(x repeats)/sqrt(k)
    CI96  = delta +/- t(0.98, welch_df) * SE

Textbook Welch two-sample t, and it is what a reader will assume unless told otherwise.

An earlier version bootstrapped the repeats instead and reported the 2nd/98th
percentiles. That is WRONG AT THIS SAMPLE SIZE and wrong in the dangerous direction: a
percentile interval resampled from 4 numbers can never be wider than the range of those
4 numbers, so it systematically under-covers. Measured on GPQA, arm vs bf16:

    nvfp4     t: [-8.73, +1.91]   bootstrap: [-6.19, -0.76]
    nvfp4a16  t: [-5.05, -0.25]   bootstrap: [-4.29, -1.26]

The bootstrap turns "cannot rule out no difference" into "significantly worse" for
nvfp4. Do not use it to decide whether an arm regressed.

--bootstrap-items ADDS A SECOND, WIDER INTERVAL
------------------------------------------------
It answers a different question: "would this difference hold on a FRESH DRAW of
questions from this benchmark?" -- so it resamples items with replacement, using the
SAME resampled index set for both arms so the pairing that cancels per-item difficulty
is preserved. It is not subject to the small-n defect above because it resamples
198-1730 items, not 4 repeats.

Off by default. The fixed item set is the apparatus every arm shares, so the t interval
is the one that ranks serving configurations; the item interval is for a claim about the
benchmark's population, and quoting it by default would hide real differences.
"""
import argparse
import ast
import glob
import json
import os
import random
import re
import sys

import statistics

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(os.path.dirname(HERE), "results")
# Set from --model in main(). Results are keyed by model first, so every loader below
# resolves through this; comparing arms across two different models is meaningless and
# the layout makes it impossible rather than merely discouraged.
RESULTS = RESULTS_ROOT
sys.path.insert(0, HERE)

HARNESS = os.environ.get(
    "MMMU_HARNESS",
    "/scratch/fsw/portfolios/coreai/projects/coreai_psx_qad/users/apanferov/prefill_decode/harnesses/MMMU/mmmu-pro")


def _repeat_dirs(bench_root, tag):
    """`tag`, `tag-r1`, ... in order. One directory per independent generation pass."""
    out = [os.path.join(bench_root, tag)]
    out += sorted(glob.glob(os.path.join(bench_root, f"{tag}-r[0-9]*")))
    return [d for d in out if os.path.isdir(d)]


BENCH_ROOT = {"gpqa": "gpqa", "ifbench": "ifbench", "mmmu": "mmmu/vision_cot",
              "ocrbench": "ocrbench", "mmlu_pro": "mmlu_pro"}


def available_tags(bench):
    """Every tag with results on disk for this benchmark, repeats collapsed.

    Used only when the configured baseline is missing. The configured arm list is
    curated -- single-engine bf16 is deliberately not in it, being a settled control
    rather than a format to rank -- but when the baseline has not landed yet that
    curation hides the only complete reference available, and the table comes back
    without anything to compare against.
    """
    root = os.path.join(RESULTS, BENCH_ROOT.get(bench, bench))
    tags = set()
    if os.path.isdir(root):
        tags = {re.sub(r"-r\d+$", "", d) for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))}
    if not tags:
        # Off-cluster there are no result directories at all, so the arm names have to
        # come from the export or nothing downstream can name a single arm.
        model = os.path.basename(RESULTS.rstrip("/"))
        path = os.path.join(SCORES_ROOT, model, f"{bench.replace('/', '_')}.json")
        if os.path.exists(path):
            with open(path) as f:
                tags = set(json.load(f).get("arms", {}))
    return sorted(tags)


def _drop_partial(per_dir, label):
    """Keep only the passes that cover every item the fullest pass covers.

    A run still in flight has a partial output file, and it is on disk under a name this
    tool happily globs. Counting it as a repeat does not fail -- it silently shifts the
    estimate, because the finished subset of a partial run is not a random subset of the
    items (it is the fast ones, which skew easy). This was visible as MMMU deltas moving
    by 0.4 points between two invocations minutes apart while 15 repeat jobs were
    running. Dropping the partial pass is right; averaging it in is not.
    """
    if not per_dir:
        return {}
    full = max(len(d) for d in per_dir)
    keep = [d for d in per_dir if len(d) == full]
    if len(keep) != len(per_dir):
        print(f"  note: {label}: ignoring {len(per_dir) - len(keep)} incomplete pass"
              f"(es) ({[len(d) for d in per_dir]} items); using {len(keep)}")
    merged = {}
    for d in keep:
        for k, v in d.items():
            merged.setdefault(k, []).append(v)
    return merged


def gpqa_items(tag):
    from gpqa_diamond import extract
    d = os.path.join(RESULTS, "gpqa", tag)
    passes = []
    for path in sorted(glob.glob(os.path.join(d, "rep*.jsonl"))):
        one = {}
        for line in open(path):
            r = json.loads(line)
            one[r["record_id"]] = int(extract(r.get("content")) == r["gold"])
        passes.append(one)
    return _drop_partial(passes, f"gpqa/{tag}")


def ifbench_items(tag):
    passes = []
    for d in _repeat_dirs(os.path.join(RESULTS, "ifbench"), tag):
        m = glob.glob(os.path.join(d, "*eval_results_strict.jsonl"))
        if not m:
            continue
        one = {}
        for line in open(m[0]):
            r = json.loads(line)
            one[r["prompt"]] = int(r["follow_all_instructions"])
        passes.append(one)
    return _drop_partial(passes, f"ifbench/{tag}")


def mmmu_items(tag):
    import random
    sys.path.insert(0, HARNESS)
    from evaluate import get_multi_choice_info, parse_multi_choice_response
    passes = []
    for d in _repeat_dirs(os.path.join(RESULTS, "mmmu", "vision_cot"), tag):
        raw = os.path.join(d, "raw.jsonl")
        if not os.path.exists(raw):
            continue
        # SEED, and re-seed per pass. MMMU's parse_multi_choice_response ends with
        #     if len(candidates) == 0:  # still not get answer, randomly choose one.
        #         pred_index = random.choice(all_choices)
        # which is unseeded. About 1% of generations are unparseable (mostly the
        # truncated ones), so every invocation scored those items differently and the
        # accuracy moved ~0.1 points between two runs on IDENTICAL files. That is
        # enough to flip a significance verdict: bf16-pd vs nvfp4pd read +1.50
        # [+0.20,+2.80]* on one run and +1.62 [-0.08,+3.32] on the next.
        # Seeding does not make the fallback correct -- it makes it reproducible, which
        # is the part that was silently untrue.
        random.seed(20260825)
        one = {}
        for line in open(raw):
            r = json.loads(line)
            i2a, choices = get_multi_choice_info(ast.literal_eval(str(r["options"])))
            pred = parse_multi_choice_response(r.get("content") or "", choices, i2a)
            one[r["id"]] = int(pred == r["answer"])
        passes.append(one)
    return _drop_partial(passes, f"mmmu/{tag}")


def ocrbench_items(tag):
    """Per-item OCRBench scores, REWEIGHTED so a flat mean reproduces the headline.

    OCRBench's own overall is an UNWEIGHTED MEAN OF CATEGORY MEANS, not a mean over
    items: an 800-item category counts exactly as much as a 200-item one. Every other
    benchmark here is a flat item mean, and the machinery in this file -- per-repeat
    scores as column means, paired bootstrap over items -- assumes that.

    Rather than special-case the aggregation everywhere, each item is scaled by

        w_i = N / (K * n_c)        c = its category, K = categories, N = items

    so that mean_i(w_i * s_i) == (1/K) * sum_c mean(category c). The existing code then
    produces the benchmark's own number with no changes.

    The caveat, since a reweighting is easy to forget: a bootstrap resample changes how
    many items each category contributes while these weights stay fixed, so the interval
    treats the category sizes as given. That is the right question here -- the arms are
    compared on a fixed benchmark -- but it is not a statement about resampling
    categories.

    Note also that `text spotting en` floors near zero for every model (1.57 for
    InternVL2.5-26B), so it contributes ~1/8 of the headline as a constant. Read the
    per-category table in summary.json before attributing a change to the overall.
    """
    import collections
    passes = []
    for d in _repeat_dirs(os.path.join(RESULTS, "ocrbench"), tag):
        path = os.path.join(d, "scored.jsonl")
        if not os.path.exists(path):
            continue
        rows = []
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except Exception:                                     # noqa: BLE001
                continue
        if not rows:
            continue
        cat = {t: c for c, ts in OCRBENCH_CATEGORIES.items() for t in ts}
        n_by_cat = collections.Counter(cat.get(r["type"], "other") for r in rows)
        K = len(n_by_cat)
        N = len(rows)
        one = {}
        for r in rows:
            n_c = n_by_cat[cat.get(r["type"], "other")]
            one[r["id"]] = r["score"] * N / (K * n_c)
        passes.append(one)
    return _drop_partial(passes, f"ocrbench/{tag}")


# Mirrors the mapping in ocrbench_score.py; kept here so this file can weight items
# without importing the scorer (which pulls in the benchmark harness).
OCRBENCH_CATEGORIES = {
    "text_recognition": ["text recognition en", "fine-grained text recognition en",
                         "full-page OCR en"],
    "text_detection": ["text grounding en", "VQA with position en"],
    "text_spotting": ["text spotting en"],
    "relationship_extraction": ["key information extraction en",
                                "key information mapping en"],
    "element_parsing": ["document parsing en", "chart parsing en", "table parsing en",
                        "formula recognition en"],
    "mathematical_calculation": ["math QA en", "text counting en"],
    "visual_text_understanding": ["document classification en", "cognition VQA en",
                                  "diagram QA en"],
    "knowledge_reasoning": ["reasoning VQA en", "science QA en", "APP agent en",
                            "ASCII art classification en"],
}

def mmlu_pro_items(tag):
    """Per-item MMLU-Pro correctness, scored with the driver's own seeded extractor.

    Rescored here from raw output rather than read from summary.json, like every other
    loader, because the pairwise tests need per-item results and a summary carries only
    the aggregate.

    SHARDS ARE GLOBBED AND DEDUPLICATED. A 12,032-item arm is generated by several jobs
    writing raw.shard{K}of{N}.jsonl side by side, so one "pass" is the union of those
    files, not one file. Deduplicating on question_id matters because a directory that
    was run unsharded and then re-run sharded holds both, and the overlap would
    otherwise be counted twice -- inflating n while quietly weighting some items double.

    The random guess for an unextractable answer is seeded and applied in question_id
    order, exactly as mmlu_pro_infer.score does it, so this file and summary.json agree
    to the item. An unseeded guess would make the same outputs score differently here
    than they did at generation time, and the difference would be small enough to look
    like noise.
    """
    import mmlu_pro_infer
    passes = []
    for d in _repeat_dirs(os.path.join(RESULTS, "mmlu_pro"), tag):
        rows, seen = [], set()
        for path in sorted(glob.glob(os.path.join(d, "raw*.jsonl"))):
            for line in open(path):
                try:
                    r = json.loads(line)
                except Exception:                                  # noqa: BLE001
                    continue
                if r["question_id"] in seen:
                    continue
                seen.add(r["question_id"])
                rows.append(r)
        if not rows:
            continue
        rng = random.Random(mmlu_pro_infer.SEED)
        one = {}
        for r in sorted(rows, key=lambda x: x["question_id"]):
            letter, _ = mmlu_pro_infer.predict(r.get("content"), rng)
            one[r["question_id"]] = int(letter == r["gold"])
        passes.append(one)
    return _drop_partial(passes, f"mmlu_pro/{tag}")


LOADERS = {"gpqa": gpqa_items, "ifbench": ifbench_items, "mmmu": mmmu_items,
           "ocrbench": ocrbench_items, "mmlu_pro": mmlu_pro_items}

_CACHE = {}


SCORES_ROOT = os.path.join(os.path.dirname(HERE), "scores")


def _exported(bench, tag):
    """Per-item scores from the versioned export, or {} if it has none for this arm.

    THE OFFLINE PATH. The generations are gitignored -- 785 MB of reproducible output --
    so a clone has no raw.jsonl to score and every test here is paired over items, which
    an aggregate cannot be un-averaged back into. bin/export_scores.py freezes exactly
    the per-item vectors these loaders produce, ~1 MB compressed, and this reads them
    back so plotting and table generation run anywhere.

    Consulted ONLY when the live loader finds nothing. On the cluster the raw data wins,
    so a stale export can never silently override a fresh result; run
    `export_scores.py --check` to find out if it has drifted.
    """
    model = os.path.basename(RESULTS.rstrip("/"))
    path = os.path.join(SCORES_ROOT, model, f"{bench.replace('/', '_')}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    arm = doc.get("arms", {}).get(tag)
    if not arm:
        return {}
    return {i: list(v) for i, v in zip(arm["ids"], arm["scores"])}


def items(bench, tag):
    """Cached per-item results. The matrix asks for the same arm k-1 times."""
    if (bench, tag) not in _CACHE:
        try:
            live = LOADERS[bench](tag)
        except Exception:                                          # noqa: BLE001
            # A missing harness import is normal off-cluster: mmmu_items needs MMMU-Pro's
            # own parser, gpqa_items needs the driver. Falling back keeps the offline
            # path working on a machine with none of that installed.
            live = {}
        _CACHE[(bench, tag)] = live or _exported(bench, tag)
    return _CACHE[(bench, tag)]


# How many items a FINISHED pass has. Used to tell "this arm was measured" from "this
# arm is still being measured", which nothing else in a results directory records.
# ocrbench is the EN subset: 7,400 of the 10,000 items, 21 of the 30 task types.
BENCH_ITEMS = {"mmmu": 1730, "gpqa": 198, "ifbench": 300, "ocrbench": 7400,
               "mmlu_pro": 12032}


def expected_items(bench):
    return BENCH_ITEMS.get(bench, 0)


def per_repeat_scores(per, ids):
    """One score per generation pass, in points. Column means of items x repeats."""
    M, k = to_matrix(per, ids)
    # float(), not the numpy scalars: they survive statistics.mean and every later
    # comparison, so `significant` ends up a numpy.bool_ and json.dump refuses it --
    # after the whole table has already printed.
    return [float(x) for x in 100.0 * M.mean(axis=0)], k


def to_matrix(per, ids):
    """items x repeats, 0/1. Ragged repeat counts are truncated to the shortest."""
    k = min(len(per[i]) for i in ids)
    return np.array([per[i][:k] for i in ids], dtype=np.float64), k


def _exact_binom_two_sided(b, c):
    """Exact two-sided p for b successes in b+c trials at q=0.5. No scipy in the overlay.

    Sums the probability of every outcome at most as likely as the observed one, which
    is the exact test rather than the chi-square approximation. The approximation is
    unreliable at exactly the point that matters here -- few discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    obs = comb(n, b)
    total = sum(comb(n, k) for k in range(n + 1) if comb(n, k) <= obs)
    # int / int, NOT total / (2.0 ** n). Both operands stay exact Python integers and
    # CPython's big-int true division is correctly rounded, so this is accurate for any
    # n. Materialising 2.0 ** n as a float overflows above n = 1023 -- fine for MMMU at
    # 1730 items, where discordant pairs number in the low hundreds, and fatal for
    # MMLU-Pro at 12,032, where they run to thousands:
    #     OverflowError: (34, 'Numerical result out of range')
    return min(1.0, total / (1 << n))


def mcnemar(bench, tag_a, tag_b):
    """Exact McNemar per repeat, for tag_a - tag_b. Returns None if data is missing.

    THE TEST. Both arms answer the identical items, so the evidence lives entirely in
    the items where they DISAGREE: b = a right & b wrong, c = a wrong & b right. Under
    the null the b+c discordant items split 50/50, so b ~ Binomial(b+c, 1/2) and the
    exact binomial tail is the p-value. Concordant items carry no information and
    correctly contribute nothing -- which is why this resolves differences that an
    unpaired binomial on n=1730 cannot.

    SE(delta) = sqrt(flip_rate / N), not sqrt(2*g*(1-g)/N). The independence formula
    overstates the uncertainty roughly 2-3x here (1.51 vs ~0.45 points measured), which
    is the difference between "no difference detectable" and three of four pairs
    resolved.

    ONE TEST PER REPEAT, NOT ONE POOLED TEST. The method assumes one binary score per
    example; we have four. Pooling the discordant counts would treat 6920 correlated
    observations as independent and inflate significance -- the same item reappears in
    every repeat. Running the test separately per repeat keeps each one exactly within
    the method's assumptions, and the reported verdict is how many of the four reject.
    Requiring all four is strictly more conservative than any pooling, and needs no
    assumption about within-item correlation.
    """
    A, B = items(bench, tag_a), items(bench, tag_b)
    if not A or not B:
        return None
    ids = sorted(set(A) & set(B))
    k = min(min(len(A[i]) for i in ids), min(len(B[i]) for i in ids))
    if not ids or k < 1:
        return None
    out = []
    for r in range(k):
        b = sum(1 for i in ids if A[i][r] and not B[i][r])
        c = sum(1 for i in ids if B[i][r] and not A[i][r])
        n = len(ids)
        flip = (b + c) / n
        out.append({"b": b, "c": c, "delta": 100.0 * (b - c) / n,
                    "se": 100.0 * (flip / n) ** 0.5, "flip": 100.0 * flip,
                    "p": _exact_binom_two_sided(b, c)})
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values. Controls family-wise error over the pairs.

    A matrix of k arms runs k(k-1)/2 tests at once -- six for four arms -- so at a 4%
    level roughly one spurious rejection is expected per matrix by chance. Holm is
    uniformly more powerful than plain Bonferroni and needs no independence assumption
    between the tests, which matters because these tests share arms and are correlated.
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, min(1.0, val))
        adj[i] = running
    return adj


def signflip(bench, tag_a, tag_b, draws=None, rng=None):
    """EXACT paired sign-flip test on per-item differences. No simulation.

    THE ESTIMAND is the expected accuracy difference over items AND generation
    randomness. Per item, d_i = mean_r(a correct) - mean_r(b correct) over the repeats.
    Under the null that the two arms are exchangeable on every item, each d_i is equally
    likely to have its sign flipped, and the p-value is the probability that
    |mean(+/-d)| reaches the observed value.

    WHY THIS HAS A CLOSED FORM. With k repeats every d_i is a multiple of 1/k, so on the
    integer lattice the null statistic is sum(eps_i * e_i) with e_i integers. Group the
    items by magnitude m: the c items of that magnitude contribute m*(2B - c) with
    B ~ Binomial(c, 1/2). The exact null distribution is therefore a convolution of a
    handful of binomials -- at k=4 there are at most four distinct magnitudes -- and it
    is computed here rather than sampled.

    This replaced 200k Monte-Carlo sign flips. The simulation agreed to three
    significant figures wherever it could resolve, but its p-value floors at
    1/(draws+1): it reported 5.0e-06 for a comparison whose exact p is 6.6e-08, so
    "smaller than we can measure" was standing in for two orders of magnitude. It is
    also faster, and it is deterministic, so a verdict cannot move between runs.

    Falls back to sampling if the differences are not on a clean lattice, which is what
    happens if a future benchmark scores something other than 0/1 per item.

    WHY NOT McNEMAR PER REPEAT, WHICH THE PAPER PRESCRIBES: it assumes one binary score
    per example. Pooling discordant counts over repeats treats correlated observations
    as independent; requiring all four per-repeat tests to reject discards the precision
    the repeats bought, and called nothing significant on Muse including a +2.53 gap.
    Averaging repeats first is the cluster-robust reading, with the item as the cluster,
    and is conservative: d_i carries its own estimation error, which inflates the spread
    across items rather than shrinking it.
    """
    import collections
    from math import comb, erfc, lgamma, sqrt

    A, B = items(bench, tag_a), items(bench, tag_b)
    if not A or not B:
        return None
    ids = sorted(set(A) & set(B))
    k = min(min(len(A[i]) for i in ids), min(len(B[i]) for i in ids))
    d = np.array([np.mean(A[i][:k]) - np.mean(B[i][:k]) for i in ids])
    n = len(d)
    obs = float(d.mean())
    se = 100.0 * float(d.std(ddof=1)) / sqrt(n)

    scaled = d * k
    if np.allclose(scaled, np.rint(scaled), atol=1e-9):
        e = np.rint(scaled).astype(int)
        obs_int = abs(int(e.sum()))
        pmf = np.array([1.0])
        for m, c in collections.Counter(np.abs(e[e != 0]).tolist()).items():
            # Binomial pmf in LOG SPACE. The direct form -- comb(c, i) into a float
            # array -- raises "int too large to convert to float" once c passes ~1050,
            # because comb(1100, 550) is about 10^330 and a float tops out at 1.8e308.
            # The count c here is items sharing one magnitude of difference, which at
            # 12,032 MMLU-Pro items reaches the thousands. Shifting by the max before
            # exponentiating keeps every term in range; the tails underflow to zero,
            # which is what they are.
            lg = lgamma(c + 1)
            logs = np.array([lg - lgamma(i + 1) - lgamma(c - i + 1)
                             for i in range(c + 1)])
            b = np.exp(logs - logs.max())
            b /= b.sum()
            g = np.zeros(2 * m * c + 1)
            g[np.arange(c + 1) * 2 * m] = b        # supported on -mc, -mc+2m, ..., mc
            pmf = np.convolve(pmf, g)
        half = (len(pmf) - 1) // 2
        idx = np.arange(len(pmf)) - half
        pval = float(pmf[np.abs(idx) >= obs_int].sum()) if obs_int else 1.0
        # BOTH one-sided tails, from the same exact null and the SIGNED statistic.
        # Stored rather than derived from `pval` later: on this discrete distribution
        # p_gt is pval/2 only when the observed sum is non-zero, and the complementary
        # tail is not 1 - pval/2 because the atom at the observed value belongs to both.
        # Storing both also means the caller never has to reason about which way round
        # the pair was cached -- it picks the tail that matches its hypothesis.
        s_int = int(e.sum())
        p_gt = float(pmf[idx >= s_int].sum())
        p_lt = float(pmf[idx <= s_int].sum())
        how = "exact"
    else:
        # Non-lattice differences: fall back to the randomization distribution's normal
        # limit, whose variance under the null is sum(d^2)/n^2 -- NOT the sample
        # variance, since the null fixes the mean at zero.
        z = obs / (sqrt(float((d ** 2).sum())) / n) if d.any() else 0.0
        pval = erfc(abs(z) / sqrt(2))
        p_gt = 0.5 * erfc(z / sqrt(2))                # P(Z >= z) = erfc(z/sqrt2)/2
        p_lt = 0.5 * erfc(-z / sqrt(2))
        how = "normal"
    return {"delta": 100.0 * obs, "n": n, "se": se, "p": min(1.0, pval),
            "p_gt": min(1.0, p_gt), "p_lt": min(1.0, p_lt), "how": how}


def mcnemar_table(benches, tags, alpha=0.05, show_mcnemar=False):
    """The decisive pairwise test: exact McNemar, Holm-corrected across the matrix."""
    if len(benches) != 1:
        print("\n(McNemar table is per benchmark; run with a single --benches)")
        return []
    bench = benches[0]
    pairs, rows = [], []
    for i, a in enumerate(tags):
        for b_ in tags[i + 1:]:
            res = mcnemar(bench, a, b_)
            if res:
                pairs.append((a, b_, res))
    if not pairs:
        return []
    # Correct on the WORST repeat: the verdict is "rejects in all four", so the family
    # of tests being controlled is the one that decides it.
    worst = [max(r["p"] for r in res) for _, _, res in pairs]
    adj = holm(worst)

    # Primary: paired randomization on the repeat-averaged per-item differences.
    flip = [signflip(bench, a, b_) for a, b_, _ in pairs]
    # UNCORRECTED, and the p_holm column is kept beside it so the multiplicity cost is
    # visible rather than hidden either way.
    #
    # Correction controls the family-wise error rate, which is the right guarantee when
    # a family is FIXED IN ADVANCE. This one is not: it is every pair among whatever
    # arms happen to be in the sweep, so its size changes as arms are added and a
    # verdict about two untouched arms changes with it. Adding lut3-nvfp4-gptq took the
    # family from 10 pairs to 15 and demoted nvfp4 - nvfp4pd on Gemma from REJECT
    # (p_holm 3.22e-02) to no (5.06e-02) on identical data. A verdict that depends on
    # what else was run that day is not a useful exploratory scan.
    #
    # The price is stated rather than glossed: at alpha=0.05 over 15 pairs, if every
    # null were true the chance of at least one spurious REJECT is about 54%, roughly
    # 0.75 per table. So this table is EXPLORATORY. The claims fixed in advance live in
    # the CONTRASTS table, whose family is a fixed list of hypotheses and so cannot grow
    # when an arm is added.
    adj_f = holm([f["p"] for f in flip])
    print(f"\nexact paired sign-flip test on {bench}, per-comparison alpha={alpha}, "
          f"UNCORRECTED over {len(pairs)} pairs (exploratory; see the contrasts table)")
    print(f"  {'pair':<26} {'delta':>7} {'SE':>5} {'flip':>6} {'b/c':>11} "
          f"{'p':>10} {'p_holm':>9}  verdict")
    for (a, b_, res), f, pa in zip(pairs, flip, adj_f):
        # Discordance, from the same per-repeat counts McNemar uses. It is the mechanism
        # behind SE = sqrt(flip/N) -- concordant items contribute nothing -- and the b/c
        # split shows whether the disagreements are one-sided or merely numerous. Shown
        # here so the diagnostics survive without a second, contradictory verdict column
        # beside them.
        fl = statistics.mean(r["flip"] for r in res)
        bb = statistics.mean(r["b"] for r in res)
        cc = statistics.mean(r["c"] for r in res)
        print(f"  {a + ' - ' + b_:<26} {f['delta']:>+7.2f} {f['se']:>5.2f} "
              f"{fl:>5.1f}% {f'{bb:.0f}/{cc:.0f}':>11} "
              f"{f['p']:>10.2e} {pa:>9.2e}  "
              f"{'REJECT' if f['p'] < alpha else 'no'}")
        rows.append({"a": a, "b": b_, "test": "signflip", "delta": f["delta"],
                     "se": f["se"], "p": f["p"], "p_holm": pa,
                     # Carried for the contrasts table, which tests directional
                     # hypotheses. This table's own verdict stays two-sided: it is a
                     # scan over every pair, and a scan has no direction to claim.
                     "p_gt": f["p_gt"], "p_lt": f["p_lt"],
                     "reject": bool(f["p"] < alpha),
                     "reject_holm": bool(pa < alpha)})

    if not show_mcnemar:
        return rows
    # OFF BY DEFAULT. The "reject in all k repeats" rule is not a calibrated procedure
    # -- it behaves like demanding k independent replications and has no stated error
    # rate -- so its verdicts contradict the primary test without a principled reason to
    # prefer them. It is also not the reference paper's procedure, which runs one test
    # on one binary score per example; the per-repeat conjunction is a workaround for
    # having four, and the sign-flip test is the better one. Kept for anyone who wants
    # the strictest reading, and because the b/c counts it computes now feed the table
    # above.
    print(f"\nexact McNemar on {bench}, per repeat, Holm-corrected over "
          f"{len(pairs)} pairs (alpha={alpha}) -- strictest reading, not calibrated")
    print(f"  {'pair':<26} {'delta':>7} {'SE':>5} {'flip':>6} {'b/c':>11} "
          f"{'p (worst)':>10} {'p_holm':>8}  verdict")
    for (a, b_, res), praw, padj in zip(pairs, worst, adj):
        d = statistics.mean(r["delta"] for r in res)
        se = statistics.mean(r["se"] for r in res)
        flip = statistics.mean(r["flip"] for r in res)
        bb = statistics.mean(r["b"] for r in res)
        cc = statistics.mean(r["c"] for r in res)
        n_rej = sum(1 for r in res if r["p"] < alpha)
        verdict = (f"REJECT ({n_rej}/{len(res)})" if padj < alpha and n_rej == len(res)
                   else f"no ({n_rej}/{len(res)} raw)")
        print(f"  {a + ' - ' + b_:<26} {d:>+7.2f} {se:>5.2f} {flip:>5.1f}% "
              f"{f'{bb:.0f}/{cc:.0f}':>11} {praw:>10.2e} {padj:>8.2e}  {verdict}")
        rows.append({"a": a, "b": b_, "delta": d, "se": se, "flip_pct": flip,
                     "b": bb, "c": cc, "p_worst_repeat": praw, "p_holm": padj,
                     "repeats_rejecting": n_rej, "n_repeats": len(res),
                     "reject": bool(padj < alpha and n_rej == len(res))})
    print("  delta and SE are means over repeats; SE = sqrt(flip_rate/N), the paired")
    print("  form. p is the WORST of the per-repeat exact tests, so REJECT means every")
    print("  repeat rejected and survived correction for all pairs simultaneously.")
    return rows


def significance_matrix(rows, tags):
    """ONE matrix, rendered from the decisive test's own verdicts.

    There used to be two: a grid of Welch-t intervals over repeats and a +/-/~ grid
    derived from it. They showed the same pairs twice and both could disagree with the
    sign-flip table printed above -- a different test on a different variance source,
    reaching a different verdict on the same comparison, with nothing saying which was
    authoritative. On a single-pass benchmark the repeat-based grids could not compute
    at all and printed a wall of placeholders beside a table of REJECTs.

    So the numbers live in the pairwise table, which already carries delta, SE, p and
    Holm-corrected p for every pair. This is purely the scan-at-a-glance view of those
    verdicts: with seven arms that is 21 pairs, worth a triangle but not worth a second
    opinion. No magnitudes here on purpose -- a reader comparing two glyphs is asking
    "which way, and is it resolved", and a number invites treating an unresolved `~`
    as a measurement.

    The diagonal is printed as `*` so the grid is square and each row lines up under
    its own column. The lower half stays blank: the comparison is antisymmetric, so it
    is the same pairs written backwards.
    """
    if not rows:
        return
    by_pair = {(r["a"], r["b"]): r for r in rows}
    present = [t for t in tags if any(t in (r["a"], r["b"]) for r in rows)]
    if len(present) < 2:
        return

    w = max(len(t) for t in present) + 2
    cw = max(6, max(len(t) for t in present) + 2)
    print("\n  + row significantly higher   - row significantly lower   "
          "~ not resolved   * self")
    print(f"{'':<{w}}" + "".join(f"{t:>{cw}}" for t in present))
    for i, r in enumerate(present):
        cells = [f"{'':>{cw}}"] * i + [f"{'*':>{cw}}"]
        for c in present[i + 1:]:
            hit, sign = by_pair.get((r, c)), 1.0
            if hit is None:
                hit, sign = by_pair.get((c, r)), -1.0
            if hit is None:
                cells.append(f"{'n/a':>{cw}}")
                continue
            d = sign * hit["delta"]
            cells.append(f"{('+' if d > 0 else '-') if hit['reject'] else '~':>{cw}}")
        print(f"{r:<{w}}" + "".join(cells))
    print("  per-comparison alpha, uncorrected -- exploratory. `~` means not "
          "resolved, not equal.")


# The hypotheses, as (a, b) tag pairs. Each is written in the DIRECTION IT IS CLAIMED --
# `a - b` comes out positive when the hypothesis holds -- so the sign of the delta reads
# as support or contradiction without a second lookup.
#
# The test behind them stays TWO-SIDED. Writing a hypothesis with a direction is a claim
# about which way the effect should go; it is not a licence to halve the p-value, which
# is what a one-sided test does and which would move verdicts on evidence that has not
# changed. So a REJECT with a NEGATIVE delta is a real result that CONTRADICTS the
# hypothesis rather than supporting it -- read the sign, not just the verdict.
#
# BASELINE still resolves to the model's own baseline if a contrast asks for it. None do
# now: the baseline is single-engine bf16 on all three models, so naming bf16 outright
# says what is being compared instead of making the reader resolve an indirection.
# (a, b, description, sides). `sides` is 1 for a directional claim, tested one-sided in
# the direction written, and 2 for a difference with no prior about which way it goes.
#
# SIDEDNESS IS A PROPERTY OF THE HYPOTHESIS, not a global setting. The first two are
# genuinely directional -- bf16 should not lose to its own 4-bit quantisation, and a
# format that keeps precision on decode should not lose to one that drops it. Testing
# those one-sided is what justifies spending the power.
#
# nvfp4-gptq vs lut3-nvfp4-gptq is not like that. Both come out of the same GPTQ
# pipeline and differ only in what the DECODE engine holds: NVFP4 against a 3-bit Lloyd
# grid. Nothing says which should win -- fewer bits is not automatically worse when the
# levels are fitted to the weight distribution. A one-sided test would encode an
# expectation nobody holds, and worse, it could not report LUT3 winning at all: the
# rejection region sits entirely on one side, so evidence the other way returns a p near
# 1 and reads as "no difference".
# The historical default. Kept so every model measured on MMMU-Pro reports exactly what
# it did before; models with no MMMU results fall back to what they have (see analyze).
DEFAULT_BENCHES = ("mmmu",)

CONTRASTS = [
    ("bf16", "nvfp4", "what W4A4 costs", 1),
    ("nvfp4pd", "nvfp4", "what NVFP4PD recovers over W4A4", 1),
    # ("nvfp4a16", "nvfp4pd", "what the mixed format still costs against W4A16", 1),
    ("nvfp4-gptq", "lut3-nvfp4-gptq", "what a LUT3 decode engine costs", 2),
    # Nemotron only -- the other models have no NVIDIA-published checkpoint, so they
    # report no data here, which is the honest rendering rather than an omission.
    # One-sided: the decode engine holds byte-identical weights and merely stops
    # quantizing activations, so it cannot be worse than the W4A4 pair for any reason
    # other than noise. Same shape as nvfp4pd > nvfp4.
    ("nvfp4-nv-pd", "nvfp4-nv", "what disabling decode act-quant recovers", 1),
    # MXFP4 models (gpt-oss-120b, kimi-k3). Their released checkpoints are weight-only,
    # so mxfp4a16 is what ships and mxfp4 adds activation quantization on the SAME weight
    # files -- the arms differ in one config field and nothing else. One-sided for the
    # same reason as bf16 > nvfp4: quantizing activations cannot help, so the only
    # question is how much it costs.
    ("mxfp4a16", "mxfp4", "what MXFP4 activations cost", 1),
]


def cross_model(collected, args, alpha=0.05):
    """The same contrasts across every model, side by side.

    A per-model table answers "does this format differ HERE". The claim worth making is
    architectural -- that the mixed format recovers what W4A4 costs -- and that is a
    statement about replication across unrelated models, which no single table shows.
    Reading it off three separate outputs by eye is exactly where a reader stops
    checking whether the verdicts actually agree.

    THE TESTS HERE ARE ONE-SIDED, and this is the only table where they are. Each
    contrast in CONTRASTS is a directional claim fixed before the data, so the tail that
    matters is the one the claim points at; the exploratory matrix above has no
    direction to claim and stays two-sided. No re-computation is involved -- the exact
    sign-flip null is symmetric and both of its tails were already stored, so the
    one-sided p is a different summary of the same null, not a different test.

    What it buys and what it costs: a one-sided test at alpha rejects on evidence a
    two-sided test would need 2*alpha for, which is more power, paid for by having no
    rejection region at all on the other side. That is only sound because the direction
    was fixed in advance -- picking it after seeing the sign is how a 0.05 test becomes
    a 0.10 one -- and it is why the REVERSED verdict exists below.

    The p_holm column is RE-CORRECTED over the contrasts in this table rather than over
    every pair in the matrix.

    That is deliberate, and it matters. Holm controls error across a declared family, so
    the family should be the hypotheses you set out to test, not every pairwise
    combination that happens to be printed. Correcting over the matrix makes a headline
    verdict depend on what else is in the table: adding the GPTQ arm took
    bf16-pd - nvfp4a16 from p_holm 2.1e-02 to 5.0e-02, flipping REJECT to no, without a
    single new observation. CONTRASTS is a fixed list of hypotheses and every one of
    them is printed whatever it says, so it is the honest family.

    Correction stays WITHIN a model: pooling three models into one family would make
    each model's verdict depend on the others, which is not what replication means.

    AN ARM THAT IS STILL GENERATING IS SKIPPED, not reported with a caveat. Its output
    file is a valid-looking result -- a plausible score over a plausible-looking number
    of items -- and the sign-flip test happily pairs it against a finished arm over the
    ITEMS THEY SHARE, which are the ones that finished first, which are the fast ones.
    Qwen's lut3-nvfp4-gptq at 1006 of 1730 items read 75.35, above its own bf16 at
    74.78, and would have put a bogus REJECT in the LUT3 row of this table. There is no
    honest number to print for a partial arm, so the row says why it is empty instead.
    Partial arms are also left out of the Holm family, since a p-value computed on a
    biased subset should not be shifting the threshold for the contrasts that are real.
    """
    print("\n" + "=" * 72)
    print("  cross-model summary")
    print("=" * 72)
    models = list(collected)

    def shortfall(c, tag):
        """'1006/1730 mmmu' for the worst benchmark this arm has not finished, or ''."""
        short = [(b, n, e) for b, (n, e) in c.get("coverage", {}).get(tag, {}).items()
                 if e and n < e]
        if not short:
            return ""
        b, n, e = min(short, key=lambda x: x[1] / x[2])
        return f"{n}/{e} {b}"

    def oriented(c, aa, bb):
        """The test for `aa - bb`, ONE-SIDED in the direction the hypothesis claims.

        Pairs are cached one way round only, so half the lookups come back reversed and
        every quantity has to be turned around with them: the delta by its sign, and the
        p-value by taking the OTHER tail. p_gt on the (bb, aa) entry is the probability
        of bb beating aa, which is the wrong hypothesis -- the same number the two-sided
        p would have hidden the direction of.

        `p_rev` is the same test against the opposite alternative. It exists because a
        one-sided test cannot reject in the direction it is not looking, so without it a
        strongly contradicted hypothesis and an unresolved one both print `no`.
        """
        r, sign, tail = c["signflip"].get((aa, bb)), 1.0, "p_gt"
        if r is None:
            r, sign, tail = c["signflip"].get((bb, aa)), -1.0, "p_lt"
        if r is None:
            return None
        rev = "p_lt" if tail == "p_gt" else "p_gt"
        # Older arm_comparison.json rows predate the one-sided tails. Fall back to the
        # two-sided p rather than crashing, and to half of it, which is what the tail
        # would be if the observed effect went the way the hypothesis claims.
        p_one = r.get(tail, r["p"] / 2 if sign * r["delta"] > 0 else 1 - r["p"] / 2)
        p_rev = r.get(rev, 1.0 - p_one)
        return {"delta": sign * r["delta"], "se": r["se"], "p": p_one,
                "p_rev": p_rev, "p_two": r["p"]}

    # Holm within each model, over the CONTRASTS present for it, from the one-sided p.
    adjusted = {}
    for m in models:
        c = collected[m]
        keys, raw = [], []
        for a, b, _, sides in CONTRASTS:
            aa = c["baseline"] if a == "BASELINE" else a
            bb = c["baseline"] if b == "BASELINE" else b
            if shortfall(c, aa) or shortfall(c, bb):
                continue
            r = oriented(c, aa, bb)
            if r is not None:
                keys.append((aa, bb))
                raw.append(r["p_two"] if sides == 2 else r["p"])
        adjusted[m] = dict(zip(keys, holm(raw))) if raw else {}
    w = max(len(m) for m in models) + 2
    for a, b, what, sides in CONTRASTS:
        rel = ">" if sides == 1 else "!="
        lbl = "p_1side" if sides == 1 else "p_2side"
        print(f"\n{what}:  H1 = {a} {rel} {b}   (delta = {a} - {b})")
        print(f"  {'model':<{w}} {'delta':>7} {'SE':>5} {lbl:>9} {'p_holm':>9}"
              f"  verdict")
        for m in models:
            c = collected[m]
            aa = c["baseline"] if a == "BASELINE" else a
            bb = c["baseline"] if b == "BASELINE" else b
            dashes = f"  {m:<{w}} {'-':>7} {'-':>5} {'-':>9} {'-':>9}"
            # Checked BEFORE the lookup: a partial arm has a row in `signflip` like any
            # other, so asking whether the test exists would answer yes and print it.
            busy = shortfall(c, aa) or shortfall(c, bb)
            if busy:
                which = aa if shortfall(c, aa) else bb
                print(f"{dashes}  still generating ({which} {busy})")
                continue
            r = oriented(c, aa, bb)
            if r is None:
                print(f"{dashes}  no data")
                continue
            padj = adjusted[m].get((aa, bb), adjusted[m].get((bb, aa), 1.0))
            # PER-COMPARISON, not Holm. These contrasts are hypotheses fixed in
            # advance and every one of them is reported whatever it says, so there is
            # no selective reporting to correct for and alpha is the error rate that
            # was actually advertised. Correcting would also make each contrast's
            # verdict depend on how many OTHER questions were asked: adding the two
            # lut3 contrasts would weaken nvfp4 - nvfp4pd without a single new
            # observation, which is the same instability that made the pairwise scan
            # unusable.
            #
            # The residual is real and not corrected away: at alpha=0.05 over these
            # contrasts, if every null were true, the chance of at least one spurious
            # REJECT per model is about 1 - 0.95^k. p_holm stays in the table so the
            # stricter reading is one column away.
            #
            # THREE outcomes, not two. A one-sided test puts its whole rejection region
            # on one side, so no amount of evidence the OTHER way can make it reject --
            # it just returns a p near 1. Printing that as `no` beside an unresolved
            # result would report "we could not tell" for a difference that is
            # resolved, in the direction the hypothesis denied. REVERSED is that case:
            # the mirror-image test rejects, so the data contradict H1 rather than
            # failing to support it. It is live -- on Muse's OCRBench, nvfp4a16 >
            # nvfp4pd comes out -0.70 with the reverse test at p = 1.1e-02.
            # REVERSED is meaningless for a two-sided contrast: there is no wrong
            # direction, the delta's sign already says which way, and the single p
            # already covers both. It exists only to stop a ONE-sided test from
            # reporting a resolved difference it was pointed away from as "no".
            pv = r["p_two"] if sides == 2 else r["p"]
            if pv < alpha:
                v = "REJECT"
            elif sides == 1 and r["p_rev"] < alpha:
                v = f"REVERSED (p={r['p_rev']:.1e})"
            else:
                v = "no"
            print(f"  {m:<{w}} {r['delta']:>+7.2f} {r['se']:>5.2f} "
                  f"{pv:>9.2e} {padj:>9.2e}  {v}")
    n1 = sum(1 for *_, sd in CONTRASTS if sd == 1)
    print(f"\n  verdict: per-comparison alpha={alpha}. {n1} of these "
          f"{len(CONTRASTS)} hypotheses are")
    print("  directional and tested ONE-SIDED in the direction written above them; the")
    print("  rest are two-sided, having no prior about which way the difference goes.")
    print("  These hypotheses are fixed in advance and all are reported,")
    print("  so there is no selective reporting to correct for and a verdict cannot be")
    print("  weakened by asking another question. p_holm is the stricter family-wise")
    print("  reading, Holm within each model over whichever p each hypothesis is")
    print("  judged on -- one-sided for the directional ones, two-sided for the rest.")
    print(f"  A one-sided test at {alpha} rejects on evidence a two-sided test would "
          f"need {2 * alpha:g}")
    print("  for, so it is the more permissive test -- it buys power by refusing to")
    print("  look the other way, which is only sound because the direction was fixed")
    print("  before the data. REVERSED, on a one-sided row, means the mirror test")
    print("  rejects: H1 is contradicted, not merely unsupported. A two-sided row needs")
    print("  no such marker -- the sign of the delta already says which way it went.")
    print("  Arms still generating are skipped: a partial pass covers the items that")
    print("  finished first, which are the fast ones, so its delta is biased.")


def main():
    ap = argparse.ArgumentParser()
    # bf16-pd, NOT bf16. The deployment being studied is disaggregated, so the
    # reference has to be the unquantized model served the same way; measuring a
    # quantized format against a single-engine baseline would fold the topology into
    # the format's number.
    #
    # bf16 and bf16-pd are also not reported AS ARMS. They existed to answer two
    # correctness questions -- does the harness reproduce the published figures, and
    # does disaggregation itself cost anything -- and both passed: bf16-pd - bf16 came
    # out at -0.06 [-1.85, +1.73] averaged over the benchmarks, i.e. topology-neutral.
    # A settled control belongs in the text, not in every results table.
    # Every model with results, unless told otherwise. The model is the OUTER loop and
    # never a dimension inside a table: arms are only comparable within one model, and a
    # matrix mixing two of them would invite reading a cell as a model comparison when
    # it is nothing of the sort. Each model gets its own baseline, its own arm list and
    # its own tables.
    ap.add_argument("--model", nargs="+", default=None,
                    help="model keys under results/ (default: all with results)")
    # Defaults come from models.json, because the right baseline differs per model:
    # it must be an arm with repeats, and bf16-pd only has them on muse-glimmer.
    ap.add_argument("--baseline", default=None)
    # The quantized formats, in the order they belong in a table: the two homogeneous
    # ones, then the phase-disaggregated one that only exists because prefill and decode
    # are served separately. An arm with no results on disk is skipped with a note
    # rather than crashing, which is what makes a default list safe mid-sweep.
    ap.add_argument("--arms", nargs="+", default=None)
    # MMMU-Pro ONLY by default. It is the only one of the three that resolves anything
    # at 4 repeats: at 1730 items its 96% intervals on a difference run about +/-1 point,
    # against +/-2.6 for GPQA (198 items) and +/-3.5 for IFBench (300). Those two put a
    # `~` on every pair no matter what is true, and averaging them in drags MMMU's
    # resolved differences back under the threshold -- the 3-benchmark average called
    # every pair unresolved while MMMU alone separated three of them.
    #
    # They are still collected and still scored; `--benches gpqa ifbench mmmu` brings
    # them back. They are just not evidence about which format is better.
    # default=None so an EXPLICIT --benches mmmu is distinguishable from the default. The
    # fallback below only applies to the default; asking for a bench by name and getting a
    # different one silently would be worse than an empty table.
    ap.add_argument("--benches", nargs="+", default=None)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matrix", nargs="+", default=None,
                    help="tags for the pairwise significance matrix "
                         "(default: baseline + arms)")
    ap.add_argument("--mcnemar", action="store_true",
                    help="also print the per-repeat exact McNemar table (strictest "
                         "reading; its all-k-repeats rule is not calibrated)")
    ap.add_argument("--no-cross", action="store_true",
                    help="skip the cross-model summary")
    args = ap.parse_args()
    args.benches_explicit = args.benches is not None
    if args.benches is None:
        args.benches = list(DEFAULT_BENCHES)

    available = sorted(d for d in os.listdir(RESULTS_ROOT)
                       if os.path.isdir(os.path.join(RESULTS_ROOT, d))) if \
        os.path.isdir(RESULTS_ROOT) else []
    # A clone has no results tree, only the export. Discovering models from it keeps the
    # offline path from depending on which summary.json files happen to be tracked.
    if os.path.isdir(SCORES_ROOT):
        available = sorted(set(available) | {
            d for d in os.listdir(SCORES_ROOT)
            if os.path.isdir(os.path.join(SCORES_ROOT, d))})
    # Split on ANY whitespace, not just the shell's. A pasted command can carry a
    # non-breaking space between two model names, which the shell does not treat as a
    # separator, so argparse receives ONE token "model-a\xa0model-b" and both look
    # missing while both are listed as available.
    models = [w for m in (args.model or []) for w in m.split()] or available
    missing = [m for m in models if m not in available]
    if missing:
        raise SystemExit(f"no results for {', '.join(missing)}; "
                         f"have: {', '.join(available) or '(none)'}")
    if not models:
        raise SystemExit(f"no results under {RESULTS_ROOT}")

    collected = {}
    for i, model in enumerate(models):
        if len(models) > 1:
            print(("\n" if i else "") + "=" * 72)
            print(f"  {model}")
            print("=" * 72)
        got = analyze(model, args)
        if got:
            collected[model] = got
    if len(collected) > 1 and not args.no_cross:
        cross_model(collected, args)


def analyze(model, args):
    """One model: raw scores, then the pairwise matrix. Called once per model."""
    reg = {}
    mj = os.path.join(os.path.dirname(HERE), "models.json")
    if os.path.exists(mj):
        with open(mj) as f:
            reg = json.load(f).get(model, {}).get("report", {})
    baseline = args.baseline or reg.get("baseline", "bf16-pd")
    arms = args.arms or reg.get("arms", ["nvfp4", "nvfp4a16", "nvfp4pd"])

    global RESULTS
    RESULTS = os.path.join(RESULTS_ROOT, model)
    # The loaders cache by (benchmark, tag) and the tags repeat across models, so the
    # cache has to be dropped when RESULTS moves or the second model is scored from the
    # first one's files -- silently, since the tag names are identical.
    _CACHE.clear()

    rng = np.random.default_rng(args.seed)
    base = {b: items(b, baseline) for b in args.benches}
    base = {b: v for b, v in base.items() if v}

    # A MISSING BASELINE IS NOT A REASON TO SKIP THE MODEL. Everything below is
    # PAIRWISE -- the sign-flip table and the matrix compare every pair of arms, and
    # none of that needs a designated reference. Only the baseline's own row does.
    # Skipping the whole model meant a sweep with three scored quantized arms and a
    # baseline still generating printed nothing at all, which reads as "no data" when
    # in fact every comparison between the arms was available.
    # FALL BACK TO THE BENCHES THIS MODEL ACTUALLY HAS. --benches defaults to mmmu, and a
    # model that never ran it -- qwen3.8-2.4t is text-only and cannot, gpt-oss and kimi-k3
    # are being measured on mmlu_pro -- reported "no results yet for this model" no matter
    # how complete its runs were. Models that DO have the requested bench are unaffected:
    # this only runs when the request came back empty.
    if not base and not getattr(args, "benches_explicit", False):
        disc = [b for b in LOADERS
                if os.path.isdir(os.path.join(RESULTS, BENCH_ROOT.get(b, b)))]
        found = {b: items(b, baseline) for b in disc}
        found = {b: v for b, v in found.items() if v}
        if not found:
            found = {b: {} for b in disc if any(items(b, a) for a in arms)}
        if found:
            print(f"  note: no {', '.join(args.benches)} results; showing "
                  f"{', '.join(found)} instead")
            args.benches = list(found)
            base = {b: v for b, v in found.items() if v}

    if not base:
        have = [b for b in args.benches if any(items(b, a) for a in arms)]
        if not have:
            print(f"  no results for benches {args.benches} on this model")
            return
        print(f"  note: baseline {baseline!r} has no results yet -- showing pairwise "
              f"comparisons between the arms that do")
        # Discovered from disk, not the curated arm list, for the reason in
        # available_tags: otherwise the one finished reference is filtered out.
        found = sorted({t for b in have for t in available_tags(b)})
        tags = [t for t in found if t != baseline and any(items(b, t) for b in have)]
        if not tags:
            print(f"  no scored arms yet for this model")
            return
        base = {b: items(b, tags[0]) for b in have}
        base = {b: v for b, v in base.items() if v}
    else:
        tags = [baseline] + [a for a in arms if a != baseline]
    print(f"raw scores over {len(base)} benchmark(s): {', '.join(base)}\n")
    # `items` is not decoration. A pass still being generated is a complete-looking row
    # -- 1 repeat, a plausible score -- and the only thing distinguishing it from a
    # finished pass is that it covers 900 of 1730 items. Those 900 are the FAST ones,
    # so the score is biased, and reading it as final is the easiest mistake this table
    # can invite.
    print(f"{'method':<12} {'score':>7} {'std':>6} {'reps':>5} {'items':>6}   per-repeat")
    report = []
    coverage = {}
    for t in tags:
        for b in base:
            per = items(b, t)
            if not per:
                continue
            sc, k = per_repeat_scores(per, sorted(per))
            # std of the repeat scores, NOT the standard error. This table describes
            # what was measured -- how much a single evaluation of this method moves
            # run to run. The inference lives in the matrix below, where the interval
            # divides by sqrt(k) and accounts for both arms.
            # A single pass has no spread. Printed as "-" rather than nan, because nan
            # in a std column reads as a broken computation instead of "this arm was
            # run once on purpose".
            sd = statistics.stdev(sc) if k > 1 else None
            label = t if len(base) == 1 else f"{t}/{b}"
            sds = f"{sd:6.2f}" if sd is not None else f"{'-':>6}"
            n_items = len(per)
            coverage.setdefault(t, {})[b] = (n_items, expected_items(b))
            flag = "" if n_items >= expected_items(b) else "  <- partial"
            print(f"{label:<12} {statistics.mean(sc):>7.2f} {sds} {k:>5} {n_items:>6}   "
                  + " ".join(f"{x:.1f}" for x in sc) + flag)
            report.append({"method": t, "benchmark": b, "score": statistics.mean(sc),
                           "std_over_repeats": sd, "repeats": k,
                           "per_repeat": sc})

    # bf16-pd is a sanity check that disaggregation serves the architecture at all, not
    # a format to rank, and on the models where it is a single pass it contributed a row
    # and a column of dashes. Kept only when it IS the baseline (muse-glimmer, where it
    # has four passes and is the reference every quantized arm is measured against).
    # bf16-pd is kept when it is the baseline, or when it has repeats to say anything
    # with. The original rule dropped it unless it WAS the baseline, which was right
    # while it was a single dash-filled pass on two of three models -- but once bf16
    # became the baseline everywhere that rule silently discarded a four-repeat arm on
    # Muse. The question "what does disaggregation alone cost" is worth a row whenever
    # it has been measured properly.
    def _reps(t):
        return max((min((len(v) for v in items(b, t).values()), default=0)
                    for b in base), default=0)

    keep = [t for t in tags
            if t != "bf16-pd" or t == baseline or _reps(t) >= 2] or tags
    pair_rows = mcnemar_table(list(base), keep, show_mcnemar=args.mcnemar)
    report += pair_rows
    matrix_tags = args.matrix or keep
    significance_matrix(pair_rows, matrix_tags)
    # A clone has no results tree -- the generations are gitignored and only the export
    # is versioned -- so the directory this derived file lands in may not exist yet.
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "arm_comparison.json"), "w") as f:
        json.dump({"model": model, "baseline": baseline, "benches": list(base),
                   "rows": report}, f, indent=2)
    return {"baseline": baseline,
            "benches": list(base),
            "coverage": coverage,
            "signflip": {(r["a"], r["b"]): r
                         for r in report if r.get("test") == "signflip"},
            "scores": {r["method"]: r for r in report if "score" in r}}


if __name__ == "__main__":
    main()
