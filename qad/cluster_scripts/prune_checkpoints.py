"""Reclaim exported-checkpoint space without breaking restart or result aggregation.

    python cluster_scripts/prune_checkpoints.py              # report only, deletes nothing
    python cluster_scripts/prune_checkpoints.py --apply      # actually delete
    python cluster_scripts/prune_checkpoints.py --grid-only  # aggressive: grid steps only

WHAT IS SAFE TO DELETE, AND WHY
-------------------------------
Two different things live under a checkpoint tag and they are NOT interchangeable:

  state/    resumable training state (fp32 master + optimizer moments + RNG).
            THIS IS WHAT RESTART USES. Never touched here. It is already minimal --
            qad.py defaults to --keep-last 1, so only the newest survives.
  weights/  exported HF checkpoints. This is what evals SERVE, and what grew to 20 TB
            because save_weights() used to fire on the --val-every cadence (25 steps),
            producing 99 per run when the sweep reads 10.

Deleting an exported checkpoint does NOT lose a result: results are JSON under
results/ and are stored separately (and committed). It only removes the ability
to RE-run an eval at that step.

So the default policy keeps a step if EITHER:
  * it is on the eval grid (a multiple of --grid, or step 0), or
  * some eval result already exists for that step anywhere in results/, or
  * it is at or past --keep-after AND on the --tail-grid (125) -- the late steps are
    what the tail-averaged recovery bars read, but at 125 resolution, not every step.
    Every-25 past 2000 wrote ~19 checkpoints per run that nothing plots.
The second clause is the important one: 300 results sit at off-grid steps (25..225,
300, 400, 600, 700) from the finer-grained early sweeps. Keeping those steps means
every eval that has ever been run can still be reproduced.

--grid-only drops the second clause. It reclaims a few TB more and forecloses
re-running those finer-grained curves. Use deliberately.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def du(path: str) -> int:
    try:
        return int(subprocess.run(["du", "-sb", path], capture_output=True, text=True)
                   .stdout.split()[0])
    except Exception:
        return 0


def evaluated_steps() -> set:
    """Every step that already has a result somewhere, so it stays reproducible."""
    out = set()
    for f in glob.glob(os.path.join(ROOT, "results", "**", "step_*.json"),
                       recursive=True):
        m = re.search(r"step_(\d+)", os.path.basename(f))
        if m:
            out.add(int(m.group(1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=250, help="eval step grid to preserve")
    ap.add_argument("--grid-only", action="store_true",
                    help="do NOT preserve off-grid steps that already have results")
    ap.add_argument("--keep-after", type=int, default=0,
                    help="keep exported steps >= N at --tail-grid resolution, on-grid or "
                         "not. The late steps are where the curves are read (the "
                         "tail-averaged recovery bars use the last 5).")
    ap.add_argument("--tail-grid", type=int, default=125,
                    help="past --keep-after, keep every multiple of this rather than "
                         "EVERY step. Mirrors the writer: qad.py exports past "
                         "--export-dense-after at the --val-every cadence, which is 125. "
                         "Set 1 to keep everything (the old behaviour).")
    ap.add_argument("--apply", action="store_true", help="delete (default: report only)")
    ap.add_argument("--filter", default="", help="only tags containing this substring")
    args = ap.parse_args()

    keep_evaluated = set() if args.grid_only else evaluated_steps()
    ev = "no" if args.grid_only else f"yes ({len(keep_evaluated)} steps)"
    ka = f"keep-all >= {args.keep_after}" if args.keep_after else "keep-after off"
    print(f"grid={args.grid}  preserve-evaluated={ev}  {ka}")

    per_tag = defaultdict(lambda: [0, 0, 0, 0])   # keep_b, drop_b, keep_n, drop_n
    victims = []
    for w in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*", "weights"))):
        tag = os.path.basename(os.path.dirname(w))
        if args.filter and args.filter not in tag:
            continue
        for d in sorted(os.listdir(w)):
            m = re.match(r"step_(\d+)$", d)
            if not m:
                continue
            step, path = int(m.group(1)), os.path.join(w, d)
            keep = ((step % args.grid == 0) or step == 0
                    or (step in keep_evaluated)
                    or (args.keep_after and step >= args.keep_after
                        and step % args.tail_grid == 0))
            n = du(path)
            i = 0 if keep else 1
            per_tag[tag][i] += n
            per_tag[tag][i + 2] += 1
            if not keep:
                victims.append(path)

    kb = sum(v[0] for v in per_tag.values()); db = sum(v[1] for v in per_tag.values())
    kn = sum(v[2] for v in per_tag.values()); dn = sum(v[3] for v in per_tag.values())
    print(f"\n{'tag':52} {'keep':>10} {'drop':>10}")
    for tag, (a, b, an, bn) in sorted(per_tag.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {tag[:50]:50} {a/1e12:8.2f}TB {b/1e12:8.2f}TB")
    print(f"\n  KEEP {kb/1e12:6.2f} TB ({kn} checkpoints)")
    print(f"  DROP {db/1e12:6.2f} TB ({dn} checkpoints)")
    print("  state/ is never touched -- that is what --resume reads.")

    if not args.apply:
        print(f"\nreport only; re-run with --apply to delete {dn} checkpoint dirs")
        return
    if not victims:
        print("\nnothing to delete")
        return
    for i, p in enumerate(victims, 1):
        shutil.rmtree(p, ignore_errors=True)
        if i % 200 == 0:
            print(f"  deleted {i}/{len(victims)}", flush=True)
    print(f"\ndeleted {len(victims)} checkpoint dirs, reclaimed ~{db/1e12:.2f} TB")


if __name__ == "__main__":
    main()
