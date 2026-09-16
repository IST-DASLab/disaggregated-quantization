# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "pillow"]
# ///
"""Offloaded disaggregated prefill, as a measurement-scaled pipeline schematic.

    uv run notebooks/schematics/fig_odp_timeline.py

One row per transformer block, time across, and per row the three states a block passes
through: read off the drive, waiting while the GPU is still on its predecessor, computed.
The first block is read inside the timed region, so compute starts after that cold read.

Each subsequent row therefore BEGINS at its read. The scheduler does make a block wait for a free slot
before that read, but drawing it would put a bar for block n+1 alongside the compute of
block n that it is waiting on -- two rows claiming the same time for the same reason. The
wait that belongs to a row is the one after its own read. The regime is the long-context one the section argues about, where
a block's compute is several times its read.

What the trace has to carry is that the COMPUTE bars never break: each starts where the one
below it ends, so the drive stays hidden behind compute, including the read of block 2
during block 1's compute. What waits is the drive, not the GPU -- the stall in front of
every read from block 3 on.

BOTH AXES ARE BROKEN because the real figures are lopsided: 36 blocks over 2 s, where a
read is 13 ms. Drawn whole, a read is 0.6% of the width and a row 3% of the height, and the
argument disappears into the ink. Two things keep the break readable rather than merely
compact: the halves are given the SAME ms per inch and the same rows per inch, so the two
staircases have the same slope and can be compared by eye; and each cut is a torn edge
rather than a corner mark, laid down the whole length of the break so that what crosses it
reads as continuing past it. The y cut falls between two rows, never through a bar. The two empty quadrants are honest rather than wasted -- the
last blocks really have not started while the first ones run.

Durations are derived from aggregate measurements, not a per-block profiler trace;
see `measured`. Colour is the RESOURCE here, not a weight
format: this figure has no phase or format axis, so red and blue are the GPU and the drive
rather than what they mean in the linear-layer schematics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import BLUE, INK, RED, plt, save, snapshot, use_style

OUT = Path(__file__).resolve().parents[1] / "figures" / "odp_timeline.pdf"
PREVIEW = Path(__file__).resolve().parent / "preview_odp.png"
KERNELS = Path(__file__).resolve().parents[2] / "qad" / "kernels" / "prefill"

MODEL, QUANT, SEQ = "Qwen/Qwen3-4B", "nvfp4", 16384
SLOTS = 2                            # the runner's double buffer
SHOWN = 4                            # blocks kept at each end of the y break
SPAN = 250.0                         # ms shown on each side of the x break

C_READ, C_COMPUTE, C_STALL = BLUE, RED, "#e2e2e2"
AXIS_FS, LEGEND_FS = 12, 11          # this figure is authored larger than the schematics


def measured():
    """Derive equal per-block intervals from measured aggregate latencies.

    Read time is the checkpoint-only loading measurement divided by block count.
    Compute intervals share the remaining zero-ssd duration after its first read,
    so the schematic ends at the measured latency in this compute-bound regime.
    These intervals include pipeline overhead; they are not isolated compute timings.
    Restoration is included in the measured total. Its bar is illustrative, with
    two blocks' worth of loading overlapped with the final compute interval;
    these aggregate measurements do not record its individual start/end times.
    """
    import csv

    def rows(name):
        with open(KERNELS / name, newline="") as f:
            return [r for r in csv.DictReader(f)
                    if r["model"] == MODEL and r["quant"] == QUANT]

    # load_floor.py writes load_floor_ssd.csv now; older runs wrote load_floor.csv.
    floor_name = next((n for n in ("load_floor_ssd.csv", "load_floor.csv")
                       if (KERNELS / n).exists()), "load_floor_ssd.csv")
    floor = rows(floor_name)[0]
    n = int(floor["n_blocks"])
    lat = {r["mode"]: float(r["latency_ms"]) for r in rows("offload_prefill.csv")
           if int(r["seq_len"]) == SEQ}
    read = float(floor["load_ms"]) / n
    compute = (lat["zero-ssd"] - read) / n
    if read <= 0 or compute < read:
        raise ValueError("This equal-block schematic requires compute-bound measurements")
    return read, compute, n, lat["resident"], lat["zero-ssd"]


T_READ, T_COMPUTE, N_BLOCKS, MS_RESIDENT, MS_OFFLOAD = measured()


def schedule(n=N_BLOCKS, slots=SLOTS, read=T_READ, compute=T_COMPUTE):
    """Per block: (slot_free_from, read_start, read_end, compute_start, compute_end).

    The dependencies are offload_forward.py's, SSDOffloadRunner.run: one fetch thread, so a
    read waits for the previous read; `slots` buffers, so it also waits for the block
    `slots` earlier to be computed out of the one it will land in; one compute stream, so a
    block computes once its weights land AND its predecessor is done. The first block
    must also be read before compute can begin.
    """
    rows, read_end, comp_end = [], [], []
    for i in range(n):
        ready = read_end[i - 1] if i else 0.0
        start = max(ready, comp_end[i - slots] if i >= slots else 0.0)
        end = start + read
        rows.append([ready, start, end])
        read_end.append(end)
        cs = max(read_end[i], comp_end[i - 1] if i else 0.0)
        rows[i] += [cs, cs + compute]
        comp_end.append(cs + compute)
    return rows


ROWS = schedule()


def draw(ax, blocks):
    for i in blocks:
        _, r0, r1, c0, c1 = ROWS[i]                  # the pre-read wait is not drawn
        for t0, t1, colour in ((r0, r1, C_READ), (r1, c0, C_STALL),
                               (c0, c1, C_COMPUTE)):
            if t1 > t0:
                ax.barh(i + 1, t1 - t0, left=t0, height=0.62, color=colour, linewidth=0,
                        zorder=3)


def draw_restore(ax):
    """Illustrate restoring two slots; this is not a separately measured duration."""
    start, end = ROWS[-1][3:5]
    duration = SLOTS * T_READ
    if start + duration > end:
        raise ValueError("Illustrated restoration does not fit under final-layer compute")
    bar = ax.barh(N_BLOCKS + 1, duration, left=start, height=0.62,
                  color=C_READ, linewidth=0, zorder=3)[0]
    bar.set_gid("decode-restore")
    # The payload is staged, but the final slot cannot be overwritten until compute ends.
    wait = ax.barh(N_BLOCKS + 1, end - start - duration, left=start + duration,
                   height=0.62, color=C_STALL, linewidth=0, zorder=3)[0]
    wait.set_gid("decode-restore-wait")
    ax.annotate("Restore decode carve-out", xy=(start, N_BLOCKS + 1),
                xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                fontsize=AXIS_FS - 3, color=INK)


def wavy_cut(ax, side, amp=0.012, waves=9):
    """A wavy rule along one side of the axes: the textbook mark for a cut axis.

    Drawn over the bars rather than through a gap in them, so a state that crosses the
    break reads as continuing past it instead of ending there. Runs the whole length of
    the cut, which corner slashes do not.
    """
    import numpy as np
    t = np.linspace(0, 1, 300)
    w = (1.0 if side in ("top", "right") else 0.0) + amp * np.sin(2 * np.pi * waves * t)
    xy = (t, w) if side in ("top", "bottom") else (w, t)
    ax.plot(*xy, color=INK, lw=0.8, transform=ax.transAxes, clip_on=False, zorder=6)


def build():
    use_style()
    lo = list(range(SHOWN))                          # first blocks, at the head of the run
    hi = list(range(N_BLOCKS - SHOWN, N_BLOCKS))     # last blocks, at the tail
    end = ROWS[-1][4]
    # Half of \\linewidth. Point sizes are absolute, so everything -- ticks, labels, key --
    # comes out proportionally larger than it did at full width, which is the point.
    fig, axes = plt.subplots(2, 2, figsize=(6, 3.6), sharex="col", sharey="row",
                             gridspec_kw=dict(wspace=0.07, hspace=0.16))
    (tl, tr), (bl, br) = axes
    draw(bl, lo)
    draw(tr, hi)
    draw_restore(tr)
    bl.annotate("Carve-out for and\nload first block", xy=(ROWS[0][4], 1),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                fontsize=AXIS_FS - 3, color=INK)

    for ax in axes.ravel():
        ax.tick_params(labelsize=AXIS_FS - 2, length=3, pad=2)
        # Half the width takes half the ticks with it: every 50 ms crowds into a smear.
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))
        ax.grid(True, axis="x", color="#ececec", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in (tl, tr):
        ax.set_ylim(hi[0] + 0.5, hi[-1] + 2.6)
        ax.set_yticks([i + 1 for i in hi])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        wavy_cut(ax, "bottom")
    for ax in (bl, br):
        ax.set_ylim(lo[0] + 0.4, lo[-1] + 1.5)
        ax.set_yticks([i + 1 for i in lo])
        wavy_cut(ax, "top")
    for ax in (tl, bl):
        ax.set_xlim(-10, SPAN - 10)
        wavy_cut(ax, "right")
    for ax in (tr, br):
        ax.set_xlim(end + 10 - SPAN, end + 10)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        wavy_cut(ax, "left")

    # supxlabel/supylabel place themselves in FIGURE coords and centre on the FIGURE, not
    # on the axes, so both are given the axes' own centre; otherwise the left margin throws
    # each of them off by half of it.
    left, right, bottom, top = 0.115, 0.995, 0.14, 0.98
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=0.07, hspace=0.16)
    fig.supxlabel("Time, ms", fontsize=AXIS_FS, color=INK, x=(left + right) / 2,
                  y=bottom - 0.125)
    fig.supylabel("Transformer block", fontsize=AXIS_FS, color=INK, x=0.035,
                  y=(bottom + top) / 2)

    # Back in the quadrant that is empty by construction -- the first blocks, long after
    # they finished -- so the key costs the trace nothing. Sized off the axis labels rather
    # than off style.py: this figure is authored larger than the schematics, and a key two
    # sizes under its own axis labels reads as an afterthought.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=t)
               for c, t in ((C_READ, "Read from SSD"), (C_COMPUTE, "Compute"),
                            (C_STALL, "Waiting for compute"))]
    leg = br.legend(handles=handles, title="Block state", loc="center", frameon=False,
                    fontsize=LEGEND_FS, title_fontsize=LEGEND_FS, alignment="left",
                    labelspacing=0.45, handlelength=1.7, handleheight=1.15,
                    handletextpad=0.6, borderpad=0.2)
    leg.get_title().set_fontweight("bold")
    return fig


if __name__ == "__main__":
    fig = build()
    save(fig, OUT)
    from PIL import Image
    Image.open(snapshot(fig)).convert("RGB").save(PREVIEW)
    print(f"wrote {PREVIEW}  ({N_BLOCKS} blocks, read {T_READ:.1f} ms, compute "
          f"{T_COMPUTE:.1f} ms, {MS_RESIDENT:.0f} ms resident "
          f"vs {MS_OFFLOAD:.0f} ms offloaded (zero-ssd))")
