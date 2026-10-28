"""Which voxels the SNR background drops as "outside the tissue" (the block's dimmest outside_pct %).

Per brain, three 5x blocks (25th, 50th and 75th percentile of block SNR at the threshold) are read
from S3 (level 2) and the cut is applied as in snr._score: outside = 10th percentile of the block,
background candidates = raw > outside. Because raw values are quantized, ties at the cut can drop
more than 10%. Figure: one row per brain with the middle z-slice of each block, dropped voxels in
colour, and the block histogram with the cut. The background shift is approximated without the
detection guard (axon voxels are < 1% of a block and the 3-SD clipping removes them): sigma-clipped
mean and SD of all voxels above the cut vs of all voxels.

Usage: uv run python scripts/plot_dim10_exclusion.py <out.png> <threshold> <subject>:<5x rescore dir> ...
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from exaspim_axon_segmentation.qc.detection_qc import _clipped
from exaspim_axon_segmentation.snr import _read, _sub, open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

out, t, specs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
key = f"p{t:g}"
rows, stats = [], []
for spec in specs:
    sid, five = spec.split(":")
    bl = [(b, json.loads(f.read_text())["url"]) for f in sorted(Path(five).glob("*/snr_rescore.json"))
          for b in json.loads(f.read_text())["blocks"] if b[key].get("snr") is not None]  # fmt: skip
    bl.sort(key=lambda x: x[0][key]["snr"])
    picks = []
    for q in (0.25, 0.5, 0.75):
        b, url = bl[int(q * (len(bl) - 1))]
        arr, lead, _, _ = open_ome(url, 2)
        lo = np.array(b["lo_vox"])
        raw = _read(arr, lead, lo, lo + np.array(b["shape"]))
        cut = np.percentile(_sub(raw), 10)
        drop = raw <= cut
        b_all, sd_all = _clipped(raw.ravel())
        b_cut, sd_cut = _clipped(raw[~drop])
        s = {"sid": sid, "q": q, "snr": b[key]["snr"], "cut": float(cut), "dropped_pct": 100 * float(drop.mean()),
             "bg_cut": b_cut, "sd_cut": sd_cut, "bg_all": b_all, "sd_all": sd_all,
             "median": float(np.median(_sub(raw)))}  # fmt: skip
        stats.append(s)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
        mid = raw.shape[0] // 2
        picks.append((raw[mid], drop[mid], np.bincount(np.clip(raw[::2, ::2, ::2], 0, 65535).astype(np.int64).ravel()), s))
        del raw, drop
    rows.append((sid, picks))
Path(out).with_suffix(".json").write_text(json.dumps(stats, indent=1))

A.apply_style("white")
fig, axs = plt.subplots(len(rows), 4, figsize=(24, 4.6 * len(rows)), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
for (sid, picks), ax_row in zip(rows, np.atleast_2d(axs), strict=True):
    for (img, drop, _, s), ax in zip(picks, ax_row[:3], strict=True):
        lo, hi = np.percentile(img, [1, 99.5])
        ax.imshow(np.clip((img - lo) / max(hi - lo, 1), 0, 1), cmap="gray", interpolation="nearest")
        ov = np.zeros((*img.shape, 4))
        ov[drop] = matplotlib.colors.to_rgba(A.BRAND["maroon"], 0.55)
        ax.imshow(ov, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        A.style_ax(ax, grid_axis=None, title=f"{sid} · SNR {s['snr']:.0f} ({int(s['q'] * 100)}th pct block) · "
                   f"dropped {s['dropped_pct']:.0f}%", title_size=11)  # fmt: skip
    ax = ax_row[3]
    for (_, _, h, s), c in zip(picks, (A.BRAND["blue"], A.BRAND["violet"], A.BRAND["teal"]), strict=True):
        x = np.nonzero(h)[0]
        hi = int(np.percentile(np.repeat(x, np.minimum(h[x], 1000)), 99.5))
        ax.plot(np.arange(hi), h[:hi] / h.sum(), color=c, lw=1.4, label=f"{int(s['q'] * 100)}th: cut {s['cut']:.0f}")
        ax.axvline(s["cut"], color=c, ls="--", lw=1)
    ax.set_xlabel("raw intensity (counts)")
    ax.set_ylabel("fraction of voxels")
    ax.legend(frameon=False, fontsize=9)
    A.style_ax(ax, title=f"{sid} · block histograms, dashed: dimmest-10% cut", title_size=11)
fig.suptitle(A.safe("voxels dropped from the SNR background as the block's dimmest 10% (red)/"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.003, A.safe(f"5x level 2, middle z-slice of each block · blocks at the 25th, 50th and 75th percentile "
                             f"of block SNR at p >= {t:g} · dropped = raw <= the block's 10th percentile (ties at the "
                             "cut included, as in the scoring)"), fontsize=10, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.01, 1, 0.985))
A.savefig(fig, out, dpi=70)
print("saved", out)
