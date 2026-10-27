"""Block SNR, same-day 1x vs 5x, at one probability threshold (with p >= 0.5 for reference).

1x: snr_1x_blocks.py results (random tissue blocks of the whole 1x image). 5x: rescore-tiles results
(the snr-tiles blocks). Blocks are the same physical size (1.28 x 1.44 x 1.44 mm) in both; the
scoring's voxel-based settings are in each scan's own voxels. Left: SNR histograms per brain (1x
and 5x overlaid, threshold solid, p >= 0.5 thin); right: all four distributions side by side.

Usage: uv run python scripts/plot_1x_vs_5x_snr.py <out.png> <threshold> <subject>:<1x json>:<5x rescore dir> ...
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

out, t, specs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
key, ref = f"p{t:g}", "p0.5"
SNR_MAX = 150


def vals(blocks, k):
    v = [b[k]["snr"] for b in blocks if b[k].get("snr") is not None and b[k].get("noise_sd", 1) > 0]
    return np.array(v, float)


data = []
for spec in specs:
    sid, one, five = spec.split(":")
    b1 = json.loads(Path(one).read_text())["blocks"]
    b5 = [
        b
        for f in sorted(Path(five).glob("*/snr_rescore.json"))
        for b in json.loads(f.read_text())["blocks"]
    ]
    data.append((sid, {"1x": (vals(b1, key), vals(b1, ref)), "5x": (vals(b5, key), vals(b5, ref))}))
A.apply_style("white")
col = {"1x": A.BRAND["teal"], "5x": A.BRAND["violet"]}
fig = plt.figure(figsize=(18, 4.6 * len(data)))
gs = fig.add_gridspec(len(data), 2, width_ratios=[1.6, 1])
bins = np.arange(0, SNR_MAX + 5, 5)
for r, (sid, d) in enumerate(data):
    ax = fig.add_subplot(gs[r, 0])
    for scope in ("1x", "5x"):
        v, v0 = d[scope]
        for x, lw, ls, a_ in ((v, 2.0, "-", 1.0), (v0, 1.0, ":", 0.7)):
            if len(x):
                h, e = np.histogram(np.clip(x, 0, SNR_MAX), bins)
                ax.stairs(h / len(x), e, color=col[scope], lw=lw, ls=ls, alpha=a_)
        ax.axvline(np.median(v), color=col[scope], lw=1, ls="--")
    ax.set_xlim(0, SNR_MAX)
    A.style_ax(
        ax, title=f"{sid} block SNR, p >= {t:g}", xlabel="block SNR", ylabel="fraction of blocks"
    )
    ax.legend(handles=[plt.Line2D([], [], color=col[s], lw=2,
                                  label=A.safe(f"{s} p >= {t:g}: median {np.median(d[s][0]):.1f} (n={len(d[s][0])}) · "
                                               f"p >= 0.5: {np.median(d[s][1]):.1f}")) for s in ("1x", "5x")],
              frameon=False, fontsize=10, loc="upper right")  # fmt: skip
ax = fig.add_subplot(gs[:, 1])
groups, labels, colors = [], [], []
for sid, d in data:
    for scope in ("1x", "5x"):
        groups.append(np.clip(d[scope][0], 0, SNR_MAX))
        labels.append(f"{sid}\n{scope}")
        colors.append(col[scope])
pos = np.arange(len(groups))
bp = ax.boxplot(groups, positions=pos, widths=0.55, showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], colors, strict=True):
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
for med in bp["medians"]:
    med.set_color(A.C["ink"])
rng = np.random.default_rng(0)
for i, (g, c) in enumerate(zip(groups, colors, strict=True)):
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=4, color=c, alpha=0.35, lw=0)
    A.value_label(ax, i, np.percentile(g, 75), f"{np.median(g):.1f}", size=12)
ax.set_xticks(pos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, SNR_MAX * 1.05)
A.style_ax(ax, title=f"all {len(data)} brains, p >= {t:g}", ylabel="block SNR")
fig.suptitle(A.safe(f"same-day 1x vs 5x / block SNR at p >= {t:g}"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe(
    "blocks 1.28 x 1.44 x 1.44 mm in both scopes · 1x: 300 random tissue blocks of the whole image "
    "(tile_000000, 1x fine-tuned TRAILMAP, 20 x 15 x 15 um voxels) · 5x: the snr-tiles blocks (v2 "
    "TRAILMAP, level 2, 4 x 3 x 3 um) rescored from the saved probability maps · same scoring code; its "
    "guard (2 voxels), shell (6) and outside cut are in each scan's own voxels · solid: threshold, dotted: "
    f"p >= 0.5 · dashed: medians · SNR > {SNR_MAX} in the last bin / at the top"), fontsize=9,
    color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
A.savefig(fig, out, dpi=100)
print("saved", out)
