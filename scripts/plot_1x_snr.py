"""Block SNR distributions of whole-image 1x results (snr_1x_blocks.py output), several brains.

Left: one histogram panel per brain (fraction of blocks vs SNR, linear), the main threshold solid and
p >= 0.5 dotted. Right: all brains side by side (box 25-75%, dots = blocks, number = median).

Usage: uv run python scripts/plot_1x_snr.py <out.png> <threshold> <label>:<json> ...
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
SNR_MAX = 80
key = f"p{t:g}"


def vals(blocks, k):
    return np.array([b[k]["snr"] for b in blocks if b[k].get("snr") is not None
                     and b[k].get("noise_sd", 1) > 0], float)  # fmt: skip


data = []
for spec in specs:
    label, path = spec.split(":", 1)
    bl = json.loads(Path(path).read_text())["blocks"]
    data.append((label, vals(bl, key), vals(bl, "p0.5")))
A.apply_style("white")
colors = [A.BRAND["blue"] if "bright" in lbl else A.BRAND["orange"] for lbl, *_ in data]
fig = plt.figure(figsize=(17, 2.9 * len(data) + 1))
gs = fig.add_gridspec(len(data), 2, width_ratios=[1.5, 1])
bins = np.arange(0, SNR_MAX + 2, 2)
for r, ((label, v, v0), c) in enumerate(zip(data, colors, strict=True)):
    ax = fig.add_subplot(gs[r, 0])
    for x, lw, ls in ((v, 2.0, "-"), (v0, 1.0, ":")):
        h, e = np.histogram(np.clip(x, 0, SNR_MAX), bins)
        ax.stairs(h / len(x), e, color=c, lw=lw, ls=ls)
    ax.axvline(np.median(v), color=c, lw=1, ls="--")
    ax.set_xlim(0, SNR_MAX)
    A.style_ax(ax, title=f"{label} · median {np.median(v):.1f} at p >= {t:g} ({np.median(v0):.1f} at 0.5) · n={len(v)}",
               ylabel="fraction of blocks", title_size=12)  # fmt: skip
    if r == len(data) - 1:
        ax.set_xlabel("block SNR", fontsize=11)
ax = fig.add_subplot(gs[:, 1])
groups = [np.clip(v, 0, SNR_MAX) for _, v, _ in data]
bp = ax.boxplot(groups, positions=range(len(groups)), widths=0.55, showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], colors, strict=True):
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
for med in bp["medians"]:
    med.set_color(A.C["ink"])
rng = np.random.default_rng(0)
for i, (g, c) in enumerate(zip(groups, colors, strict=True)):
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=5, color=c, alpha=0.4, lw=0)
    A.value_label(ax, i, np.percentile(g, 75), f"{np.median(g):.1f}", size=13)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([lbl.replace(" (", "\n(") for lbl, *_ in data], fontsize=10)
ax.set_ylim(0, SNR_MAX * 1.05)
A.style_ax(ax, title=f"all 1x brains, p >= {t:g}", ylabel="block SNR")
fig.suptitle(A.safe(f"same-day 1x / block SNR at p >= {t:g}"), x=0.01, ha="left", fontfamily=A.HEAD,
             fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.004, A.safe(
    "binned 1x scans (tile_000000, 20 x 15 x 15 um), 1x fine-tuned TRAILMAP · 300 random tissue blocks "
    "of 1.28 x 1.44 x 1.44 mm per brain (>= 90% inside the level-4 tissue mask) · SNR = (mean detected "
    "- clipped background) / background SD · solid: threshold, dotted: p >= 0.5, dashed: median · "
    f"SNR > {SNR_MAX} drawn at the top / last bin · blue: bright group, orange: dim"), fontsize=9,
    color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
A.savefig(fig, out, dpi=100)
print("saved", out)
