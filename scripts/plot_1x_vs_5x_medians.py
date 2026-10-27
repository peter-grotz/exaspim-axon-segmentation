"""One point per brain: median block SNR at 1x vs at 5x (same threshold), with the middle 50%.

Usage: uv run python scripts/plot_1x_vs_5x_medians.py <out.png> <threshold> <subject>:<group>:<1x json>:<5x rescore dir> ...
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
key = f"p{t:g}"


def vals(blocks):
    return np.array([b[key]["snr"] for b in blocks if b[key].get("snr") is not None
                     and b[key].get("noise_sd", 1) > 0], float)  # fmt: skip


col = {"bright": A.BRAND["blue"], "dim": A.BRAND["orange"]}
A.apply_style("white")
fig, ax = plt.subplots(figsize=(8.5, 7.5))
for spec in specs:
    sid, group, one, five = spec.split(":")
    v1 = vals(json.loads(Path(one).read_text())["blocks"])
    v5 = vals(
        [
            b
            for f in sorted(Path(five).glob("*/snr_rescore.json"))
            for b in json.loads(f.read_text())["blocks"]
        ]
    )
    c = col.get(group, A.BRAND["gray"])
    q1, q5 = np.percentile(v1, [25, 50, 75]), np.percentile(v5, [25, 50, 75])
    ax.plot([q1[0], q1[2]], [q5[1]] * 2, color=c, lw=1.5, alpha=0.5)
    ax.plot([q1[1]] * 2, [q5[0], q5[2]], color=c, lw=1.5, alpha=0.5)
    ax.scatter(q1[1], q5[1], s=70, color=c, zorder=3, edgecolor="white")
    ax.annotate(sid, (q1[1], q5[1]), xytext=(6, 5), textcoords="offset points", fontsize=10)
lim = ax.get_xlim()[1], ax.get_ylim()[1]
for k in (1, 2, 4):
    xs = np.linspace(0, lim[0], 10)
    ax.plot(xs, k * xs, color=A.C["grid"], lw=1, ls="--", zorder=0)
    ax.text(lim[1] / k * 0.97 if lim[1] / k < lim[0] else lim[0] * 0.97, min(lim[1], k * lim[0]) * 0.97,
            f"{k}x", color=A.C["muted"], fontsize=9, ha="right", va="top")  # fmt: skip
ax.set_xlim(0, lim[0])
ax.set_ylim(0, lim[1])
A.style_ax(ax, title=f"median block SNR, 1x vs 5x · p >= {t:g}", xlabel="1x median SNR",
           ylabel="5x median SNR", grid_axis="both")  # fmt: skip
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=col[g], label=g) for g in col]
          + [plt.Line2D([], [], marker="o", ls="", color=A.BRAND["gray"], label="not assigned")],
          frameon=False, loc="upper left")  # fmt: skip
fig.text(0.01, 0.005, A.safe("dot: median, lines: 25-75% of blocks · dashed: 5x = 1x, 2x, 4x · same-day "
                             "1x (1x fine-tuned TRAILMAP, 300 tissue blocks) and 5x (v2, snr-tiles blocks)"),
         fontsize=8, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 1))
A.savefig(fig, out, dpi=110)
print("saved", out)
