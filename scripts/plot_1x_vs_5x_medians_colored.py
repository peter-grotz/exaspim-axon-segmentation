"""1x vs 5x median block SNR per brain, twice: coloured by genotype, and by 5x acquisition date.

Genotype from the ExM Dataset Summary (results/exm_dataset_summary.json); date from the 5x
acquisition name in results/run_snrtiles_all.json.

Usage: uv run python scripts/plot_1x_vs_5x_medians_colored.py <out.png> <threshold> <subject>:<1x json>:<5x rescore dir> ...
"""

import json
import re
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
subs = json.loads(Path("results/run_snrtiles_all.json").read_text())
sheet = {str(d.get("Sample")).split("_")[0]: d for d in json.loads(Path("results/exm_dataset_summary.json").read_text())}


def vals(blocks):
    return np.array([b[key]["snr"] for b in blocks if b[key].get("snr") is not None
                     and b[key].get("noise_sd", 1) > 0], float)  # fmt: skip


rows = []
for spec in specs:
    sid, one, five = spec.split(":")
    v1 = vals(json.loads(Path(one).read_text())["blocks"])
    v5 = vals([b for f in sorted(Path(five).glob("*/snr_rescore.json")) for b in json.loads(f.read_text())["blocks"]])
    g = (sheet.get(sid, {}).get("Genotype") or "unknown").replace("C57BL/6J + ", "").replace(" / ", "/").strip()
    date = np.datetime64(re.search(r"(\d{4}-\d{2}-\d{2})", subs[sid]["root"]).group(1))
    rows.append({"sid": sid, "q1": np.percentile(v1, [25, 50, 75]), "q5": np.percentile(v5, [25, 50, 75]),
                 "genotype": g, "date": date})  # fmt: skip

genos = sorted({r["genotype"] for r in rows}, key=lambda g: (-sum(r["genotype"] == g for r in rows), g))
gcol = {g: A.CATEGORICAL[i % len(A.CATEGORICAL)] for i, g in enumerate(genos)}
d = np.array([r["date"].astype(float) for r in rows])
dnorm = matplotlib.colors.Normalize(d.min(), d.max())
cmap = matplotlib.colormaps["viridis"]
xmax = max(r["q1"][1] for r in rows) * 1.15
ymax = max(r["q5"][2] for r in rows) * 1.05

A.apply_style("white")
fig, axs = plt.subplots(1, 2, figsize=(19, 8.5))
for ax, mode in zip(axs, ("genotype", "date"), strict=True):
    for k in (1, 2, 4):
        xs = np.linspace(0, xmax, 10)
        ax.plot(xs, k * xs, color=A.C["grid"], lw=1, ls="--", zorder=0)
        ax.text(min(xmax, ymax / k) * 0.98, min(ymax, k * xmax) * 0.98, f"{k}x", color=A.C["muted"],
                fontsize=9, ha="right", va="top")  # fmt: skip
    for r, dv in zip(rows, d, strict=True):
        c = gcol[r["genotype"]] if mode == "genotype" else cmap(dnorm(dv))
        q1, q5 = r["q1"], r["q5"]
        ax.plot([q1[1]] * 2, [q5[0], q5[2]], color=c, lw=1.8, alpha=0.55)
        ax.scatter(q1[1], q5[1], s=90, color=c, zorder=3, edgecolor="white")
        ax.annotate(r["sid"], (q1[1], q5[1]), xytext=(6, 5), textcoords="offset points", fontsize=10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    A.style_ax(ax, title=f"coloured by {'genotype' if mode == 'genotype' else '5x acquisition date'} · p >= {t:g}",
               xlabel="1x median SNR", ylabel="5x median SNR", grid_axis="both", title_size=14)  # fmt: skip
axs[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=9, color=gcol[g],
                                  label=f"{g} (n={sum(r['genotype'] == g for r in rows)})") for g in genos],
              frameon=False, loc="upper right", fontsize=9)  # fmt: skip
sm = matplotlib.cm.ScalarMappable(norm=dnorm, cmap=cmap)
cb = fig.colorbar(sm, ax=axs[1], fraction=0.04, pad=0.02)
ticks = np.linspace(d.min(), d.max(), 5)
cb.set_ticks(ticks)
cb.set_ticklabels([str(np.datetime64(int(x), "D")) for x in ticks])
cb.set_label("5x acquisition date (oldest -> newest)")
cb.outline.set_visible(False)
fig.suptitle(A.safe("1x vs 5x median block SNR/"), x=0.01, ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=20)
fig.text(0.01, 0.01, A.safe(f"dot: median, vertical line: 25-75% of 5x blocks · dashed: 5x = 1x, 2x, 4x · {len(rows)} brains with a "
                            "same-day 1x scan · 851474: 5x blocks inside its intensity outline (1.03) · genotype: "
                            "ExM Dataset Summary"), fontsize=9, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
A.savefig(fig, out, dpi=100)
print("saved", out)
