"""SNR distribution per tile and per sample from snr-tiles results (Allen plotting style).

Usage: uv run python scripts/plot_snrtiles.py <results/snrtiles> <out prefix>
Each sample folder holds snr_tiles/<tile>/snr_blocks.json. Samples whose tiles were selected
without a brain mask are marked with "*".
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from allen_style import BRAND, C, apply_style, safe, savefig, style_ax  # noqa: E402

root, out = Path(sys.argv[1]), sys.argv[2]
rows = []
for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
    for f in sorted(sdir.glob("snr_tiles/*/snr_blocks.json")):
        d = json.loads(f.read_text())
        for b in d.get("blocks", []):
            n = b.get("native", {})
            if not isinstance(n, dict) or n.get("snr") is None:
                continue
            rows.append({"sample": sdir.name, "tile": int(f.parent.name.split("_")[1]),
                         "selection": d.get("selection", "?"), "snr": n["snr"],
                         "local_snr": n.get("local_snr"), "axon_fraction": n["axon_fraction"]})  # fmt: skip
samples = sorted({r["sample"] for r in rows})
# "*": the sample had no brain mask (most of its blocks unmasked); a few edge tiles without a mask
# in a masked sample do not mark it
star = {s: np.mean([r["selection"].startswith("unmasked") for r in rows if r["sample"] == s]) > 0.5
        for s in samples}  # fmt: skip
label = {s: s + ("*" if star[s] else "") for s in samples}
summary = {}
for s in samples:
    v = np.array([r["snr"] for r in rows if r["sample"] == s])
    lv = np.array([r["local_snr"] for r in rows if r["sample"] == s and r["local_snr"] is not None])
    tiles = sorted({r["tile"] for r in rows if r["sample"] == s})
    summary[label[s]] = {"blocks": len(v), "tiles": len(tiles), "snr_median": float(np.median(v)),
                         "snr_p10": float(np.percentile(v, 10)), "snr_p90": float(np.percentile(v, 90)),
                         "local_snr_median": float(np.median(lv)) if len(lv) else None,
                         "per_tile_median": {t: float(np.median([r["snr"] for r in rows if r["sample"] == s
                                                                 and r["tile"] == t])) for t in tiles}}  # fmt: skip
Path(out + ".json").write_text(json.dumps(summary, indent=1))

apply_style("white")
order = sorted(samples, key=lambda s: summary[label[s]]["snr_median"])
fig, ax = plt.subplots(2, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1.1, 1]})
data = [[r["snr"] for r in rows if r["sample"] == s] for s in order]
bp = ax[0].boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
for patch, s in zip(bp["boxes"], order, strict=True):
    patch.set_facecolor(BRAND["gray1"] if star[s] else BRAND["blue"])
    patch.set_alpha(0.85)
for k in ("medians",):
    for m in bp[k]:
        m.set_color("black")
for i, s in enumerate(order, 1):
    ax[0].text(i, summary[label[s]]["snr_p90"] * 1.04, f"{summary[label[s]]['snr_median']:.0f}",
               ha="center", va="bottom", fontsize=11, fontfamily=A.HEAD, fontweight="bold")  # fmt: skip
ax[0].set_xticks(range(1, len(order) + 1), [label[s] for s in order], rotation=45, ha="right")
style_ax(ax[0], title="axon SNR per block, by sample", ylabel="SNR (block, TRAILMAP detections)")
# per-tile median heatmap
tiles_max = max(r["tile"] for r in rows) + 1
M = np.full((len(order), tiles_max), np.nan)
for i, s in enumerate(order):
    for t, m in summary[label[s]]["per_tile_median"].items():
        M[i, t] = m
im = ax[1].imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=np.nanpercentile(M, 98))
ax[1].set_yticks(range(len(order)), [label[s] for s in order])
ax[1].set_xticks(range(tiles_max))
ax[1].set_xlabel("tile index")
ax[1].set_title(safe("median SNR per tile/"), loc="left", fontfamily=A.HEAD)
cb = fig.colorbar(im, ax=ax[1], fraction=0.025, pad=0.01)
cb.set_label("median block SNR")
n_blocks = len(rows)
fig.text(0.01, 0.005, safe(f"n = {n_blocks} blocks · up to 40 non-overlapping 1.28 × 1.44 × 1.44 mm "
                           "blocks per tile, ≥ 90% inside the flat-field brain mask · level 2 "
                           "(2.99 × 2.99 × 4 um) · boxes: 25–75%, whiskers 1.5 IQR, outliers not shown "
                           "· * = no brain mask (random blocks, gray)"),
         fontsize=10, color=C["muted"] if "muted" in C else "#555555")  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 1))
savefig(fig, out + ".png", dpi=150)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_tile_median"} for k, v in summary.items()}, indent=0)[:2500])
