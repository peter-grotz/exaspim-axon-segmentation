"""Figures and summary of an snr-pair run placed by metadata only (no image matching).

Each scope is scored on its own detections; the 5x also at the 1x voxel size (mean-pooled image,
max-pooled detection). Optionally compares with a registered run of the same 1x blocks.

Usage: uv run python scripts/plot_snr_pair_rough.py <snr_pair dir> <out prefix> [<registered dir>]
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d_dir, out = Path(sys.argv[1]), sys.argv[2]
reg = json.loads((Path(sys.argv[3]) / "snr_pair.json").read_text()) if len(sys.argv) > 3 else None
d = json.loads((d_dir / "snr_pair.json").read_text())
B = [b for b in d["blocks"] if "skipped" not in b and b.get("a", {}).get("snr") is not None
     and b.get("b", {}).get("snr") is not None]  # fmt: skip


def get(bs, *k):
    out_ = []
    for b in bs:
        v = b
        for key in k:
            v = v.get(key) if isinstance(v, dict) else None
        out_.append(np.nan if v is None else v)
    return np.array(out_, float)


rows = {
    "1x SNR (15x15x20 um)": get(B, "a", "snr"),
    "5x SNR (3x3x4 um)": get(B, "b", "snr"),
    "5x SNR at 1x voxel size": get(B, "b", "at_a_voxel", "snr"),
    "1x noise / background": get(B, "a", "noise_sd") / get(B, "a", "background"),
    "5x noise / background (3 um)": get(B, "b", "noise_sd") / get(B, "b", "background"),
    "5x noise / background at 1x voxel": get(B, "b", "at_a_voxel", "noise_sd")
    / get(B, "b", "at_a_voxel", "background"),
    "1x detected %": 100 * get(B, "a", "axon_fraction"),
    "5x detected % (3 um)": 100 * get(B, "b", "axon_fraction"),
    "5x detected % at 1x voxel": 100 * get(B, "b", "at_a_voxel", "axon_fraction"),
}
summary = {"blocks": len(B), "of": len(d["blocks"])}
for k, v in rows.items():
    summary[k] = {"median": float(np.nanmedian(v)), "p25": float(np.nanpercentile(v, 25)),
                  "p75": float(np.nanpercentile(v, 75))}  # fmt: skip
ratio = rows["5x SNR at 1x voxel size"] / rows["1x SNR (15x15x20 um)"]
summary["5x at 1x voxel / 1x SNR"] = {"median": float(np.nanmedian(ratio)),
                                     "p25": float(np.nanpercentile(ratio, 25)),
                                     "p75": float(np.nanpercentile(ratio, 75))}  # fmt: skip
if reg is not None:  # same 1x blocks, 5x placed by registration: how much does placement matter
    rb = {b["block"]: b for b in reg["blocks"]}
    pairs = [(b, rb[b["block"]]) for b in B if b["block"] in rb and "b" in rb[b["block"]]]
    dist = [np.linalg.norm(np.subtract(b["centre_b_um"], r["centre_b_um"])) for b, r in pairs]
    x = np.array([r["b"].get("at_a_voxel", {}).get("snr") or r["cross"]["b_matched"]["snr"]
                  for _, r in pairs], float)  # fmt: skip
    y = np.array([b["b"]["at_a_voxel"]["snr"] for b, _ in pairs], float)
    summary["placement offset vs registration (um)"] = {
        "median": float(np.median(dist)), "p25": float(np.percentile(dist, 25)),
        "p75": float(np.percentile(dist, 75))}  # fmt: skip
Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(f"{k:42s}", v if not isinstance(v, dict) else "median %.3g [%.3g, %.3g]" % tuple(v.values()))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
a, b5 = rows["1x SNR (15x15x20 um)"], rows["5x SNR at 1x voxel size"]
lim = np.nanmax([a, b5]) * 1.1
ax[0].scatter(b5, a, c="#2e6db8", label="5x at the 1x voxel size")
ax[0].scatter(rows["5x SNR (3x3x4 um)"], a, c="#b84a2e", marker="x", label="5x at its own voxel size")
for bb, x_, y_ in zip(B, b5, a, strict=True):
    ax[0].annotate(str(bb["block"]), (x_, y_), fontsize=7, alpha=0.6)
ax[0].plot([0, lim], [0, lim], "k--", lw=0.8)
ax[0].set(xlabel="5x SNR", ylabel="1x SNR", title="per block, each scope on its own detections")
ax[0].legend()
labels = ["1x", "5x\n3 um", "5x at\n1x voxel"]
ax[1].boxplot([rows["1x SNR (15x15x20 um)"], rows["5x SNR (3x3x4 um)"], b5], tick_labels=labels)
ax[1].set(ylabel="SNR", title=f"{len(B)} blocks")
for x_ in ax:
    x_.grid(alpha=0.3)
fig.suptitle("791116 · same-day 1x vs 5x raw · blocks placed by array centres (metadata only, "
             "~2 mm accuracy)")  # fmt: skip
fig.tight_layout()
fig.savefig(f"{out}_scatter.png", dpi=90)
