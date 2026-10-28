"""Keep only the 5x rescore blocks that lie inside a brain outline (brain_outline_intensity.py); QC figure.

A block is kept when >= 90% of its extent lies inside the outline mask (same rule as the flat-field
masks). Writes the kept blocks to <out dir>/<tile>/snr_rescore.json in the rescore layout, so the
plotting scripts can read it in place of the full rescore directory.

Usage: uv run python scripts/apply_outline_to_blocks.py <outline npz> <rescore dir> <out dir> <qc.png>
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

npz, src, dst, qc = sys.argv[1:5]
z = np.load(npz)
m, org, sc = z["mask"], z["origin_um"], z["scale_um"]
keep, drop = [], []
for f in sorted(Path(src).glob("*/snr_rescore.json")):
    d = json.loads(f.read_text())
    kept = []
    for b in d["blocks"]:
        half = np.array(b["shape"]) * np.array(b["voxel_um"]) / 2
        c = np.array(b["centre_um"])
        lo = np.maximum(np.floor((c - half - org) / sc).astype(int), 0)
        hi = np.minimum(np.ceil((c + half - org) / sc).astype(int), m.shape)
        inside = float(m[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].mean()) if np.all(hi > lo) else 0.0
        b["inside_outline"] = round(inside, 3)
        (keep if inside >= 0.9 else drop).append((c, half, b["p0.9"].get("snr")))
        if inside >= 0.9:
            kept.append(b)
    d["blocks"] = kept
    d["outline"] = {"file": npz, "rule": "inside >= 0.9"}
    (Path(dst) / f.parent.name).mkdir(parents=True, exist_ok=True)
    (Path(dst) / f.parent.name / "snr_rescore.json").write_text(json.dumps(d, indent=1))
print(f"blocks kept {len(keep)}, dropped {len(drop)}")

A.apply_style("white")
fig, axs = plt.subplots(1, 2, figsize=(22, 7), gridspec_kw={"width_ratios": [1.6, 1]})
for ax, (ah, aw), proj in ((axs[0], (0, 2), m.any(1)), (axs[1], (1, 2), m.any(0))):
    e = [org[aw] / 1e3, (org[aw] + m.shape[aw] * sc[aw]) / 1e3, (org[ah] + m.shape[ah] * sc[ah]) / 1e3, org[ah] / 1e3]
    ax.imshow(proj, cmap="Greys", vmin=0, vmax=4, extent=e, interpolation="nearest")
    for pts, col in ((drop, A.BRAND["maroon"]), (keep, A.BRAND["teal"])):
        for c, half, _ in pts:
            ax.add_patch(Rectangle(((c[aw] - half[aw]) / 1e3, (c[ah] - half[ah]) / 1e3), 2 * half[aw] / 1e3,
                                   2 * half[ah] / 1e3, fill=False, ec=col, lw=0.7))  # fmt: skip
    ax.set_aspect("equal")
A.style_ax(axs[0], grid_axis=None, title="side view (z down) · outline shaded", xlabel="x (mm)", ylabel="z (mm)")
A.style_ax(axs[1], grid_axis=None, title="top view", xlabel="x (mm)", ylabel="y (mm)")
fig.suptitle(A.safe(f"{Path(npz).stem.split('_')[0]} / 5x blocks vs brain outline: kept {len(keep)} (teal), "
                    f"dropped {len(drop)} (red)"), x=0.01, ha="left", fontfamily=A.HEAD, fontweight="bold",
             fontsize=18)  # fmt: skip
fig.text(0.01, 0.01, A.safe(f"outline {npz} (threshold {float(z['thresh']):g}) · kept when >= 90% of the "
                            "block lies inside"), fontsize=10, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
A.savefig(fig, qc, dpi=90)
print("saved", qc)
