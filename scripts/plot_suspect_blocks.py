"""Image MIPs of sampled blocks in chosen tiles, to check what an edge tile's mask let through.

Usage: uv run python scripts/plot_suspect_blocks.py <subject> <snr results dir> <out.png> <tile idx>...
Reads each sampled block from the raw tile at level 4 (16 x 12 x 12 um voxels), MIP over z.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

from exaspim_axon_segmentation.snr import _read, open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

sid, res, out, tiles = sys.argv[1], Path(sys.argv[2]), sys.argv[3], [int(t) for t in sys.argv[4:]]
root = json.loads(Path("results/run_snrtiles_all.json").read_text())[sid]["root"].rstrip("/")
LEVEL, HALF = 4, np.array([640.0, 720.0, 720.0])
rows = []
for t in tiles:
    name = f"tile_{t:06d}_ch_561"
    url = next(f"{root}/{n}" for n in (f"{name}.ome.zarr", f"{name}.zarr")
               if (res / "snr_tiles" / name).exists() or True)  # fmt: skip
    try:
        arr, lead, sc, tr = open_ome(url, LEVEL)
    except Exception:
        url = f"{root}/{name}.zarr"
        arr, lead, sc, tr = open_ome(url, LEVEL)
    for b in json.loads((res / "snr_tiles" / name / "snr_blocks.json").read_text())["blocks"]:
        c = np.array(b["centre_um"])
        lo = np.maximum(np.floor((c - HALF - tr) / sc).astype(int), 0)
        hi = np.ceil((c + HALF - tr) / sc).astype(int)
        rows.append((t, b["block"], b["native"].get("snr"), b["native"].get("qc", {}).get("verdict"),
                     _read(arr, lead, lo, hi).max(0)))  # fmt: skip
A.apply_style("white")
n = len(rows)
cols = 8
fig, axs = plt.subplots(
    -(-n // cols), cols, figsize=(2.4 * cols, 2.7 * -(-n // cols)), squeeze=False
)
for ax in axs.ravel():
    ax.axis("off")
for ax, (t, k, snr, verdict, mip) in zip(axs.ravel(), rows, strict=False):
    lo, hi = np.percentile(mip, 1), np.percentile(mip, 99.5)
    ax.imshow(np.clip((mip - lo) / max(hi - lo, 1), 0, 1), cmap="gray")
    s = "-" if snr is None else f"{snr:.0f}"
    ax.set_title(A.safe(f"tile {t} · b{k}\nSNR {s} · {verdict}"), fontsize=9)
fig.suptitle(A.safe(f"{sid} / sampled blocks in tiles {', '.join(map(str, tiles))} (MIP over z, "
                    f"level {LEVEL}, own contrast per block)"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=16)  # fmt: skip
fig.tight_layout(rect=(0, 0, 1, 0.97))
A.savefig(fig, out, dpi=90)
print("saved", out, n, "blocks")
