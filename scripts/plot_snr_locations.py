"""Where the snr-blocks patches were sampled: block outlines on whole-tile overview projections.

Usage: uv run python scripts/plot_snr_locations.py <out.png> <snr dir> [<snr dir> ...]
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from exaspim_axon_segmentation.snr import _multiscales, _read, _store, open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import zarr  # noqa: E402

out, dirs = sys.argv[1], [Path(p) for p in sys.argv[2:]]
fig, axs = plt.subplots(len(dirs), 2, figsize=(22, 8.5 * len(dirs)),
                        gridspec_kw={"width_ratios": [2.4, 1]}, squeeze=False)  # fmt: skip
cmap, normc = plt.get_cmap("viridis"), Normalize(0, 12)
for r, d_dir in enumerate(dirs):
    d = json.loads((d_dir / "snr_blocks.json").read_text())
    url = d["url"]
    levels = len(_multiscales(zarr.open_group(_store(url), mode="r"))[0]["datasets"])
    lv = min(4, levels - 1)
    a, lead, sc, tr = open_ome(url, lv)
    v = _read(a, lead, (0, 0, 0), a.shape[lead:])
    half = np.asarray(d["config"]["block_um"]) / 2
    name = url.split("aind-open-data/")[1].split("/")[0].split("_")[0]
    for c, ax_proj in enumerate((0, 2)):  # top view (z-MIP) and side view (x-MIP)
        m = v.max(ax_proj)
        m = np.log1p(np.clip(m - np.percentile(m, 30), 0, None))
        lo, hi = np.percentile(m, [2, 99.7])
        keep = [i for i in range(3) if i != ax_proj]
        ext = [tr[keep[1]], tr[keep[1]] + m.shape[1] * sc[keep[1]],
               tr[keep[0]] + m.shape[0] * sc[keep[0]], tr[keep[0]]]  # fmt: skip
        ax = axs[r, c]
        ax.imshow(np.clip((m - lo) / (hi - lo), 0, 1), cmap="gray", extent=ext, aspect="auto"
                  if c else "equal")  # fmt: skip
        for b in d["blocks"]:
            ctr = np.asarray(b["centre_um"])
            snr = b.get("native", {}).get("snr")
            col = "#ff3b3b" if snr is None else cmap(normc(snr))
            ax.add_patch(Rectangle((ctr[keep[1]] - half[keep[1]], ctr[keep[0]] - half[keep[0]]),
                                   2 * half[keep[1]], 2 * half[keep[0]], fill=False, ec=col,
                                   lw=1.8))  # fmt: skip
            ax.text(ctr[keep[1]], ctr[keep[0]], str(b["block"]), color=col, ha="center",
                    va="center", fontsize=9, weight="bold")  # fmt: skip
        ax.set_title(f"{name} · {'top view (z max projection)' if c == 0 else 'side view (x max projection)'}"
                     f" · level {lv} ({sc[2]:.0f} µm px)")  # fmt: skip
        ax.set_xlabel(f"{'xyz'[2 - keep[1]]} (µm)")
        ax.set_ylabel(f"{'xyz'[2 - keep[0]]} (µm)")
sm = plt.cm.ScalarMappable(norm=normc, cmap=cmap)
fig.colorbar(sm, ax=axs, fraction=0.015, label="block SNR (red = not scored)")
fig.suptitle("20 random 1.3 × 1.4 × 1.4 mm blocks per brain, coloured by SNR", y=0.995)
fig.savefig(out, dpi=60, bbox_inches="tight")
print("wrote", out)
