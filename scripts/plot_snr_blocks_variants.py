"""Native vs resampled TRAILMAP on snr-blocks patches: image, native detections, resampled ones.

Usage: uv run python scripts/plot_snr_blocks_variants.py <snr dir> <out.png> <title> [n]
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d_dir, out, title = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
d = json.loads((d_dir / "snr_blocks.json").read_text())
B = sorted([b for b in d["blocks"] if b.get("native", {}).get("snr") is not None],
           key=lambda b: b["native"]["snr"])  # fmt: skip
pick = [B[i] for i in np.linspace(0, len(B) - 1, min(n, len(B))).round().astype(int)]


def gray(m):
    m = np.log1p(np.clip(m - np.percentile(m, 50), 0, None))
    lo, hi = np.percentile(m, [1, 99.8])
    return np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)


def overlay(img, prob):
    rgb = np.stack([img] * 3, -1) * 0.8
    on = prob / 255.0 >= 0.5
    rgb[on] = rgb[on] * 0.2 + np.array([1.0, 0.45, 0.1]) * 0.8
    return rgb


fig, axs = plt.subplots(len(pick), 3, figsize=(18, 6 * len(pick)))
for r, b in enumerate(pick):
    zn = np.load(d_dir / f"block{b['block']:03d}_native.npz")
    img = gray(zn["raw_mip"].astype(float))
    nat = b["native"]
    axs[r, 0].imshow(img, cmap="gray")
    axs[r, 0].set_title(f"block {b['block']} · image ({nat['voxel_um'][2]:.2f} µm px)")
    axs[r, 1].imshow(overlay(img, zn["prob_mip"]))
    axs[r, 1].set_title(f"native · {100 * nat['axon_fraction']:.2f}% · SNR {nat['snr']:.1f}")
    f = d_dir / f"block{b['block']:03d}_resampled.npz"
    if f.exists() and isinstance(b.get("resampled"), dict) and b["resampled"].get("snr"):
        zr = np.load(f)
        res = b["resampled"]
        axs[r, 2].imshow(overlay(gray(zr["raw_mip"].astype(float)), zr["prob_mip"]))
        axs[r, 2].set_title(f"resampled to 2.99 µm · {100 * res['axon_fraction']:.2f}% · SNR "
                            f"{res['snr']:.1f}")  # fmt: skip
    for a in axs[r]:
        a.axis("off")
fig.suptitle(title, y=0.998)
fig.tight_layout()
fig.savefig(out, dpi=55)
print("wrote", out)
