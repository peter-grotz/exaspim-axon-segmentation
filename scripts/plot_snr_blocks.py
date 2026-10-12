"""Preview MIPs of an snr-blocks run: image and TRAILMAP probability per block, sorted by SNR.

Usage: uv run python scripts/plot_snr_blocks.py <snr dir> <out.png> <title> [n]
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d_dir, out, title = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
d = json.loads((d_dir / "snr_blocks.json").read_text())
B = [b for b in d["blocks"] if b.get("native", {}).get("snr") is not None]
B = sorted(B, key=lambda b: b["native"]["snr"])
pick = [B[i] for i in np.linspace(0, len(B) - 1, min(n, len(B))).round().astype(int)]
cols = 4
rows = -(-len(pick) // 2)
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
for i, b in enumerate(pick):
    z = np.load(d_dir / f"block{b['block']:03d}_native.npz")
    m = np.log1p(np.clip(z["raw_mip"] - np.percentile(z["raw_mip"], 50), 0, None))
    lo, hi = np.percentile(m, [1, 99.8])
    img = np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)
    p = z["prob_mip"] / 255.0
    rgb = np.stack([img] * 3, -1) * 0.8
    on = p >= 0.5
    rgb[on] = rgb[on] * 0.2 + np.array([1.0, 0.45, 0.1]) * 0.8
    nat = b["native"]
    r, c = i // 2, (i % 2) * 2
    axs[r, c].imshow(img, cmap="gray")
    axs[r, c].set_title(f"block {b['block']} · image (7.52 µm px)", fontsize=10)
    axs[r, c + 1].imshow(rgb)
    axs[r, c + 1].set_title(f"TRAILMAP p ≥ 0.5 · {100 * nat['axon_fraction']:.2f}% · SNR {nat['snr']:.1f}",
                            fontsize=10)  # fmt: skip
for a in axs.flat:
    a.axis("off")
fig.suptitle(title, y=0.995)
fig.tight_layout()
fig.savefig(out, dpi=60)
print("wrote", out)
