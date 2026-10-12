"""Side-by-side 1x and 5x patches of an snr-pair run, with TRAILMAP's detections overlaid.

Images and probabilities are z max projections over the whole patch (saved by snr-pair).

Usage: uv run python scripts/plot_pair_patches.py <snr_pair dir> <out.png> [n_patches]
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d_dir, out = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
d = json.loads((d_dir / "snr_pair.json").read_text())
B = [b for b in d["blocks"] if b.get("a", {}).get("snr") is not None
     and b.get("b", {}).get("snr") is not None]  # fmt: skip
B = sorted(B, key=lambda b: b["a"]["snr"])
pick = [B[i] for i in np.linspace(0, len(B) - 1, n).round().astype(int)]  # across the 1x SNR range


def gray(m):
    m = np.log1p(np.clip(m - np.percentile(m, 50), 0, None))
    lo, hi = np.percentile(m, [1, 99.8])
    return np.clip((m - lo) / (hi - lo + 1e-6), 0, 1)


def overlay(img, prob):
    p = prob / 255.0
    rgb = np.stack([img] * 3, -1) * 0.8
    on = p >= 0.5
    rgb[on] = rgb[on] * 0.2 + np.array([1.0, 0.45, 0.1]) * 0.8
    return rgb


fig, axs = plt.subplots(n, 4, figsize=(20, 5.2 * n))
for r, b in enumerate(pick):
    k = b["block"]
    za, zb = np.load(d_dir / f"block{k:03d}_a.npz"), np.load(d_dir / f"block{k:03d}_b.npz")
    ia, ib = gray(za["raw_mip"].astype(float)), gray(zb["raw_mip"].astype(float))
    va, vb = b["a"]["voxel_um"], b["b"]["voxel_um"]
    ea = [0, ia.shape[1] * va[2] / 1000, ia.shape[0] * va[1] / 1000, 0]
    eb = [0, ib.shape[1] * vb[2] / 1000, ib.shape[0] * vb[1] / 1000, 0]
    a, c = b["a"], b["b"]
    panels = [
        (ia, ea, f"patch {k} · 1x image (15 µm px)"),
        (overlay(ia, za["prob_mip"]), ea,
         f"1x + TRAILMAP (p ≥ 0.5, orange) · {100 * a['axon_fraction']:.2f}% · SNR {a['snr']:.1f}"),
        (ib, eb, f"5x image (level 4, 12 µm px) · {b.get('b_tile', '')}"),
        (overlay(ib, zb["prob_mip"]), eb,
         f"5x + TRAILMAP (p ≥ 0.5, orange) · {100 * c['axon_fraction']:.3f}% · SNR {c['snr']:.1f}"),
    ]  # fmt: skip
    for col, (img, ext, title) in enumerate(panels):
        ax = axs[r, col]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, extent=ext)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mm")
fig.suptitle("791116 · same-day 1x vs 5x patches (1.3 × 1.4 × 1.4 mm each; z max projection over "
             "the whole patch) · 5x level 4 (12×12×16 µm) · 5x placed from metadata, ~2 mm from the 1x patch · sorted by 1x "
             "SNR", y=0.999)  # fmt: skip
fig.tight_layout()
fig.savefig(out, dpi=60)
print("wrote", out, [b["block"] for b in pick])
