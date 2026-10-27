"""Several 1x brains side by side at one threshold, from plot_wholebrain_1x.py caches.

Rows: brains. Columns: whole-volume image MIP, mid-depth slab image, the same slab with TRAILMAP
detections (max probability in the slab >= threshold) in magenta.

Usage: uv run python scripts/plot_1x_brains_p09.py <out.png> <threshold> <label>:<cache.npz> ...
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

out, t, specs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
VOX = (20.0, 15.04, 15.04)


def norm(img, lo_pct, hi_pct):
    v = img[img > 0] if (img > 0).any() else img
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return np.clip((np.log1p(img) - np.log1p(lo)) / max(np.log1p(hi) - np.log1p(lo), 1e-6), 0, 1) ** 0.8


A.apply_style("white")
fig, axs = plt.subplots(len(specs), 3, figsize=(20, 5.3 * len(specs)), squeeze=False)
for r, spec in enumerate(specs):
    label, path = spec.split(":", 1)
    z = np.load(path)
    slab = z["slab_mip"].astype(np.float32) / 255
    img_mip = np.kron(z["img_mip"], np.ones((4, 4), np.float32))[: slab.shape[0], : slab.shape[1]]
    s0, s1 = z["slab"]
    ext = [0, slab.shape[1] * VOX[2] / 1e3, slab.shape[0] * VOX[1] / 1e3, 0]
    g_mip, g_slab = norm(img_mip, 5, 99.5), norm(z["img_slab"], 5, 99.8)
    m = slab >= t
    rgba = np.zeros((*m.shape, 4))
    rgba[..., 0], rgba[..., 2], rgba[..., 3] = 1, 1, m * 0.9
    panels = ((g_mip, None, "whole volume, max over z"),
              (g_slab, None, f"slab {(s1 - s0) * VOX[0]:.0f} um at mid depth"),
              (g_slab, rgba, f"slab · detections p >= {t:g} · {100 * m.mean():.2f}% of pixels"))  # fmt: skip
    for c, (g, over, name) in enumerate(panels):
        ax = axs[r, c]
        ax.imshow(g, cmap="gray", extent=ext, interpolation="nearest")
        if over is not None:
            ax.imshow(over, extent=ext, interpolation="nearest")
        A.style_ax(ax, grid_axis=None, title=f"{label} · {name}", title_size=12)
        ax.set_xlabel("x (mm)", fontsize=9)
        ax.set_ylabel("y (mm)", fontsize=9)
fig.suptitle(A.safe(f"same-day 1x brains / TRAILMAP (1x fine-tuned) detections at p >= {t:g}"),
             x=0.01, ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.004, A.safe("binned 1x scans, tile_000000, 20 x 15.04 x 15.04 um voxels · image MIP from "
                             "pyramid level 2 · magenta: max probability across the slab >= threshold"),
         fontsize=10, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.015, 1, 0.97))
A.savefig(fig, out, dpi=90)
print("saved", out)
