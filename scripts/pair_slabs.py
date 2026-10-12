"""Thin slabs of matched snr-pair blocks: 1x, 5x resampled onto the 1x grid, 5x native, overlay.

Usage: uv run python scripts/pair_slabs.py <snr_pair.json> <out.png> <block> [<block> ...]
"""

import json
import sys

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter

from exaspim_axon_segmentation.snr import open_ome
from exaspim_axon_segmentation.snr_pair import PairConfig, RawTiles, _crop, _to_a

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(sys.argv[1]))
out, ks = sys.argv[2], [int(k) for k in sys.argv[3:]]
cfg = PairConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["config"].items()})
aff, t = np.asarray(d["affine"]["A"]), np.asarray(d["affine"]["t"])
tiles = RawTiles(d["raw_tiles"], "capsule/code/snr/791116_5x_bigstitcher_affine.xml", (1.0, 0.748, 0.748))
la = open_ome(d["url_a"], cfg.level_a)
half = np.asarray(cfg.block_um) / 2
DEPTHS = (0.25, 0.5, 0.75)


def nm(x, ref=None):
    ref = x if ref is None else ref
    lo, hi = np.percentile(ref, [1, 99.7])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


fig, axs = plt.subplots(len(ks) * len(DEPTHS), 4, figsize=(20, 5 * len(ks) * len(DEPTHS)))
r = 0
for k in ks:
    b = next(x for x in d["blocks"] if x["block"] == k)
    pa = np.asarray(b["centre_a_um"])
    dvec = np.asarray(b["refine_um"])
    raw_a, oa = _crop(la, pa, half)
    i = next(j for j, n in tiles.names.items() if n == b["b_tile"])
    (_, _, sc0, tr0), lv = open_ome(tiles.url(i), 0), open_ome(tiles.url(i), cfg.level_b)
    m = np.diag([*sc0, 1.0]) @ tiles.fused_to_voxel(i)
    m[:3, 3] += tr0
    lin, shift = m[:3, :3] @ aff, m[:3, :3] @ (t + dvec) + m[:3, 3]
    c = np.array([[z, y, x] for z in (0, 1) for y in (0, 1) for x in (0, 1)])
    pts = (lin @ (oa + c * np.array(raw_a.shape) * la[2]).T).T + shift
    raw_b, ob = _crop(lv, (pts.min(0) + pts.max(0)) / 2, (pts.max(0) - pts.min(0)) / 2 + 2 * lv[2])
    size = np.maximum(np.round(la[2] / lv[2]).astype(int), 1)
    b_on_a = _to_a(uniform_filter(raw_b, size=tuple(size)), ob, lv[2], lin, shift, oa, la[2],
                   raw_a.shape)  # fmt: skip
    cm = b["cross"]
    for f in DEPTHS:
        zi = int(f * raw_a.shape[0])
        sa = raw_a[zi : zi + 2].max(0)  # 40 um: two 1x planes
        sb = b_on_a[zi : zi + 2].max(0)
        # native 5x: same physical plane centre, 40 um thick (tilt between the volumes ~2 deg)
        pc = oa + (np.array([zi + 1, 0, 0]) + np.array([0, *raw_a.shape[1:]]) / 2) * la[2]
        pc5 = lin @ pc + shift
        zb = int(round((pc5[0] - ob[0]) / lv[2][0]))
        nz = int(round(40 / lv[2][0]))
        sn = raw_b[max(zb - nz // 2, 0) : zb + nz // 2].max(0)
        ext = [0, raw_a.shape[2] * la[2][2] / 1000, raw_a.shape[1] * la[2][1] / 1000, 0]
        A, B = nm(sa), nm(sb)
        axs[r, 0].imshow(A, cmap="gray", extent=ext)
        axs[r, 0].set_title(f"block {k} · depth {zi * 20:.0f} um · 1x, 40 um slab (15 um px)")
        axs[r, 1].imshow(B, cmap="gray", extent=ext)
        axs[r, 1].set_title("5x averaged onto the 1x grid, same slab")
        axs[r, 2].imshow(nm(sn), cmap="gray")
        axs[r, 2].set_title("5x native (3 um px), same plane (unrotated crop)")
        axs[r, 2].axis("off")
        axs[r, 3].imshow(np.stack([A, B, A], -1), extent=ext)
        axs[r, 3].set_title(f"overlay 1x magenta / 5x green · match NCC {b['match_ncc']:.2f} · "
                            f"SNR 1x {cm['a_matched']['snr']:.1f} / 5x {cm['b_matched']['snr']:.1f}",
                            fontsize=9)  # fmt: skip
        r += 1
fig.tight_layout()
fig.savefig(out, dpi=60)
print("wrote", out)
