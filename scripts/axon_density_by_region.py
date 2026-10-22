"""Axon density per broad CCF region from per-bin TRAILMAP counts and a CCF annotation in the crop.

density = detected axon voxels (p >= threshold) / all level-0 voxels of the region (its bins x 16^3), per hemisphere
(split at the sample midline: the middle of the labelled brain along y, per z/x column).

Usage (isolated env with brainglobe-atlasapi):
  uvx --python 3.11 --with brainglobe-atlasapi --with tifffile --with matplotlib python \
      scripts/axon_density_by_region.py <counts.npz> <ccf_annotation_in_crop.tif> <crop.json> <out prefix>
"""

import json
import sys

import matplotlib
import numpy as np
import tifffile
from brainglobe_atlasapi import BrainGlobeAtlas

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

counts_npz, anno_tif, crop_json, out = sys.argv[1:5]
BROAD = ["OLF", "Isocortex", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "CB", "P", "MY"]
SHOW = {"Isocortex": "CTX (isocortex)"}
c = np.load(counts_npz)
det, seen = c["detected"].astype(np.float64), c["predicted"].astype(np.float64)
crop = json.load(open(crop_json))
lo, hi = np.array(crop["crop_lo_vox"]), np.array(crop["crop_hi_vox"])
det, seen = det[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]], seen[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
anno = tifffile.imread(anno_tif)
assert anno.shape == det.shape, (anno.shape, det.shape)
atlas = BrainGlobeAtlas("allen_mouse_25um")
ids = {a: atlas.structures[a]["id"] for a in BROAD}
lut = {}
for sid in np.unique(anno):
    if sid == 0:
        continue
    path = atlas.structures[int(sid)]["structure_id_path"] if int(sid) in atlas.structures else []
    lut[int(sid)] = next((a for a in BROAD if ids[a] in path), None)
broad = np.full(anno.shape, -1, np.int8)
for sid, a in lut.items():
    if a is not None:
        broad[anno == sid] = BROAD.index(a)
# hemisphere: per (z, x) column, the middle of the labelled extent along y (axis 1)
lab = anno > 0
ys = np.arange(anno.shape[1])[None, :, None]
ymin = np.where(lab, ys, 10**6).min(1, keepdims=True)
ymax = np.where(lab, ys, -1).max(1, keepdims=True)
mid = (ymin + ymax) / 2.0
side = np.where(ys < mid, 0, 1)  # y runs left -> right: 0 = left, 1 = right
vox_um3 = 5.0 * 3.76 * 3.76
rows = []
for i, a in enumerate(BROAD):
    m = broad == i
    r = {"region": SHOW.get(a, a), "acronym": a}
    for h, name in ((None, "both"), (0, "left"), (1, "right")):
        mm = m if h is None else m & (side == h)
        n = mm.sum() * 16**3  # every bin is 16^3 level-0 voxels (all-zero chunks are not stored)
        r[f"density_{name}"] = float(det[mm].sum() / n) if n else float("nan")
    r["region_volume_mm3_expanded"] = float(m.sum() * 16**3 * vox_um3 / 1e9)
    r["axon_volume_mm3_expanded"] = float(det[m].sum() * vox_um3 / 1e9)
    r["coverage"] = float(seen[m].sum() / max(m.sum() * 16**3, 1))  # fraction of region predicted
    rows.append(r)
json.dump(rows, open(out + ".json", "w"), indent=1)
print(f"{'region':16s} {'density':>8s} {'left':>8s} {'right':>8s} {'axon mm3':>9s} {'region mm3':>10s} {'covered':>8s}")
for r in rows:
    print(f"{r['region']:16s} {100 * r['density_both']:7.2f}% {100 * r['density_left']:7.2f}% "
          f"{100 * r['density_right']:7.2f}% {r['axon_volume_mm3_expanded']:9.2f} "
          f"{r['region_volume_mm3_expanded']:10.1f} {r['coverage']:7.0%}")  # fmt: skip
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(rows))
ax.bar(x - 0.2, [100 * r["density_left"] for r in rows], 0.4, label="left", color="#2e6db8")
ax.bar(x + 0.2, [100 * r["density_right"] for r in rows], 0.4, label="right", color="#b84a2e")
ax.set_xticks(x, [r["region"] for r in rows], rotation=30, ha="right")
ax.set_ylabel("detected axon (% of region volume, p ≥ 0.5)")
ax.set_title("850231 · TRAILMAP axon density per broad CCF region (1x unbinned, 0.05× SSC, tile 0)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(out + ".png", dpi=90)
print("wrote", out + ".json", out + ".png")
