"""Fraction of detected axon per broad CCF region, styled like the projection heatmap of
Nature s41586-026-11026-0 (Fig. c): regions as rows, samples as columns, 0..1 colour scale.

A column is one sample (or hemisphere): its detected axon voxels per region / its detected axon
voxels in all listed regions. "Spinal cord" counts tissue posterior to the CCF (which ends in the
medulla), so it holds caudal medulla too.

Usage (isolated env with brainglobe-atlasapi):
  python scripts/plot_axon_fraction_heatmap.py <out prefix> <name>=<counts.npz>,<annotation.tif>,<crop.json>,<crop.tif> ...
"""

import json
import sys

import matplotlib
import numpy as np
import tifffile
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
ROWS = ["OLF", "CTX", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "CB", "P", "MY", "Spinal cord"]
CCF = {"CTX": "Isocortex"}  # CTX row = isocortex: OLF, HPF and CTXsp are their own rows
atlas = BrainGlobeAtlas("allen_mouse_25um")
cols, names = [], []
for arg in sys.argv[2:]:
    name, files = arg.split("=", 1)
    counts_npz, anno_tif, crop_json, crop_tif = files.split(",")
    c = np.load(counts_npz)
    crop = json.load(open(crop_json))
    lo, hi = np.array(crop["crop_lo_vox"]), np.array(crop["crop_hi_vox"])
    det = c["detected"][lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float64)
    anno = tifffile.imread(anno_tif)
    img = tifffile.imread(crop_tif).astype(np.float32)
    ids = {r: atlas.structures[CCF.get(r, r)]["id"] for r in ROWS[:-1]}
    row_of = np.full(anno.shape, -1, np.int8)
    for sid in np.unique(anno):
        if sid and int(sid) in atlas.structures:
            path = atlas.structures[int(sid)]["structure_id_path"]
            k = next((i for i, r in enumerate(ROWS[:-1]) if ids[r] in path), -1)
            row_of[anno == sid] = k
    # spinal cord: unlabelled tissue posterior (low x: x runs posterior -> anterior) of the medulla
    s = gaussian_filter(np.log1p(img), 1.0)
    tissue = s > np.percentile(s[anno > 0], 2)
    my_x = np.argwhere(row_of == ROWS.index("MY"))[:, 2]
    caudal = np.zeros(anno.shape, bool)
    caudal[:, :, : int(np.percentile(my_x, 1))] = True
    row_of[(anno == 0) & tissue & caudal] = len(ROWS) - 1
    # hemispheres: per (z, x) column, the middle of the labelled extent along y (left -> right)
    lab = row_of >= 0
    ys = np.arange(anno.shape[1])[None, :, None]
    ymin = np.where(lab, ys, 10**6).min(1, keepdims=True)
    ymax = np.where(lab, ys, -1).max(1, keepdims=True)
    left = ys < (ymin + ymax) / 2.0
    for tag, sel in (("", np.ones(anno.shape, bool)), (" left", left), (" right", ~left)):
        per = np.array([det[(row_of == i) & sel].sum() for i in range(len(ROWS))])
        cols.append(per / per.sum())
        names.append(name + tag)
        print(name + tag, {r: round(float(v), 3) for r, v in zip(ROWS, per / per.sum(), strict=True)})
M = np.array(cols).T
json.dump({"rows": ROWS, "columns": names, "fraction_of_axons": M.tolist()}, open(out + ".json", "w"),
          indent=1)  # fmt: skip

plt.rcParams.update({"font.family": "Arial", "font.size": 13})
fig = plt.figure(figsize=(2.6 + 0.6 * len(names), 7.0))
ax = fig.add_axes([0.42, 0.17, 0.5, 0.64])
im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(ROWS)), ROWS)
ax.set_xticks(range(len(names)), names, rotation=90)
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
cax = fig.add_axes([0.42, 0.89, 0.5, 0.022])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
cb.outline.set_linewidth(0.8)
cax.set_title("Fraction of axons", fontsize=13, pad=6)
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print("wrote", out + ".png")
