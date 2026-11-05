"""3D render of a raw (uncleaned) skeleton of a slap2_trailmap.py probability map over the SLAP2 image.

Skeleton: p >= --thr thinned with skimage (3D), no cleanup; neighbouring skeleton voxels (26-connected) are
joined into line segments. Image: green channel averaged to 1 um (4 x 4 px in xy), volume-rendered.

Usage:
  uv run --with pyvista --with scikit-image --with matplotlib python scripts/slap2_skeleton_3d.py \
      <REFERENCE.tif> <trailmap npz> <grid key 4x3x3> <run index> <out.png> [--thr 0.5]
"""

import argparse
import itertools
import json
import sys

import matplotlib
import numpy as np
import pyvista as pv
import tifffile
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

p = argparse.ArgumentParser()
for k in ("tif", "npz", "key", "run", "out"):
    p.add_argument(k)
p.add_argument("--thr", type=float, default=0.5)
a = p.parse_args()
d = np.load(a.npz)
res = json.loads(str(d["results"]))[int(a.run)]
g = np.array([float(v) for v in a.key.split("x")])  # z, y, x um
prob = d[f"prob_{a.key}_{a.run}"].astype(np.float32)
sk = skeletonize(prob >= a.thr)
vox = np.argwhere(sk)
lut = {tuple(v): i for i, v in enumerate(vox)}
edges = []
for i, v in enumerate(vox):
    for o in itertools.product((-1, 0, 1), repeat=3):
        if o > (0, 0, 0):  # each neighbour pair once
            j = lut.get((v[0] + o[0], v[1] + o[1], v[2] + o[2]))
            if j is not None:
                edges.append((i, j))
xyz = ((vox + 0.5) * g)[:, ::-1]  # (x, y, z) um
lines = pv.PolyData(xyz, lines=np.hstack([[2, i, j] for i, j in edges]) if edges else None)
length = float(np.linalg.norm(xyz[[e[0] for e in edges]] - xyz[[e[1] for e in edges]], axis=1).sum())
print("skeleton voxels", len(vox), "segments", len(edges), "length mm", round(length / 1e3, 2))

n = len(tifffile.TiffFile(a.tif).pages)
img = tifffile.imread(a.tif, key=range(0, n, 2))  # green, (z, y, x) at 1 x 0.25 x 0.25 um
img = img.reshape(img.shape[0], img.shape[1] // 4, 4, img.shape[2] // 4, 4).mean((2, 4))  # 1 um voxels
lo, hi = np.percentile(img, [50, 99.9])
vol = pv.ImageData(dimensions=img.shape[::-1], spacing=(1.0, 1.0, 1.0), origin=(0.375, 0.375, 0.0))
vol.point_data["v"] = np.clip((img - lo) / (hi - lo), 0, 1).ravel()  # C order (z, y, x) = vtk x fastest
del img

views = [("oblique", (-0.6, 1.0, -0.55)), ("xy, looking down z", (0, 0, -1)), ("xz, side", (0, 1, 0))]
shots = []
for name, dirn in views:
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("black")
    pl.add_volume(vol, scalars="v", cmap="gray", opacity=[0, 0, 0.02, 0.15, 0.5], clim=(0, 1), show_scalar_bar=False)
    pl.add_mesh(lines, color="#FF5A36", line_width=2.5, render_lines_as_tubes=True)
    c = np.array(vol.center)
    pl.camera.focal_point = c
    pl.camera.position = c + np.array(dirn) / np.linalg.norm(dirn) * 1500
    pl.camera.up = (0, -1, 0) if name.startswith("xy") else (0, 0, -1)  # depth (z) points down in side views
    pl.reset_camera()
    pl.camera.zoom(1.25)
    pl.add_text(name, font_size=10, color="white")
    shots.append(pl.screenshot(None, return_img=True))
    pl.close()

A.apply_style("white")
fig, axs = plt.subplots(1, 3, figsize=(27, 7.6))
for ax, s in zip(axs, shots, strict=True):
    ax.imshow(s)
    ax.axis("off")
fig.suptitle(A.safe(f"raw skeleton of {res['weights'].split('/')[0]} on slap2 878030 dmd1 (3d)/"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=18)  # fmt: skip
fig.text(0.01, 0.01, A.safe(f"orange: p >= {a.thr:g} on the {a.key.replace('x', ' x ')} um grid (z, y, x), thinned, "
                            f"no cleanup ({length / 1e3:.1f} mm); gray: green channel at 1 um, volume rendered; "
                            "depth (z) points down in the oblique and side views"), fontsize=10,
         color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
A.savefig(fig, a.out, dpi=110)
print("saved", a.out)
