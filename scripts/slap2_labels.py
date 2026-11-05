"""Partial-label training targets for a SLAP2 stack with one traced neuron: positive / confident background / ignore.

Grid: 1 um isotropic (green channel averaged over 4 x 4 px in xy; planes are 1 um apart).
  positive   every voxel the trace passes through and its core (within 1 um of the centreline, 2 um in z for the
             axial blur), with no exception (soma, field edge), plus the shell out to 2 um (3 um in z) where the
             image is bright (local excess > --k_pos noise SDs), outside the soma
  ignore     a band around the trace (to 4 um, 6 um in z) and the rest of the soma sphere (--soma_um);
             every bright structure off the trace (local excess > --k_ign noise SDs); everything the existing
             model calls p >= --p_ign (its 4 x 3 x 3 um map, upsampled); all dilated by --grow_um; outside the
             imaged field
  background only what is left: dim, not tube-like, not found by the model, inside the imaged field
Local excess = gaussian(0.7 um) - gaussian(6 um); noise = 1.4826 x MAD of it inside the field.

Labels use the trainer's codes (train.labels.Label: 0 ignore, 1 axon = positive, 2 background). Output npz: image
(float32, raw counts at 1 um, z/y/x), label, voxel_um, stats; --volume also writes a training volume directory
(raw.zarr + labels.zarr) for `exaspim-axon finetune`, marked split "all" (every voxel trains and validates).
Plus an overlay figure.

Usage:
  uv run --with matplotlib python scripts/slap2_labels.py <REFERENCE.tif> <swc> <trailmap npz> <run index> \
      <out.npz> <out.png>
"""

import argparse
import json
import sys

import matplotlib
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from exaspim_axon_segmentation.train.labels import Label

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

p = argparse.ArgumentParser()
for k in ("tif", "swc", "npz", "run", "out", "png"):
    p.add_argument(k)
p.add_argument("--key", default="4x3x3")
p.add_argument("--k_pos", type=float, default=3.0)
p.add_argument("--k_ign", type=float, default=4.0)
p.add_argument("--p_ign", type=float, default=0.1)
p.add_argument("--grow_um", type=float, default=2.0)
p.add_argument("--soma_um", type=float, default=10.0)
p.add_argument("--volume", help="also write a training volume directory here")
a = p.parse_args()
SRC = np.array([1.0, 0.25, 0.25])
V = np.array([1.0, 1.0, 1.0])  # target voxel (z, y, x) um

n = len(tifffile.TiffFile(a.tif).pages)
img = tifffile.imread(a.tif, key=range(0, n, 2))
img = img.reshape(img.shape[0], img.shape[1] // 4, 4, img.shape[2] // 4, 4).mean((2, 4)).astype(np.float32)
shape = np.array(img.shape)
field = np.ones(shape, bool)
colmax = np.abs(img).max((0, 1))
field[:, :, colmax <= np.percentile(colmax, 5) * 1.0001] = False  # blank columns at the x edges
xs = np.flatnonzero(field[0, 0])
field[:, :, : xs[0] + 2] = False  # and the first / last 2 um of the imaged columns (edge ramp)
field[:, :, xs[-1] - 1 :] = False
print("grid", shape, "field x um", xs[0], xs[-1], flush=True)

# trace -> distance map (um, anisotropic: z scaled so the z radii map onto the xy radii)
sw = np.loadtxt(a.swc, comments="#")
idx = {int(r[0]): k for k, r in enumerate(sw)}
um = np.stack([sw[:, 4], sw[:, 3], sw[:, 2]], 1) * SRC + np.array([0, 0.125, 0.125])  # pixel centres
pts = []
for k, q in enumerate(sw[:, 6]):
    if int(q) in idx:
        r0, r1 = um[idx[int(q)]], um[k]
        m = max(int(np.ceil(np.linalg.norm(r1 - r0) / 0.25)), 1)
        pts.append(r0 + (r1 - r0) * np.linspace(0, 1, m + 1)[:, None])
pts = np.concatenate(pts)
seed = np.zeros(shape, bool)
seed[tuple(np.clip(np.floor(pts / V).astype(int), 0, shape - 1).T)] = True
ZS = 2.0  # z distances count half: 2 um in z ~ 1 um in xy
dist = distance_transform_edt(~seed, sampling=(V[0] / ZS, V[1], V[2]))  # "xy-equivalent" um

# local excess and its noise inside the field
exc = gaussian_filter(img, 0.7) - gaussian_filter(img, 6.0)
med = np.median(exc[field])
noise = 1.4826 * float(np.median(np.abs(exc[field] - med)))
bright_pos = exc > med + a.k_pos * noise
bright_ign = exc > med + a.k_ign * noise

# existing model's probability, upsampled to 1 um (its voxel i covers [i g, (i + 1) g))
d = np.load(a.npz)
g = np.array([float(v) for v in a.key.split("x")])
prob = d[f"prob_{a.key}_{a.run}"].astype(np.float32)
x0 = int(d["x0"])
zi = [np.clip(np.floor((np.arange(s) + 0.5) * V[i] / g[i]).astype(int), 0, prob.shape[i] - 1)
      for i, s in enumerate(shape)]  # fmt: skip
if x0:
    raise SystemExit("trailmap npz was cropped in x; map x0 before upsampling")
p_up = prob[np.ix_(*zi)]

lab = np.full(shape, Label.BACKGROUND, np.uint8)
core = dist <= 1.0
shell = (dist <= 2.0) & bright_pos
band = dist <= 4.0
root = um[int(np.flatnonzero(sw[:, 6] == -1)[0])]
zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
soma = ((zz + 0.5 - root[0]) / ZS) ** 2 + (yy + 0.5 - root[1]) ** 2 + (xx + 0.5 - root[2]) ** 2 <= a.soma_um**2
grow = int(round(a.grow_um / V[1]))
off = binary_dilation((bright_ign | (p_up >= a.p_ign)) & ~band, iterations=grow)
lab[off | band | soma | ~field] = Label.IGNORE
lab[shell & ~soma] = Label.AXON
lab[core | seed] = Label.AXON  # every trace voxel is positive, wherever it is
assert (lab[seed] == Label.AXON).all() and not (lab[~field] == Label.BACKGROUND).any()
stats = {
    "grid_um": V.tolist(),
    "shape": shape.tolist(),
    "noise": noise,
    "positive_fraction": float((lab == Label.AXON).mean()),
    "ignore_fraction": float((lab == Label.IGNORE).mean()),
    "background_fraction": float((lab == Label.BACKGROUND).mean()),
    "trace_voxels": int(seed.sum()),
    "trace_voxels_positive": int((lab[seed] == Label.AXON).sum()),
    "ignore_off_trace_bright": float((bright_ign & ~band).mean()),
    "ignore_off_trace_model": float(((p_up >= a.p_ign) & ~band).mean()),
    "trace_um": float(np.linalg.norm(np.diff(pts, axis=0), axis=1)[np.linalg.norm(np.diff(pts, axis=0), axis=1) < 1].sum()),
    "args": vars(a) | {"tif": a.tif.split("/")[-1], "swc": a.swc.split("/")[-1]},
}
print(json.dumps({k: v for k, v in stats.items() if k != "args"}), flush=True)
np.savez_compressed(a.out, image=img, label=lab, voxel_um=V, stats=json.dumps(stats))
if a.volume:
    from pathlib import Path

    from exaspim_axon_segmentation.io import Box
    from exaspim_axon_segmentation.io.ngff import LevelTransform
    from exaspim_axon_segmentation.train.volume import write_training_volume

    prov = {"cube": "slap2-878030-dmd1", "brain": "878030", "split": "all", "modality": "slap2 in vivo 2p",
            "image": a.tif.split("/")[-1], "swc": a.swc.split("/")[-1], "voxel_um": V.tolist(),
            "voxels": {k.name.lower(): int((lab == k).sum()) for k in (Label.AXON, Label.BACKGROUND, Label.IGNORE)},
            "label_args": stats["args"]}  # fmt: skip
    write_training_volume(Path(a.volume), img, lab, LevelTransform(scale=tuple(V), translation=tuple(V / 2)),
                          Box.from_arrays(np.zeros(3, int), shape), prov)  # fmt: skip

# ---- overlay: xy slices through the arbor and an xz slice through the soma, plus xy max projection
A.apply_style("white")
cz = [int(round(root[0])) + o for o in (-60, -25, 0, 25, 55)]
cz = [c for c in cz if 0 <= c < shape[0]]
cy = int(round(root[1]))
COL = {Label.AXON: (1.0, 0.24, 0.24, 0.85), Label.IGNORE: (1.0, 0.82, 0.1, 0.33)}


def over(ax, im, lb, ext, title):
    lo_, hi_ = np.percentile(im[np.isfinite(im)], [1, 99.7])
    ax.imshow(im, cmap="gray", vmin=lo_, vmax=hi_, extent=ext, interpolation="nearest")
    if lb is not None:
        rgba = np.zeros((*lb.shape, 4))
        for v, c in COL.items():
            rgba[lb == v] = c
        ax.imshow(rgba, extent=ext, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10, fontfamily=A.HEAD, fontweight="bold")
    ax.grid(False)
    ax.tick_params(labelsize=7)


fig = plt.figure(figsize=(24, 21))
gs = fig.add_gridspec(3, 3, height_ratios=[shape[1], shape[1], shape[0]], hspace=0.16, wspace=0.08)
ext_xy = [0, shape[2], shape[1], 0]
for c_, z in enumerate(cz[1:4]):  # 25 um above, through, and 25 um below the soma
    for r_, show in enumerate((False, True)):
        ax = fig.add_subplot(gs[r_, c_])
        sl = img[max(z - 1, 0) : z + 2].max(0)  # 3-plane max
        over(ax, sl, lab[z] if show else None, ext_xy,
             f"xy, z = {z} um" + (" + labels" if show else ""))  # fmt: skip
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("y (um)", fontsize=8)
sl = img[:, max(cy - 1, 0) : cy + 2].max(1)
for c_, show in enumerate((False, True)):
    ax = fig.add_subplot(gs[2, c_])
    over(ax, sl, lab[:, cy] if show else None, [0, shape[2], shape[0], 0],
         f"xz through the soma, y = {cy} um" + (" + labels" if show else ""))  # fmt: skip
    ax.set_xlabel("x (um)", fontsize=8)
    ax.set_ylabel("z (um)", fontsize=8)
z = cz[2]  # zoom on the soma plane
r0, c0 = int(root[1]) - 45, int(root[2]) - 70
ax = fig.add_subplot(gs[2, 2])
sl = img[max(z - 1, 0) : z + 2].max(0)[r0 : r0 + 90, c0 : c0 + 140]
over(ax, sl, lab[z, r0 : r0 + 90, c0 : c0 + 140], [c0, c0 + 140, r0 + 90, r0], f"zoom, xy z = {z} um + labels")
ax.set_xlabel("x (um)", fontsize=8)
ax.set_ylabel("y (um)", fontsize=8)
h = [plt.Rectangle((0, 0), 1, 1, color=COL[Label.AXON][:3], label=f"positive (every trace voxel) {100 * stats['positive_fraction']:.2f}%"),
     plt.Rectangle((0, 0), 1, 1, color=COL[Label.IGNORE][:3], alpha=0.5,
                   label=f"ignore (near trace, soma, untraced bright or model-detected) {100 * stats['ignore_fraction']:.1f}%"),
     plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#999",
                   label=f"background (no overlay) {100 * stats['background_fraction']:.1f}%")]  # fmt: skip
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(A.safe("slap2 878030 dmd1 partial labels at 1 um/"), x=0.01, ha="left", fontfamily=A.HEAD,
             fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.035, A.safe(
    f"slices are 3-plane max projections of the green channel. positive: within 1 um of the trace (2 um in z), "
    f"or 2 um (3 um in z) where bright; ignore: 4 um band around the trace, {a.soma_um:g} um soma sphere, off-trace "
    f"voxels with local excess > {a.k_ign:g} sd or co_ft2 p >= {a.p_ign:g}, grown {a.grow_um:g} um, and the field "
    f"edges; background: the rest"), fontsize=10, color=A.C["muted"])  # fmt: skip
fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.07)
A.savefig(fig, a.png, dpi=100)
print("saved", a.out, a.png)
