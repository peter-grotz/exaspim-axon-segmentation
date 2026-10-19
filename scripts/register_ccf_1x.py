"""Rough Allen CCF registration of a 1x tile with ANTsPy (affine + SyN) and the BrainGlobe atlas.

Input: a tissue crop (TIFF, z y x) of the registration channel and its voxel size divided by the
sample's expansion. The crop is reoriented to the atlas (BrainGlobe "asr": anterior, superior,
right origins) from the acquisition axis directions, registered, and the CCF annotation is warped
back into the crop's own grid.

Usage (isolated env):
  uvx --python 3.11 --with antspyx --with brainglobe-atlasapi --with tifffile --with matplotlib \
      python scripts/register_ccf_1x.py <crop.tif> <out dir> <vz,vy,vx um> <orientation> [syn]
orientation: one letter per input axis, the side it starts from (e.g. "ilp": z from inferior,
y from left, x from posterior).
"""

import json
import sys
from pathlib import Path

import ants
import matplotlib
import numpy as np
import tifffile
from brainglobe_atlasapi import BrainGlobeAtlas

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

crop, out = Path(sys.argv[1]), Path(sys.argv[2])
vox = np.array([float(v) for v in sys.argv[3].split(",")])
orient = sys.argv[4]
do_syn = len(sys.argv) > 5 and sys.argv[5] == "syn"
out.mkdir(parents=True, exist_ok=True)

atlas = BrainGlobeAtlas("allen_mouse_25um")
assert atlas.orientation == "asr"
pair = {"a": ("a", 1), "p": ("a", -1), "s": ("s", 1), "i": ("s", -1), "r": ("r", 1), "l": ("r", -1)}
src = [pair[c] for c in orient]  # for each input axis: (atlas axis letter, +1 same start / -1 flipped)
perm = [next(i for i, (ax, _) in enumerate(src) if ax == t) for t in "asr"]
flips = [src[i][1] for i in perm]


def to_atlas(arr):
    a = np.transpose(arr, perm)
    return a[tuple(slice(None, None, f) for f in flips)]


def from_atlas(arr):
    a = arr[tuple(slice(None, None, f) for f in flips)]
    return np.transpose(a, np.argsort(perm))


img = tifffile.imread(crop).astype(np.float32)
mov_np = to_atlas(img)
mov_np = np.log1p(np.clip(mov_np - np.percentile(mov_np, 5), 0, None))
mov_np /= np.percentile(mov_np, 99.5)
spacing = tuple(vox[perm].tolist())
moving = ants.from_numpy(np.ascontiguousarray(mov_np), spacing=spacing)
ref = atlas.reference.astype(np.float32)
ref /= np.percentile(ref, 99.5)
fixed = ants.from_numpy(ref, spacing=(25.0, 25.0, 25.0))

init = ants.registration(fixed, moving, type_of_transform="Affine", aff_metric="mattes",
                         aff_iterations=(500, 300, 100, 30), aff_shrink_factors=(8, 4, 2, 1),
                         aff_smoothing_sigmas=(3, 2, 1, 0), verbose=False)  # fmt: skip
transforms = init["fwdtransforms"]
inverse = init["invtransforms"]
kind = "affine"
if do_syn:
    syn = ants.registration(fixed, moving, type_of_transform="SyNOnly",
                            initial_transform=init["fwdtransforms"][0],
                            reg_iterations=(80, 40, 10), syn_metric="mattes", verbose=False)  # fmt: skip
    transforms, inverse, kind = syn["fwdtransforms"], syn["invtransforms"], "affine+SyN"
warped = ants.apply_transforms(fixed, moving, transforms).numpy()
anno = ants.from_numpy(atlas.annotation.astype(np.float32), spacing=(25.0, 25.0, 25.0))
whichinv = [True] + [False] * (len(inverse) - 1) if kind == "affine" else [True, False]
anno_in = ants.apply_transforms(moving, anno, inverse, interpolator="nearestNeighbor",
                                whichtoinvert=whichinv if kind == "affine" else None).numpy()  # fmt: skip
tifffile.imwrite(out / "ccf_annotation_in_crop.tif", from_atlas(anno_in).astype(np.uint32))
for i, t in enumerate(transforms):
    Path(t).replace(out / f"sample_to_ccf_{i}{''.join(Path(t).suffixes)}")
nmi = float(ants.image_mutual_information(fixed, ants.from_numpy(warped, spacing=(25.0,) * 3)))
json.dump({"crop": str(crop), "voxel_um_deexpanded": vox.tolist(), "orientation": orient,
           "perm_to_asr": perm, "flips": flips, "transform": kind, "mutual_information": nmi},
          open(out / "registration.json", "w"), indent=1)  # fmt: skip


def edges(lab):
    e = np.zeros(lab.shape, bool)
    e[1:] |= lab[1:] != lab[:-1]
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    return e


fig, axs = plt.subplots(2, 3, figsize=(21, 13))
A = atlas.annotation
for c, axis in enumerate((0, 1, 2)):  # coronal, horizontal, sagittal in atlas space
    k = A.shape[axis] // 2 if axis != 2 else A.shape[2] // 3
    w = np.take(warped, k, axis)
    lab = np.take(A, k, axis)
    r = np.take(ref, k, axis)
    for r_, base, name in ((0, w, "sample (registered)"), (1, r, "CCF template")):
        rgb = np.stack([np.clip(base, 0, 1)] * 3, -1)
        rgb[edges(lab)] = [1.0, 0.3, 0.1]
        axs[r_, c].imshow(rgb)
        axs[r_, c].set_title(f"{name} · {['coronal', 'horizontal', 'sagittal'][c]} slice {k} · CCF "
                             "boundaries", fontsize=10)  # fmt: skip
        axs[r_, c].axis("off")
fig.suptitle(f"{crop.stem} → Allen CCF 25 µm ({kind}), MI {nmi:.3f}")
fig.tight_layout()
fig.savefig(out / "qc_ccf_overlay.png", dpi=60)
print("done", kind, "MI", nmi, "perm", perm, "flips", flips)
