"""Brain outline from intensity for tiles without a flat-field mask; QC figure with every step.

Per tile, pyramid level 6 (64 x 48 x 48 um voxels), Gaussian-smoothed (sigma SMOOTH voxels). The
background differs between tiles and across each tile (illumination), and the brain never fills
a tile's whole depth, so each tile's background at (y, x) is a low percentile over z (BG_Q) and the
image is divided by it. Bright artifact sheets lie at the top and bottom of each tile's z stack,
separated from the brain by a background gap: in each (y, x) column, the dimmest slice within the
outer EDGE fraction of the stack at each end marks the gap, and everything beyond it is blanked. Tiles are placed on one
grid (max where they overlap); tissue = ratio >= threshold, 1-voxel opening, largest 3D region,
CLOSE-voxel closing, holes filled slice by slice in z, y and x. Up to three thresholds are drawn; the first is saved.

Outputs: <out>/<subject>_outline6.npz (mask, origin_um, scale_um), <out>/<subject>_outline6_qc.png.

Usage: uv run python scripts/brain_outline_intensity.py <subject> <out dir> [--thresh 1.04 1.03 1.06]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, gaussian_filter, label

from exaspim_axon_segmentation.snr import _read, open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("subject")
p.add_argument("out")
p.add_argument("--thresh", type=float, nargs="+", default=[1.04, 1.03, 1.06])
p.add_argument("--level", type=int, default=6)
p.add_argument("--smooth", type=float, default=3.0)
p.add_argument("--bg-q", type=float, default=3.0)
p.add_argument("--edge", type=float, default=0.2)
p.add_argument("--close", type=int, default=4)
a = p.parse_args()
root = json.loads(Path("results/run_snrtiles_all.json").read_text())[a.subject]["root"].rstrip("/")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cache = out / f"{a.subject}_level{a.level}.npz"
if cache.exists():
    z = np.load(cache)
    names = sorted({k.split("__")[0] for k in z.files})
    T = {n: {f: z[f"{n}__{f}"] for f in ("v", "sc", "tr")} for n in names}
else:
    fs = fsspec.filesystem("s3", anon=True)
    names = sorted(x.split("/")[-1] for x in fs.ls(root.removeprefix("s3://")) if "_ch_561" in x)

    def read(name):
        arr, lead, sc, tr = open_ome(f"{root}/{name}", a.level)
        return name, {"v": _read(arr, lead, [0, 0, 0], arr.shape[lead:]).astype(np.uint16), "sc": sc, "tr": tr}

    with ThreadPoolExecutor(4) as ex:
        T = dict(ex.map(read, names))
    np.savez_compressed(cache, **{f"{n}__{f}": t[f] for n, t in T.items() for f in ("v", "sc", "tr")})

# 1. per-tile ratio to background
sc = T[names[0]]["sc"]
for t in T.values():
    s = gaussian_filter(t["v"].astype(np.float32), a.smooth)
    t["bg"] = np.percentile(s, a.bg_q, axis=0)
    t["r"] = s / np.maximum(t["bg"], 1)
    k = int(a.edge * len(s))
    iz = np.arange(len(s))[:, None, None]
    top = np.argmin(t["r"][:k], axis=0)  # background gap below the top sheet, per column
    bot = len(s) - k + np.argmin(t["r"][-k:], axis=0)  # and above the bottom sheet
    cut = (iz < top) | (iz > bot)
    t["r"][cut] = 1.0
    t["sheets"] = round(float(cut.mean()) * 100, 1)
# 2. one grid
org = np.min([t["tr"] for t in T.values()], 0)
shape = np.max([np.round((t["tr"] - org) / sc).astype(int) + t["v"].shape for t in T.values()], 0)
R = np.zeros(shape, np.float32)
RAW = np.zeros(shape, np.float32)
for t in T.values():
    o = np.round((t["tr"] - org) / sc).astype(int)
    sl = tuple(slice(o[i], o[i] + t["v"].shape[i]) for i in range(3))
    R[sl] = np.maximum(R[sl], t["r"])
    RAW[sl] = np.maximum(RAW[sl], t["v"])
# 3. threshold, largest region, fill
def outline(th):
    m = binary_opening(R >= th, iterations=1)
    lab, n = label(m)
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    if a.close:  # bridge gaps in the dim hemisphere's boundary so its interior fills
        m = np.pad(m, a.close)
        m = binary_closing(m, iterations=a.close)[(slice(a.close, -a.close),) * 3]
    for ax in range(3):
        m = np.stack([binary_fill_holes(s) for s in np.moveaxis(m, ax, 0)], ax)
    return m


MS = [outline(th) for th in a.thresh]
m = MS[0]
np.savez_compressed(out / f"{a.subject}_outline6.npz", mask=m, origin_um=org, scale_um=sc, thresh=a.thresh[0])
VOL = [x.sum() * np.prod(sc) / 1e9 for x in MS]
vol_mm3 = VOL[0]
print(f"{a.subject}: % of each tile blanked as sheet {[t['sheets'] for t in T.values()]}", flush=True)
print(f"{a.subject}: brain " + ", ".join(f"{v:.0f} mm^3 at {th:g}" for v, th in zip(VOL, a.thresh)), flush=True)

# QC figure
lo, hi = np.percentile(RAW[::4, ::4, ::4], [2, 99.7])


def gray(img):
    return np.clip((np.log1p(img) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1) ** 0.8


def ext(ax_h, ax_w):  # extent in mm for an image with axes (ax_h, ax_w) of the grid
    return [org[ax_w] / 1e3, (org[ax_w] + shape[ax_w] * sc[ax_w]) / 1e3,
            (org[ax_h] + shape[ax_h] * sc[ax_h]) / 1e3, org[ax_h] / 1e3]  # fmt: skip


LCOL = [A.BRAND["teal"], A.BRAND["orange"], A.BRAND["violet"]]


def draw(ax, img, sel, ah, aw, title):
    e = ext(ah, aw)
    ax.imshow(gray(img), cmap="gray", extent=e, vmin=0, vmax=1, interpolation="nearest")
    hs = (org[ah] + (np.arange(shape[ah]) + 0.5) * sc[ah]) / 1e3
    ws = (org[aw] + (np.arange(shape[aw]) + 0.5) * sc[aw]) / 1e3
    for k in range(len(MS) - 1, -1, -1):
        mk = sel(MS[k])
        if mk.any():
            ax.contour(ws, hs, mk, levels=[0.5], colors=[LCOL[k]], linewidths=1.6 if k == 0 else 0.9)
    ax.set_aspect("equal")
    A.style_ax(ax, grid_axis=None, title=title, title_size=12)


A.apply_style("white")
fig = plt.figure(figsize=(26, 25))
gs = fig.add_gridspec(4, 4, height_ratios=[0.8, 1, 1, 1], hspace=0.3, wspace=0.15)
axh = fig.add_subplot(gs[0, 0])
vals = R[::2, ::2, ::2].ravel()
axh.hist(np.clip(vals, 0.9, 1.4), bins=200, color=A.BRAND["blue"], log=True)
for k, th in enumerate(a.thresh):
    axh.axvline(th, color=LCOL[k], lw=2, label=f"{th:g}: {VOL[k]:.0f} mm^3")
axh.legend(frameon=False, fontsize=10)
axh.set_xlabel("smoothed intensity / tile background")
axh.set_ylabel("voxels (log)")
A.style_ax(axh, title="1. ratio to background (sheets blanked)", title_size=12)
draw(fig.add_subplot(gs[0, 1:3]), RAW.max(1), lambda x: x.any(1), 0, 2, "2. side view (max over y, z down)")
draw(fig.add_subplot(gs[0, 3]), RAW.max(2), lambda x: x.any(2), 0, 1, "3. end view (max over x, z down)")
for k, q in enumerate((0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.05, 0.97)):
    iz = int(q * shape[0])
    draw(fig.add_subplot(gs[1 + k // 4, k % 4]), RAW[iz], lambda x: x[iz], 1, 2,
         f"{4 + k}. slice z = {(org[0] + iz * sc[0]) / 1e3:.1f} mm")  # fmt: skip
for k, q in enumerate((0.2, 0.4, 0.6, 0.8)):
    iy = int(q * shape[1])
    draw(fig.add_subplot(gs[3, k]), RAW[:, iy], lambda x: x[:, iy], 0, 2,
         f"{12 + k}. slice y = {(org[1] + iy * sc[1]) / 1e3:.1f} mm (z down)")  # fmt: skip
fig.suptitle(A.safe(f"{a.subject} / brain outline from intensity, step by step"),
             x=0.01, ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.01, A.safe(
    f"level {a.level} ({sc[0]:.0f} x {sc[1]:.0f} x {sc[2]:.0f} um voxels), Gaussian smoothed (sigma {a.smooth:g} "
    f"voxels), divided per tile by its background (the {a.bg_q:g}th percentile over z at each y, x) · tissue = "
    f"ratio >= threshold (teal {a.thresh[0]:g}, orange {a.thresh[1:2]}, violet {a.thresh[2:3]}), 1-voxel opening, "
    f"largest 3D region, {a.close}-voxel closing, holes filled slice by slice in z, y and x · top/bottom artifact sheets blanked per tile · "
    "image: raw intensity, log scale"), fontsize=10, color=A.C["muted"])  # fmt: skip
A.savefig(fig, out / f"{a.subject}_outline6_qc.png", dpi=80)
print("saved", out / f"{a.subject}_outline6_qc.png")
