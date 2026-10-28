"""QC views of a texture-based brain outline (brain_outline_unmasked.py windows): every step shown.

Rows: (1) top view (max over z) of the raw tiles at level 5 with the brain-envelope contour; (2) side
view (max over y, z down) with the same; (3) five z-slices through the brain with the envelope
and the streak flag drawn on them. Envelope: windows whose bright tail (p99.9 - median) >= 3 counts,
smoothed (Gaussian sigma 1.5 windows) > 0.3, largest region, holes filled. Streak flag: x/y
intensity-change ratio >= 2.5 (smoothed > 0.5).

Usage: uv run python scripts/outline_qc_views.py <subject> <out.png>
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

from exaspim_axon_segmentation.snr import _read, open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

sid, out = sys.argv[1], sys.argv[2]
LEVEL, WIN = 5, 8
z = np.load(f"results/outlines/{sid}_windows.npz")
root = json.loads(Path("results/run_snrtiles_all.json").read_text())[sid]["root"].rstrip("/")
fs = fsspec.filesystem("s3", anon=True)
names = sorted(x.split("/")[-1] for x in fs.ls(root.removeprefix("s3://")) if "_ch_561" in x)


def tile(name):
    arr, lead, sc, tr = open_ome(f"{root}/{name}", LEVEL)
    v = _read(arr, lead, [0, 0, 0], arr.shape[lead:])
    tail, ratio = z[f"{name}__tail"], z[f"{name}__ratio"]
    env = gaussian_filter((tail >= 3).astype(np.float32), 1.5) > 0.3
    lab, n = label(env)
    if n:
        env = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    env = binary_fill_holes(env)
    streak = gaussian_filter((ratio >= 2.5).astype(np.float32), 1.0) > 0.5
    up = lambda m: np.kron(m, np.ones((WIN, WIN, WIN), bool))  # noqa: E731  window grid -> voxels
    return name, {"v": v, "sc": sc, "tr": tr, "env": up(env), "streak": up(streak)}


with ThreadPoolExecutor(4) as ex:
    T = dict(ex.map(tile, names))
allv = np.concatenate([t["v"][::4, ::4, ::4].ravel() for t in T.values()])
lo, hi = np.percentile(allv, 5), np.percentile(allv, 99.5)


def show(ax, img, ext):
    g = np.clip((np.log1p(img) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1) ** 0.7
    ax.imshow(g, cmap="gray", extent=ext, interpolation="nearest", vmin=0, vmax=1)


A.apply_style("white")
fig = plt.figure(figsize=(26, 22))
gs = fig.add_gridspec(3, 5, height_ratios=[1.2, 0.9, 1.0], hspace=0.25)
ax_top = fig.add_subplot(gs[0, :3])
ax_str = fig.add_subplot(gs[0, 3:])
ax_side = fig.add_subplot(gs[1, :])
zmin = min(t["tr"][0] for t in T.values())
zmax = max(t["tr"][0] + t["v"].shape[0] * t["sc"][0] for t in T.values())
for t in T.values():
    sc, tr, v = t["sc"], t["tr"], t["v"]
    n = np.minimum(v.shape, t["env"].shape)
    env, stk = t["env"][: n[0], : n[1], : n[2]], t["streak"][: n[0], : n[1], : n[2]]
    ext_xy = [tr[2] / 1e3, (tr[2] + n[2] * sc[2]) / 1e3, (tr[1] + n[1] * sc[1]) / 1e3, tr[1] / 1e3]
    show(ax_top, v[: n[0], : n[1], : n[2]].max(0), ext_xy)
    show(ax_str, v[: n[0], : n[1], : n[2]].max(0), ext_xy)
    ys = (tr[1] + (np.arange(n[1]) + 0.5) * sc[1]) / 1e3
    xs = (tr[2] + (np.arange(n[2]) + 0.5) * sc[2]) / 1e3
    ax_top.contour(xs, ys, env.any(0), levels=[0.5], colors=[A.BRAND["teal"]], linewidths=1.2)
    ax_str.contourf(xs, ys, (stk & env).mean(0), levels=[0.15, 1], colors=[A.BRAND["maroon"]], alpha=0.45)
    ext_xz = [tr[2] / 1e3, (tr[2] + n[2] * sc[2]) / 1e3, (tr[0] + n[0] * sc[0]) / 1e3, tr[0] / 1e3]
    show(ax_side, v[: n[0], : n[1], : n[2]].max(1), ext_xz)
    zs = (tr[0] + (np.arange(n[0]) + 0.5) * sc[0]) / 1e3
    ax_side.contour(xs, zs, env.any(1), levels=[0.5], colors=[A.BRAND["teal"]], linewidths=1.2)
for ax, title in ((ax_top, "1. top view (max over z) · teal: brain envelope"),
                  (ax_str, "2. streak flag (red) inside the envelope"),
                  (ax_side, "3. side view (max over y, z down) · teal: brain envelope")):  # fmt: skip
    ax.set_aspect("equal")
    ax.autoscale_view()
    if ax is not ax_side:
        ax.invert_yaxis()
    else:
        ax.set_ylim(zmax / 1e3, zmin / 1e3)
    A.style_ax(ax, grid_axis=None, title=title, title_size=14)
# 4. z-slices
qs = (0.2, 0.35, 0.5, 0.65, 0.8)
for k, q in enumerate(qs):
    ax = fig.add_subplot(gs[2, k])
    zc = zmin + q * (zmax - zmin)
    for t in T.values():
        sc, tr, v = t["sc"], t["tr"], t["v"]
        iz = int((zc - tr[0]) / sc[0])
        if not 0 <= iz < min(v.shape[0], t["env"].shape[0]):
            continue
        n = np.minimum(v.shape, t["env"].shape)
        ext_xy = [tr[2] / 1e3, (tr[2] + n[2] * sc[2]) / 1e3, (tr[1] + n[1] * sc[1]) / 1e3, tr[1] / 1e3]
        show(ax, v[iz, : n[1], : n[2]], ext_xy)
        ys = (tr[1] + (np.arange(n[1]) + 0.5) * sc[1]) / 1e3
        xs = (tr[2] + (np.arange(n[2]) + 0.5) * sc[2]) / 1e3
        ax.contour(xs, ys, t["env"][iz, : n[1], : n[2]], levels=[0.5], colors=[A.BRAND["teal"]], linewidths=1)
        stk = t["streak"][iz, : n[1], : n[2]] & t["env"][iz, : n[1], : n[2]]
        if stk.any():
            ax.contour(xs, ys, stk, levels=[0.5], colors=[A.BRAND["maroon"]], linewidths=1)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    A.style_ax(ax, grid_axis=None, title=f"4. slice z = {zc / 1e3:.1f} mm", title_size=12)
fig.suptitle(A.safe(f"{sid} / brain outline, step by step (level {LEVEL}, 32 x 24 x 24 um voxels)"), x=0.01,
             ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.01, A.safe(
    "envelope: windows of 8^3 voxels (256 x 191 x 191 um) with a bright tail (p99.9 - median >= 3 counts), "
    "smoothed (sigma 1.5 windows) > 0.3, largest region, holes filled · streak flag: x/y intensity-change "
    "ratio >= 2.5 (smoothed > 0.5) · teal: envelope boundary, red: streak-flagged region"),
    fontsize=10, color=A.C["muted"])  # fmt: skip
A.savefig(fig, out, dpi=80)
print("saved", out)
