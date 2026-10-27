"""Brain outline for tiles without a flat-field mask, from coarse image texture; QC figure.

Per tile, pyramid level 5 (32 x 24 x 24 um voxels) is cut into windows of 8 x 8 x 8 voxels
(256 x 191 x 191 um). A window is tissue when it is not empty background (99.9th percentile minus
median >= 3 counts, as the snr-tiles tissue check) and not stripe/streak noise: streaks are
elongated along y, so their intensity changes far more across x than along y (mean |dI/dx| /
mean |dI/dy| >= RATIO; 851474: streak blocks 3.1-3.7, tissue 0.9-2.1). The tissue windows are then
cleaned (largest connected region over the tile grid, small holes filled).

Outputs per brain: <out>/<subject>_outline.npz (per tile: window grid origin, window size, mask)
and <out>/<subject>_outline_qc.png: top-view image MIP | outline (fraction of the depth that is
tissue) | sampled blocks kept (inside >= 90%) or dropped.

Usage: uv run python scripts/brain_outline_unmasked.py <subject> <mip cache dir> <out dir> [--ratio 2.5]
"""

import argparse
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
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("subject")
p.add_argument("mips")
p.add_argument("out")
p.add_argument("--ratio", type=float, default=2.5)
p.add_argument("--level", type=int, default=5)
p.add_argument("--win", type=int, default=8)
a = p.parse_args()
root = json.loads(Path("results/run_snrtiles_all.json").read_text())[a.subject]["root"].rstrip("/")
fs = fsspec.filesystem("s3", anon=True)
names = sorted(x.split("/")[-1] for x in fs.ls(root.removeprefix("s3://")) if "_ch_561" in x)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def tile_windows(name):
    arr, lead, sc, tr = open_ome(f"{root}/{name}", a.level)
    shape = np.array(arr.shape[lead:])
    n = shape // a.win
    tail = np.zeros(n, np.float32)
    ratio = np.zeros(n, np.float32)
    for iz in range(n[0]):  # one slab of windows at a time
        z0 = iz * a.win
        v = _read(arr, lead, [z0, 0, 0], [z0 + a.win, n[1] * a.win, n[2] * a.win])
        w = (
            v.reshape(a.win, n[1], a.win, n[2], a.win)
            .transpose(1, 3, 0, 2, 4)
            .reshape(n[1], n[2], -1)
        )
        tail[iz] = np.percentile(w, 99.9, axis=2) - np.median(w, axis=2)
        gx = np.abs(np.diff(v, axis=2))[:, :, : n[2] * a.win - 1]
        gy = np.abs(np.diff(v, axis=1))[:, : n[1] * a.win - 1, :]
        gx = (
            np.pad(gx, ((0, 0), (0, 0), (0, 1)))
            .reshape(a.win, n[1], a.win, n[2], a.win)
            .mean((0, 2, 4))
        )
        gy = (
            np.pad(gy, ((0, 0), (0, 1), (0, 0)))
            .reshape(a.win, n[1], a.win, n[2], a.win)
            .mean((0, 2, 4))
        )
        ratio[iz] = gx / np.maximum(gy, 1e-3)
    return name, {"tail": tail, "ratio": ratio, "scale": sc * a.win, "origin": tr}


cache = out / f"{a.subject}_windows.npz"
if cache.exists():
    z = np.load(cache)
    tiles = {n: {f: z[f"{n}__{f}"] for f in ("tail", "ratio", "scale", "origin")} for n in names}
else:
    with ThreadPoolExecutor(4) as ex:
        tiles = dict(ex.map(tile_windows, names))
    np.savez_compressed(cache, **{f"{n}__{f}": t[f] for n, t in tiles.items()
                                  for f in ("tail", "ratio", "scale", "origin")})  # fmt: skip
for t in tiles.values():
    # brain envelope: windows with a bright tail, smoothed so dim tissue between them counts, largest
    # region, holes filled; minus streak windows (direction ratio, smoothed)
    base = gaussian_filter((t["tail"] >= 3).astype(np.float32), 1.5) > 0.3
    lab, n = label(base)
    if n:
        base = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    base = binary_fill_holes(base)
    streak = gaussian_filter((t["ratio"] >= a.ratio).astype(np.float32), 1.0) > 0.5
    t["tissue"] = base & ~streak
np.savez_compressed(out / f"{a.subject}_outline.npz",
                    **{f"{k}__{f}": v[f] for k, v in tiles.items() for f in ("tissue", "scale", "origin")})  # fmt: skip

# blocks: inside fraction from the outline windows
HALF = np.array([640.0, 720.0, 720.0])
keep, drop = [], []
for f in sorted(Path(f"results/snrtiles_v2/{a.subject}/snr_tiles").glob("*/snr_blocks.json")):
    name = next(n for n in names if n.startswith(f.parent.name))
    t = tiles[name]
    for b in json.loads(f.read_text())["blocks"]:
        c = np.array(b["centre_um"])
        lo = np.floor((c - HALF - t["origin"]) / t["scale"]).astype(int)
        hi = np.ceil((c + HALF - t["origin"]) / t["scale"]).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, t["tissue"].shape)
        inside = (
            t["tissue"][lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].mean()
            if np.all(hi > lo)
            else 0
        )
        (keep if inside >= 0.9 else drop).append((c, inside, b["native"].get("snr")))
print(f"{a.subject}: blocks kept {len(keep)}, dropped {len(drop)}", flush=True)

# QC figure
A.apply_style("white")
mips = {Path(f).stem: np.load(f) for f in sorted((Path(a.mips) / a.subject).glob("*.npz"))}
fig, axs = plt.subplots(1, 3, figsize=(24, 7.5))
allv = np.concatenate([m["mip"].ravel() for m in mips.values()])
lo_, hi_ = np.percentile(allv, 5), np.percentile(allv, 99)
for m in mips.values():
    img = (
        np.clip((np.log1p(m["mip"]) - np.log1p(lo_)) / (np.log1p(hi_) - np.log1p(lo_)), 0, 1) ** 0.7
    )
    sc, tr = m["sc"], m["tr"]
    e = [
        tr[2] / 1e3,
        (tr[2] + img.shape[1] * sc[2]) / 1e3,
        (tr[1] + img.shape[0] * sc[1]) / 1e3,
        tr[1] / 1e3,
    ]
    for ax in (axs[0], axs[2]):
        ax.imshow(img, cmap="gray", extent=e, interpolation="nearest", vmin=0, vmax=1)
for t in tiles.values():
    frac = t["tissue"].mean(0)  # share of the depth that is tissue, top view
    sc, tr = t["scale"], t["origin"]
    e = [
        tr[2] / 1e3,
        (tr[2] + frac.shape[1] * sc[2]) / 1e3,
        (tr[1] + frac.shape[0] * sc[1]) / 1e3,
        tr[1] / 1e3,
    ]
    axs[1].imshow(frac, cmap="magma", extent=e, vmin=0, vmax=0.6, interpolation="nearest")
    for ax in (axs[0], axs[2]):
        ys = (tr[1] + (np.arange(frac.shape[0]) + 0.5) * sc[1]) / 1e3
        xs = (tr[2] + (np.arange(frac.shape[1]) + 0.5) * sc[2]) / 1e3
        if frac.max() > 0.05:
            ax.contour(xs, ys, frac, levels=[0.05], colors=[A.BRAND["teal"]], linewidths=1.0)
for pts, col in ((keep, A.BRAND["teal"]), (drop, A.BRAND["maroon"])):
    for c, inside, snr in pts:
        axs[2].add_patch(Rectangle(((c[2] - 720) / 1e3, (c[1] - 720) / 1e3), 1.44, 1.44, fill=False,
                                   ec=col, lw=0.6, alpha=0.8))  # fmt: skip
xs = [m["tr0"][2] for m in mips.values()] + [m["tr0"][2] + m["ext"][2] for m in mips.values()]
ys = [m["tr0"][1] for m in mips.values()] + [m["tr0"][1] + m["ext"][1] for m in mips.values()]
for ax, title in zip(axs, ("image (top view, max over z) + outline", "outline: share of depth that is tissue",
                           f"blocks: kept {len(keep)} (teal), dropped {len(drop)} (red)"), strict=True):  # fmt: skip
    ax.set_xlim(min(xs) / 1e3, max(xs) / 1e3)
    ax.set_ylim(max(ys) / 1e3, min(ys) / 1e3)
    ax.set_aspect("equal")
    A.style_ax(ax, grid_axis=None, title=title, title_size=13)
fig.suptitle(A.safe(f"{a.subject} / brain outline from image texture (no flat-field mask)"), x=0.01,
             ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=18)  # fmt: skip
fig.text(0.01, 0.01, A.safe(
    f"windows of {a.win}^3 voxels at level {a.level} (~256 x 191 x 191 um); brain = windows with a bright "
    f"tail (p99.9 - median >= 3 counts), smoothed, largest region, holes filled; minus streak windows "
    f"(x/y intensity-change ratio >= {a.ratio}, smoothed; streaks run along y) · teal contour: tissue in >= 5% of the depth · a block is kept when >= 90% of it "
    "lies in tissue windows (same rule as the flat-field masks)"), fontsize=10, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
A.savefig(fig, out / f"{a.subject}_outline_qc.png", dpi=100)
print("saved", out / f"{a.subject}_outline_qc.png")
