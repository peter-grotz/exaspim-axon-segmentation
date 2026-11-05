"""View a SLAP2 structural reference stack with an SNT reconstruction (SWC in pixel coordinates) on top.

The REFERENCE tif is (2 * Z, Y, X) float32 with the green and red channels interleaved or in blocks (detected
from plane-to-plane correlation); voxels are 0.25 x 0.25 x 1 um (acquisition.json). Output: max projections
(XY, XZ, YZ) per channel, each with and without the reconstruction, and the stack's depth profile.

Usage:
  uvx --with tifffile --with matplotlib --with numpy python scripts/slap2_view_swc.py <REFERENCE.tif> <swc> <out.png>
"""

import sys

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

tif, swc_path, out = sys.argv[1:4]
XY_UM, Z_UM = 0.25, 1.0
st = tifffile.imread(tif)  # 1.7 GB float32
n = st.shape[0] // 2


def corr(i, j):
    a, b = np.asarray(st[i], np.float32).ravel()[::7], np.asarray(st[j], np.float32).ravel()[::7]
    return float(np.corrcoef(a, b)[0, 1])


mid = n // 2 * 2
inter = corr(mid, mid + 2) > corr(mid, mid + 1)  # same channel two planes on when interleaved
ch = {"green": st[0::2] if inter else st[:n], "red": st[1::2] if inter else st[n:]}
print("layout", "interleaved" if inter else "blocked", "planes per channel", n, st.shape, st.dtype)

sw = np.loadtxt(swc_path, comments="#")
idx = {int(r[0]): k for k, r in enumerate(sw)}
xyz = sw[:, 2:5]
seg = [(xyz[idx[int(p)]], xyz[k]) for k, p in enumerate(sw[:, 6]) if int(p) in idx]
seg = np.array(seg)  # (S, 2, 3) pixel x, y, z
print("swc nodes", len(sw), "segments", len(seg), "extent px", xyz.min(0), xyz.max(0))

A.apply_style("white")
views = ["xy (max over z)", "xz (max over y)", "yz (max over x)"]
fig = plt.figure(figsize=(26, 22))
gs = fig.add_gridspec(4, 3, width_ratios=[1280, 208 * 4, 208 * 4], height_ratios=[800, 208 * 4, 800, 208 * 4],
                      hspace=0.12, wspace=0.06)  # fmt: skip
stats = {}
for r, (cname, v) in enumerate(ch.items()):
    vol = np.asarray(v, np.float32)
    stats[cname] = np.percentile(vol, [1, 50, 99, 99.9]).round(1).tolist()
    mips = [vol.max(0), vol.max(1), vol.max(2).T]  # (Y, X), (Z, X), (Y, Z)
    for c, vname in enumerate(views):
        im = mips[c]
        lo, hi = np.percentile(im, [1, 99.7])
        for k, overlay in enumerate((False, True)):
            if c == 0:
                ax = fig.add_subplot(gs[2 * r, k])
            elif c == 1:
                ax = fig.add_subplot(gs[2 * r + 1, 0]) if not overlay else None
            else:
                ax = fig.add_subplot(gs[2 * r, 2]) if overlay else None
            if ax is None:
                continue
            if c == 1:  # xz: depth down, stretched to the true 1 um : 0.25 um aspect
                ext = [0, im.shape[1] * XY_UM, im.shape[0] * Z_UM, 0]
            elif c == 2:  # yz: depth across
                ext = [0, im.shape[1] * Z_UM, im.shape[0] * XY_UM, 0]
            else:
                ext = [0, im.shape[1] * XY_UM, im.shape[0] * XY_UM, 0]
            ax.imshow(im, cmap="gray", vmin=lo, vmax=hi, extent=ext, interpolation="nearest", aspect="equal")
            if overlay or c == 1:
                um = (seg + 0.5) * np.array([XY_UM, XY_UM, Z_UM])  # pixel centres -> um (x, y, z)
                hv = (0, 1) if c == 0 else (0, 2) if c == 1 else (2, 1)  # (horizontal, vertical) axes
                ax.add_collection(LineCollection(um[:, :, list(hv)], colors="#FF3C3C", linewidths=0.6, alpha=0.8))
            ax.set_title(f"{cname}  {vname}" + ("  + reconstruction" if (overlay or c == 1) else ""), loc="left",
                         fontsize=11, fontfamily=A.HEAD, fontweight="bold")  # fmt: skip
            ax.tick_params(labelsize=7)
            ax.grid(False)
            ax.set_xlabel("x (um)" if c != 2 else "z (um)", fontsize=8)
            ax.set_ylabel("y (um)" if c != 1 else "z (um)", fontsize=8)
    del vol
fig.suptitle(A.safe("slap2 878030 dmd1 reference stack with the snt reconstruction/"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe(f"max projections, gray 1-99.7th percentile per panel; voxels 0.25 x 0.25 x 1 um; "
                             f"{len(sw)} swc nodes (snt, pixel coords) in red; channel percentiles 1/50/99/99.9: "
                             f"{stats}"), fontsize=9, color=A.C["muted"])  # fmt: skip
A.savefig(fig, out, dpi=110)
print("saved", out, stats)
