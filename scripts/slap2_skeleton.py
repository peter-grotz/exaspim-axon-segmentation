"""Skeletonize + clean up the slap2_trailmap.py probability maps and compare the skeletons with the SNT trace.

Uses the production skeletonization (``skeletonize_probability``, default ``SkeletonConfig``: hysteresis
0.2 / 0.5, component volume, spur pruning, gap bridging, piece length and isolation filters, gap
connection). Scores in um against the trace (densified to 0.25 um): recall = share of trace length within
`tol` of the skeleton; on-trace = share of skeleton length within `tol` of the trace (a lower bound on
precision: other labelled processes in the field are untraced). Writes one SWC per model (um, z/y/x as
SWC x/y/z = x, y, z) and a figure.

Usage:
  uv run --with matplotlib python scripts/slap2_skeleton.py <trailmap npz> <grid key 4x3x3> <swc> <out dir> \
      [--tol 4 --tol 8]
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig, skeletonize_probability

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

p = argparse.ArgumentParser()
for k in ("npz", "key", "swc", "out"):
    p.add_argument(k)
p.add_argument("--tol", type=float, action="append")
a = p.parse_args()
tols = a.tol or [4.0, 8.0]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
d = np.load(a.npz)
res = json.loads(str(d["results"]))
x0 = int(d["x0"])
g = np.array([float(v) for v in a.key.split("x")])
SRC = np.array([1.0, 0.25, 0.25])

# trace as segments in um (z, y, x), densified to 0.25 um points with their segment lengths
sw = np.loadtxt(a.swc, comments="#")
idx = {int(r[0]): k for k, r in enumerate(sw)}
um = np.stack([sw[:, 4], sw[:, 3], sw[:, 2] - x0], 1) * SRC
seg = np.array([(um[idx[int(q)]], um[k]) for k, q in enumerate(sw[:, 6]) if int(q) in idx])
pts, wts = [], []
for r0, r1 in seg:
    n = max(int(np.ceil(np.linalg.norm(r1 - r0) / 0.25)), 1)
    pts.append(r0 + (r1 - r0) * (np.arange(n) + 0.5)[:, None] / n)
    wts.append(np.full(n, np.linalg.norm(r1 - r0) / n))
tpts, twts = np.concatenate(pts), np.concatenate(wts)
trace_mm = twts.sum() / 1e3

grid = SimpleNamespace(scale=tuple(g), translation=tuple(0.5 * g))  # voxel centres, as the trace raster
region = SimpleNamespace(start=(0, 0, 0))
runs = [(i, r) for i, r in enumerate(res) if np.allclose(r["grid_um"], g)]
summary, skels = [], []
for i, r in runs:
    prob = np.clip(np.round(d[f"prob_{a.key}_{i}"].astype(np.float32) * 255), 0, 255).astype(np.uint8)
    sk = skeletonize_probability(prob, grid, region, SkeletonConfig())
    c, e = sk.coordinates_um, sk.edges
    mid = (c[e[:, 0]] + c[e[:, 1]]) / 2
    elen = np.linalg.norm(c[e[:, 0]] - c[e[:, 1]], axis=1)
    d_sk = cKDTree(tpts).query(mid)[0] if len(mid) else np.zeros(0)  # skeleton -> trace
    d_tr = cKDTree(c).query(tpts)[0] if len(c) else np.full(len(tpts), np.inf)  # trace -> skeleton
    name = r["weights"].split("/")[0]
    row = {"weights": name, "skeleton_mm": round(sk.length_um / 1e3, 3), "trace_mm": round(trace_mm, 3),
           "pieces": len(sk.pieces()), "gap_filled_mm": round(sk.filled_length_um / 1e3, 3)}  # fmt: skip
    for t in tols:
        row[f"recall@{t:g}um"] = round(float(twts[d_tr <= t].sum() / twts.sum()), 3)
        row[f"on_trace@{t:g}um"] = round(float(elen[d_sk <= t].sum() / max(elen.sum(), 1e-9)), 3)
    print(json.dumps(row), flush=True)
    summary.append(row)
    sk.write_swc(out / f"skeleton_{a.key}_{name}.swc")
    skels.append((name, row, c, e, d_sk))
json.dump(summary, open(out / f"skeleton_{a.key}.json", "w"), indent=1)

# figure: input mip with the trace, then each skeleton coloured by distance to the trace
x = d[f"x_{a.key}"].astype(np.float32)
A.apply_style("white")
cols = 1 + len(skels)
hz, hy = x.shape[0] * g[0], x.shape[1] * g[1]
fig, axs = plt.subplots(2, cols, figsize=(5.4 * cols, 8.8), gridspec_kw={"height_ratios": [hy, hz]})
ext = [[0, x.shape[2] * g[2], hy, 0], [0, x.shape[2] * g[2], hz, 0]]
T, NEAR, FAR = "#00B4D8", A.BRAND["blue"], "#FF5A36"
for rr, (axis, hv) in enumerate(((0, (2, 1)), (1, (2, 0)))):
    for c_ in range(cols):
        ax = axs[rr, c_]
        mip = x.max(axis)
        ax.imshow(mip, cmap="gray", vmin=0, vmax=np.percentile(mip, 99.5), extent=ext[rr], interpolation="nearest",
                  alpha=1.0 if c_ == 0 else 0.45)  # fmt: skip
        ax.add_collection(LineCollection(seg[:, :, list(hv)], colors=T, linewidths=0.6 if c_ == 0 else 1.6,
                                         alpha=0.9 if c_ == 0 else 0.35))  # fmt: skip
        if c_ > 0:
            name, row, c, e, d_sk = skels[c_ - 1]
            near = d_sk <= tols[0]
            for m, col in ((near, NEAR), (~near, FAR)):
                ax.add_collection(LineCollection(np.stack([c[e[m, 0]], c[e[m, 1]]], 1)[:, :, list(hv)], colors=col,
                                                 linewidths=1.0))  # fmt: skip
            if rr == 0:
                ax.set_title(f"{name}\nskeleton {row['skeleton_mm']:.2f} mm in {row['pieces']} pieces; recall "
                             f"{row[f'recall@{tols[0]:g}um']:.2f}, on-trace {row[f'on_trace@{tols[0]:g}um']:.2f} "
                             f"(@{tols[0]:g} um)", loc="left", fontsize=9.5, fontfamily=A.HEAD,
                             fontweight="bold")  # fmt: skip
        elif rr == 0:
            ax.set_title(f"input + snt trace ({trace_mm:.2f} mm)", loc="left", fontsize=9.5, fontfamily=A.HEAD,
                         fontweight="bold")  # fmt: skip
        ax.set_xlim(ext[rr][0], ext[rr][1])
        ax.set_ylim(ext[rr][2], 0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("y (um)" if rr == 0 else "z (um)", fontsize=8)
h = [plt.Line2D([], [], color=T, lw=2, alpha=0.6, label="snt trace"),
     plt.Line2D([], [], color=NEAR, lw=2, label=f"skeleton within {tols[0]:g} um of the trace"),
     plt.Line2D([], [], color=FAR, lw=2, label="skeleton off the trace")]  # fmt: skip
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))
fig.suptitle(A.safe(f"trailmap skeletons on slap2 878030 dmd1, {a.key.replace('x', ' x ')} um grid (z, y, x)/"),
             x=0.01, ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=16)  # fmt: skip
fig.text(0.01, 0.005, A.safe("top xy, bottom xz (max projection of the input underneath); production skeletonization "
                             "(threshold 0.2, core 0.5, spurs < 10 um pruned, 30 um bridging, pieces >= 100 um, "
                             "isolation 500 / 50 um, 50 um gap connection); recall = trace length within the distance "
                             "of the skeleton; on-trace = skeleton length within it of the trace (lower bound on "
                             "precision)"), fontsize=8, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
A.savefig(fig, out / f"skeleton_{a.key}.png", dpi=120)
print("saved", out)
