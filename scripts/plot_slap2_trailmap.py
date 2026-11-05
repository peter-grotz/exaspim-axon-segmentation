"""Max projections of a slap2_trailmap.py result: network input, probability per model, and the trace.

Usage: uvx --with matplotlib --with numpy python scripts/plot_slap2_trailmap.py <in.npz> <grid key e.g. 4x3x3> <out.png>
"""

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

src, key, out = sys.argv[1:4]
d = np.load(src)
res = json.loads(str(d["results"]))
g = [float(v) for v in key.split("x")]
x, lab = d[f"x_{key}"].astype(np.float32), d[f"label_{key}"]
runs = [(i, r) for i, r in enumerate(res) if "x".join(f"{v:g}" for v in r["grid_um"]) == key]
A.apply_style("white")
cols = 2 + len(runs)
fig, axs = plt.subplots(2, cols, figsize=(5.2 * cols, 8.5), gridspec_kw={"height_ratios": [x.shape[1] * g[1], x.shape[0] * g[0]]})
ext_xy = [0, x.shape[2] * g[2], x.shape[1] * g[1], 0]
ext_xz = [0, x.shape[2] * g[2], x.shape[0] * g[0], 0]
panels = [("input (scaled)", x, "gray", (0, 1)), ("trace (rasterized)", lab.astype(np.float32), "gray_r", (0, 1))]
panels += [(f"{r['weights'].split('/')[0]}\nrecall {r['recall']:.2f}, on-trace {r['on_trace_share']:.2f}, "
            f"p>=0.5 {100 * r['positive_fraction']:.1f}%", d[f"prob_{key}_{i}"].astype(np.float32), "magma", (0, 1))
           for i, r in runs]  # fmt: skip
for c, (t, v, cm, lim) in enumerate(panels):
    for rr, (axis, ext) in enumerate(((0, ext_xy), (1, ext_xz))):
        ax = axs[rr, c]
        ax.imshow(v.max(axis), cmap=cm, vmin=lim[0], vmax=lim[1] if c != 0 else np.percentile(v.max(axis), 99.5),
                  extent=ext, interpolation="nearest")  # fmt: skip
        if c >= 2:
            ax.contour(np.linspace(0, ext[1], v.shape[2]), np.linspace(0, ext[2], v.shape[1 if rr == 0 else 0]),
                       lab.max(axis), levels=[0.5], colors="cyan", linewidths=0.4)  # fmt: skip
        ax.grid(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("y (um)" if rr == 0 else "z (um)", fontsize=8)
        if rr == 0:
            ax.set_title(t, loc="left", fontsize=10, fontfamily=A.HEAD, fontweight="bold")
fig.suptitle(A.safe(f"trailmap on slap2 878030 dmd1, green, {key.replace('x', ' x ')} um grid (z, y, x)/"), x=0.01,
             ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=16)  # fmt: skip
fig.text(0.01, 0.005, A.safe("max projections, top xy, bottom xz; cyan: snt trace; recall = trace points with "
                             "p >= 0.5 within 1 voxel; on-trace = p >= 0.5 voxels within 1 voxel of the trace (lower bound "
                             "on precision: other labelled cells are untraced)"), fontsize=8, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
A.savefig(fig, out, dpi=120)
print("saved", out)
