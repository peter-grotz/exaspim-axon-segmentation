"""QC map of an expand run: region footprint, predicted cubes, followed axons by round, somata.

Usage: uv run python scripts/render_expand.py <expand dir> <out.png> [--footprint <npz>]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path.home() / ".claude/skills/allen-plotting-style/scripts"))
import allen_style as A

parser = argparse.ArgumentParser()
parser.add_argument("expand_dir", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--footprint", type=Path)
args = parser.parse_args()
A.apply_style("white")
summary = json.loads((args.expand_dir / "summary.json").read_text())
g = np.load(args.expand_dir / "graph.npz")
coords, edges, types, rnd, followed = g["coords"], g["edges"], g["types"], g["round"], g["followed"]
source = g["source"] if "source" in g else np.zeros(len(coords), int)  # 1 = robo-traced

fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
if args.footprint and args.footprint.exists():
    z = np.load(args.footprint)
    fp, lo, cell = z["footprint"], z["lo"], z["cell"]
    hi = lo + np.multiply(fp.shape, cell)
    fill = np.zeros((*fp.shape, 4))
    fill[fp] = (*plt.matplotlib.colors.to_rgb(A.BRAND["page"]), 1.0)
    ax.imshow(fill, extent=(lo[1], hi[1], hi[0], lo[0]), interpolation="nearest", zorder=0)
half = summary["cube_um"] / 2
for c in summary["cubes"]:
    ax.add_patch(Rectangle((c[2] - half, c[1] - half), 2 * half, 2 * half, fill=False,
                           ec=A.BRAND["gray1"], lw=0.6, zorder=1))  # fmt: skip


def draw(mask, colour, width, z):
    e = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    seg = np.stack([coords[e[:, 0]][:, [2, 1]], coords[e[:, 1]][:, [2, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors=colour, linewidths=width, zorder=z))


draw(~followed & (types == 3), "#B9B9FF", 0.4, 2)  # dendrites of detected somata
draw(~followed & (types == 2), A.BRAND["gray2"], 0.4, 2)  # other axon cable, not followed
n_rounds = max(int(rnd.max()), 1)
cmap = plt.get_cmap("plasma")
for k in range(n_rounds + 1):
    draw(followed & (rnd == k), cmap(0.1 + 0.8 * k / n_rounds), 1.2, 3)
draw(followed & (source == 1), A.BRAND["blue"], 1.6, 3.5)  # robo-traced cable (bridges, runs)
for s in summary["somata_reached"]:
    ax.scatter(s["soma_um"][2], s["soma_um"][1], marker="*", s=180, c=A.BRAND["maroon"], zorder=4)
ax.autoscale_view()
ax.set_aspect("equal")
ax.invert_yaxis()
ax.tick_params(length=0, labelsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_xlabel("x (um)", fontsize=9)
ax.set_ylabel("y (um)", fontsize=9)
subject = Path(summary["asset"]).name.split("_")[1]
axons = f" · {summary['axons']} axons" if "axons" in summary else ""
ax.set_title(
    f"{subject} · axons followed out of md · stop: {summary['stop']}/\n"
    f"{len(summary['rounds'])} rounds · {len(summary['cubes'])} cubes · "
    f"{summary['followed_mm']:.0f} mm followed{axons} · "
    f"{len(summary['somata_reached'])} somata reached (stars)",
    loc="left", fontsize=13, fontfamily=A.HEAD,
)  # fmt: skip
fig.text(0.01, 0.005, "colour = round the cable was added (dark = md, light = later) · blue = "
         "cable traced by the robo tracer · grey = cable not followed\npale blue = dendrites of "
         "detected somata · grey squares = predicted cubes",
         fontsize=9, color=A.C["muted"])  # fmt: skip
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150)
print("wrote", args.out)
