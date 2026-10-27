"""Where the SNR blocks are: every sampled block's footprint on each brain's top-view MIP.

One panel per brain: the raw tiles' MIP (cached by plot_snr_report.py) placed by stage coordinates,
tile outlines, and each sampled block as a rectangle (its y-x footprint; blocks at different depths
overlap), coloured by its block SNR, or grey when it has no SNR (no detections / skipped).

Usage: uv run python scripts/plot_block_locations.py <results dir> <mip cache dir> <out.png>
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

root, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
LABEL = sys.argv[4] if len(sys.argv) > 4 else "p >= 0.5"
VMAX = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
C1 = {"665465", "719653", "787423", "852470", "853533"}
HALF = np.array([1440.0, 1440.0]) / 2  # y, x
subjects = sorted((p.name for p in root.iterdir() if (p / "snr_tiles").is_dir()),
                  key=lambda s: (0 if s in C1 else 2 if s == "787425" else 1, s))  # fmt: skip
norm = Normalize(0, VMAX)
cmap = plt.get_cmap("magma")
A.apply_style("white")
ncol = 4  # brains per row; each is two panels: image | blocks
nrow = -(-len(subjects) // ncol)
fig, axs = plt.subplots(nrow, 2 * ncol, figsize=(4.6 * 2 * ncol, 3.9 * nrow), squeeze=False)
for ax in axs.ravel():
    ax.axis("off")
for k, s in enumerate(subjects):
    ax, axb = axs[k // ncol, 2 * (k % ncol)], axs[k // ncol, 2 * (k % ncol) + 1]
    mips = {f.stem: np.load(f) for f in sorted((cache / s).glob("*.npz"))}
    allv = np.concatenate([m["mip"].ravel() for m in mips.values()])
    lo, hi = np.percentile(allv, 5), np.percentile(allv, 99.0)
    for m in mips.values():
        img = np.clip((np.log1p(m["mip"]) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1)
        sc, tr = m["sc"], m["tr"]
        ax.imshow(img**0.7, cmap="gray", interpolation="nearest", vmin=0, vmax=1,
                  extent=[tr[2] / 1e3, (tr[2] + img.shape[1] * sc[2]) / 1e3,
                          (tr[1] + img.shape[0] * sc[1]) / 1e3, tr[1] / 1e3])  # fmt: skip
        for a_ in (ax, axb):
            a_.add_patch(Rectangle((m["tr0"][2] / 1e3, m["tr0"][1] / 1e3), m["ext"][2] / 1e3,
                                   m["ext"][1] / 1e3, fill=False, ec=A.BRAND["orange"], lw=0.5,
                                   ls="--"))  # fmt: skip
    n_snr = n_none = 0
    for f in sorted((root / s / "snr_tiles").glob("*/snr_blocks.json")):
        for b in json.loads(f.read_text())["blocks"]:
            y, x = b["centre_um"][1], b["centre_um"][2]
            snr = b["native"].get("snr")
            if snr is None:
                n_none += 1
                kw = {"fill": False, "ec": A.BRAND["gray2"], "lw": 0.4}
            else:
                n_snr += 1
                kw = {"fill": True, "fc": cmap(norm(snr)), "ec": "none", "alpha": 0.45}
            axb.add_patch(Rectangle(((x - HALF[1]) / 1e3, (y - HALF[0]) / 1e3), 2 * HALF[1] / 1e3,
                                   2 * HALF[0] / 1e3, **kw))  # fmt: skip
    ext_x = [(m["tr0"][2], m["tr0"][2] + m["ext"][2]) for m in mips.values()]
    ext_y = [(m["tr0"][1], m["tr0"][1] + m["ext"][1]) for m in mips.values()]
    for a_ in (ax, axb):
        a_.set_xlim(min(e[0] for e in ext_x) / 1e3, max(e[1] for e in ext_x) / 1e3)
        a_.set_ylim(max(e[1] for e in ext_y) / 1e3, min(e[0] for e in ext_y) / 1e3)
        a_.set_aspect("equal")
    axb.set_facecolor(A.BRAND["page"])
    axb.axis("on")
    axb.set_xticks([])
    axb.set_yticks([])
    for sp in axb.spines.values():
        sp.set_visible(False)
    ax.set_title(A.safe(f"{s} {'c1' if s in C1 else '?' if s == '787425' else 'c2'} · "
                        f"{n_snr} blocks with SNR, {n_none} without"), loc="left", fontsize=10,
                 color=A.BRAND["blue"] if s in C1 else A.BRAND["orange"])  # fmt: skip
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs.ravel().tolist(),
                  fraction=0.012, pad=0.01)  # fmt: skip
cb.set_label(f"block SNR (capped at {VMAX:g})", fontsize=10)
fig.suptitle(A.safe(f"SNR blocks per brain · {LABEL}/"), x=0.01, ha="left", fontfamily=A.HEAD,
             fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe("top view, max over z of the raw tiles (level 7) · orange dashed: tiles · "
                             "each sampled 1.28 x 1.44 x 1.44 mm block as its y-x footprint, coloured "
                             "by SNR (blocks at different depths overlap; darker = several stacked) · grey outline: no SNR · "
                             "left: image, right: blocks, same axes"),
         fontsize=10, color=A.C["muted"])  # fmt: skip
A.savefig(fig, out, dpi=90)
print("saved", out)
