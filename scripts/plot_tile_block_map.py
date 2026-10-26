"""Whole-brain map of snr-tiles sampling: low-resolution MIP, brain mask, tile grid, blocks.

Per tile (cached in <cache>/<tile>.npz): the raw tile's MIP at a coarse pyramid level (top view:
max over z; side view: max over y) and the brain-mask tissue fraction on the 32-voxel grid
snr-tiles uses (low-res mask copy where the flat-field step saved one). Blocks: every grid block
that qualifies (>= 90% inside the mask) and the ones the run sampled (from its snr_blocks.json).
Everything is placed by each tile's own OME-Zarr translation (stage coordinates, no stitching).

Usage: uv run python scripts/plot_tile_block_map.py <subject> <snr results dir> <cache dir> <out.png>
<snr results dir> holds snr_tiles/<tile>/snr_blocks.json; subject roots come from
results/run_snrtiles_all.json.
"""

import json
import sys
from pathlib import Path

import fsspec
import matplotlib
import numpy as np

from exaspim_axon_segmentation.qc.tile_mask import blocks_from_fraction, mask_fraction
from exaspim_axon_segmentation.snr import open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

sid, res_dir, cache, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
MIP_LEVEL, BIN, BLOCK_UM, MIN_TISSUE = 6, 32, np.array([1280.0, 1440.0, 1440.0]), 0.9
sub = json.loads(Path("results/run_snrtiles_all.json").read_text())[sid]
root, mask_root = sub["root"].rstrip("/"), (sub["mask"] or "").rstrip("/")
fs = fsspec.filesystem("s3", anon=True)
names = sorted(p.split("/")[-1] for p in fs.ls(root.removeprefix("s3://")) if "_ch_561" in p)
stem = lambda n: n.split(".")[0]  # noqa: E731
masks, low_res = {}, {}
if mask_root:
    for p in fs.ls(mask_root.removeprefix("s3://")):
        masks[stem(p.rstrip("/").split("/")[-1])] = "s3://" + p.rstrip("/")
    art = mask_root.split("/flatfield_correction/")[0] + "/flatfield_correction/artifacts"
    for d in fs.ls(art.removeprefix("s3://")):
        for p in fs.ls(d):
            leaf = p.rstrip("/").split("/")[-1]
            if leaf.endswith("_mask_low_res.zarr"):
                low_res[stem(leaf)] = "s3://" + p.rstrip("/")
cache.mkdir(parents=True, exist_ok=True)

tiles = []
for n in names:
    url = f"{root}/{n}"
    a0, _, sc0, tr0 = open_ome(url, 0)
    shape0 = np.array(a0.shape[-3:])
    f = cache / f"{stem(n)}.npz"
    if not f.exists():
        a, lead, sc, tr = open_ome(url, MIP_LEVEL)
        vol = np.asarray(a[(0,) * lead]).astype(np.float32)
        frac = None
        if stem(n) in masks:
            frac, _ = mask_fraction(masks[stem(n)], bin=BIN, low_res=low_res.get(stem(n)))
        np.savez_compressed(f, mip_xy=vol.max(0), mip_zx=vol.max(1), sc=sc, tr=tr,
                            frac=np.zeros(0) if frac is None else np.round(frac * 255).astype(np.uint8))  # fmt: skip
        print(n, "cached", flush=True)
    z = np.load(f)
    frac = z["frac"].astype(np.float32) / 255 if z["frac"].size else None
    bins = np.maximum(np.round(BLOCK_UM / (sc0 * BIN)).astype(int), 1)
    qual = []
    if frac is not None:
        starts, _ = blocks_from_fraction(frac, bins, 10**9, min_tissue=MIN_TISSUE)
        qual = tr0 + ((starts + bins / 2) * BIN) * sc0
    j = res_dir / "snr_tiles" / stem(n) / "snr_blocks.json"
    sampled = [b["centre_um"] for b in json.loads(j.read_text())["blocks"]] if j.exists() else []
    tiles.append({"name": n, "idx": int(n.split("_")[1]), "tr0": tr0, "ext": shape0 * sc0,
                  "z": z, "frac": frac, "sc0": sc0, "qual": np.array(qual), "sampled": np.array(sampled)})  # fmt: skip

A.apply_style("white")
fig, axes = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1, 1]})
views = [("top view (max over z)", "mip_xy", (1, 2), "y (mm)"),
         ("side view (max over y)", "mip_zx", (0, 2), "z (mm)")]  # fmt: skip
allv = np.concatenate([t["z"]["mip_xy"].ravel() for t in tiles])
lo, hi = np.percentile(allv, 1), np.percentile(allv, 99.7)
half = BLOCK_UM / 2
for ax, (title, key, (ra, ca), ylab) in zip(axes, views, strict=True):
    for t in tiles:  # image, drawn in stage coordinates; overlaps: later tiles on top
        img = np.clip((np.log1p(t["z"][key]) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1)
        sc, tr = t["z"]["sc"], t["z"]["tr"]
        e = [tr[ca] / 1e3, (tr[ca] + img.shape[1] * sc[ca]) / 1e3,
             (tr[ra] + img.shape[0] * sc[ra]) / 1e3, tr[ra] / 1e3]  # fmt: skip
        ax.imshow(img, cmap="gray", extent=e, interpolation="nearest", zorder=1, alpha=0.9)
    for t in tiles:  # brain mask outline per tile
        if t["frac"] is None:
            continue
        m = t["frac"].max(axis=0 if key == "mip_xy" else 1)
        step = t["sc0"] * BIN
        ys = (t["tr0"][ra] + (np.arange(m.shape[0]) + 0.5) * step[ra]) / 1e3
        xs = (t["tr0"][ca] + (np.arange(m.shape[1]) + 0.5) * step[ca]) / 1e3
        ax.contour(xs, ys, m, levels=[0.5], colors=[A.BRAND["teal"]], linewidths=1.2, zorder=3)
    for t in tiles:  # tile grid
        x0, y0 = t["tr0"][ca] / 1e3, t["tr0"][ra] / 1e3
        ax.add_patch(Rectangle((x0, y0), t["ext"][ca] / 1e3, t["ext"][ra] / 1e3, fill=False,
                               ec=A.BRAND["orange"], lw=1.3, ls="--", zorder=4))  # fmt: skip
        ax.text(x0 + 0.15, y0 + 0.15, f"{t['idx']}" + ("" if t["frac"] is not None else "*"),
                color=A.BRAND["orange"], fontsize=11, va="top", ha="left", zorder=6,
                fontfamily=A.HEAD, fontweight="bold")  # fmt: skip
    for t in tiles:  # qualifying blocks (outlines) and sampled blocks (filled)
        for pts, kw in ((t["qual"], {"fill": False, "ec": A.BRAND["blue"], "lw": 0.5, "alpha": 0.35}),
                        (t["sampled"], {"fill": True, "fc": A.BRAND["violet"], "ec": A.BRAND["violet"],
                                        "lw": 0.8, "alpha": 0.22})):  # fmt: skip
            for c in pts:
                ax.add_patch(Rectangle(((c[ca] - half[ca]) / 1e3, (c[ra] - half[ra]) / 1e3),
                                       BLOCK_UM[ca] / 1e3, BLOCK_UM[ra] / 1e3, zorder=5, **kw))  # fmt: skip
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()
    A.style_ax(ax, title=title, xlabel="x (mm)", ylabel=ylab, grid_axis=None)
n_q = sum(len(t["qual"]) for t in tiles)
n_s = sum(len(t["sampled"]) for t in tiles)
fig.suptitle(A.safe(f"{sid} / tiles, brain mask and sampled blocks"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=22)  # fmt: skip
fig.text(0.01, 0.015, A.safe(
    f"561 nm raw tiles, MIP at level {MIP_LEVEL} (log scale), placed by stage coordinates · "
    f"orange dashed: tiles (* = no mask) · teal: brain mask outline (per tile) · blue outlines: "
    f"blocks >= 90% inside the mask (n = {n_q}) · violet: sampled blocks (n = {n_s}; projections "
    f"stack, so darker = several blocks along the hidden axis) · block 1.28 x 1.44 x 1.44 mm"),
    fontsize=11, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
A.savefig(fig, out, dpi=110)
print("saved", out, "qualifying", n_q, "sampled", n_s)
