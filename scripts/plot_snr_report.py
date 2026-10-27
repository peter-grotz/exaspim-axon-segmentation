"""SNR report for snr-tiles runs: per-sample distributions, per-tile distributions, brain maps.

Figures (Allen style, small multiples, coloured by cluster):
  <out>_samples.png  block SNR histogram per sample (fraction of blocks, log; SNR symlog); solid =
                     all scored blocks, dashed = blocks passing QC ("ok")
  <out>_tiles.png    per sample: SNR of every block by tile (dots) with the tile median (bar)
  <out>_maps.png     per sample: low-resolution MIP of the raw tiles placed by stage coordinates,
                     each tile coloured by its median block SNR with the value written on it

Usage: uv run python scripts/plot_snr_report.py <results/snrtiles_v2> <mip cache dir> <out prefix>
Subject roots come from results/run_snrtiles_all.json; the MIP of each tile (max over z at pyramid
level 7) is cached as <mip cache>/<subject>/<tile>.npz.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import matplotlib
import numpy as np

from exaspim_axon_segmentation.snr import open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

root, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
LABEL = (
    sys.argv[4] if len(sys.argv) > 4 else "p >= 0.5"
)  # detection threshold, shown in every title
MIP_LEVEL = 7
C1 = {"665465", "719653", "787423", "852470", "853533"}  # cluster 1 (user's grouping)
UNASSIGNED = {"787425"}
subs = json.loads(Path("results/run_snrtiles_all.json").read_text())


def colour(s):
    if s in UNASSIGNED:
        return A.BRAND["gray"]
    return A.BRAND["blue"] if s in C1 else A.BRAND["orange"]


def cluster(s):
    return "?" if s in UNASSIGNED else ("c1" if s in C1 else "c2")


data, undefined = {}, {}
for sdir in sorted(p for p in root.iterdir() if (p / "snr_tiles").is_dir()):
    rows = []
    for f in sorted(sdir.glob("snr_tiles/*/snr_blocks.json")):
        d = json.loads(f.read_text())
        for b in d["blocks"]:
            n = b["native"]
            if n.get("snr") is None:
                continue
            if n.get("noise_sd", 1.0) <= 0:  # background clipped to one camera value: SNR undefined
                undefined[sdir.name] = undefined.get(sdir.name, 0) + 1
                continue
            rows.append((int(f.parent.name.split("_")[1]), n["snr"], n.get("qc", {}).get("verdict"),
                         b["centre_um"][1], b["centre_um"][2]))  # fmt: skip
    star = all(json.loads(f.read_text()).get("selection", "").startswith("unmasked")
               for f in list(sdir.glob("snr_tiles/*/snr_blocks.json"))[:3])  # fmt: skip
    data[sdir.name] = {"rows": rows, "star": star}
# order like the reference figure: cluster 1, then 2, then unassigned; by subject id within each
order = sorted(data, key=lambda s: ({"c1": 0, "c2": 1, "?": 2}[cluster(s)], s))
label = {s: f"{s}{'*' if data[s]['star'] else ''} {cluster(s)}" for s in order}
A.apply_style("white")
ncol = 6
nrow = -(-len(order) // ncol)
note = (f"{len(order)} samples · block SNR = (mean of detected axon voxels - clipped background) / "
        "background SD · blocks 1.28 x 1.44 x 1.44 mm at 2.99 x 2.99 x 4 um, tissue only · "
        "* = no brain mask · "
        f"{sum(undefined.values())} blocks with zero background SD (SNR undefined) left out")  # fmt: skip

# midline from the tissue blocks: the brain is roughly symmetric, so in each 1.44 mm band along x
# (anteroposterior) the midline lies halfway between the outermost tissue blocks in y; a line fitted
# through those midpoints allows a slightly rotated brain
from scipy.stats import mannwhitneyu  # noqa: E402

lr = json.loads(Path("results/lr_axis.json").read_text())
HALF_Y = 720.0  # blocks this close to the midline straddle it: excluded
midline, hemi = {}, {}
for s in order:
    yx = np.array([(r[3], r[4]) for r in data[s]["rows"]])
    band = np.floor(yx[:, 1] / 1440.0)
    mids = [((yx[band == k, 0].min() + yx[band == k, 0].max()) / 2, yx[band == k, 1].mean())
            for k in np.unique(band) if (band == k).sum() >= 3]  # fmt: skip
    mids = np.array(mids)
    if len(mids) >= 3:
        a, b = np.polyfit(mids[:, 1], mids[:, 0], 1)
    else:
        a, b = 0.0, float(np.median(yx[:, 0]))
    midline[s] = (a, b, yx[:, 1].min(), yx[:, 1].max())
    sign = 1 if lr[s] == "Right_to_left" else -1  # +y points to the left
    side, dist = {}, []
    for r in data[s]["rows"]:
        d = (r[3] - (a * r[4] + b)) * sign  # > 0: left of the midline
        dist.append(d)
        side.setdefault("left" if d > HALF_Y else "right" if d < -HALF_Y else "midline", []).append(
            r[1]
        )
    hemi[s] = side
    data[s]["dist"] = np.array(dist)
    # a tile's side: where most of its blocks lie (rough; tiles can straddle the midline)
    data[s]["tile_side"] = {t: ("left" if np.median(data[s]["dist"][[r[0] == t for r in data[s]["rows"]]]) > 0 else "right")
                            for t in {r[0] for r in data[s]["rows"]}}  # fmt: skip

# 1. per-sample histograms
SNR_MAX = 150  # linear axes end here; larger values are drawn in the last bin / at the top edge
bins = np.arange(0, SNR_MAX + 5, 5)
fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.4 * nrow), sharex=True, sharey=True,
                        squeeze=False)  # fmt: skip
for ax in axs.ravel()[len(order) :]:
    ax.axis("off")
for ax, s in zip(axs.ravel(), order, strict=False):
    L, R = np.array(hemi[s].get("left", [])), np.array(hemi[s].get("right", []))
    for vals, ls, lw in ((L, "-", 1.6), (R, "--", 1.3)):
        if len(vals):
            h, e = np.histogram(np.clip(vals, 0, bins[-1]), bins)
            ax.stairs(h / len(vals), e, color=colour(s), lw=lw, ls=ls)
    ax.set_xlim(0, SNR_MAX)
    A.style_ax(ax, grid_axis="y")
    ax.set_title(A.safe(f"{label[s]} · L {np.median(L):.0f} (n={len(L)}) · R {np.median(R):.0f} (n={len(R)})"),
                 loc="left", fontsize=9, color=colour(s))  # fmt: skip
for ax in axs[:, 0]:
    ax.set_ylabel("fraction of blocks", fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("block SNR", fontsize=9)
fig.suptitle(A.safe(f"axon SNR per block, by sample · {LABEL}/"), x=0.01, ha="left", fontfamily=A.HEAD,
             fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe(note + f" · solid: left hemisphere, dashed: right (each normalized to its own n; blocks within 0.72 mm of the estimated midline left out) · SNR > {SNR_MAX} "
                             "counted in the last bin"), fontsize=9,
         color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
A.savefig(fig, f"{out}_samples.png", dpi=110)

# 2. per-tile distributions
fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), sharey=True, squeeze=False)
for ax in axs.ravel()[len(order) :]:
    ax.axis("off")
rng = np.random.default_rng(0)
for ax, s in zip(axs.ravel(), order, strict=False):
    rows = data[s]["rows"]
    ts = data[s]["tile_side"]
    left = sorted(t for t in ts if ts[t] == "left")
    right = sorted(t for t in ts if ts[t] == "right")
    pos = {t: i for i, t in enumerate(left)} | {t: len(left) + 1 + i for i, t in enumerate(right)}
    for t, x in pos.items():
        v = np.array([r[1] for r in rows if r[0] == t])
        jit = x + rng.uniform(-0.25, 0.25, len(v))
        ax.scatter(jit[v <= SNR_MAX], v[v <= SNR_MAX], s=3, color=colour(s), alpha=0.35, lw=0)
        ax.scatter(jit[v > SNR_MAX], np.full((v > SNR_MAX).sum(), SNR_MAX), s=8, marker="^",
                   color=colour(s), lw=0)  # fmt: skip
        ax.plot([x - 0.38, x + 0.38], [min(np.median(v), SNR_MAX)] * 2, color=A.C["ink"], lw=1.2)
    ax.axvline(len(left), color=A.BRAND["teal"], lw=1.2, ls="--")
    ax.text(len(left) / 2 - 0.5, SNR_MAX * 1.0, "L", ha="center", va="top", fontsize=10,
            fontfamily=A.HEAD, fontweight="bold")  # fmt: skip
    ax.text(len(left) + 1 + len(right) / 2 - 0.5, SNR_MAX * 1.0, "R", ha="center", va="top",
            fontsize=10, fontfamily=A.HEAD, fontweight="bold")  # fmt: skip
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([str(t) for t in pos], fontsize=6)
    ax.set_ylim(0, SNR_MAX * 1.03)
    A.style_ax(ax, grid_axis="y")
    ax.set_title(A.safe(label[s]), loc="left", fontsize=10, color=colour(s))
for ax in axs[:, 0]:
    ax.set_ylabel("block SNR", fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel("tile index", fontsize=9)
fig.suptitle(A.safe(f"axon SNR per block, by tile · {LABEL}/"), x=0.01, ha="left", fontfamily=A.HEAD,
             fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe(note + f" · dots: blocks, bar: tile median, triangles: SNR > {SNR_MAX} · tiles grouped by hemisphere (where most of their blocks lie relative to the estimated midline), teal line between"),
         fontsize=9,
         color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
A.savefig(fig, f"{out}_tiles.png", dpi=110)

# 3. brain maps
fs = fsspec.filesystem("s3", anon=True)


def tile_mip(s, name):
    f = cache / s / f"{name.split('.')[0]}.npz"
    if not f.exists():
        url = f"{subs[s]['root'].rstrip('/')}/{name}"
        try:
            a, lead, sc, tr = open_ome(url, MIP_LEVEL)
        except IndexError:  # shallower pyramid: its coarsest level
            a, lead, sc, tr = open_ome(url, -1)
        _, _, sc0, tr0 = open_ome(url, 0)
        a0 = open_ome(url, 0)[0]
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, mip=np.asarray(a[(0,) * lead]).max(0), sc=sc, tr=tr, tr0=tr0,
                            ext=np.asarray(a0.shape[-3:]) * sc0)  # fmt: skip
    return np.load(f)


maps = {}
for s in order:
    base = subs[s]["root"].rstrip("/").removeprefix("s3://")
    names = sorted(p.split("/")[-1] for p in fs.ls(base) if "_ch_561" in p)
    with ThreadPoolExecutor(8) as ex:
        maps[s] = dict(zip(names, ex.map(lambda n, s=s: tile_mip(s, n), names), strict=True))
    print(s, "mips ready", flush=True)
summary = {}
for s in order:
    L, R = np.array(hemi[s].get("left", [])), np.array(hemi[s].get("right", []))
    p = float(mannwhitneyu(L, R).pvalue) if len(L) > 5 and len(R) > 5 else float("nan")
    summary[s] = {"left_median": float(np.median(L)) if len(L) else None,
                  "right_median": float(np.median(R)) if len(R) else None,
                  "n_left": len(L), "n_right": len(R), "n_midline": len(hemi[s].get("midline", [])),
                  "left_over_right": float(np.median(L) / np.median(R)) if len(L) and len(R) else None,
                  "mann_whitney_p": p, "y_axis": lr[s],
                  "undefined_snr_blocks": undefined.get(s, 0)}  # fmt: skip
Path(f"{out}_hemispheres.json").write_text(json.dumps(summary, indent=1))
fig, ax = plt.subplots(figsize=(1.0 * len(order) + 3, 6))
for i, s in enumerate(order):
    L, R = hemi[s].get("left", []), hemi[s].get("right", [])
    for x, v, mk in ((i - 0.17, L, "o"), (i + 0.17, R, "s")):
        if v:
            q = np.percentile(v, [25, 50, 75])
            ax.plot([x, x], [q[0], q[2]], color=colour(s), lw=2, alpha=0.5)
            ax.plot(x, q[1], mk, color=colour(s), ms=7, mfc="white" if mk == "s" else colour(s))
    if L and R:
        ax.plot([i - 0.17, i + 0.17], [np.median(L), np.median(R)], color=A.C["muted"], lw=0.8)
        r = summary[s]["left_over_right"]
        pv = summary[s]["mann_whitney_p"]
        ax.text(i, ax.get_ylim()[1] if False else 0, "", fontsize=1)
        ax.annotate(A.safe(f"L/R {r:.2f}" + ("*" if pv < 0.01 else "")), (i, max(np.percentile(L, 75), np.percentile(R, 75))),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)  # fmt: skip
ax.set_xticks(range(len(order)))
ax.set_xticklabels([label[s] for s in order], rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, None)
A.style_ax(ax, title=f"left vs right hemisphere, block SNR · {LABEL}", ylabel="block SNR")
fig.text(0.01, 0.005, A.safe("filled circle: left, open square: right (median and 25-75%) · L/R: ratio "
                             "of medians, * Mann-Whitney p < 0.01 · midline halfway between the outermost tissue "
                             "blocks in y along x (rough estimate); blocks within 0.72 mm of it excluded · left/right from each "
                             "acquisition's y-axis direction"), fontsize=9, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.03, 1, 1))
A.savefig(fig, f"{out}_hemispheres.png", dpi=110)

allmed = [np.median([r[1] for r in data[s]["rows"] if r[0] == t]) for s in order
          for t in {r[0] for r in data[s]["rows"]}]  # fmt: skip
norm = Normalize(vmin=0, vmax=float(np.percentile(allmed, 98)))
cmap = plt.get_cmap("magma")
ncol_m = 4  # brains per row; each brain is two panels: image | tiles with metrics
nrow_m = -(-len(order) // ncol_m)
fig, axs = plt.subplots(nrow_m, 2 * ncol_m, figsize=(5.0 * 2 * ncol_m, 3.9 * nrow_m), squeeze=False)
for ax in axs.ravel():
    ax.axis("off")
for k, s in enumerate(order):
    axi, axt = axs[k // ncol_m, 2 * (k % ncol_m)], axs[k // ncol_m, 2 * (k % ncol_m) + 1]
    mips = maps[s]
    allv = np.concatenate([m["mip"].ravel() for m in mips.values()])
    lo, hi = np.percentile(allv, 5), np.percentile(allv, 99.0)
    for name, m in mips.items():
        img = np.clip((np.log1p(m["mip"]) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1)
        sc, tr = m["sc"], m["tr"]
        axi.imshow(img**0.7, cmap="gray", interpolation="nearest", zorder=1, vmin=0, vmax=1,
                   extent=[tr[2] / 1e3, (tr[2] + img.shape[1] * sc[2]) / 1e3,
                           (tr[1] + img.shape[0] * sc[1]) / 1e3, tr[1] / 1e3])  # fmt: skip
    for name, m in mips.items():
        t = int(name.split("_")[1])
        v = [r[1] for r in data[s]["rows"] if r[0] == t]
        x0, y0 = m["tr0"][2] / 1e3, m["tr0"][1] / 1e3
        w, h = m["ext"][2] / 1e3, m["ext"][1] / 1e3
        axi.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=A.BRAND["orange"], lw=0.5,
                                ls="--", zorder=2))  # fmt: skip
        if v:
            med = float(np.median(v))
            axt.add_patch(Rectangle((x0, y0), w, h, fc=cmap(norm(med)), ec="white", lw=0.8,
                                    zorder=2))  # fmt: skip
            txt = f"{med:.0f}\nn={len(v)}"
            tc = "white" if norm(med) < 0.6 else "black"
        else:
            axt.add_patch(Rectangle((x0, y0), w, h, fc=A.BRAND["page"], ec=A.BRAND["gray2"], lw=0.6,
                                    ls=":", zorder=2))  # fmt: skip
            txt, tc = "-", A.C["muted"]
        axt.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=7, zorder=3,
                 color=tc, fontfamily=A.HEAD, fontweight="bold")  # fmt: skip
    a, b, xa, xb = midline[s]
    sm_ = summary[s]
    # same limits on both panels, y increasing downward (image convention), so they line up
    ext_x = [(m["tr0"][2], m["tr0"][2] + m["ext"][2]) for m in mips.values()]
    ext_y = [(m["tr0"][1], m["tr0"][1] + m["ext"][1]) for m in mips.values()]
    xl = (min(e[0] for e in ext_x) / 1e3, max(e[1] for e in ext_x) / 1e3)
    yl = (max(e[1] for e in ext_y) / 1e3, min(e[0] for e in ext_y) / 1e3)
    for ax in (axi, axt):
        ax.plot([xa / 1e3, xb / 1e3], [(a * xa + b) / 1e3, (a * xb + b) / 1e3],
                color=A.BRAND["teal"], lw=1.2, ls="--", zorder=4)  # fmt: skip
        ax.set_aspect("equal")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
    axi.set_title(A.safe(f"{label[s]} · median {np.median([r[1] for r in data[s]['rows']]):.0f}"),
                  loc="left", fontsize=11, color=colour(s))  # fmt: skip
    if sm_["left_median"] is not None and sm_["right_median"] is not None:
        axt.set_title(A.safe(f"L {sm_['left_median']:.0f} · R {sm_['right_median']:.0f}"),
                      loc="left", fontsize=10, color=A.C["ink"])  # fmt: skip
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.015, pad=0.01)
cb.set_label("tile median block SNR", fontsize=10)
fig.suptitle(A.safe(f"median axon SNR per tile · {LABEL}/"), x=0.01, ha="left", fontfamily=A.HEAD,
             fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe(note + " · top view, MIP over z at level 7 (log), tiles by stage "
                             "coordinates (they overlap ~15%) · number: tile median, n blocks · "
                             "dotted: no tissue blocks · teal dashed: estimated midline · colour scale linear, "
                             "capped at the 98th percentile of tile medians"), fontsize=9, color=A.C["muted"])  # fmt: skip
A.savefig(fig, f"{out}_maps.png", dpi=110)
print("saved", out)
