"""Brains at 1x and 5x, one threshold: whole-brain views, SNR histograms and box plots.

Per brain (one row): 1x mid-depth slab (image, and with detections), four 5x blocks spanning the
brain's SNR range (block MIP and with detections; max-over-z previews of 1.28 mm blocks), and the
1x vs 5x block-SNR histograms. Detections: max probability along the projection >= threshold.

Usage: uv run python scripts/plot_1x_5x_pair_figure.py <token file> <out.png> <threshold>
       <subject>:<1x wholebrain cache npz>:<1x snr json>:<5x rescore dir> ...
"""

import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

tok, out, t, specs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
key = f"p{t:g}"
auth = "Basic " + base64.b64encode(f"{Path(tok).read_text().strip()}:".encode()).decode()
state = json.loads(Path("results/snrtiles_v2_state.json").read_text())
VOX1 = (20.0, 15.04, 15.04)
MIPS = Path("results/tile_mips")
SNR_MAX = 150
NORM = Normalize(0, 120)
CMAP = plt.get_cmap("magma")


def snrs(blocks):
    return np.array([b[key]["snr"] for b in blocks if b[key].get("snr") is not None
                     and b[key].get("noise_sd", 1) > 0], float)  # fmt: skip


def lognorm(img, lo_pct, hi_pct):
    v = img[img > 0] if (img > 0).any() else img
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return (
        np.clip((np.log1p(img) - np.log1p(lo)) / max(np.log1p(hi) - np.log1p(lo), 1e-6), 0, 1)
        ** 0.8
    )


def overlay(mask):
    rgba = np.zeros((*mask.shape, 4))
    rgba[..., 0], rgba[..., 2], rgba[..., 3] = 1, 1, mask * 0.9
    return rgba


def block_preview(sid, tile, block):
    cid = state[sid]["ids"][-1]
    u = (f"https://codeocean.allenneuraldynamics.org/api/v1/computations/{cid}/results/urls?path="
         f"snr_tiles/{tile}/block{block:03d}_native.npz")  # fmt: skip
    url = json.load(
        urllib.request.urlopen(
            urllib.request.Request(u, headers={"Authorization": auth}), timeout=60
        )
    )["download_url"]
    z = np.load(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    return z["raw_mip"].astype(np.float32), z["prob_mip"].astype(np.float32) / 255


A.apply_style("white")
fig = plt.figure(figsize=(30, 7.2 * len(specs)))
outer = fig.add_gridspec(len(specs), 5, width_ratios=[2.4, 2.4, 1.4, 0.8, 0.5], wspace=0.18,
                         hspace=0.3, left=0.12, right=0.99)  # fmt: skip
# genotype per sample, from the ExM Dataset Summary sheet (results/exm_dataset_summary.json)
sheet = json.loads(Path("results/exm_dataset_summary.json").read_text())
GENOTYPE = {str(d.get("Sample")).split("_")[0]: d.get("Genotype") for d in sheet}
for r, spec in enumerate(specs):
    sid, cache, one, five = spec.split(":")
    # 1x slab
    z = np.load(cache)
    slab = z["slab_mip"].astype(np.float32) / 255
    g = lognorm(z["img_slab"], 5, 99.8)
    s0, s1 = z["slab"]
    ext = [0, slab.shape[1] * VOX1[2] / 1e3, slab.shape[0] * VOX1[1] / 1e3, 0]
    gl = outer[r, 0].subgridspec(1, 2, wspace=0.05)
    pos = outer[r, 0].get_position(fig)
    geno = (GENOTYPE.get(sid) or "genotype unknown").replace(" / ", "/\n").replace(" + ", " +\n")
    fig.text(0.005, (pos.y0 + pos.y1) / 2, A.safe(f"{geno}\n{sid}"), ha="left", va="center",
             fontsize=15, fontfamily=A.HEAD, fontweight="bold", color=A.C["ink"])  # fmt: skip
    for c, over in enumerate((None, overlay(slab >= t))):
        ax = fig.add_subplot(gl[0, c])
        ax.imshow(g, cmap="gray", extent=ext, interpolation="nearest")
        if over is not None:
            ax.imshow(over, extent=ext, interpolation="nearest")
        A.style_ax(ax, grid_axis=None, title_size=12,
                   title=f"{sid} 1x · {(s1 - s0) * VOX1[0]:.0f} um slab" + ("" if over is None else f" · detections p >= {t:g}"))  # fmt: skip
        ax.set_xlabel("x (mm)", fontsize=9)
        if c == 1:
            ax.set_yticklabels([])
    # 5x whole brain: image (tile MIPs) and the sampled blocks coloured by SNR (5x was segmented in
    # blocks only, so there is no whole-brain 5x detection map)
    rows5 = [(f.parent.name, b) for f in sorted(Path(five).glob("*/snr_rescore.json"))
             for b in json.loads(f.read_text())["blocks"] if b[key].get("snr") is not None]  # fmt: skip
    mips = [np.load(f) for f in sorted((MIPS / sid).glob("*.npz"))]
    allv = np.concatenate([m["mip"].ravel() for m in mips])
    lo_, hi_ = np.percentile(allv, 5), np.percentile(allv, 99)
    xs = [m["tr0"][2] for m in mips] + [m["tr0"][2] + m["ext"][2] for m in mips]
    ys = [m["tr0"][1] for m in mips] + [m["tr0"][1] + m["ext"][1] for m in mips]
    g5 = outer[r, 1].subgridspec(1, 2, wspace=0.05)
    for c in range(2):
        ax = fig.add_subplot(g5[0, c])
        if c == 0:
            for m in mips:
                img = (
                    np.clip(
                        (np.log1p(m["mip"]) - np.log1p(lo_)) / (np.log1p(hi_) - np.log1p(lo_)), 0, 1
                    )
                    ** 0.7
                )
                sc, tr = m["sc"], m["tr"]
                ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1,
                          extent=[tr[2] / 1e3, (tr[2] + img.shape[1] * sc[2]) / 1e3,
                                  (tr[1] + img.shape[0] * sc[1]) / 1e3, tr[1] / 1e3])  # fmt: skip
        else:
            ax.set_facecolor(A.BRAND["page"])
            for _, b in rows5:
                y, x = b["centre_um"][1], b["centre_um"][2]
                ax.add_patch(Rectangle(((x - 720) / 1e3, (y - 720) / 1e3), 1.44, 1.44, ec="none",
                                       fc=CMAP(NORM(b[key]["snr"])), alpha=0.5))  # fmt: skip
            ax.set_yticklabels([])
        ax.set_xlim(min(xs) / 1e3, max(xs) / 1e3)
        ax.set_ylim(max(ys) / 1e3, min(ys) / 1e3)
        ax.set_aspect("equal")
        A.style_ax(ax, grid_axis=None, title_size=12,
                   title=f"{sid} 5x · whole brain (top view)" if c == 0 else f"{sid} 5x · blocks by SNR, p >= {t:g}")  # fmt: skip
        ax.set_xlabel("x (mm)", fontsize=9)
    # SNR histograms
    ax = fig.add_subplot(outer[r, 2])
    v1 = snrs(json.loads(Path(one).read_text())["blocks"])
    v5 = snrs([b for _, b in rows5])
    bins = np.arange(0, SNR_MAX + 5, 5)
    for v, c, lab in ((v1, A.BRAND["teal"], "1x"), (v5, A.BRAND["violet"], "5x")):
        h, e = np.histogram(np.clip(v, 0, SNR_MAX), bins)
        ax.stairs(
            h / len(v),
            e,
            color=c,
            lw=2,
            label=A.safe(f"{lab}: median {np.median(v):.1f} (n={len(v)})"),
        )
        ax.axvline(np.median(v), color=c, lw=1, ls="--")
    ax.set_xlim(0, SNR_MAX)
    ax.legend(frameon=False, fontsize=11)
    A.style_ax(
        ax,
        title=f"{sid} block SNR, p >= {t:g}",
        xlabel="block SNR",
        ylabel="fraction of blocks",
        title_size=13,
    )
    axb = fig.add_subplot(outer[r, 3])
    groups = [np.clip(v1, 0, SNR_MAX), np.clip(v5, 0, SNR_MAX)]
    bp = axb.boxplot(groups, positions=[0, 1], widths=0.55, showfliers=False, patch_artist=True)
    rng = np.random.default_rng(0)
    for k, (gv, c) in enumerate(zip(groups, (A.BRAND["teal"], A.BRAND["violet"]), strict=True)):
        bp["boxes"][k].set_facecolor(c)
        bp["boxes"][k].set_alpha(0.35)
        bp["medians"][k].set_color(A.C["ink"])
        axb.scatter(k + rng.uniform(-0.18, 0.18, len(gv)), gv, s=4, color=c, alpha=0.35, lw=0)
        A.value_label(axb, k, np.percentile(gv, 75), f"{np.median(gv):.1f}", size=13)
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["1x", "5x"])
    axb.set_ylim(0, SNR_MAX * 1.05)
    A.style_ax(axb, title=f"{sid}", ylabel="block SNR", title_size=13)
    # ratio of medians, 5x / 1x, with a bootstrap 95% interval (blocks resampled within each scope)
    axr = fig.add_subplot(outer[r, 4])
    boot = [
        np.median(rng.choice(v5, len(v5))) / np.median(rng.choice(v1, len(v1))) for _ in range(2000)
    ]
    ratio_m = np.median(v5) / np.median(v1)
    lo_r, hi_r = np.percentile(boot, [2.5, 97.5])
    axr.bar([0], [ratio_m], width=0.6, color=A.BRAND["violet"], alpha=0.6)
    axr.errorbar(
        [0], [ratio_m], yerr=[[ratio_m - lo_r], [hi_r - ratio_m]], color=A.C["ink"], capsize=6
    )
    A.value_label(axr, 0, hi_r, f"{ratio_m:.1f}x", size=16)
    axr.set_xticks([0])
    axr.set_xticklabels(["5x / 1x"])
    axr.set_xlim(-0.6, 0.6)
    axr.set_ylim(0, max(6.0, hi_r * 1.25))
    A.style_ax(axr, title="SNR ratio", ylabel="median 5x / median 1x", title_size=13)
fig.suptitle(A.safe(f"1x vs 5x SNR · p >= {t:g}/"), x=0.01,
             ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=22)  # fmt: skip
fig.text(0.01, 0.005, A.safe(
    "1x: binned scan (tile_000000), 1x fine-tuned TRAILMAP, 20 x 15 x 15 um, 200 um slab at mid depth; "
    "SNR from 300 random tissue blocks · 5x: v2 TRAILMAP at level 2 (4 x 3 x 3 um), snr-tiles blocks "
    "rescored at the threshold; 5x was segmented in sampled blocks only (no whole-brain 5x detection map): "
    "blocks shown as 1.44 mm footprints coloured by SNR (0-120, magma) · boxes: 25-75%, dots: blocks · "
    "ratio: median 5x SNR / median 1x SNR, bar = 95% bootstrap interval · magenta: max probability >= threshold · SNR > "
    f"{SNR_MAX} in the last bin"), fontsize=10, color=A.C["muted"])  # fmt: skip
A.savefig(fig, out, dpi=80)
print("saved", out)
