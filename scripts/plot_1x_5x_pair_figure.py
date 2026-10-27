"""Two brains, 1x vs 5x at one threshold: SNR distributions plus what TRAILMAP detected in each.

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

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

tok, out, t, specs = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
key = f"p{t:g}"
auth = "Basic " + base64.b64encode(f"{Path(tok).read_text().strip()}:".encode()).decode()
state = json.loads(Path("results/snrtiles_v2_state.json").read_text())
VOX1 = (20.0, 15.04, 15.04)
SNR_MAX = 150


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
outer = fig.add_gridspec(len(specs), 3, width_ratios=[2.6, 2.4, 1.5], wspace=0.14, hspace=0.3)
for r, spec in enumerate(specs):
    sid, cache, one, five = spec.split(":")
    # 1x slab
    z = np.load(cache)
    slab = z["slab_mip"].astype(np.float32) / 255
    g = lognorm(z["img_slab"], 5, 99.8)
    s0, s1 = z["slab"]
    ext = [0, slab.shape[1] * VOX1[2] / 1e3, slab.shape[0] * VOX1[1] / 1e3, 0]
    gl = outer[r, 0].subgridspec(1, 2, wspace=0.05)
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
    # 5x blocks across the SNR range
    rows5 = [(f.parent.name, b) for f in sorted(Path(five).glob("*/snr_rescore.json"))
             for b in json.loads(f.read_text())["blocks"] if b[key].get("snr") is not None]  # fmt: skip
    rows5.sort(key=lambda x: x[1][key]["snr"])
    picks = [rows5[int(q * (len(rows5) - 1))] for q in (0.1, 0.4, 0.7, 0.95)]
    gm = outer[r, 1].subgridspec(2, 4, wspace=0.04, hspace=0.18)
    for c, (tile, b) in enumerate(picks):
        raw, prob = block_preview(sid, tile, b["block"])
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.7)
        img = np.clip((raw - lo) / max(hi - lo, 1), 0, 1)
        for rr, over in enumerate((None, overlay(prob >= t))):
            ax = fig.add_subplot(gm[rr, c])
            ax.imshow(img, cmap="gray", interpolation="nearest")
            if over is not None:
                ax.imshow(over, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if rr == 0:
                ax.set_title(A.safe(f"5x t{int(tile.split('_')[1])} b{b['block']} · SNR {b[key]['snr']:.0f}"),
                             fontsize=10, loc="left")  # fmt: skip
            elif c == 0:
                ax.set_ylabel(A.safe(f"p >= {t:g}"), fontsize=10)
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
fig.suptitle(A.safe(f"same-day 1x vs 5x / TRAILMAP detections and block SNR at p >= {t:g}"), x=0.01,
             ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=22)  # fmt: skip
fig.text(0.01, 0.005, A.safe(
    "1x: binned scan (tile_000000), 1x fine-tuned TRAILMAP, 20 x 15 x 15 um, 200 um slab at mid depth; "
    "SNR from 300 random tissue blocks · 5x: v2 TRAILMAP at level 2 (4 x 3 x 3 um), snr-tiles blocks "
    "rescored at the threshold; previews are max-over-z projections of 1.28 x 1.44 x 1.44 mm blocks at the "
    "10th, 40th, 70th and 95th SNR percentile · magenta: max probability >= threshold · SNR > "
    f"{SNR_MAX} in the last bin"), fontsize=10, color=A.C["muted"])  # fmt: skip
A.savefig(fig, out, dpi=80)
print("saved", out)
