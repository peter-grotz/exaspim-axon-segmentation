"""TRAILMAP detections at several probability thresholds, on saved per-block previews.

A block's preview holds max-over-z projections of the raw image and of the probability, so "max
p along the projection >= t" is exactly the projection of the detections at threshold t.
Columns: image, then detections (magenta) at each threshold; the number is the fraction of the
projection's pixels detected. Blocks: N per brain, chosen across tiles and SNR.

Usage: uv run python scripts/plot_threshold_sweep.py <token file> <out.png> <subject>[:<n>] ...
"""

import base64
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

tok, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
TH = (0.5, 0.7, 0.9, 0.95, 0.99)
auth = "Basic " + base64.b64encode(f"{Path(tok).read_text().strip()}:".encode()).decode()
state = json.loads(Path("results/snrtiles_v2_state.json").read_text())
jobs = []
for spec in specs:
    sid, n = (spec.split(":") + ["6"])[:2]
    bl = [(f.parent.name, b) for f in sorted(Path(f"results/snrtiles_v2/{sid}/snr_tiles").glob("*/snr_blocks.json"))
          for b in json.loads(f.read_text())["blocks"] if b["native"].get("snr") is not None]  # fmt: skip
    bl.sort(key=lambda x: x[1]["native"]["snr"])
    pick = np.linspace(0, len(bl) - 1, int(n)).round().astype(int)  # low to high SNR
    jobs += [(sid, state[sid]["ids"][-1], *bl[i]) for i in pick]


def fetch(job):
    sid, cid, name, b = job
    u = (f"https://codeocean.allenneuraldynamics.org/api/v1/computations/{cid}/results/urls?path="
         f"snr_tiles/{name}/block{b['block']:03d}_native.npz")  # fmt: skip
    url = json.load(
        urllib.request.urlopen(
            urllib.request.Request(u, headers={"Authorization": auth}), timeout=60
        )
    )["download_url"]
    z = np.load(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    return sid, name, b, z["raw_mip"].astype(np.float32), z["prob_mip"].astype(np.float32) / 255


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, jobs))
A.apply_style("white")
cols = 1 + len(TH)
fig, axs = plt.subplots(len(res), cols, figsize=(2.1 * cols, 2.2 * len(res)), squeeze=False)
for ax in axs.ravel():
    ax.axis("off")
for r, (sid, name, b, raw, prob) in enumerate(res):
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.7)
    img = np.clip((raw - lo) / max(hi - lo, 1), 0, 1)
    axs[r, 0].imshow(img, cmap="gray", interpolation="nearest")
    n = b["native"]
    axs[r, 0].set_title(A.safe(f"{sid} t{int(name.split('_')[1])} b{b['block']} · SNR {n['snr']:.0f} · "
                               f"{n.get('qc', {}).get('verdict', '')}"), fontsize=7, loc="left")  # fmt: skip
    for c, t in enumerate(TH, start=1):
        m = prob >= t
        rgba = np.zeros((*prob.shape, 4))
        rgba[..., 0], rgba[..., 2], rgba[..., 3] = 1, 1, m * 0.85
        axs[r, c].imshow(img, cmap="gray", interpolation="nearest")
        axs[r, c].imshow(rgba, interpolation="nearest")
        axs[r, c].set_title(
            A.safe(f"p >= {t} · {100 * m.mean():.1f}% of pixels"), fontsize=7, loc="left"
        )
fig.suptitle(A.safe("TRAILMAP detections by probability threshold (max-over-z projections of 1.28 mm "
                    "blocks; rows ordered by SNR within each brain)"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=12)  # fmt: skip
fig.tight_layout(rect=(0, 0, 1, 0.99))
A.savefig(fig, out, dpi=85)
print("saved", out, len(res))
