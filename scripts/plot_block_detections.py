"""Block MIPs with TRAILMAP detections overlaid, from an snr-tiles run's saved per-block previews.

For each tile, up to N evenly spaced sampled blocks: raw MIP (own contrast) and the same with the
probability MIP overlaid (magenta, opacity = p, shown where p >= 0.5). The per-block previews
(block###_native.npz: raw_mip, prob_mip at the segmentation level) are fetched from Code Ocean.

Usage: uv run python scripts/plot_block_detections.py <token file> <subject> <computation id> <out.png> [N] [p]
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

tok, sid, cid, out = sys.argv[1:5]
per_tile = int(sys.argv[5]) if len(sys.argv) > 5 else 4
thr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
auth = "Basic " + base64.b64encode(f"{Path(tok).read_text().strip()}:".encode()).decode()
B = f"https://codeocean.allenneuraldynamics.org/api/v1/computations/{cid}/results/urls?path="
jobs = []
for f in sorted(Path(f"results/snrtiles_v2/{sid}/snr_tiles").glob("*/snr_blocks.json")):
    bl = json.loads(f.read_text())["blocks"]
    bl = [b for b in bl if "skipped" not in b["native"]]
    for b in bl[:: max(1, len(bl) // per_tile)][:per_tile]:
        jobs.append((f.parent.name, b))


def fetch(job):
    name, b = job
    path = f"snr_tiles/{name}/block{b['block']:03d}_native.npz"
    req = urllib.request.Request(B + path, headers={"Authorization": auth})
    url = json.load(urllib.request.urlopen(req, timeout=60))["download_url"]
    z = np.load(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
    return name, b, z["raw_mip"].astype(np.float32), z["prob_mip"].astype(np.float32) / 255


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, jobs))
A.apply_style("white")
cols = 8  # 4 blocks per row, each raw | overlay
rows = -(-len(res) * 2 // cols)
fig, axs = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.15 * rows), squeeze=False)
for ax in axs.ravel():
    ax.axis("off")
for k, (name, b, raw, prob) in enumerate(res):
    a0, a1 = axs.ravel()[2 * k], axs.ravel()[2 * k + 1]
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.7)
    img = np.clip((raw - lo) / max(hi - lo, 1), 0, 1)
    a0.imshow(img, cmap="gray", interpolation="nearest")
    a1.imshow(img, cmap="gray", interpolation="nearest")
    rgba = np.zeros((*prob.shape, 4))
    rgba[..., 0], rgba[..., 2] = 1.0, 1.0
    rgba[..., 3] = np.where(prob >= thr, 0.9, 0)
    a1.imshow(rgba, interpolation="nearest")
    n = b["native"]
    snr = "-" if n.get("snr") is None else f"{n['snr']:.0f}"
    a0.set_title(A.safe(f"tile {int(name.split('_')[1])} · b{b['block']}"), fontsize=7, loc="left")
    a1.set_title(A.safe(f"SNR(0.5) {snr} · {n.get('qc', {}).get('verdict', '')} · "
                        f"{100 * (prob >= thr).mean():.1f}% px"), fontsize=7, loc="left")  # fmt: skip
fig.suptitle(A.safe(f"{sid} / TRAILMAP detections, {per_tile} blocks per tile (MIP over z of each "
                    f"1.28 mm block; left image, right detections p >= {thr} in magenta; % px = share of the "
                    "projection detected)"),
             x=0.01, ha="left", fontfamily=A.HEAD, fontweight="bold", fontsize=13)  # fmt: skip
fig.tight_layout(rect=(0, 0, 1, 0.985))
A.savefig(fig, out, dpi=80)
print("saved", out, len(res), "blocks")
