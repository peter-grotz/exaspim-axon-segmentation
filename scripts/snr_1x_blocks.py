"""Block SNR on a whole-image TRAILMAP result (predict-image), e.g. a binned 1x scan.

Blocks are the same physical size as the 5x snr-tiles blocks (1.28 x 1.44 x 1.44 mm; 64 x 96 x 96
voxels at 20 x 15.04 x 15.04 um), on a regular grid, kept where >= 90% lies in the tissue mask
(Otsu on the log intensity of pyramid level 4, largest connected region), and up to N drawn at
random (fixed seed). Each block's raw voxels come from the image (level 0) and its probabilities
from the run's image/probability.zarr (p x 255, streamed from Code Ocean); it is scored with the
same function as the 5x blocks (exaspim_axon_segmentation.snr._score) at each threshold, plus the
QC signal-present check. Note the voxel-based settings (guard 2 voxels, shell 6, outside 10th
percentile) are in 1x voxels here, i.e. ~5x larger in um than at 5x level 2.

Usage: uv run python scripts/snr_1x_blocks.py <computation id> <image url> <out.json>
       [--n 300] [--thresholds 0.5,0.9] [--threads 8]
"""

import argparse
import base64
import json
import math
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
from scipy.ndimage import label

from exaspim_axon_segmentation.qc.detection_qc import DetectionQcConfig, signal_present
from exaspim_axon_segmentation.snr import SnrConfig, _read, _score, open_ome

p = argparse.ArgumentParser()
p.add_argument("cid")
p.add_argument("url")
p.add_argument("out")
p.add_argument("--n", type=int, default=300)
p.add_argument("--thresholds", default="0.5,0.9")
p.add_argument("--threads", type=int, default=8)
a = p.parse_args()
ts = [float(v) for v in a.thresholds.split(",")]
tok = Path("/Users/peter.grotz/projects/agent/secrets/codeocean_api_token").read_text().strip()
auth = "Basic " + base64.b64encode(f"{tok}:".encode()).decode()
B = "https://codeocean.allenneuraldynamics.org/api/v1"


def retry(fn):
    for k in range(8):
        try:
            return fn()
        except Exception:
            if k == 7:
                raise
            time.sleep(5 * (k + 1))


def get(path):
    q = f"/computations/{a.cid}/results/urls?path={urllib.parse.quote(path)}"
    req = urllib.request.Request(B + q, headers={"Authorization": auth})
    url = retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))["download_url"]
    return retry(lambda: urllib.request.urlopen(url, timeout=120).read())


arr0, lead, sc0, tr0 = open_ome(a.url, 0)
shape0 = np.array(arr0.shape[lead:])
arr4, lead4, sc4, _ = open_ome(a.url, 4)
low = np.asarray(arr4[(0,) * lead4]).astype(np.float32)
# tissue: Otsu on log intensity at level 4, largest region
v = np.log1p(low[low > 0])
h, e = np.histogram(v, 256)
c = (e[:-1] + e[1:]) / 2
w = np.cumsum(h)
m = np.cumsum(h * c)
otsu = c[np.argmax((m[-1] * w - m * w[-1]) ** 2 / np.maximum(w * (w[-1] - w), 1))]
tissue = np.log1p(low) > otsu
lab, n = label(tissue)
tissue = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
f = np.round(sc4 / sc0).astype(int)  # level-0 voxels per level-4 voxel
block = np.round(np.array([1280.0, 1440.0, 1440.0]) / sc0).astype(int)  # 64, 96, 96
cells = block // f  # level-4 cells per block
grid = (np.array(tissue.shape) // cells).astype(int)
sub = tissue[: grid[0] * cells[0], : grid[1] * cells[1], : grid[2] * cells[2]]
frac = sub.reshape(grid[0], cells[0], grid[1], cells[1], grid[2], cells[2]).mean((1, 3, 5))
cand = np.argwhere(frac >= 0.9)
rng = np.random.default_rng(0)
pick = cand[rng.choice(len(cand), min(a.n, len(cand)), replace=False)]
print(f"tissue {100 * tissue.mean():.1f}% of level 4; {len(cand)} qualifying blocks; scoring {len(pick)}",
      flush=True)  # fmt: skip

za = json.loads(get("image/probability.zarr/0/.zarray"))
ch = np.array(za["chunks"])
comp = numcodecs.get_codec(za["compressor"])
chunk_cache: dict = {}


def chunk(idx):
    key = tuple(int(i) for i in idx)
    if key not in chunk_cache:
        try:
            buf = get("image/probability.zarr/0/" + "/".join(map(str, key)))
            chunk_cache[key] = np.frombuffer(comp.decode(buf), np.uint8).reshape(ch)
        except Exception:  # chunk never written (all zero)
            chunk_cache[key] = np.zeros(ch, np.uint8)
    return chunk_cache[key]


def prob_box(lo, hi):
    out = np.zeros(tuple(hi - lo), np.uint8)
    for idx in np.ndindex(*(((hi - 1) // ch) - (lo // ch) + 1)):
        ci = lo // ch + np.array(idx)
        c0 = ci * ch
        a0, a1 = np.maximum(lo, c0), np.minimum(hi, c0 + ch)
        out[tuple(slice(s, t) for s, t in zip(a0 - lo, a1 - lo, strict=True))] = chunk(ci)[
            tuple(slice(s, t) for s, t in zip(a0 - c0, a1 - c0, strict=True))]  # fmt: skip
    return out


def score_block(g):
    lo = g * block
    hi = np.minimum(lo + block, shape0)
    raw = _read(arr0, lead, lo, hi)
    stored = prob_box(lo, hi)
    prob = stored.astype(np.float32) / 255
    row = {"lo_vox": lo.tolist(), "shape": (hi - lo).tolist(),
           "centre_um": (tr0 + (lo + hi) / 2 * sc0).round(1).tolist(),
           "signal_present": signal_present(raw, DetectionQcConfig())}  # fmt: skip
    for t in ts:
        cut = math.floor(t * 255 + 0.5)
        metrics, _ = _score(raw, prob, SnrConfig(axon_p=(cut - 0.5) / 255))
        row[f"p{t:g}"] = metrics
    return row


t0 = time.time()
with ThreadPoolExecutor(a.threads) as ex:
    rows = list(ex.map(score_block, pick))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"url": a.url, "computation": a.cid, "voxel_um": sc0.tolist(),
                                   "block_vox": block.tolist(), "thresholds": ts,
                                   "n_qualifying": int(len(cand)), "blocks": rows}, indent=1))  # fmt: skip
for t in ts:
    s = [r[f"p{t:g}"]["snr"] for r in rows if r[f"p{t:g}"]["snr"] is not None]
    print(f"p >= {t}: {len(s)} blocks with SNR, median {np.median(s):.1f}", flush=True)
print(f"saved {a.out} in {(time.time() - t0) / 60:.1f} min")
