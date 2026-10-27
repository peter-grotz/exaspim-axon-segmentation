"""Whole-brain view of a predict-image result: image and TRAILMAP detections, full MIP and a thin slab.

Streams the run's image/probability.zarr chunks from Code Ocean and keeps, at full in-plane
resolution, the max over z of the probability for the whole volume and for a slab of ``--slab``
planes at the middle of the tissue. The image is read from the source OME-Zarr: the full MIP from a
coarser level (upsampled to the probability grid), the slab at level 0. Rows: full MIP, slab;
columns: image, detections p >= 0.5, p >= 0.9 (magenta over the image).

Usage: uv run python scripts/plot_wholebrain_1x.py <computation id> <image url> <title> <out.png>
       [--slab 10] [--threads 12]
"""

import argparse
import base64
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numcodecs
import numpy as np

from exaspim_axon_segmentation.snr import _read, open_ome

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, "/Users/peter.grotz/.claude/skills/allen-plotting-style/scripts")
import allen_style as A  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("cid")
p.add_argument("url")
p.add_argument("title")
p.add_argument("out")
p.add_argument("--slab", type=int, default=10)
p.add_argument("--threads", type=int, default=12)
a = p.parse_args()
tok = Path("/Users/peter.grotz/projects/agent/secrets/codeocean_api_token").read_text().strip()
auth = "Basic " + base64.b64encode(f"{tok}:".encode()).decode()
B = "https://codeocean.allenneuraldynamics.org/api/v1"
cache = Path(f"results/wholebrain_1x/{a.cid[:8]}.npz")


def retry(fn):
    for k in range(8):
        try:
            return fn()
        except Exception:
            if k == 7:
                raise
            time.sleep(5 * (k + 1))


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        B + path, data, {"Authorization": auth, "Content-Type": "application/json"}
    )
    return retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))


def get(path):
    url = api(f"/computations/{a.cid}/results/urls?path={urllib.parse.quote(path)}")["download_url"]
    return retry(lambda: urllib.request.urlopen(url, timeout=120).read())


arr0, lead, sc0, tr0 = open_ome(a.url, 0)
shape0 = np.array(arr0.shape[lead:])
if not cache.exists():
    root = "image/probability.zarr"
    za = json.loads(get(f"{root}/0/.zarray"))
    at = json.loads(get(f"{root}/.zattrs"))
    prov = (
        at
        if "region_box" in at
        else next(v for v in at.values() if isinstance(v, dict) and "region_box" in v)
    )
    start = np.array(prov["region_box"]["start"])
    ch = np.array(za["chunks"])
    comp = numcodecs.get_codec(za["compressor"])
    jobs = []
    for z in api(f"/computations/{a.cid}/results", {"path": f"{root}/0"})["items"]:
        if z["type"] != "folder":
            continue
        for y in api(f"/computations/{a.cid}/results", {"path": z["path"]})["items"]:
            if y["type"] != "folder":
                continue
            for x in api(f"/computations/{a.cid}/results", {"path": y["path"]})["items"]:
                if x["type"] == "file":
                    jobs.append((int(z["name"]), int(y["name"]), int(x["name"]), x["path"]))
    print(len(jobs), "chunks", flush=True)
    mip = np.zeros(shape0[1:], np.uint8)
    slab_mip = np.zeros(shape0[1:], np.uint8)
    zc = shape0[0] // 2
    s0, s1 = zc - a.slab // 2, zc - a.slab // 2 + a.slab

    def load(j):
        zi, yi, xi, path = j
        return np.array([zi, yi, xi]), np.frombuffer(comp.decode(get(path)), np.uint8).reshape(ch)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for idx, v in ex.map(load, jobs):
            lo = start + idx * ch
            hi = np.minimum(lo + ch, shape0)
            if np.any(hi <= lo):
                continue
            v = v[: hi[0] - lo[0], : hi[1] - lo[1], : hi[2] - lo[2]]
            ys, xs = slice(lo[1], hi[1]), slice(lo[2], hi[2])
            np.maximum(mip[ys, xs], v.max(0), out=mip[ys, xs])
            a0, a1 = max(s0, lo[0]), min(s1, hi[0])
            if a1 > a0:
                np.maximum(
                    slab_mip[ys, xs], v[a0 - lo[0] : a1 - lo[0]].max(0), out=slab_mip[ys, xs]
                )
            done += 1
            if done % 1000 == 0:
                print(f"{done}/{len(jobs)} chunks · {(time.time() - t0) / 60:.1f} min", flush=True)
    # image: full MIP from a coarser level, slab at level 0
    lvl = 2
    arr2, lead2, _, _ = open_ome(a.url, lvl)
    img_mip = np.zeros(arr2.shape[lead2 + 1 :], np.float32)
    for z in range(0, arr2.shape[lead2], 32):
        sl = np.asarray(arr2[(0,) * lead2 + (slice(z, z + 32),)]).astype(np.float32)
        np.maximum(img_mip, sl.max(0), out=img_mip)
    img_slab = _read(arr0, lead, [s0, 0, 0], [s1, shape0[1], shape0[2]]).max(0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, mip=mip, slab_mip=slab_mip, img_mip=img_mip, img_slab=img_slab,
                        slab=[s0, s1], level=lvl)  # fmt: skip
z = np.load(cache)
mip, slab_mip = z["mip"].astype(np.float32) / 255, z["slab_mip"].astype(np.float32) / 255
img_mip = np.kron(z["img_mip"], np.ones((4, 4), np.float32))[: mip.shape[0], : mip.shape[1]]
img_slab = z["img_slab"]
s0, s1 = z["slab"]


def norm(img, lo_pct, hi_pct):
    lo, hi = np.percentile(img[img > 0] if (img > 0).any() else img, [lo_pct, hi_pct])
    return np.clip((np.log1p(img) - np.log1p(lo)) / max(np.log1p(hi) - np.log1p(lo), 1e-6), 0, 1)


A.apply_style("white")
fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))
ext = [0, mip.shape[1] * sc0[2] / 1e3, mip.shape[0] * sc0[1] / 1e3, 0]
rows = ((img_mip, mip, "whole volume, max over z", 5, 99.5),
        (img_slab, slab_mip, f"slab {(s1 - s0) * sc0[0]:.0f} um at mid depth, max over z", 5, 99.8))  # fmt: skip
for r, (img, prob, name, lo, hi) in enumerate(rows):
    g = norm(img, lo, hi) ** 0.8
    for c, t in enumerate((None, 0.5, 0.9)):
        ax = axs[r, c]
        ax.imshow(g, cmap="gray", extent=ext, interpolation="nearest")
        if t is not None:
            m = prob >= t
            rgba = np.zeros((*m.shape, 4))
            rgba[..., 0], rgba[..., 2], rgba[..., 3] = 1, 1, m * 0.9
            ax.imshow(rgba, extent=ext, interpolation="nearest")
            label = f"detections p >= {t} · {100 * m.mean():.2f}% of pixels"
        else:
            label = "image"
        A.style_ax(ax, grid_axis=None, title=f"{name} · {label}", title_size=12)
        ax.set_xlabel("x (mm)", fontsize=10)
        ax.set_ylabel("y (mm)", fontsize=10)
fig.suptitle(A.safe(f"{a.title} / TRAILMAP (1x fine-tuned) on the whole 1x image"), x=0.01, ha="left",
             fontfamily=A.HEAD, fontweight="bold", fontsize=20)  # fmt: skip
fig.text(0.01, 0.005, A.safe(f"{a.url.split('aind-open-data/')[-1]} · {shape0.tolist()} voxels at "
                             f"{sc0[1]:.2f} x {sc0[2]:.2f} x {sc0[0]:.0f} um · whole-volume image MIP "
                             "from pyramid level 2 · magenta: max probability along z >= threshold"),
         fontsize=10, color=A.C["muted"])  # fmt: skip
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
A.savefig(fig, a.out, dpi=100)
print("saved", a.out)
