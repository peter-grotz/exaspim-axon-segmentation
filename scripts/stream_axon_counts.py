"""Stream a Code Ocean probability.zarr result and count detected axon voxels per coarse bin.

Bins are ``bin`` level-0 voxels per axis (16 for a 1x tile: 80 x 60 x 60 um, the registration
grid), aligned to the image's level-0 origin. Writes counts of p >= threshold and the summed
probability per bin, plus the number of predicted voxels per bin.

Usage: uv run python scripts/stream_axon_counts.py <computation id> <level-0 shape z,y,x> <out.npz>
       [--bin 16] [--threshold 128] [--threads 12]
"""

import argparse
import base64
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("cid")
p.add_argument("shape")
p.add_argument("out")
p.add_argument("--bin", type=int, default=16)
p.add_argument("--threshold", type=int, default=128)
p.add_argument("--threads", type=int, default=12)
a = p.parse_args()
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


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(B + path, data, {"Authorization": auth, "Content-Type": "application/json"})
    return retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))


def get(path):
    url = api(f"/computations/{a.cid}/results/urls?path={urllib.parse.quote(path)}")["download_url"]
    return retry(lambda: urllib.request.urlopen(url, timeout=120).read())


root = "image/probability.zarr"
za = json.loads(get(f"{root}/0/.zarray"))
at = json.loads(get(f"{root}/.zattrs"))
prov = at if "region_box" in at else next(v for v in at.values() if isinstance(v, dict) and "region_box" in v)
start = np.array(prov["region_box"]["start"])
ch = np.array(za["chunks"])
comp = numcodecs.get_codec(za["compressor"])
shape0 = np.array([int(v) for v in a.shape.split(",")])
nb = -(-shape0 // a.bin)
det = np.zeros(nb, np.uint32)
psum = np.zeros(nb, np.float32)
seen = np.zeros(nb, np.uint32)
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


def load(j):
    zi, yi, xi, path = j
    v = np.frombuffer(comp.decode(get(path)), np.uint8).reshape(ch)
    return np.array([zi, yi, xi]), v


t0, done = time.time(), 0
with ThreadPoolExecutor(a.threads) as ex:
    for idx, v in ex.map(load, jobs):
        lo = start + idx * ch  # level-0 voxel of the chunk's first voxel
        hi = np.minimum(lo + ch, shape0)
        if np.any(hi <= lo):
            continue
        v = v[: hi[0] - lo[0], : hi[1] - lo[1], : hi[2] - lo[2]]
        off = lo % a.bin
        size = -(-(off + v.shape) // a.bin) * a.bin
        buf = np.zeros(tuple(size), np.float32)
        sl = tuple(slice(o, o + s) for o, s in zip(off, v.shape, strict=True))
        b0 = lo // a.bin
        n = size // a.bin
        dest = tuple(slice(s, s + k) for s, k in zip(b0, n, strict=True))

        def pooled(x):
            buf[:] = 0
            buf[sl] = x
            return buf.reshape(n[0], a.bin, n[1], a.bin, n[2], a.bin).sum((1, 3, 5))

        det[dest] += pooled(v >= a.threshold).astype(np.uint32)
        psum[dest] += pooled(v / 255.0)
        seen[dest] += pooled(np.ones(v.shape, np.float32)).astype(np.uint32)
        done += 1
        if done % 2000 == 0:
            print(f"{done}/{len(jobs)} chunks · {(time.time() - t0) / 60:.0f} min", flush=True)
np.savez_compressed(a.out, detected=det, prob_sum=psum, predicted=seen, bin=a.bin,
                    threshold=a.threshold, region_start=start)  # fmt: skip
print("wrote", a.out, "detected voxels", int(det.sum()), "predicted", int(seen.sum()))
