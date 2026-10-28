"""Save every voxel of the scored 1x blocks: raw counts, stored probability and the tissue mask.

For each block in a snr_1x_blocks.py result (same boxes), the raw voxels come from the image (level
0), the probabilities (p x 255) from the run's image/probability.zarr on Code Ocean, and the tissue
mask is the one the blocks were chosen with (Otsu on log intensity at pyramid level 4, largest
connected region), taken onto level 0 by nearest neighbour. As a check, each block is rescored with
snr._score at p >= 0.9 and compared with the saved SNR.

Output: <out dir>/<subject>.npz with raw_NNN (uint16), prob_NNN (uint8, p x 255), tissue_NNN
(bool) per block, and lo_vox, shape, centre_um, voxel_um; <out dir>/<subject>.json with the check.

Usage: uv run python scripts/dump_1x_blocks.py <out dir> <subject>:<snr_1x json> ... [--threads 8]
"""

import argparse
import base64
import functools
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

from exaspim_axon_segmentation.snr import SnrConfig, _read, _score, open_ome

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("specs", nargs="+")
p.add_argument("--threads", type=int, default=8)
a = p.parse_args()
tok = Path("/Users/peter.grotz/projects/agent/secrets/codeocean_api_token").read_text().strip()
auth = "Basic " + base64.b64encode(f"{tok}:".encode()).decode()
B = "https://codeocean.allenneuraldynamics.org/api/v1"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def retry(fn):
    for k in range(8):
        try:
            return fn()
        except Exception:
            if k == 7:
                raise
            time.sleep(5 * (k + 1))


def tissue_mask(url):
    """Otsu on log intensity at level 4, largest region (as in snr_1x_blocks.py)."""
    arr4, lead4, sc4, _ = open_ome(url, 4)
    low = np.asarray(arr4[(0,) * lead4]).astype(np.float32)
    v = np.log1p(low[low > 0])
    h, e = np.histogram(v, 256)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    otsu = c[np.argmax((m[-1] * w - m * w[-1]) ** 2 / np.maximum(w * (w[-1] - w), 1))]
    t = np.log1p(low) > otsu
    lab, _ = label(t)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1), sc4


for spec in a.specs:
    sid, js = spec.split(":")
    d = json.loads(Path(js).read_text())
    cid, url = d["computation"], d["url"]

    def get(path, cid=cid):
        q = f"/computations/{cid}/results/urls?path={urllib.parse.quote(path)}"
        req = urllib.request.Request(B + q, headers={"Authorization": auth})
        u = retry(lambda: json.load(urllib.request.urlopen(req, timeout=60)))["download_url"]
        return retry(lambda: urllib.request.urlopen(u, timeout=120).read())

    arr0, lead, sc0, _ = open_ome(url, 0)
    tissue, sc4 = tissue_mask(url)
    f = np.round(sc4 / sc0).astype(int)
    za = json.loads(get("image/probability.zarr/0/.zarray"))
    ch = np.array(za["chunks"])
    comp = numcodecs.get_codec(za["compressor"])

    @functools.lru_cache(maxsize=256)
    def chunk(key, get=get, ch=ch, comp=comp):
        try:
            buf = get("image/probability.zarr/0/" + "/".join(map(str, key)))
            return np.frombuffer(comp.decode(buf), np.uint8).reshape(ch)
        except Exception:  # chunk never written (all zero)
            return np.zeros(ch, np.uint8)

    def prob_box(lo, hi, ch=ch, chunk=chunk):
        o = np.zeros(tuple(hi - lo), np.uint8)
        for idx in np.ndindex(*(((hi - 1) // ch) - (lo // ch) + 1)):
            ci = lo // ch + np.array(idx)
            c0 = ci * ch
            a0, a1 = np.maximum(lo, c0), np.minimum(hi, c0 + ch)
            src = chunk(tuple(int(i) for i in ci))
            o[tuple(slice(s, t) for s, t in zip(a0 - lo, a1 - lo, strict=True))] = src[
                tuple(slice(s, t) for s, t in zip(a0 - c0, a1 - c0, strict=True))]  # fmt: skip
        return o

    def one(b, arr0=arr0, lead=lead, tissue=tissue, f=f, prob_box=prob_box):
        lo = np.array(b["lo_vox"])
        hi = lo + np.array(b["shape"])
        raw = _read(arr0, lead, lo, hi)
        stored = prob_box(lo, hi)
        idx = [
            np.minimum(np.arange(s, t) // k, n - 1)
            for s, t, k, n in zip(lo, hi, f, tissue.shape, strict=True)
        ]
        ts = tissue[np.ix_(*idx)]
        cut = math.floor(0.9 * 255 + 0.5)
        m, _ = _score(raw, stored.astype(np.float32) / 255, SnrConfig(axon_p=(cut - 0.5) / 255))
        return raw.astype(np.uint16), stored, ts, m["snr"], b["p0.9"]["snr"]

    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(one, d["blocks"]))
    arrays = {}
    for i, (raw, stored, ts, _, _) in enumerate(res):
        arrays |= {f"raw_{i:03d}": raw, f"prob_{i:03d}": stored, f"tissue_{i:03d}": ts}
    np.savez_compressed(out / f"{sid}.npz", **arrays,
                        lo_vox=np.array([b["lo_vox"] for b in d["blocks"]]),
                        shape=np.array([b["shape"] for b in d["blocks"]]),
                        centre_um=np.array([b["centre_um"] for b in d["blocks"]]),
                        voxel_um=np.asarray(sc0))  # fmt: skip
    diffs = [abs(r[3] - r[4]) / abs(r[4]) for r in res if r[3] is not None and r[4] is not None]
    mism = sum((r[3] is None) != (r[4] is None) for r in res)
    check = {"subject": sid, "source": js, "url": url, "computation": cid, "blocks": len(res),
             "max_rel_diff_snr_p0.9": max(diffs) if diffs else None, "none_mismatches": mism,
             "tissue_fraction_in_blocks": float(np.mean([r[2].mean() for r in res])),
             "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
    (out / f"{sid}.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check), flush=True)
    del res, arrays
