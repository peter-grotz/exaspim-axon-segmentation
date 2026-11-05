"""Run TRAILMAP on a SLAP2 structural stack resampled to a chosen voxel size, scored against an SNT trace.

The green channel of the REFERENCE tif (0.25 x 0.25 x 1 um, planes interleaved green/red) is low-passed
(gaussian, sigma = 0.4 x the downsampling factor per axis) and resampled (linear) to each requested grid
(z, y, x um), restricted to the imaged field (columns with signal). Intensities use the repo's foreground
scaling over the whole field. The network runs in 96^3 tiles (68^3 out) on the CPU, fp32.

Scoring against the trace (pixel coords, densified to 0.25 um steps): recall = share of trace points with
p >= 0.5 within `tol` voxels; on-trace share = share of p >= 0.5 voxels within `tol` voxels of the trace (a lower
bound on precision: other labelled cells are in the image but not traced).

Usage:
  uv run python scripts/slap2_trailmap.py <REFERENCE.tif> <swc> <out.npz> \
      --grid 4,3,3 --grid 1.5,1.5,1.5 --weights trailmap-axon-210409 --weights <best.safetensors>
"""

import argparse
import json
import resource
import time

import numpy as np
import tifffile
import torch
from scipy.ndimage import binary_dilation, gaussian_filter, zoom

from exaspim_axon_segmentation.config import NormalizationConfig
from exaspim_axon_segmentation.inference.normalize import scaling_from_samples
from exaspim_axon_segmentation.model import load_model

p = argparse.ArgumentParser()
p.add_argument("tif")
p.add_argument("swc")
p.add_argument("out")
p.add_argument("--grid", action="append", required=True, help="target voxel z,y,x in um")
p.add_argument("--weights", action="append", required=True)
p.add_argument("--tol", type=int, default=1)
a = p.parse_args()
SRC = np.array([1.0, 0.25, 0.25])  # z, y, x um
IN, CROP = 96, 14
torch.set_num_threads(4)

n_pages = len(tifffile.TiffFile(a.tif).pages)
green = tifffile.imread(a.tif, key=range(0, n_pages, 2))  # green planes (interleaved, see slap2_view_swc.py)
cols = np.flatnonzero(np.abs(green).max((0, 1)) > 0)  # the imaged field (blank bands at the x edges)
x0, x1 = int(cols[0]), int(cols[-1]) + 1
green = green[:, :, x0:x1]
sw = np.loadtxt(a.swc, comments="#")
idx = {int(r[0]): k for k, r in enumerate(sw)}
pts = []
for k, par in enumerate(sw[:, 6]):  # densify every segment to 0.25 um steps, (z, y, x) um
    q = np.array([sw[k, 4] * SRC[0], sw[k, 3] * SRC[1], (sw[k, 2] - x0) * SRC[2]])
    if int(par) in idx:
        j = idx[int(par)]
        r = np.array([sw[j, 4] * SRC[0], sw[j, 3] * SRC[1], (sw[j, 2] - x0) * SRC[2]])
        n = max(int(np.ceil(np.linalg.norm(q - r) / 0.25)), 1)
        pts.append(r + (q - r) * np.linspace(0, 1, n + 1)[1:, None])
    else:
        pts.append(q[None])
pts = np.concatenate(pts)
print("field x", x0, x1, "stack", green.shape, "trace points", len(pts), flush=True)


def predict(model, vol):
    pad = [(CROP, CROP + (-(s) % (IN - 2 * CROP))) for s in vol.shape]
    v = np.pad(vol, pad, mode="reflect")
    out = np.zeros([s - 2 * CROP for s in v.shape], np.float32)
    step = IN - 2 * CROP
    for z in range(0, out.shape[0], step):
        for y in range(0, out.shape[1], step):
            for x in range(0, out.shape[2], step):
                t = v[z : z + IN, y : y + IN, x : x + IN]
                if t.shape != (IN, IN, IN):  # last tile: pad up (only for volumes < one tile)
                    t = np.pad(t, [(0, IN - s) for s in t.shape], mode="reflect")
                with torch.inference_mode():
                    o = model(torch.from_numpy(np.ascontiguousarray(t))[None, None]).numpy()[0, 0]
                sz = out[z : z + step, y : y + step, x : x + step].shape
                out[z : z + step, y : y + step, x : x + step] = o[: sz[0], : sz[1], : sz[2]]
    return out[: vol.shape[0], : vol.shape[1], : vol.shape[2]]


res, saved = [], {}
for g in a.grid:
    tgt = np.array([float(x) for x in g.split(",")])
    f = tgt / SRC
    vol = zoom(gaussian_filter(green, 0.4 * f), 1 / f, order=1).astype(np.float32)
    cfg = NormalizationConfig(foreground_min_counts=1.0)
    sc = scaling_from_samples(vol.ravel()[:: max(vol.size // 2_000_000, 1)], cfg)
    x = sc(vol)
    lab = np.zeros(vol.shape, bool)
    vi = np.clip(np.floor(pts / tgt).astype(int), 0, np.array(vol.shape) - 1)
    lab[tuple(vi.T)] = True
    near = binary_dilation(lab, np.ones((3, 3, 3), bool), iterations=a.tol)
    key = g.replace(",", "x")
    saved[f"x_{key}"] = x.astype(np.float16)
    saved[f"label_{key}"] = lab
    for w in a.weights:
        t0 = time.time()
        prob = predict(load_model(w), x)
        hit = prob >= 0.5
        hit_near = binary_dilation(hit, np.ones((3, 3, 3), bool), iterations=a.tol)
        r = {
            "grid_um": tgt.tolist(),
            "shape": list(vol.shape),
            "weights": w.split("/")[-3] + "/" + w.split("/")[-1] if w.endswith(".safetensors") else w,
            "scaling": sc.as_dict(),
            "recall": float(hit_near[tuple(vi.T)].mean()),
            "on_trace_share": float((hit & near).sum() / max(hit.sum(), 1)),
            "positive_fraction": float(hit.mean()),
            "trace_voxel_fraction": float(lab.mean()),
            "seconds": round(time.time() - t0, 1),
            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        }
        print(json.dumps(r), flush=True)
        res.append(r)
        saved[f"prob_{key}_{len(res) - 1}"] = prob.astype(np.float16)
np.savez_compressed(a.out, results=json.dumps(res), x0=x0, **saved)
print("saved", a.out)
