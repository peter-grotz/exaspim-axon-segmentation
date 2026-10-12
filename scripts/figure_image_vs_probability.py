"""Image vs TRAILMAP probability on a small fused exaSPIM crop (README figure).

Usage: uv run python scripts/figure_image_vs_probability.py <asset> <x,y,z um> <weights> <out.png>
       [--side-um 800] [--depth-um 256] [--slab-um 40]
Runs on CPU on a crop of ~5 M voxels; a watchdog stops it above 7 GB resident memory.
"""

import argparse
import os
import threading
import time

import matplotlib
import numpy as np
import resource

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("asset")
p.add_argument("centre")
p.add_argument("weights")
p.add_argument("out")
p.add_argument("--side-um", type=float, default=800)
p.add_argument("--depth-um", type=float, default=256)
p.add_argument("--slab-um", type=float, default=40)
a = p.parse_args()


def watchdog(limit=7e9):
    while True:  # peak resident set (bytes on macOS, kB on Linux)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak * (1 if os.uname().sysname == "Darwin" else 1024) > limit:
            print("memory watchdog: stopping", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=watchdog, daemon=True).start()

import torch  # noqa: E402

from exaspim_axon_segmentation.config import InferenceConfig  # noqa: E402
from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array  # noqa: E402
from exaspim_axon_segmentation.inference.predict import TilePredictor  # noqa: E402
from exaspim_axon_segmentation.model import load_model  # noqa: E402
from exaspim_axon_segmentation.snr import _predict, _read, open_ome  # noqa: E402

torch.set_num_threads(4)
x, y, z = (float(v) for v in a.centre.split(","))
arr, lead, sc, tr = open_ome(f"{a.asset.rstrip('/')}/fusion/fused.zarr", 2)
c = np.array([z, y, x])
half = np.array([a.depth_um, a.side_um, a.side_um]) / 2
lo = np.round((c - half - tr) / sc).astype(int)
hi = np.round((c + half - tr) / sc).astype(int)
raw = _read(arr, lead, lo, hi)
print("crop", raw.shape, "voxels", raw.size, flush=True)
norm = InferenceConfig(weights=a.weights).normalization
scaling = estimate_scaling_array(raw, raw > np.percentile(raw, 10), norm)
pred = TilePredictor(load_model(a.weights), torch.device("cpu"), torch.float32)
t0 = time.time()
prob = _predict(pred, scaling(raw), 96)
print("predicted in %.0f s" % (time.time() - t0), flush=True)
np.savez_compressed(a.out.replace(".png", ".npz"), raw=raw, prob=(prob * 255).astype(np.uint8),
                    lo_um=tr + lo * sc, voxel_um=sc)  # fmt: skip
n = int(round(a.slab_um / sc[0]))
mid = raw.shape[0] // 2
img = raw[mid - n // 2 : mid + n // 2].max(0)
pm = prob[mid - n // 2 : mid + n // 2].max(0)
lo_v, hi_v = np.percentile(img, [1, 99.7])
img = np.clip((img - lo_v) / (hi_v - lo_v), 0, 1)
ext = [0, raw.shape[2] * sc[2], raw.shape[1] * sc[1], 0]
fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
ax[0].imshow(img, cmap="gray", extent=ext)
ax[0].set_title("exaSPIM image (fused level 2, 3 µm px)")
ax[1].imshow(pm, cmap="magma", vmin=0, vmax=1, extent=ext)
ax[1].set_title("TRAILMAP axon probability (fine-tuned)")
rgb = np.stack([img, img, img], -1) * 0.75
rgb[..., 0] = np.maximum(rgb[..., 0], pm)
rgb[..., 1] = np.maximum(rgb[..., 1], 0.35 * pm)
ax[2].imshow(np.clip(rgb, 0, 1), extent=ext)
ax[2].set_title("overlay: probability in orange")
for x_ in ax:
    x_.set_xlabel("µm")
fig.suptitle(f"{os.path.basename(a.asset.rstrip('/')).split('_processed')[0]} · {a.slab_um:.0f} µm "
             f"slab around x {x:.0f}, y {y:.0f}, z {z:.0f} µm", y=0.99)  # fmt: skip
fig.tight_layout()
fig.savefig(a.out, dpi=100)
print("wrote", a.out)
