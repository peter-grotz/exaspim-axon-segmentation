"""Re-score the b side of an snr-pair run at another pyramid level of b's raw tiles, on CPU.

Same patches, same tiles, same mapping as the run; b is read at LEVEL, segmented by TRAILMAP and
scored exactly as a was. Writes a copy of snr_pair.json whose ``b`` is the new level, and MIPs.

Usage: uv run python scripts/pair_level_rescore.py <snr_pair dir> <out dir> <level> <weights>
       [--block K] [--merge]

The level is built by averaging level 2 in float (2^(level-2) per axis), not read from the stored
pyramid: stored levels are rounded to integers, and at level 4 (4096 raw voxels per voxel) the noise
is far below one count, so rounding erases it. With --block K only patch K is scored (one process
per patch keeps memory flat); --merge collects the per-patch results into snr_pair.json.
"""

import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np


def watchdog(limit=10e9):
    while True:
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
from exaspim_axon_segmentation.snr import SnrConfig, _predict, open_ome, score  # noqa: E402
from exaspim_axon_segmentation.snr_pair import PairConfig, RawTiles, _crop, _corners  # noqa: E402

src, out, level, weights = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
d = json.loads((src / "snr_pair.json").read_text())
cfg = PairConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["config"].items()})
aff, t = np.asarray(d["affine"]["A"]), np.asarray(d["affine"]["t"])
tiles = RawTiles(d["raw_tiles"], "capsule/code/snr/791116_5x_bigstitcher_affine.xml",
                 (1.0, 0.748, 0.748))  # fmt: skip
la = open_ome(d["url_a"], cfg.level_a)
norm = InferenceConfig(weights=weights).normalization
pred = TilePredictor(load_model(weights), torch.device("cpu"), torch.float32)
sc = SnrConfig(axon_p=cfg.axon_p)
half = np.asarray(cfg.block_um) / 2
cache = {}
only = int(sys.argv[sys.argv.index("--block") + 1]) if "--block" in sys.argv else None
if "--merge" in sys.argv:
    for b in d["blocks"]:
        f = out / f"row{b['block']:03d}.json"
        if f.exists():
            b["b_native_level2"], b["b"] = b.get("b"), json.loads(f.read_text())
    (out / "snr_pair.json").write_text(json.dumps(d, indent=1))
    print("merged", out / "snr_pair.json")
    sys.exit(0)
fac = 2 ** (level - 2)
for b in d["blocks"]:
    if "b_tile" not in b or (only is not None and b["block"] != only):
        continue
    t0 = time.time()
    raw_a, oa = _crop(la, np.asarray(b["centre_a_um"]), half)
    i = next(j for j, n in tiles.names.items() if n == b["b_tile"])
    if i not in cache:
        cache[i] = (open_ome(tiles.url(i), 0), open_ome(tiles.url(i), 2))
    (_, _, sc0, tr0), lv = cache[i]
    m = np.diag([*sc0, 1.0]) @ tiles.fused_to_voxel(i)
    m[:3, 3] += tr0
    lin, shift = m[:3, :3] @ aff, m[:3, :3] @ (t + np.asarray(b["refine_um"])) + m[:3, 3]
    pts = (lin @ _corners(oa, raw_a.shape, la[2]).T).T + shift
    raw2, ob = _crop(lv, (pts.min(0) + pts.max(0)) / 2, (pts.max(0) - pts.min(0)) / 2)
    cut = tuple(slice(0, (n // fac) * fac) for n in raw2.shape)
    shp = [(n // fac, fac) for n in raw2.shape]
    raw = raw2[cut].reshape(shp[0] + shp[1] + shp[2]).mean((1, 3, 5))  # float: no rounding
    del raw2
    row = {"shape": list(raw.shape), "voxel_um": (lv[2] * fac).tolist(), "level": level,
           "built": f"level 2 averaged {fac}x{fac}x{fac} in float"}  # fmt: skip
    try:
        s = estimate_scaling_array(raw, raw > np.percentile(raw, 10), norm)
        prob = _predict(pred, s(raw), 96)
        row |= score(raw, prob, sc) | {"scaling": [s.offset, s.scale]}
        np.savez_compressed(out / f"block{b['block']:03d}_b.npz", raw_mip=raw.max(0),
                            prob_mip=np.round(prob.max(0) * 255).astype(np.uint8))  # fmt: skip
    except ValueError as error:
        row["skipped"] = f"no foreground: {error}"
    (out / f"row{b['block']:03d}.json").write_text(json.dumps(row))
    f = src / f"block{b['block']:03d}_a.npz"
    if f.exists():
        (out / f.name).write_bytes(f.read_bytes())
    print(b["block"], raw.shape, "SNR", row.get("snr"), "noise", row.get("noise_sd"),
          "%.0fs" % (time.time() - t0), flush=True)  # fmt: skip
