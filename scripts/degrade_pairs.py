"""Extract aligned 1x / averaged-5x block pairs (same tissue, 1x grid) to measure the 1x degradation.

Uses a registered snr-pair run (whole-brain affine + local refinement per block). Per block: the 1x
block and the 5x raw tile at level 2 averaged in float over each 1x voxel's footprint, resampled
onto the 1x grid. One block per call keeps memory flat.

Usage: uv run python scripts/degrade_pairs.py <registered snr_pair dir> <out dir> <block>
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from exaspim_axon_segmentation.snr import open_ome
from exaspim_axon_segmentation.snr_pair import PairConfig, RawTiles, _corners, _crop, _to_a

src, out, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
d = json.loads((src / "snr_pair.json").read_text())
b = next(x for x in d["blocks"] if x["block"] == k)
cfg = PairConfig(**{kk: tuple(v) if isinstance(v, list) else v for kk, v in d["config"].items()})
aff, t = np.asarray(d["affine"]["A"]), np.asarray(d["affine"]["t"])
tiles = RawTiles(d["raw_tiles"], "capsule/code/snr/791116_5x_bigstitcher_affine.xml", (1.0, 0.748, 0.748))
la = open_ome(d["url_a"], cfg.level_a)
raw_a, oa = _crop(la, np.asarray(b["centre_a_um"]), np.asarray(cfg.block_um) / 2)
i = next(j for j, n in tiles.names.items() if n == b["b_tile"])
(_, _, sc0, tr0), lv = open_ome(tiles.url(i), 0), open_ome(tiles.url(i), 2)
m = np.diag([*sc0, 1.0]) @ tiles.fused_to_voxel(i)
m[:3, 3] += tr0
lin, shift = m[:3, :3] @ aff, m[:3, :3] @ (t + np.asarray(b["refine_um"])) + m[:3, 3]
pts = (lin @ _corners(oa, raw_a.shape, la[2]).T).T + shift
raw_b, ob = _crop(lv, (pts.min(0) + pts.max(0)) / 2, (pts.max(0) - pts.min(0)) / 2 + 4 * lv[2])
size = tuple(int(x) for x in np.maximum(np.round(la[2] / lv[2]).astype(int), 1))
b_on_a = _to_a(uniform_filter(raw_b, size=size), ob, lv[2], lin, shift, oa, la[2], raw_a.shape)
inside = _to_a(np.ones(raw_b.shape, np.float32), ob, lv[2], lin, shift, oa, la[2], raw_a.shape,
               order=0) > 0  # fmt: skip
np.savez_compressed(out / f"pair{k:03d}.npz", one=raw_a.astype(np.float32),
                    five=b_on_a.astype(np.float32), inside=inside, voxel_um=la[2])  # fmt: skip
print(k, raw_a.shape, "inside %.2f" % inside.mean(), flush=True)
