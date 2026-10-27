"""Download a finished rescore run (JSON, logs) and check it against the original snr-tiles run.

Checks: exit status, every tile with scored blocks rescored, the same blocks, and that the p >= 0.5
rescoring reproduces the original block SNR. Not exactly: neighbouring blocks share their 1-2
outermost voxel planes (block edges are rounded outward), and probability.zarr keeps the block written
last there, so a block's rescored edge planes can hold its neighbour's prediction. Expected: median
SNR difference ~1e-4, a few blocks with few detections more; flagged if the median exceeds 1e-3 or
QC verdicts change in more than 1% of blocks.

Usage: uv run python scripts/verify_rescore.py <token file> <computation id> <dest dir> <original dir>
"""

import json
import subprocess
import sys
from pathlib import Path

tok, cid, dest, orig = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
import time  # noqa: E402

for attempt in range(5):  # results can still be syncing just after a run completes
    if subprocess.run([sys.executable, "scripts/co_download.py", tok, cid, str(dest), "rescore",
                       "rescore_status.txt", "rescore.log", "--no-npz"],
                      capture_output=True).returncode == 0:  # fmt: skip
        break
    time.sleep(60 * (attempt + 1))
problems = []
status = (
    (dest / "rescore_status.txt").read_text().strip()
    if (dest / "rescore_status.txt").exists()
    else "missing"
)
if status != "exit 0":
    problems.append(f"status: {status}")
import numpy as np  # noqa: E402

rel, n_blocks, n_verdict, n_tiles = [], 0, 0, 0
for f in sorted((orig / "snr_tiles").glob("*/snr_blocks.json")):
    o = {b["block"]: b for b in json.loads(f.read_text())["blocks"] if "skipped" not in b["native"]}
    if not o:
        continue
    r = dest / "rescore" / f.parent.name / "snr_rescore.json"
    if not r.exists():
        problems.append(f"{f.parent.name}: not rescored")
        continue
    n_tiles += 1
    new = {b["block"]: b for b in json.loads(r.read_text())["blocks"]}
    if set(new) != set(o):
        problems.append(f"{f.parent.name}: blocks differ ({len(new)} vs {len(o)})")
    for k, b in o.items():
        if k not in new:
            continue
        n_blocks += 1
        a, c = b["native"], new[k]["p0.5"]
        if a.get("snr") is not None and c.get("snr") is not None:
            rel.append(abs(a["snr"] - c["snr"]) / max(abs(a["snr"]), 1e-12))
        elif (a.get("snr") is None) != (c.get("snr") is None):
            n_verdict += 1
        if a.get("qc", {}).get("verdict") != c["qc"].get("verdict"):
            n_verdict += 1
rel = np.array(rel) if rel else np.zeros(1)
if np.median(rel) > 1e-3:
    problems.append(f"p0.5 SNR differs from the original: median {np.median(rel):.1e}")
if n_verdict > 0.01 * max(n_blocks, 1):
    problems.append(f"QC verdict or SNR presence changed in {n_verdict} of {n_blocks} blocks")
print(f"{cid[:8]}: status {status}, {n_tiles} tiles, {n_blocks} blocks, p0.5 vs original SNR "
      f"difference median {np.median(rel):.1e}, p90 {np.percentile(rel, 90):.1e}, "
      f"max {rel.max():.1e}; verdict changes {n_verdict}")  # fmt: skip
print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems[:20]))
