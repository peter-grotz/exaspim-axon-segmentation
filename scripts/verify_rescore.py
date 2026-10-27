"""Download a finished rescore run (JSON, logs) and check it against the original snr-tiles run.

Checks: exit status, every tile with scored blocks rescored, the same blocks, and that the p >= 0.5
rescoring reproduces the original block metrics (stored probabilities are 8-bit; at 0.5 the
detected voxels are identical, so differences should be at rounding level).

Usage: uv run python scripts/verify_rescore.py <token file> <computation id> <dest dir> <original dir>
"""

import json
import subprocess
import sys
from pathlib import Path

tok, cid, dest, orig = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
subprocess.run([sys.executable, "scripts/co_download.py", tok, cid, str(dest), "rescore",
                "rescore_status.txt", "rescore.log", "--no-npz"], check=True, capture_output=True)  # fmt: skip
problems = []
status = (
    (dest / "rescore_status.txt").read_text().strip()
    if (dest / "rescore_status.txt").exists()
    else "missing"
)
if status != "exit 0":
    problems.append(f"status: {status}")
keys = ("snr", "local_snr", "median_snr", "signal", "background", "noise_sd", "axon_voxels")
worst, n_blocks, n_verdict, n_tiles = 0.0, 0, 0, 0
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
        for key in keys:
            if a.get(key) is not None and c.get(key) is not None:
                worst = max(worst, abs(a[key] - c[key]) / max(abs(a[key]), 1e-12))
            elif (a.get(key) is None) != (c.get(key) is None):
                problems.append(f"{f.parent.name} b{k}: {key} {a.get(key)} vs {c.get(key)}")
        if a.get("qc", {}).get("verdict") != c["qc"].get("verdict"):
            n_verdict += 1
if worst > 1e-6:
    problems.append(f"p0.5 differs from the original by up to {worst:.2e}")
print(f"{cid[:8]}: status {status}, {n_tiles} tiles, {n_blocks} blocks, p0.5 vs original max "
      f"relative difference {worst:.1e}, QC verdict changes {n_verdict}")  # fmt: skip
print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems[:20]))
