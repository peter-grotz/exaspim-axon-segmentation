"""Download a finished snr-tiles run (JSON and logs) and check it really completed.

Checks: our own exit status (snr_tiles_status.txt), the package commit the capsule ran, every tile's
snr_blocks.json marked complete, blocks per tile, the tissue check's rejections, that each tile with
blocks has probability.zarr and histograms on Code Ocean, and GPU out-of-memory messages in the log.

Usage: uv run python scripts/verify_snrtiles_run.py <token file> <computation id> <dest dir> [commit]
"""

import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

tok, cid, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
want = sys.argv[4] if len(sys.argv) > 4 else None
auth = "Basic " + base64.b64encode(f"{Path(tok).read_text().strip()}:".encode()).decode()
B = "https://codeocean.allenneuraldynamics.org/api/v1/computations/" + cid


def ls(path):
    req = urllib.request.Request(B + "/results", json.dumps({"path": path}).encode(),
                                 {"Authorization": auth, "Content-Type": "application/json"})  # fmt: skip
    return json.load(urllib.request.urlopen(req, timeout=60))["items"]


subprocess.run([sys.executable, "scripts/co_download.py", tok, cid, str(dest), "snr_tiles",
                "snr_tiles_status.txt", "snr_tiles.log", "SOURCE_COMMIT", "--no-npz"],
               check=True, capture_output=True)  # fmt: skip
problems = []
status = (
    (dest / "snr_tiles_status.txt").read_text().strip()
    if (dest / "snr_tiles_status.txt").exists()
    else "missing"
)
if status != "exit 0":
    problems.append(f"status: {status}")
log = (
    (dest / "snr_tiles.log").read_text(errors="replace")
    if (dest / "snr_tiles.log").exists()
    else ""
)
commit = next(
    (ln.split("@")[-1][:7] for ln in log.splitlines() if "exaspim-axon-segmentation.git@" in ln),
    None,
)
src = dest / "SOURCE_COMMIT"
if src.exists():
    commit = src.read_text().strip().split("@")[-1][:7]
if want and commit and commit != want[:7]:
    problems.append(f"commit {commit}, expected {want[:7]}")
n_oom = log.lower().count("out of memory")
tiles = sorted((dest / "snr_tiles").glob("*/snr_blocks.json"))
rows, total, rejected, checked = [], 0, 0, 0
remote = {i["name"]: i for i in ls("snr_tiles")}
for f in tiles:
    d = json.loads(f.read_text())
    n = len(d["blocks"])
    total += n
    tc = d.get("tissue_check", {})
    rejected += tc.get("rejected", 0)
    checked += tc.get("checked", 0)
    if not d.get("complete"):
        problems.append(f"{f.parent.name}: not complete")
    if n:
        names = {i["name"] for i in ls(f"snr_tiles/{f.parent.name}")}
        if "probability.zarr" not in names:
            problems.append(f"{f.parent.name}: no probability.zarr")
        if sum(x.endswith("_hist.npz") for x in names) != n:
            problems.append(
                f"{f.parent.name}: histograms for {sum(x.endswith('_hist.npz') for x in names)} of {n} blocks"
            )
    rows.append((f.parent.name, n, tc.get("checked"), tc.get("rejected"), d.get("selection")))
missing = sorted(set(remote) - {f.parent.name for f in tiles})
if missing:
    problems.append(f"tiles without snr_blocks.json: {missing}")
secs = [b["seconds"] for f in tiles for b in json.loads(f.read_text())["blocks"] if "seconds" in b]
print(f"{cid[:8]}: status {status}, commit {commit}, {len(tiles)} tiles, {total} blocks, "
      f"tissue check rejected {rejected} of {checked}, median {sorted(secs)[len(secs) // 2] if secs else 0:.1f} s/block, "
      f"OOM messages {n_oom}")  # fmt: skip
for r in rows:
    print(f"  {r[0]}: {r[1]} blocks (checked {r[2]}, rejected {r[3]}, {r[4]})")
print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
