"""Download selected result folders of a Code Ocean computation.

Usage: uv run python scripts/co_download.py <token file> <computation id> <dest> <folder>...

Zarr folders (e.g. an expand run's probability.zarr) are skipped unless --zarr is given;
--no-npz skips .npz files (e.g. snr-tiles preview MIPs).
"""

import base64
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

token = Path(sys.argv[1]).read_text().strip()
with_zarr = "--zarr" in sys.argv
skip_npz = "--no-npz" in sys.argv  # e.g. snr-tiles preview MIPs
cid, dest, roots = sys.argv[2], sys.argv[3], [a for a in sys.argv[4:] if a not in ("--zarr", "--no-npz")]
BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
AUTH = "Basic " + base64.b64encode(f"{token}:".encode()).decode()


def api(path, body=None):
    data = json.dumps(body).encode() if body else None
    headers = {"Authorization": AUTH, "Content-Type": "application/json"}
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, data, headers)))


files, stack = [], []
for top in api(f"/computations/{cid}/results", {"path": ""})["items"]:
    if top["type"] == "file" and top["path"] != "output":
        files.append(top["path"])
    elif top["type"] == "folder" and top["path"] in roots:
        stack.append(top["path"])
while stack:
    for item in api(f"/computations/{cid}/results", {"path": stack.pop()})["items"]:
        if item["type"] == "folder":
            if with_zarr or not item["path"].endswith(".zarr"):
                stack.append(item["path"])
        elif not (skip_npz and item["path"].endswith(".npz")):
            files.append(item["path"])


def get(path):
    url = api(f"/computations/{cid}/results/urls?path={urllib.parse.quote(path)}")["download_url"]
    target = os.path.join(dest, path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    urllib.request.urlretrieve(url, target)


with ThreadPoolExecutor(16) as pool:
    list(pool.map(get, files))
print(cid[:8], len(files), "files ->", dest)
