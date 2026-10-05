"""Wait for Code Ocean computations to finish: uv run python scripts/co_wait.py <token> <id>..."""

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

auth = "Basic " + base64.b64encode(f"{Path(sys.argv[1]).read_text().strip()}:".encode()).decode()
pending = set(sys.argv[2:])
while pending:
    for cid in sorted(pending):
        url = f"https://codeocean.allenneuraldynamics.org/api/v1/computations/{cid}"
        r = json.load(
            urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": auth}))
        )
        if r["state"] in ("completed", "failed"):
            print(time.strftime("%H:%M"), cid[:8], r["state"], r.get("end_status"),
                  "results" if r.get("has_results") else "NO RESULTS",
                  f"{r.get('run_time', 0) / 60:.0f} min", flush=True)  # fmt: skip
            pending.discard(cid)
    if pending:
        time.sleep(60)
