"""Launch Code Ocean computations with at most N running at once (relaunching fast failures).

Usage: uv run python scripts/co_queue.py <token file> <jobs.json> <state.json> [--max 8] [--gap 20]
jobs.json: {name: {"parameters": [...], "data_assets": [...]}}; capsule id inside as "_capsule".
A run that ends "failed" with no results within 5 minutes is treated as not started and retried
(up to 3 times). Launches are --gap seconds apart. State (run ids, outcomes) is written to
state.json after every poll.
"""

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

tok, jobs_file, state_file = sys.argv[1:4]
limit = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 8
gap = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 20
auth = "Basic " + base64.b64encode(f"{Path(tok).read_text().strip()}:".encode()).decode()
B = "https://codeocean.allenneuraldynamics.org/api/v1"


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(B + path, data, {"Authorization": auth, "Content-Type": "application/json"})
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except OSError:
            time.sleep(10 * (k + 1))
    raise RuntimeError(path)


jobs = json.loads(Path(jobs_file).read_text())
cap = jobs.pop("_capsule")
state = json.loads(Path(state_file).read_text()) if Path(state_file).exists() else {}
for n in jobs:
    state.setdefault(n, {"tries": 0, "status": "pending", "ids": []})
while True:
    running = 0
    for n, s in state.items():
        if s["status"] != "running":
            continue
        r = api(f"/computations/{s['ids'][-1]}")
        if r["state"] != "completed":
            running += 1
            continue
        quick_fail = r.get("end_status") == "failed" and not r.get("has_results") and r.get("run_time", 0) < 300
        if quick_fail and s["tries"] < 3:
            s["status"] = "pending"
            print(time.strftime("%H:%M"), n, "failed to start; will retry", flush=True)
        else:
            s["status"] = f"{r.get('end_status')} ({'results' if r.get('has_results') else 'NO RESULTS'}, {r.get('run_time', 0) / 60:.0f} min)"
            print(time.strftime("%H:%M"), n, s["status"], flush=True)
    for n, s in state.items():
        if s["status"] == "pending" and running < limit:
            r = api("/computations", {"capsule_id": cap} | jobs[n])
            s["ids"].append(r["id"])
            s["tries"] += 1
            s["status"] = "running"
            running += 1
            print(time.strftime("%H:%M"), n, "launched", r["id"], flush=True)
            Path(state_file).write_text(json.dumps(state, indent=1))
            time.sleep(gap)
    Path(state_file).write_text(json.dumps(state, indent=1))
    if all(s["status"] not in ("pending", "running") for s in state.values()):
        break
    time.sleep(300)
print("all done")
