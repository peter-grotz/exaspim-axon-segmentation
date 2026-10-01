"""Run the pipeline capsule over many assets, a few at a time, and collect the light outputs.

Usage::

    uv run python scripts/batch_codeocean.py <token file> <jobs.json> <out dir> [--parallel 8]

``jobs.json`` lists ``{"subject", "asset", "annotation"}``. Progress is kept in
``<out dir>/state.json`` so the script can be restarted; finished runs have their skeleton folder
and QC figures downloaded to ``<out dir>/<subject>/`` (not the probability maps).
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
CAPSULE = "cde3a47b-a776-4a6c-94c3-e070c2092cc0"
WEIGHTS = "/data/trailmap_finetune_v2/best.safetensors"
MAX_ATTEMPTS = 5
RETRY_WAIT_S = 15 * 60
MODEL_ASSET = {"id": "52a5d51b-5ba7-4e5a-a923-65e4dd04c359", "mount": "trailmap_finetune_v2"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("token", type=Path)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    auth = "Basic " + base64.b64encode(f"{args.token.read_text().strip()}:".encode()).decode()

    def api(path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": auth, "Content-Type": "application/json"}
        return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, data, headers)))

    jobs = json.loads(args.jobs.read_text())
    state_path = args.out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    while True:
        running = [s for s, v in state.items() if v["state"] == "running"]
        for subject in running:
            r = api(f"/computations/{state[subject]['id']}")
            if r["state"] in ("completed", "failed"):
                ok = r.get("end_status") == "succeeded" and r.get("has_results")
                attempts = state[subject].get("attempts", 1)
                if not ok and not r.get("has_results") and attempts < MAX_ATTEMPTS:
                    # usually "Selected Machine Unavailable": no GPU free, nothing ran; retry later
                    print(time.strftime("%H:%M"), subject, "not started; will retry", flush=True)
                    state[subject] = {"state": "retry", "attempts": attempts,
                                      "after": time.time() + RETRY_WAIT_S * attempts}  # fmt: skip
                    continue
                state[subject]["state"] = "done" if ok else "failed"
                if ok:
                    tag = f"{subject}_MD_L2"
                    subprocess.run([sys.executable, "scripts/co_download.py", str(args.token),
                                    state[subject]["id"], str(args.out / subject),
                                    f"{tag}_skeleton", "qc"], check=False)  # fmt: skip
                print(time.strftime("%H:%M"), subject, state[subject]["state"], flush=True)
        todo = [
            j for j in jobs
            if state.get(j["subject"], {}).get("state", "retry") == "retry"
            and state.get(j["subject"], {}).get("after", 0) <= time.time()
        ]  # fmt: skip
        waiting = any(v["state"] == "retry" for v in state.values())
        free = args.parallel - sum(v["state"] == "running" for v in state.values())
        for job in todo[: max(free, 0)]:
            params = ["predict", job["asset"], "MD", "2", "0", "160", "1", WEIGHTS, "0.2",
                      job["annotation"]]  # fmt: skip
            r = api("/computations", {"capsule_id": CAPSULE, "parameters": params,
                                      "data_assets": [MODEL_ASSET]})  # fmt: skip
            attempts = state.get(job["subject"], {}).get("attempts", 0) + 1
            state[job["subject"]] = {"id": r["id"], "state": "running", "asset": job["asset"],
                                     "attempts": attempts}  # fmt: skip
            print(time.strftime("%H:%M"), job["subject"], "launched", r["id"], flush=True)
        state_path.write_text(json.dumps(state, indent=1))
        if all(v["state"] != "running" for v in state.values()) and not todo and not waiting:
            break
        time.sleep(60)
    print("all done:", sum(v["state"] == "done" for v in state.values()), "succeeded,",
          sum(v["state"] == "failed" for v in state.values()), "failed")  # fmt: skip


if __name__ == "__main__":
    main()
