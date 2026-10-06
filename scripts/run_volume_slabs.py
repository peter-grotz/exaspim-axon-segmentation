"""Segment and skeletonize a large box on Code Ocean: slabs in parallel, then one skeleton run.

Usage: uv run python scripts/run_volume_slabs.py <token file> <state.json>

State (slab boxes, computation ids, data assets, final run) is kept in state.json, so the script
can be restarted. Slabs run ``box`` mode without skeletons and with one fixed intensity scaling;
each finished slab becomes a data asset; the ``slabs`` mode run mounts them all.
"""

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://codeocean.allenneuraldynamics.org/api/v1"
CAPSULE = "cde3a47b-a776-4a6c-94c3-e070c2092cc0"
WEIGHTS = {"id": "52a5d51b-5ba7-4e5a-a923-65e4dd04c359", "mount": "trailmap_finetune_v2"}
token = Path(sys.argv[1]).read_text().strip()
state_path = Path(sys.argv[2])
AUTH = "Basic " + base64.b64encode(f"{token}:".encode()).decode()
PARALLEL, MAX_ATTEMPTS = 3, 4


def api(path, body=None):
    for attempt in range(6):
        try:
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(
                BASE + path, data, {"Authorization": AUTH, "Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                t = r.read()
                return json.loads(t) if t else {}
        except OSError as error:
            print(time.strftime("%H:%M"), "api error", error, flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("Code Ocean API unreachable")


def save(state):
    state_path.write_text(json.dumps(state, indent=1))


state = json.loads(state_path.read_text())
while True:
    slabs = state["slabs"]
    for s in slabs:  # update running slabs
        if s.get("cid") and s["status"] == "running":
            r = api(f"/computations/{s['cid']}")
            if r["state"] == "completed":
                if r.get("has_results") and r.get("end_status") == "succeeded":
                    s["status"] = "predicted"
                    print(
                        time.strftime("%H:%M"),
                        s["name"],
                        "predicted in",
                        round(r["run_time"] / 3600, 2),
                        "h",
                        flush=True,
                    )
                else:
                    s["status"] = "retry" if s["attempts"] < MAX_ATTEMPTS else "failed"
                    print(
                        time.strftime("%H:%M"),
                        s["name"],
                        "ended",
                        r.get("end_status"),
                        "results",
                        r.get("has_results"),
                        "->",
                        s["status"],
                        flush=True,
                    )
        if s["status"] == "predicted" and not s.get("asset"):
            a = api(
                "/data_assets",
                {
                    "name": f"{state['subject']} slab {s['name']} prediction ({s['cid'][:8]})",
                    "mount": s["name"],
                    "tags": ["exaspim", "axon-segmentation", "slab", state["subject"]],
                    "source": {"computation": {"id": s["cid"], "path": f"{state['subject']}_box"}},
                },
            )
            s["asset"] = a["id"]
            print(time.strftime("%H:%M"), s["name"], "data asset", a["id"], flush=True)
        pending = s["status"] == "predicted" and s.get("asset") and not s.get("asset_ready")
        if pending and api(f"/data_assets/{s['asset']}")["state"] == "ready":
            s["asset_ready"] = True
            print(time.strftime("%H:%M"), s["name"], "asset ready", flush=True)
    save(state)
    running = sum(s["status"] == "running" for s in slabs)
    for s in slabs:  # launch waiting slabs
        if running >= PARALLEL:
            break
        if s["status"] in ("waiting", "retry"):
            params = [
                "box",
                state["asset"],
                s["lo"],
                s["hi"],
                "/data/trailmap_finetune_v2/best.safetensors",
                "6",
                "no",
                state["scaling"],
            ]
            r = api(
                "/computations",
                {"capsule_id": CAPSULE, "parameters": params, "data_assets": [WEIGHTS]},
            )
            s.update(
                cid=r["id"],
                status="running",
                attempts=s.get("attempts", 0) + 1,
                launched=time.strftime("%Y-%m-%d %H:%M"),
            )
            running += 1
            print(
                time.strftime("%H:%M"),
                "launched",
                s["name"],
                r["id"],
                "attempt",
                s["attempts"],
                flush=True,
            )
            save(state)
            time.sleep(30)
    if any(s["status"] == "failed" for s in slabs):
        print("a slab failed", MAX_ATTEMPTS, "times; stopping", flush=True)
        break
    if all(s.get("asset_ready") for s in slabs):
        final = state.setdefault("final", {})
        if not final.get("cid"):
            assets = [WEIGHTS] + [{"id": s["asset"], "mount": s["name"]} for s in slabs]
            r = api(
                "/computations",
                {"capsule_id": CAPSULE, "parameters": ["slabs", "6", "no"], "data_assets": assets},
            )
            final.update(cid=r["id"], status="running", attempts=final.get("attempts", 0) + 1)
            print(time.strftime("%H:%M"), "launched the skeleton run", r["id"], flush=True)
            save(state)
        else:
            r = api(f"/computations/{final['cid']}")
            if r["state"] == "completed":
                ok = r.get("has_results") and r.get("end_status") == "succeeded"
                print(
                    time.strftime("%H:%M"),
                    "skeleton run",
                    r.get("end_status"),
                    "results",
                    r.get("has_results"),
                    round(r["run_time"] / 3600, 2),
                    "h",
                    flush=True,
                )
                if ok or final["attempts"] >= MAX_ATTEMPTS:
                    final["status"] = "done" if ok else "failed"
                    save(state)
                    break
                final["cid"] = None
                save(state)
    time.sleep(300)
