"""Collect manual traces (Janelia Workstation / Horta SWCs) per subject and check their format.

Usage: uv run python scripts/collect_traces.py <batch dir> <root> [<root> ...]
Writes <batch>/traces_index.json: per subject, the de-duplicated trace files with compartment,
node count and the share of nodes inside the subject's fused image.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from exaspim_axon_segmentation.io import ExaspimAsset
from exaspim_axon_segmentation.train.swc import read_swc

batch, roots = Path(sys.argv[1]), [Path(r) for r in sys.argv[2:]]
jobs = {j["subject"]: j for j in json.loads((batch / "jobs.json").read_text())}
found: dict[str, dict[str, dict]] = {}
seen: set[str] = set()
for root in roots:
    for path in sorted(root.rglob("*.swc")):
        try:
            with path.open(errors="replace") as f:
                head = [f.readline() for _ in range(3)]
        except OSError:
            continue
        if not head[0].startswith("# ORIGINAL_SOURCE Janelia Workstation"):
            continue  # not a manual Horta/Workstation trace
        m = re.search(r"exaSPIM[_-](\d{6})", str(path)) or re.search(r"N\d{3}-(\d{6})", path.name)
        if not m or m.group(1) not in jobs:
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        name = path.name.lower()
        comp = "axon" if "axon" in name else "dendrite" if "dend" in name else "other"
        has_offset = any(h.upper().startswith("# OFFSET") for h in head)
        found.setdefault(m.group(1), {})[str(path)] = {"compartment": comp, "offset": has_offset}

index = {}
for subject, files in sorted(found.items()):
    fused = ExaspimAsset(jobs[subject]["asset"]).fused
    tf = fused.transforms[0]
    hi = (np.asarray(fused.shape(0)) - 0.5) * np.asarray(tf.scale) + np.asarray(tf.translation)
    records = []
    for path, info in files.items():
        swc = read_swc(path)
        inside = (
            np.all((swc.zyx_um >= 0) & (swc.zyx_um <= hi), axis=1).mean() if len(swc.ids) else 0
        )
        records.append(
            {"path": path, **info, "nodes": len(swc.ids), "inside": round(float(inside), 3)}
        )
    ok = [r for r in records if r["inside"] > 0.9 and r["offset"]]
    index[subject] = {"files": records, "usable": len(ok)}
    print(
        subject,
        f"{len(records)} trace files, {len(ok)} usable "
        f"(axon {sum(r['compartment'] == 'axon' for r in ok)}, "
        f"dendrite {sum(r['compartment'] == 'dendrite' for r in ok)})",
        flush=True,
    )
(batch / "traces_index.json").write_text(json.dumps(index, indent=1))
