"""Compare two snr-tiles runs on the same tiles: every per-block metric and QC value, and timing.

Usage: uv run python scripts/compare_snrtiles_runs.py <old run dir> <new run dir>
Each dir holds snr_tiles/<tile>/snr_blocks.json. Blocks are matched by tile and centre.
"""

import json
import sys
from pathlib import Path

import numpy as np

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])


def blocks(d):
    out = {}
    for f in sorted(d.glob("snr_tiles/*/snr_blocks.json")):
        for b in json.loads(f.read_text())["blocks"]:
            out[(f.parent.name, tuple(np.round(b["centre_um"], 1)))] = b
    return out


def flatten(m):
    qc = {f"qc.{x}": v for x, v in m.get("qc", {}).items()}
    return {x: v for x, v in m.items() if x != "qc"} | qc


A, B = blocks(old_dir), blocks(new_dir)
common = sorted(set(A) & set(B))
print(f"blocks: old {len(A)}, new {len(B)}, matched {len(common)}")
worst, diffs, n_values = 0.0, [], 0
for k in common:
    a, b = A[k].get("native", {}), B[k].get("native", {})
    flat_a, flat_b = flatten(a), flatten(b)
    for key in sorted(set(flat_a) | set(flat_b)):
        va, vb = flat_a.get(key), flat_b.get(key)
        n_values += 1
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (va, vb))
        if numeric:
            r = abs(va - vb) / max(abs(va), 1e-12)
            worst = max(worst, r)
            if r > 1e-6:
                diffs.append((k, key, va, vb, r))
        elif va != vb:
            diffs.append((k, key, va, vb, None))
print(
    f"{n_values} values compared; max relative difference {worst:.2e}; "
    f"{len(diffs)} differ by > 1e-6"
)
for d in diffs[:30]:
    print("  ", d)


def timing(bs):
    rows = [b for b in bs.values() if "timing" in b]
    keys = sorted({x for b in rows for x in b["timing"]})
    sec = np.array([b["seconds"] for b in rows])
    print(
        f"  {len(rows)} blocks, seconds per block: median {np.median(sec):.1f}, "
        f"total {sec.sum() / 60:.1f} min"
    )
    for x in keys:
        print(f"    {x:18s} median {np.median([b['timing'].get(x, 0) for b in rows]):5.2f} s")


print("old timing:")
timing(A)
print("new timing:")
timing(B)
