"""Re-draw the batch skeleton maps in the Allen Institute plotting style.

Usage: uv run python scripts/render_maps_allen.py <batch dir> [subject ...]
Uses the allen-plotting-style helper (fonts + palette); region footprints are cached in
<batch>/<subject>/qc/region_footprint.npz so re-renders skip the CCF annotation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/skills/allen-plotting-style/scripts"))
import allen_style as A

from exaspim_axon_segmentation.io import CcfOntology, ExaspimAsset
from exaspim_axon_segmentation.qc.skeleton import render_skeleton_map

A.apply_style("white")
batch = Path(sys.argv[1])
only = set(sys.argv[2:])
md = CcfOntology.load().resolve(["MD"])
for job in json.loads((batch / "jobs.json").read_text()):
    subject = job["subject"]
    if only and subject not in only:
        continue
    skeleton = batch / subject / f"{subject}_MD_L2_skeleton"
    if not (skeleton / "skeleton.swc").exists():
        print(subject, "no skeleton", flush=True)
        continue
    asset = ExaspimAsset(job["asset"], annotation_root=job["annotation"] or None)
    render_skeleton_map(
        skeleton, asset, md, batch / subject / "qc" / f"{subject}_MD_L2_skeleton_map.png",
        head_font=A.HEAD, cache=batch / subject / "qc" / "region_footprint.npz",
    )  # fmt: skip
    print(subject, "ok", flush=True)
