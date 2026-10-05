"""Collect per-sample numbers and slide-sized images for the batch deck.

Usage: uv run python scripts/deck_prepare.py results/batch_md [--untraced]
Writes <batch>/deck/samples.json and <batch>/deck/img/<subject>_{map,overview}.jpg; with
--untraced, numbers and maps come from the trace-removed skeletons (remove_traced.py) where a
subject has traces, plus its before/after figure, into samples_untraced.json.
"""

import json
import sys
from pathlib import Path

from PIL import Image

batch = Path(sys.argv[1])
untraced = "--untraced" in sys.argv[2:]
out = batch / "deck"
removal = (
    json.loads((batch / "untraced_summary.json").read_text())
    if untraced and (batch / "untraced_summary.json").exists()
    else {}
)
(out / "img").mkdir(parents=True, exist_ok=True)
jobs = {j["subject"]: j for j in json.loads((batch / "jobs.json").read_text())}
state = json.loads((batch / "state.json").read_text())
samples = []
for subject in sorted(jobs):
    tag = f"{subject}_MD_L2"
    traced = subject in removal
    folder = f"{tag}_skeleton_untraced" if traced else f"{tag}_skeleton"
    summary_path = batch / subject / folder / "summary.json"
    qc = batch / subject / "qc"
    if state.get(subject, {}).get("state") != "done" or not summary_path.exists():
        samples.append({"subject": subject, "ok": False})
        continue
    s = json.loads(summary_path.read_text())
    d = s.get("dendrites", {})
    images = {}
    for kind, name, width in (
        ("map", f"{tag}_skeleton_map{'_untraced' if traced else ''}.png", 2200),
        ("overview", f"{tag}_skeleton.png", 1500),
        ("removed", f"{tag}_traced_removed.png", 2400),
    ):
        src = qc / name
        if src.exists():
            im = Image.open(src).convert("RGB")
            im.thumbnail((width, width))
            dst = out / "img" / f"{subject}_{kind}.jpg"
            im.save(dst, quality=85)
            images[kind] = {"path": str(dst.resolve()), "w": im.width, "h": im.height}
    samples.append(
        {
            "subject": subject,
            "ok": True,
            "asset": jobs[subject]["asset"].rsplit("/", 1)[-1],
            "skeleton_mm": s["length_in_region_um"] / 1e3,
            "pieces": len(s.get("pieces", [])),
            "region_mm3": s["region_volume_mm3"],
            "somata": d.get("somata"),
            "dendrite_mm": d.get("dendrite_um", 0) / 1e3 if d else None,
            "axon_mm": d.get("axon_candidate_um", 0) / 1e3 if d else None,
            "axon_pieces": len(s.get("axon_candidate_pieces", [])),
            "images": images,
            "traced": removal.get(subject),
        }
    )
(out / ("samples_untraced.json" if untraced else "samples.json")).write_text(
    json.dumps(samples, indent=1)
)
ok = [x for x in samples if x["ok"]]
print(
    len(ok),
    "samples;",
    sum("map" in x["images"] for x in ok),
    "with maps;",
    sum("overview" in x["images"] for x in ok),
    "with overviews",
)
