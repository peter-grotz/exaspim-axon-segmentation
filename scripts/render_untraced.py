"""Draw untraced-only maps and before/after figures for subjects whose traces overlapped.

Usage: uv run python scripts/render_untraced.py <batch dir> [subject ...]
Needs remove_traced.py output and the cached region footprints from render_maps_allen.py.
Writes qc/<tag>_skeleton_map_untraced.png and, where cable was removed,
qc/<tag>_traced_removed.png (before | after, Allen style).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / ".claude/skills/allen-plotting-style/scripts"))
import allen_style as A

from exaspim_axon_segmentation.io import CcfOntology, ExaspimAsset
from exaspim_axon_segmentation.qc.skeleton import MAP_PALETTE, render_skeleton_map
from exaspim_axon_segmentation.train.swc import read_swc

A.apply_style("white")
TRACE, REMOVED = A.BRAND["violet"], A.BRAND["maroon"]


def lines(ax, zyx, edges, colour, width, alpha=1.0, z=2):
    seg = np.stack([zyx[edges[:, 0]][:, [2, 1]], zyx[edges[:, 1]][:, [2, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors=colour, linewidths=width, alpha=alpha, zorder=z))


def frame(ax, cache):
    z = np.load(cache)
    footprint, grown, lo, cell = z["footprint"], z["grown"], z["lo"], z["cell"]
    hi = lo + np.multiply(footprint.shape, cell)
    fill = np.zeros((*footprint.shape, 4))
    fill[footprint] = (*plt.matplotlib.colors.to_rgb(MAP_PALETTE["footprint"]), 1.0)
    ax.imshow(fill, extent=(lo[1], hi[1], hi[0], lo[0]), interpolation="nearest", zorder=0)
    yy = lo[0] + (np.arange(footprint.shape[0]) + 0.5) * cell[0]
    xx = lo[1] + (np.arange(footprint.shape[1]) + 0.5) * cell[1]
    ax.contour(xx, yy, grown.astype(float), levels=[0.5], colors=MAP_PALETTE["outline"],
               linestyles="--", linewidths=1.0, zorder=1)  # fmt: skip
    m = 0.04 * max(hi - lo)
    ax.set_xlim(lo[1] - m, hi[1] + m)
    ax.set_ylim(hi[0] + m, lo[0] - m)
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MAP_PALETTE["spine"])
    ax.set_xlabel("x (um)", fontsize=9)
    ax.set_ylabel("y (um)", fontsize=9)
    return (hi[0] - lo[0]) / (hi[1] - lo[1]), lo, hi


def typed(ax, swc, axon_width=2.0):
    zyx, e, t = swc.zyx_um, swc.edges(), swc.types
    both_d = (t[e[:, 0]] == 3) & (t[e[:, 1]] == 3)
    lines(ax, zyx, e[both_d], MAP_PALETTE["dendrite"], 0.5)
    lines(ax, zyx, e[~both_d], MAP_PALETTE["axon"], axon_width, z=3)


batch = Path(sys.argv[1])
only = set(sys.argv[2:])
jobs = {j["subject"]: j for j in json.loads((batch / "jobs.json").read_text())}
report = json.loads((batch / "untraced_summary.json").read_text())
index = json.loads((batch / "traces_index.json").read_text())
md = CcfOntology.load().resolve(["MD"])
for subject in sorted(jobs):
    stats = report.get(subject)
    if only and subject not in only:
        continue
    tag = f"{subject}_MD_L2"
    qc = batch / subject / "qc"
    cache = qc / "region_footprint.npz"
    original = batch / subject / f"{tag}_skeleton"
    if not (original / "skeleton.swc").exists() or not cache.exists():
        print(subject, "skipped (no skeleton or footprint cache)", flush=True)
        continue
    job = jobs[subject]
    asset = ExaspimAsset(job["asset"], annotation_root=job["annotation"] or None)
    untraced = batch / subject / f"{tag}_skeleton_untraced"
    if stats is not None:
        render_skeleton_map(untraced, asset, md, qc / f"{tag}_skeleton_map_untraced.png",
                            head_font=A.HEAD, cache=cache,
                            title=f"{subject} · md · untraced cable only · "
                                  f"{stats['after']['total_um'] / 1e3:.0f} mm/")  # fmt: skip

    before = read_swc(original / "skeleton.swc")
    after = read_swc(untraced / "skeleton.swc") if stats is not None else before
    kept = np.isfinite(cKDTree(after.zyx_um).query(before.zyx_um, distance_upper_bound=0.01)[0])
    e = before.edges()
    removed = ~(kept[e[:, 0]] & kept[e[:, 1]])
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), layout="constrained")
    aspect = None
    for ax in axes:
        aspect, lo, hi = frame(ax, cache)
    fig.set_size_inches(19, 9.5 * aspect + 1.4)
    files = index.get(subject, {}).get("files", [])
    for f in files:  # traces near the region, thin violet
        t = read_swc(f["path"])
        inside = np.all((t.zyx_um[:, 1:] >= lo - 200) & (t.zyx_um[:, 1:] <= hi + 200), axis=1)
        te = t.edges()
        te = te[inside[te[:, 0]] & inside[te[:, 1]]]
        lines(axes[0], t.zyx_um, te, TRACE, 0.8, alpha=0.9, z=4)
    typed(axes[0], before)
    lines(axes[0], before.zyx_um, e[removed], REMOVED, 1.6, z=5)
    typed(axes[1], after)
    summary = json.loads((original / "summary.json").read_text()).get("dendrites", {})
    if stats is None:
        left = "before · no manual traces found for this sample/"
        right = (f"after · unchanged · axon candidates "
                 f"{summary.get('axon_candidate_um', 0) / 1e3:.0f} mm · dendrites "
                 f"{summary.get('dendrite_um', 0) / 1e3:.0f} mm/")  # fmt: skip
        head = f"{subject} · md · no manual traces: nothing removed/"
    else:
        b, a = stats["before"], stats["after"]
        left = (f"before · traces (violet) and the predicted cable they overlap, removed "
                f"(maroon) · {stats['removed_um'] / 1e3:.0f} mm/")  # fmt: skip
        right = (f"after · axon candidates {b['axon_candidate_um'] / 1e3:.0f} -> "
                 f"{a['axon_candidate_um'] / 1e3:.0f} mm · dendrites "
                 f"{b['dendrite_um'] / 1e3:.0f} -> {a['dendrite_um'] / 1e3:.0f} mm/")  # fmt: skip
        head = (f"{subject} · md · {stats['trace_files']} manual trace files · predicted cable "
                f"within {stats['near_um']:.0f} um of a trace removed/")  # fmt: skip
    axes[0].set_title(left, loc="left", fontsize=13, fontfamily=A.HEAD)
    axes[1].set_title(right, loc="left", fontsize=13, fontfamily=A.HEAD)
    fig.suptitle(head, x=0.01, ha="left", fontsize=18, fontweight="bold", fontfamily=A.HEAD)
    fig.savefig(qc / f"{tag}_before_after.png", dpi=150)
    plt.close(fig)
    print(subject, "before/after" + ("" if stats else " (no traces)"), flush=True)
