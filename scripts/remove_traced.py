"""Remove predicted skeleton that overlaps manual traces; keep only untraced cable.

Usage: uv run python scripts/remove_traced.py <batch dir> [--near-um 10] [--min-piece-um 100]

Per subject with traces (``traces_index.json`` from collect_traces.py): predicted nodes within
``near_um`` of any trace (axon or dendrite, densified to 2 µm) are removed, then any piece that
lost nodes and is now shorter than ``min_piece_um`` is dropped too (cut-off stubs). Writes
``<subject>_MD_L2_skeleton_untraced/`` (skeleton.swc, swc/, axon_candidates.swc, axon_swc/,
summary.json) and ``<batch>/untraced_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.classify.graph import SkeletonGraph
from exaspim_axon_segmentation.classify.traces import densify
from exaspim_axon_segmentation.postprocess.skeleton import AxonSkeleton, SkeletonConfig
from exaspim_axon_segmentation.train.swc import read_swc


def trace_points(files: list[dict]) -> np.ndarray:
    """All trace nodes, densified to 2 µm, as (z, y, x) µm."""
    points = []
    for record in files:
        swc = read_swc(record["path"])
        graph = densify(SkeletonGraph(swc.zyx_um, swc.edges()), 2.0)
        points.append(graph.coordinates_um)
    return np.concatenate(points) if points else np.zeros((0, 3))


def lengths_by_type(coords, edges, types) -> dict[str, float]:
    length = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    both = types[edges[:, 0]] == types[edges[:, 1]]
    out = {"total_um": float(length.sum())}
    for name, t in (("dendrite_um", 3), ("axon_candidate_um", 2)):
        out[name] = float(length[both & (types[edges[:, 0]] == t)].sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--near-um", type=float, default=10.0)
    parser.add_argument("--min-piece-um", type=float, default=100.0)
    args = parser.parse_args()
    index = json.loads((args.batch / "traces_index.json").read_text())
    report = {}
    for subject, entry in sorted(index.items()):
        tag = f"{subject}_MD_L2"
        src = args.batch / subject / f"{tag}_skeleton"
        if not (src / "skeleton.swc").exists():
            continue
        files = [f for f in entry["files"] if f["offset"] and f["inside"] > 0.9]
        traced = trace_points(files)
        swc = read_swc(src / "skeleton.swc")
        coords, edges, types = swc.zyx_um, swc.edges(), swc.types
        dist, _ = cKDTree(traced).query(coords, distance_upper_bound=args.near_um)
        overlap = np.isfinite(dist)

        # drop overlapping nodes, then the stubs they leave behind
        keep = ~overlap
        n = len(coords)
        graph = coo_matrix((np.ones(len(edges)), tuple(edges.T)), shape=(n, n))
        _, before_piece = connected_components(graph, directed=False)
        touched = np.zeros(before_piece.max() + 1, dtype=bool)
        touched[before_piece[overlap]] = True
        kept_edges = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]
        sub = coo_matrix((np.ones(len(kept_edges)), tuple(kept_edges.T)), shape=(n, n))
        _, piece = connected_components(sub, directed=False)
        seg = np.linalg.norm(coords[kept_edges[:, 0]] - coords[kept_edges[:, 1]], axis=1)
        plen = np.bincount(piece[kept_edges[:, 0]], weights=seg, minlength=piece.max() + 1)
        stub = touched[before_piece] & (plen[piece] < args.min_piece_um)
        keep &= ~stub

        skeleton = AxonSkeleton(np.zeros((1, 1, 1), dtype=bool), coords, edges, SkeletonConfig(),
                                np.zeros(n, dtype=bool), types).subset(keep)  # fmt: skip
        out = args.batch / subject / f"{tag}_skeleton_untraced"
        out.mkdir(parents=True, exist_ok=True)
        skeleton.write_swc(out / "skeleton.swc")
        pieces = skeleton.write_pieces(out / "swc", f"{tag}_untraced")
        axon = skeleton.subset(skeleton.node_types == 2)
        axon.write_swc(out / "axon_candidates.swc")
        axon_pieces = axon.write_pieces(out / "axon_swc", f"{tag}_untraced_axon")

        summary = json.loads((src / "summary.json").read_text())
        before = lengths_by_type(coords, edges, types)
        after = lengths_by_type(skeleton.coordinates_um, skeleton.edges, skeleton.node_types)
        # how much of the traced cable inside the processed box the prediction found (alignment)
        lo, hi = coords.min(axis=0) - 50, coords.max(axis=0) + 50
        in_box = traced[np.all((traced >= lo) & (traced <= hi), axis=1)]
        hit = np.isfinite(cKDTree(coords).query(in_box, distance_upper_bound=args.near_um)[0])
        stats = {
            "trace_files": len(files), "before": before, "after": after,
            "removed_um": before["total_um"] - after["total_um"],
            "trace_um_near_region": float(len(in_box) * 2.0),
            "trace_coverage": float(hit.mean()) if len(in_box) else None,
            "pieces": len(pieces), "axon_candidate_pieces": len(axon_pieces),
            "near_um": args.near_um, "min_piece_um": args.min_piece_um,
        }  # fmt: skip
        summary.update({
            "pieces": pieces, "axon_candidate_pieces": axon_pieces,
            "length_in_region_um": after["total_um"],
            "dendrites": summary.get("dendrites", {}) | {
                "dendrite_um": after["dendrite_um"],
                "axon_candidate_um": after["axon_candidate_um"],
            },
            "traced_removed": stats,
        })  # fmt: skip
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        report[subject] = stats
        cov = "–" if stats["trace_coverage"] is None else f"{stats['trace_coverage']:.0%}"
        removed_mm, total_mm = stats["removed_um"] / 1e3, before["total_um"] / 1e3
        print(f"{subject}: removed {removed_mm:.1f} of {total_mm:.1f} mm "
              f"(axon cand. {before['axon_candidate_um'] / 1e3:.1f} -> "
              f"{after['axon_candidate_um'] / 1e3:.1f} mm) · {len(files)} trace files · traced "
              f"cable near MD {stats['trace_um_near_region'] / 1e3:.1f} mm, covered {cov}",
              flush=True)  # fmt: skip
    (args.batch / "untraced_summary.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
