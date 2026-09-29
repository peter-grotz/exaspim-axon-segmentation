"""Skeletonize a probability map into an axon graph with physical lengths.

Pipeline: threshold → hysteresis (keep components with a confident core) → drop specks →
per-component 3D thinning → branch graph (``skan``) on the anisotropic grid → prune short spurs
(thinning artefacts) → SWC forest + summary.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cc3d
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import breadth_first_order, connected_components
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

from exaspim_axon_segmentation.io import Box, LevelTransform

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkeletonConfig:
    """Skeletonization settings.

    Attributes
    ----------
    threshold : float
        Probability defining the neurite mask. 0.2 maximized trace recall on 822178 MD (held-out
        slab: 0.79 at 0.88 precision near traces, vs 0.69 / 0.91 at 0.3).
    core : float
        A component is kept only if some voxel exceeds this probability.
    min_volume_um3 : float
        Components smaller than this are dropped before thinning.
    spur_um : float
        Terminal branches shorter than this are pruned (repeated ``prune_rounds`` times).
    prune_rounds : int
        Pruning passes; each pass can expose new short terminals.
    """

    threshold: float = 0.2
    core: float = 0.5
    min_volume_um3: float = 100.0
    spur_um: float = 10.0
    prune_rounds: int = 2


@dataclass(frozen=True, eq=False)
class AxonSkeleton:
    """Skeleton voxels and graph of a probability map.

    Attributes
    ----------
    voxels : numpy.ndarray of bool
        Skeleton voxels on the probability map's grid (region-local).
    coordinates_um : numpy.ndarray, shape (n, 3)
        Physical (z, y, x) position of every skeleton node.
    edges : numpy.ndarray, shape (m, 2)
        Node index pairs of the skeleton graph.
    config : SkeletonConfig
        Settings used.
    """

    voxels: NDArray[np.bool_]
    coordinates_um: NDArray[np.float64]
    edges: NDArray[np.int64]
    config: SkeletonConfig

    @property
    def edge_lengths_um(self) -> NDArray[np.float64]:
        """Physical length of every edge."""
        a, b = self.coordinates_um[self.edges[:, 0]], self.coordinates_um[self.edges[:, 1]]
        return np.linalg.norm(a - b, axis=1)

    @property
    def length_um(self) -> float:
        """Total skeleton length."""
        return float(self.edge_lengths_um.sum())

    def length_inside_um(
        self, inside: NDArray[np.bool_], grid: LevelTransform, region: Box
    ) -> float:
        """Length of edges whose both endpoints fall inside a mask on the same grid."""
        index = np.floor(
            (self.coordinates_um - np.asarray(grid.translation)) / np.asarray(grid.scale) + 0.5
        ).astype(int) - np.asarray(region.start)
        index = np.clip(index, 0, np.asarray(inside.shape) - 1)
        node_in = inside[tuple(index.T)]
        keep = node_in[self.edges[:, 0]] & node_in[self.edges[:, 1]]
        return float(self.edge_lengths_um[keep].sum())

    def write_swc(self, path: Path) -> None:
        """Write the skeleton as an SWC forest (one root per connected piece), in µm (x, y, z).

        Node type is 0 (undefined): the network does not distinguish axon from dendrite yet.
        """
        n = len(self.coordinates_um)
        graph = _adjacency(self.edges, n)
        n_parts, part = connected_components(graph, directed=False)
        parent = np.full(n, -1, dtype=np.int64)
        order = []
        for p in range(n_parts):
            members = np.flatnonzero(part == p)
            degree = np.asarray(graph[members].sum(axis=1)).ravel()
            root = members[np.argmin(degree)]  # an end point where there is one
            nodes, predecessors = breadth_first_order(graph, root, directed=False)
            order.extend(nodes.tolist())
            parent[nodes] = predecessors[nodes]
        ids = np.empty(n, dtype=np.int64)
        ids[np.asarray(order)] = np.arange(1, n + 1)
        lines = [
            "# exaspim-axon-segmentation skeleton (TRAILMAP probability, physical um)",
            f"# config {asdict(self.config)}",
        ]
        xyz = self.coordinates_um[:, ::-1]
        for node in order:
            p = parent[node]
            x, y, z = xyz[node]
            lines.append(f"{ids[node]} 0 {x:.3f} {y:.3f} {z:.3f} 1.0 {ids[p] if p >= 0 else -1}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")


def skeletonize_probability(
    probability: NDArray[np.uint8], grid: LevelTransform, region: Box, config: SkeletonConfig
) -> AxonSkeleton:
    """Skeletonize a uint8 (p × 255) probability map covering ``region`` of ``grid``."""
    scale = np.asarray(grid.scale)
    mask = probability > round(config.threshold * 255)
    labels, n = cc3d.connected_components(mask, connectivity=26, return_N=True, out_dtype=np.uint32)
    stats = cc3d.statistics(labels)
    core = np.zeros(n + 1, dtype=bool)
    core[np.unique(labels[probability > round(config.core * 255)])] = True
    min_voxels = config.min_volume_um3 / float(np.prod(scale))

    voxels = np.zeros(mask.shape, dtype=bool)
    kept = 0
    for k in range(1, n + 1):
        if not core[k] or stats["voxel_counts"][k] < min_voxels:
            continue
        bbox = stats["bounding_boxes"][k]
        voxels[bbox] |= skeletonize(labels[bbox] == k)
        kept += 1
    log.info("Skeletonized %d of %d components (p > %.2f)", kept, n, config.threshold)

    skeleton = Skeleton(voxels, spacing=scale)
    for _ in range(config.prune_rounds):
        branches = summarize(skeleton, separator="_")
        spurs = branches.index[
            (branches["branch_type"] == 1) & (branches["branch_distance"] < config.spur_um)
        ].to_numpy()
        if spurs.size == 0:
            break
        skeleton = skeleton.prune_paths(spurs)
    coordinates = skeleton.coordinates * scale + (
        np.asarray(region.start) * scale + grid.translation
    )
    graph = skeleton.graph.tocoo()
    upper = graph.row < graph.col
    edges = np.stack([graph.row[upper], graph.col[upper]], axis=1).astype(np.int64)
    final = np.zeros(mask.shape, dtype=bool)
    final[tuple(np.round(skeleton.coordinates).astype(int).T)] = True
    result = AxonSkeleton(final, coordinates, edges, config)
    log.info("Skeleton: %d nodes, %.1f mm total length", len(coordinates), result.length_um / 1e3)
    return result


def _adjacency(edges: NDArray[np.int64], n: int):
    from scipy.sparse import coo_matrix

    data = np.ones(len(edges) * 2, dtype=np.int8)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
