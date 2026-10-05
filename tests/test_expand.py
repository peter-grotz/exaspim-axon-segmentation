import numpy as np

from exaspim_axon_segmentation.expand import (
    AXON,
    DENDRITE,
    DoneArea,
    ExpandConfig,
    Graph,
    _exits,
    _extend_followed,
    _path_from_region,
    _stitch,
)
from exaspim_axon_segmentation.io import Box, LevelTransform, RegionRoi
from exaspim_axon_segmentation.postprocess.skeleton import AxonSkeleton, SkeletonConfig


def _region() -> RegionRoi:
    # a 400 µm cube region at the origin, on a 20 µm grid
    mask = np.ones((20, 20, 20), dtype=bool)
    return RegionRoi(
        frozenset(), mask, Box((0, 0, 0), (20, 20, 20)), LevelTransform((20.0,) * 3, (0.0,) * 3)
    )


def _line(start, stop, step=3.0, kind=AXON) -> AxonSkeleton:
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    n = int(np.linalg.norm(stop - start) // step) + 1
    pts = start + np.linspace(0, 1, n)[:, None] * (stop - start)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    return AxonSkeleton(np.zeros((1, 1, 1), bool), pts, edges, SkeletonConfig(),
                        np.zeros(n, bool), np.full(n, kind))  # fmt: skip


def test_done_area_contains_region_and_cubes():
    done = DoneArea(_region())
    assert done.contains(np.array([[100.0, 100.0, 100.0]]))[0]
    assert not done.contains(np.array([[100.0, 100.0, 700.0]]))[0]
    done.add_cube(np.array([200.0, 200.0, 800.0]), 400.0)
    assert done.contains(np.array([[100.0, 100.0, 700.0]]))[0]
    assert not done.contains(np.array([[100.0, 100.0, 700.0]]), n_cubes=0)[0]
    grid = LevelTransform((10.0,) * 3, (0.0,) * 3)
    mask = done.mask_on(grid, Box((0, 0, 30), (40, 40, 110)), n_cubes=1)  # x 300..1100 µm
    assert mask[:38, :38, :9].all()  # the region, x < 390 µm (inside its edge voxels)
    assert not mask[:, :, 12:29].any()  # gap between region and cube (x 420..590)
    assert mask[:, :, 31:69].all()  # the cube, x 610..990
    assert not mask[:, :, 72:].any()  # beyond it


def test_exit_found_where_axon_leaves_region_then_followed_after_stitch():
    done = DoneArea(_region())
    graph = Graph()
    graph.add(_line([200, 200, 100], [200, 200, 390]), 0)  # runs out through the +x face
    followed = graph.types == AXON
    config = ExpandConfig()
    dist = _path_from_region(graph, followed, done, done.roi)
    exits = _exits(graph, followed, done, config, dist)
    assert len(exits) == 1 and exits[0, 3] > 0.99  # pointing along +x
    # the next cube's cable continues from 396 µm; stitching joins it, and it becomes followed
    added = graph.add(_line([200, 200, 396], [200, 200, 700]), 1)
    _stitch(graph, added, config.join_um, config.join_any_um)
    followed = _extend_followed(graph, followed)
    assert followed[added].all()
    # dendrite cable touching the axon is not followed
    graph.add(_line([200, 200, 703], [200, 200, 800], kind=DENDRITE), 1)
    _stitch(graph, np.arange(graph.n - 33, graph.n), config.join_um, config.join_any_um)
    followed = _extend_followed(graph, followed)
    assert not followed[graph.types == DENDRITE].any()


def test_stitch_joins_ends_ahead_across_a_gap_but_not_sideways():
    config = ExpandConfig()
    graph = Graph()
    graph.add(_line([200, 200, 100], [200, 200, 390]), 0)
    ahead = graph.add(_line([200, 200, 407], [200, 200, 600]), 1)  # 17 µm ahead
    _stitch(graph, ahead, config.join_um, config.join_any_um)
    side = graph.add(_line([200, 215, 150], [200, 215, 350]), 1)  # parallel, 15 µm to the side
    _stitch(graph, side, config.join_um, config.join_any_um)
    _, piece = __import__("scipy.sparse.csgraph", fromlist=["x"]).connected_components(
        graph.adjacency(), directed=False
    )
    assert piece[ahead[0]] == piece[0]
    assert piece[side[0]] != piece[0]
