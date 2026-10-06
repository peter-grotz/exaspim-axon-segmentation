import numpy as np

from exaspim_axon_segmentation.expand import (
    AXON,
    DENDRITE,
    DoneArea,
    ExpandConfig,
    Followed,
    Graph,
    _exits,
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
    state = Followed(graph, followed)
    added = graph.add(_line([200, 200, 396], [200, 200, 700]), 1)
    _stitch(graph, added, config.join_um, config.join_any_um)
    followed = state.grow(graph, config)
    assert followed[added].all()
    # dendrite cable touching the axon is not followed
    graph.add(_line([200, 200, 703], [200, 200, 800], kind=DENDRITE), 1)
    _stitch(graph, np.arange(graph.n - 33, graph.n), config.join_um, config.join_any_um)
    followed = state.grow(graph, config)
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


class _FakeTracer:
    """Links every start straight to the piece ahead, across the gap."""

    def __init__(self, target_xyz):
        self.target = np.asarray(target_xyz, float)

    def trace(self, starts, frags):
        out = []
        for s in starts:
            p0 = s.path[-1]
            pts = p0 + np.linspace(0, 1, 6)[:, None] * (self.target - p0)
            _, j = frags.tree.query(self.target)
            out.append(dict(status="linked", link=int(frags.piece[j]), p_link=0.9,
                            trace=pts, steps=5, path_um=0.0, rejected=[]))  # fmt: skip
        return out


def test_trace_ends_links_a_gap_and_closes_the_end():
    from exaspim_axon_segmentation.expand import TRACED, _trace_ends

    done = DoneArea(_region())
    graph = Graph()
    graph.add(_line([200, 200, 100], [200, 200, 300]), 0)
    other = graph.add(_line([200, 200, 340], [200, 200, 390]), 0)  # 40 µm gap: no geometric join
    followed = np.zeros(graph.n, bool)
    followed[: other[0]] = True
    tracer = _FakeTracer([340.0, 200.0, 200.0])  # x, y, z of the far piece's near end
    state = Followed(graph, followed)
    followed, ended, counts = _trace_ends(graph, state, np.zeros(0, bool), done, done.roi,
                                          tracer, ExpandConfig(), None, 0)  # fmt: skip
    assert counts["linked"] >= 1
    assert followed[other].all()
    assert (graph.source == TRACED).sum() == 5
    # the far piece's own outer end (x = 390) stays open for the next pass; the near tip closed
    assert ended[np.argmin(np.abs(graph.coords[:, 2] - 300))]


def test_followed_axon_goes_straight_through_a_crossing_fibre_without_adopting_it():
    config = ExpandConfig()
    graph = Graph()
    graph.add(_line([200, 200, 100], [200, 200, 300]), 0)  # the followed axon, along +x
    state = Followed(graph, np.ones(graph.n, bool))
    ahead = graph.add(_line([200, 200, 303], [200, 200, 600]), 1)  # its continuation
    cross = graph.add(_line([200, 50, 450], [200, 350, 450]), 1)  # a fibre crossing it (along y)
    j_ahead = ahead[np.argmin(np.abs(graph.coords[ahead, 2] - 450))]
    j_cross = cross[np.argmin(np.abs(graph.coords[cross, 1] - 200))]
    graph.edges = np.concatenate([graph.edges, [[ahead[0] - 1, ahead[0]], [j_ahead, j_cross]]])
    followed = state.grow(graph, config)  # no tracer: the straightest arm within 45° is kept
    assert followed[ahead].all()
    assert not followed[cross].any()
    assert state.stats["arms_rejected"] >= 1
    # growing again does not revisit the rejected arms
    assert not state.grow(graph, config)[cross].any()


def test_prune_twigs_drops_short_side_branches_only():
    from exaspim_axon_segmentation.expand import _prune_twigs

    main = _line([200, 200, 100], [200, 200, 400])  # 300 µm along x
    k = len(main.coordinates_um) // 2
    j = main.coordinates_um[k]
    twig = j + np.array([[0, 3.0, 0], [0, 6.0, 0], [0, 9.0, 0]])  # 9 µm side twig
    branch = j + np.array([[0, -3.0 * i, 0] for i in range(1, 11)])  # 30 µm side branch
    pts = np.concatenate([main.coordinates_um, twig, branch])
    n0, nt = len(main.coordinates_um), len(twig)
    chain = [[n0 + nt + i, n0 + nt + i + 1] for i in range(9)]
    edges = np.concatenate([main.edges, [[k, n0], [n0, n0 + 1], [n0 + 1, n0 + 2]],
                            [[k, n0 + nt]], chain])  # fmt: skip
    sk = AxonSkeleton(np.zeros((1, 1, 1), bool), pts, edges, SkeletonConfig(),
                      np.zeros(len(pts), bool), np.full(len(pts), AXON))  # fmt: skip
    out = _prune_twigs(sk, 15.0)
    assert len(out.coordinates_um) == len(pts) - nt


def test_open_end_at_a_branch_point_whose_arms_were_rejected():
    from exaspim_axon_segmentation.expand import _open_ends

    graph = Graph()
    graph.add(_line([200, 200, 100], [200, 200, 300]), 0)
    arm = graph.add(_line([200, 203, 300], [200, 300, 300]), 1)
    graph.edges = np.concatenate([graph.edges, [[arm[0] - 1, arm[0]]]])
    followed = np.zeros(graph.n, bool)
    followed[: arm[0]] = True
    ends = np.flatnonzero(_open_ends(graph, followed))
    assert arm[0] - 1 in ends  # the end touching unfollowed cable is still an open end
