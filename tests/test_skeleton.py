import numpy as np

from exaspim_axon_segmentation.io import Box, LevelTransform
from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig, skeletonize_probability
from exaspim_axon_segmentation.train import read_swc

GRID = LevelTransform((4.0, 3.0, 3.0), (1.5, 1.0, 1.0))


def _probability() -> np.ndarray:
    p = np.zeros((30, 40, 140), dtype=np.uint8)
    p[14:17, 19:22, 20:120] = 230  # 100-voxel tube along x (300 um)
    p[14:17, 22:37, 70:73] = 230  # 15-voxel side branch along y (~45 um)
    p[14:17, 16:19, 40:43] = 230  # 2-voxel spur (~6 um): should be pruned
    p[5, 5, 5] = 250  # isolated speck: dropped
    return p


def test_skeleton_length_pruning_and_swc(tmp_path):
    region = Box((10, 20, 30), (40, 60, 170))
    skel = skeletonize_probability(
        _probability(), GRID, region, SkeletonConfig(spur_um=10.0, anchor_um=0)
    )
    # main tube ~ 300 um (minus thinning end effects) + branch ~ 45 um; spur and speck removed
    assert 300 < skel.length_um < 360
    assert skel.voxels[5, 5, 5] == 0
    # physical coordinates carry the region offset and translation
    assert np.all(skel.coordinates_um[:, 2] > (30 + 20) * 3.0)
    skel.write_swc(tmp_path / "s.swc")
    swc = read_swc(tmp_path / "s.swc")
    assert len(swc.ids) == len(skel.coordinates_um)
    assert (swc.parents == -1).sum() == 1  # one connected piece
    assert np.isclose(swc.path_length_um(), skel.length_um, rtol=1e-6)


def test_min_piece_drops_short_isolated_fragments():
    p = _probability()
    p[25, 5:9, 100] = 230  # isolated 4-voxel (~12 um) fragment
    region = Box((0, 0, 0), p.shape)
    kept = skeletonize_probability(p, GRID, region, SkeletonConfig(min_piece_um=30.0, anchor_um=0))
    assert not kept.voxels[25, 5:9, 100].any()
    loose = skeletonize_probability(
        p,
        GRID,
        region,
        SkeletonConfig(min_piece_um=0.0, min_volume_um3=0, anchor_um=0, speckle_max_um=0.0),
    )
    assert loose.voxels[25, 5:9, 100].any()


def test_speckle_removal_drops_short_single_plane_pieces_only():
    p = _probability()
    p[25, 5:17, 100] = 230  # ~36 um fragment confined to one z-plane: speckle
    region = Box((0, 0, 0), p.shape)
    off = SkeletonConfig(min_volume_um3=0, min_piece_um=30.0, anchor_um=0, speckle_max_um=0.0)
    on = SkeletonConfig(min_volume_um3=0, min_piece_um=30.0, anchor_um=0, speckle_max_um=60.0)
    assert skeletonize_probability(p, GRID, region, off).voxels[25, 5:17, 100].any()
    kept = skeletonize_probability(p, GRID, region, on)
    assert not kept.voxels[25, 5:17, 100].any()
    # the 300 um tube thins to a single-plane centerline but is long, so it survives
    assert kept.length_um > 300


def test_bridging_joins_aligned_gap_into_one_cable():
    p = np.zeros((20, 30, 200), dtype=np.uint8)
    p[9:12, 14:17, 10:60] = 230  # 150 um
    p[9:12, 14:17, 66:116] = 230  # 150 um, after an 18 um gap on the same line
    p[9:12, 2:5, 150:170] = 230  # short parallel piece far away
    region = Box((0, 0, 0), p.shape)
    joined = skeletonize_probability(
        p, GRID, region, SkeletonConfig(min_piece_um=250.0, anchor_um=0)
    )
    assert joined.length_um > 250  # the two halves count as one cable and survive
    apart = skeletonize_probability(
        p, GRID, region, SkeletonConfig(min_piece_um=250.0, bridge_um=0)
    )
    assert apart.length_um == 0


def test_isolation_keeps_pieces_near_anchor_cable_only():
    p = np.zeros((20, 60, 260), dtype=np.uint8)
    p[9:12, 10:13, 10:250] = 230  # 720 um anchor cable along x
    p[9:12, 16:40, 100:103] = 230  # ~70 um side piece starting ~9 um from it: attached
    p[9:12, 30:55, 200:203] = 230  # ~75 um piece ~57 um from the anchor: isolated
    region = Box((0, 0, 0), p.shape)
    config = SkeletonConfig(min_piece_um=60.0, anchor_um=500.0, attach_um=20.0)
    kept = skeletonize_probability(p, GRID, region, config)
    assert kept.voxels[10, 20:35, 100:103].any()
    assert not kept.voxels[10, 32:53, 200:203].any()
    assert kept.voxels[10, 11, 20:240].any()


def test_connection_follows_dim_signal_but_not_background():
    p = np.zeros((20, 30, 200), dtype=np.uint8)
    p[9:12, 14:17, 10:90] = 230  # 240 um
    p[9:12, 14:17, 90:98] = 30  # dim stretch (p ~ 0.12, below threshold)
    p[9:12, 14:17, 98:180] = 230  # 246 um
    region = Box((0, 0, 0), p.shape)
    base = {"min_piece_um": 100.0, "anchor_um": 0.0, "bridge_um": 0.0}
    joined = skeletonize_probability(p, GRID, region, SkeletonConfig(**base))
    assert joined.edges.size and joined.filled.any()
    assert _pieces(joined) == 1
    p[9:12, 14:17, 90:98] = 0  # same gap over pure background: not connected
    apart = skeletonize_probability(p, GRID, region, SkeletonConfig(**base))
    assert _pieces(apart) == 2


def _pieces(skel) -> int:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(skel.coordinates_um)
    graph = coo_matrix((np.ones(len(skel.edges)), tuple(skel.edges.T)), shape=(n, n))
    return connected_components(graph, directed=False)[0]


def test_pieces_are_written_as_horta_swcs_in_absolute_micrometers(tmp_path):
    p = np.zeros((20, 40, 200), dtype=np.uint8)
    p[9:12, 5:8, 10:190] = 230  # long piece
    p[9:12, 30:33, 40:100] = 230  # shorter, separate piece
    region = Box((2, 3, 4), (22, 43, 204))
    config = SkeletonConfig(min_piece_um=50.0, anchor_um=0, connect_um=0, bridge_um=0)
    skel = skeletonize_probability(p, GRID, region, config)
    records = skel.write_pieces(tmp_path, "S_MD")
    assert [r["file"] for r in records] == ["S_MD_0001.swc", "S_MD_0002.swc"]
    assert records[0]["length_um"] > records[1]["length_um"]
    for record in records:
        path = tmp_path / record["file"]
        lines = path.read_text().splitlines()
        assert lines[0].startswith("# ORIGINAL_SOURCE")
        assert lines[1].startswith("# OFFSET") and lines[2].startswith("# COLOR")
        swc = read_swc(path)
        assert list(swc.ids) == list(range(1, len(swc.ids) + 1))
        assert all(parent < node for node, parent in zip(swc.ids, swc.parents, strict=True))
        assert (swc.parents == -1).sum() == 1
        assert np.isclose(swc.path_length_um(), record["length_um"])
        # absolute positions after applying OFFSET are skeleton nodes
        nearest = np.min(
            np.linalg.norm(swc.zyx_um[:, None] - skel.coordinates_um[None], axis=2), axis=1
        )
        assert nearest.max() < 1e-3


def test_dendrites_of_somata_are_separated_from_passing_axons():
    from exaspim_axon_segmentation.postprocess.dendrites import DendriteConfig, label_dendrites

    x = np.arange(0.0, 900.0, 3.0)
    dendrite = np.stack(
        [np.full_like(x, 50.0), np.full_like(x[:100], 50.0).repeat(3)[: len(x)], x], 1
    )
    dendrite = np.stack([np.full(100, 50.0), np.full(100, 50.0), np.arange(100) * 3.0 + 30.0], 1)
    broken = dendrite[-1] + np.stack([np.zeros(20), np.zeros(20), np.arange(20) * 3.0 + 30.0], 1)
    axon = np.stack([np.full(400, 50.0), np.full(400, 900.0), np.arange(400) * 3.0 - 300.0], 1)
    coords = np.concatenate([dendrite, broken, axon])

    def chain(start, n):
        return np.stack([np.arange(start, start + n - 1), np.arange(start + 1, start + n)], 1)

    edges = np.concatenate([chain(0, 100), chain(100, 20), chain(120, 400)])
    soma = np.array([[50.0, 50.0, 0.0]])
    label = label_dendrites(coords, edges, soma, DendriteConfig())
    assert label[:100].all()  # connected to the soma
    assert label[100:120].all()  # broken-off piece, agglomerated onto the dendrite
    assert not label[120:].any()  # axon passing 850 µm away
