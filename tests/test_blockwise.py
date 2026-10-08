import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components

from exaspim_axon_segmentation.io import Box, LevelTransform
from exaspim_axon_segmentation.postprocess.blockwise import BlockConfig, skeletonize_blockwise
from exaspim_axon_segmentation.postprocess.skeleton import (
    SkeletonConfig,
    _adjacency,
    skeletonize_probability,
)


def _tubes(shape=(60, 160, 160)):
    vol = np.zeros(shape, np.float32)
    t = np.linspace(0, 1, 2000)
    for a, b in (
        ((30, 5, 5), (30, 154, 150)),
        ((10, 150, 20), (50, 10, 140)),
        ((5, 80, 5), (55, 80, 155)),
    ):
        p = np.asarray(a) + t[:, None] * (np.subtract(b, a))
        vol[tuple(np.round(p).astype(int).T)] = 1
    vol = gaussian_filter(vol, (0.8, 1.2, 1.2))
    return np.clip(vol / vol.max() * 255 * 1.6, 0, 255).astype(np.uint8)


def test_blockwise_matches_whole_volume_across_block_boundaries():
    prob = _tubes()
    grid = LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0))
    region = Box((0, 0, 0), prob.shape)
    cfg = SkeletonConfig(anchor_um=0)
    whole = skeletonize_probability(prob, grid, region, cfg)
    blocks = skeletonize_blockwise(
        prob, grid, region, cfg, BlockConfig(block_vox=40, halo_vox=16, workers=1)
    )
    assert abs(blocks.length_um - whole.length_um) / whole.length_um < 0.03
    _, pw = connected_components(_adjacency(whole.edges, len(whole.coordinates_um)), directed=False)
    _, pb = connected_components(
        _adjacency(blocks.edges, len(blocks.coordinates_um)), directed=False
    )
    assert pb.max() == pw.max()  # no tube broken where blocks meet


def test_blockwise_connects_an_end_onto_a_fibre_in_the_next_block():
    # a fibre along x, and a second one ending 12 um short of its side; the gap is dim (p ~ 0.1)
    vol = np.zeros((40, 120, 120), np.float32)
    vol[20, 60, 5:115] = 1
    vol[20, 8:54, 60] = 1  # ends 6 voxels (18 um) before the first fibre
    vol = gaussian_filter(vol, (0.8, 1.2, 1.2))
    prob = np.clip(vol / vol.max() * 255 * 1.6, 0, 255)
    prob[19:22, 54:59, 59:62] = np.maximum(prob[19:22, 54:59, 59:62], 26)  # dim stretch
    prob = prob.astype(np.uint8)
    grid = LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0))
    region = Box((0, 0, 0), prob.shape)
    cfg = SkeletonConfig(anchor_um=0, min_piece_um=30.0)
    whole = skeletonize_probability(prob, grid, region, cfg)
    blocks = skeletonize_blockwise(
        prob, grid, region, cfg, BlockConfig(block_vox=56, halo_vox=24, workers=1)
    )
    for sk in (whole, blocks):
        _, piece = connected_components(
            _adjacency(sk.edges, len(sk.coordinates_um)), directed=False
        )
        assert piece.max() == 0  # joined into one piece
    assert abs(blocks.length_um - whole.length_um) / whole.length_um < 0.03


def test_slab_source_reads_overlapping_slabs_as_one_volume(tmp_path):
    import zarr

    from exaspim_axon_segmentation.postprocess.blockwise import SlabSource

    prob = _tubes()
    cuts = [(0, 90), (70, 160)]  # two slabs along x, overlapping by 20 voxels
    slabs = []
    for k, (a, b) in enumerate(cuts):
        uri = str(tmp_path / f"slab{k}.zarr")
        g = zarr.open_group(uri, mode="w")
        arr = g.create_array("0", shape=(60, 160, b - a), chunks=(32, 32, 32), dtype="uint8")
        arr[:] = prob[:, :, a:b]
        slabs.append((uri, (100, 200, 300 + a), (160, 360, 300 + b)))
    src = SlabSource(slabs)
    assert src.shape == prob.shape
    np.testing.assert_array_equal(src[(slice(0, 60), slice(0, 160), slice(0, 160))], prob)
    grid = LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0))
    cfg = SkeletonConfig(anchor_um=0)
    a = skeletonize_blockwise(
        prob, grid, Box((0, 0, 0), prob.shape), cfg, BlockConfig(40, 16, workers=1)
    )
    b = skeletonize_blockwise(
        src, grid, Box((0, 0, 0), prob.shape), cfg, BlockConfig(40, 16, workers=1)
    )
    assert abs(a.length_um - b.length_um) < 1e-6


def test_fast_swc_forest_matches_pieces():
    from exaspim_axon_segmentation.postprocess.skeleton import AxonSkeleton

    prob = _tubes()
    grid = LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0))
    sk = skeletonize_probability(
        prob, grid, Box((0, 0, 0), prob.shape), SkeletonConfig(anchor_um=0, min_piece_um=0)
    )
    swc = sk.to_swc()
    pos = {i: k for k, i in enumerate(swc.ids)}
    assert all(
        p == -1 or pos[p] < pos[i] for i, p in zip(swc.ids, swc.parents, strict=True)
    )  # parents first
    assert (swc.parents == -1).sum() == len(sk.pieces())  # one root per piece
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        rec = sk.write_pieces(Path(d), "t")
        expected = sk.pieces()
        assert [r["nodes"] for r in rec] == [len(nodes) for nodes, _ in expected]
        assert np.allclose([r["length_um"] for r in rec], [length for _, length in expected])
        text = (Path(d) / "t_0001.swc").read_text().splitlines()
        rows = [ln.split() for ln in text if not ln.startswith("#")]
        assert rows[0][6] == "-1" and all(int(r[6]) < int(r[0]) for r in rows[1:])
    assert isinstance(sk, AxonSkeleton)
