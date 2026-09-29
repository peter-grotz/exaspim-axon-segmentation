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
    skel = skeletonize_probability(_probability(), GRID, region, SkeletonConfig(spur_um=10.0))
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
