import numpy as np

from exaspim_axon_segmentation.io.ngff import LevelTransform
from exaspim_axon_segmentation.train.coarse import CoarseLabelConfig, coarse_grid, coarse_labels
from exaspim_axon_segmentation.train.labels import Label
from exaspim_axon_segmentation.train.trainer import TrainConfig, degrade


def test_coarse_grid_centres_pooled_voxels():
    g = coarse_grid(LevelTransform((4.0, 3.0, 3.0), (2.0, 1.5, 1.5)), (5, 5, 5))
    assert g.scale == (20.0, 15.0, 15.0)
    assert g.translation == (10.0, 7.5, 7.5)  # centre of fine voxels 0..4


def test_coarse_labels_axon_line_and_envelope():
    grid = LevelTransform((20.0, 15.0, 15.0), (0.0, 0.0, 0.0))
    raw = np.zeros((10, 20, 20), np.float32)
    raw[5, 10, :] = 10.0
    pts = np.array([[100.0, 150.0, x] for x in np.arange(0, 300, 4.0)])
    lab, stats = coarse_labels(raw, grid, np.zeros(3, int), np.ones(raw.shape, bool), pts,
                               CoarseLabelConfig(hard_negatives=False))  # fmt: skip
    assert (lab[5, 10, :] == Label.AXON).all()
    assert lab[0, 0, 0] == Label.IGNORE  # beyond the 60 um envelope
    assert stats["voxels"]["axon"] == 20


def test_degrade_blurs_and_adds_noise():
    x = np.zeros((16, 16, 16), np.float32)
    x[8, 8, 8] = 1.0
    cfg = TrainConfig(degrade_blur=(1.5, 0.5), degrade_gain=(1.0, 1.0), degrade_noise_sd=0.01,
                      degrade_jitter=0.0)  # fmt: skip
    y = degrade(x, np.random.default_rng(0), cfg)
    assert y[8, 8, 8] < 0.5 and y[7, 8, 8] > 0.05  # spread mostly along z
    assert 0.003 < y[0].std() < 0.02
