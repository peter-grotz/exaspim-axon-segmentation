import numpy as np

from exaspim_axon_segmentation.io import Box, LevelTransform
from exaspim_axon_segmentation.postprocess import Category, CleanupConfig, Soma, cleanup

GRID = LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0))


def _volume() -> np.ndarray:
    p = np.zeros((20, 60, 60), dtype=np.uint8)
    p[10, 10, 5:55] = 250  # long confident "axon"
    p[10, 30, 5:55] = 180  # long but never confident: no core
    p[5, 50, 50] = 250  # confident speck: too small
    p[15, 40:59, 20] = 250  # confident line touching a soma at its end
    return p


def test_hysteresis_and_speckle():
    out = cleanup(_volume(), GRID, Box.full((20, 60, 60)), CleanupConfig(min_volume_um3=100))
    assert out[10, 10, 30] == Category.KEPT
    assert out[10, 30, 30] == Category.NO_CONFIDENT_CORE
    assert out[5, 50, 50] == Category.TOO_SMALL
    assert out[0, 0, 0] == Category.BACKGROUND


def test_soma_connected_removal():
    soma = Soma(center_um=(15 * 4.0, 58 * 3.0, 20 * 3.0), radius_um=6.0)
    config = CleanupConfig(min_volume_um3=100, remove_soma_connected=True)
    out = cleanup(_volume(), GRID, Box.full((20, 60, 60)), config, somata=[soma])
    assert out[15, 41, 20] == Category.SOMA_CONNECTED
    assert out[10, 10, 30] == Category.KEPT


def test_region_offset_applies_to_somata():
    region = Box((100, 100, 100), (120, 160, 160))
    soma = Soma(center_um=(115 * 4.0, 158 * 3.0, 120 * 3.0), radius_um=6.0)
    config = CleanupConfig(min_volume_um3=100, remove_soma_connected=True)
    out = cleanup(_volume(), GRID, region, config, somata=[soma])
    assert out[15, 41, 20] == Category.SOMA_CONNECTED
