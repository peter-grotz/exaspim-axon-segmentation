import numpy as np

from exaspim_axon_segmentation.inference import IntensityScaling, plan_tiles
from exaspim_axon_segmentation.io import Box


def test_tiles_cover_region_exactly_once():
    region = Box((5, 10, 0), (75, 50, 33))
    tiles = plan_tiles(region, tile_output=20)
    covered = np.zeros(region.shape, dtype=int)
    for tile in tiles:
        dst, src = tile.write_box(region)
        assert dst.shape == src.shape
        covered[dst.slices] += 1
        assert tile.in_box.shape == (48, 48, 48)
        assert np.all(np.subtract(tile.out_box.start, tile.in_box.start) == 14)
    assert np.all(covered == 1)
    assert {t.chunk for t in tiles} == {
        (i, j, k) for i in range(4) for j in range(2) for k in range(2)
    }


def test_intensity_scaling_clips():
    scaling = IntensityScaling(offset=10, scale=100)
    out = scaling(np.array([0, 10, 60, 110, 5000], dtype=np.uint16))
    np.testing.assert_allclose(out, [0, 0, 0.5, 1, 1])
    assert out.dtype == np.float32


def test_normalization_config_modes():
    import pytest
    from pydantic import ValidationError

    from exaspim_axon_segmentation.config import NormalizationConfig
    from exaspim_axon_segmentation.inference import estimate_scaling

    assert (
        estimate_scaling(NormalizationConfig(mode="gain", gain=2), None, 0, None).scale == 32767.5
    )
    scaling = estimate_scaling(NormalizationConfig(mode="range", low=5, high=305), None, 0, None)
    assert (scaling.offset, scaling.scale) == (5, 300)
    with pytest.raises(ValidationError):
        NormalizationConfig(mode="range", low=10, high=5)
