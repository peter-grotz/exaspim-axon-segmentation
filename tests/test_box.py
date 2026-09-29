import pytest

from exaspim_axon_segmentation.io import Box


def test_shape_slices_center():
    box = Box((1, 2, 3), (4, 6, 9))
    assert box.shape == (3, 4, 6)
    assert box.slices == (slice(1, 4), slice(2, 6), slice(3, 9))
    assert box.center == (2, 4, 6)


def test_pad_then_clip():
    box = Box((0, 5, 5), (3, 8, 10)).pad(2).clip((4, 9, 11))
    assert box == Box((0, 3, 3), (4, 9, 11))


def test_rejects_inverted():
    with pytest.raises(ValueError):
        Box((2, 0, 0), (1, 1, 1))
