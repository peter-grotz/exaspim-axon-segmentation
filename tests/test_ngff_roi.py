import numpy as np
import pytest

from exaspim_axon_segmentation.io import Box, LevelTransform, NgffImage, RegionRoi


def test_transform_round_trip():
    tf = LevelTransform((8.0, 5.984, 5.984), (3.5, 2.618, 2.618))
    index = np.array([[0, 0, 0], [10, 20, 30]])
    np.testing.assert_allclose(tf.to_index(tf.to_physical(index)), index)


def test_physical_extent_is_voxel_edges():
    tf = LevelTransform((2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    lo, hi = tf.physical_extent(Box((0, 0, 0), (3, 3, 3)))
    np.testing.assert_allclose(lo, 0.0)
    np.testing.assert_allclose(hi, 6.0)


def test_level_parsing(labels_and_image):
    labels_uri, _, _ = labels_and_image
    labels = NgffImage(labels_uri)
    assert labels.n_levels == 2
    assert labels.shape(0) == (6, 6, 6)
    assert labels.transforms[1].translation == (4.0, 4.0, 4.0)
    assert labels.level_for_spacing(15.0) == 1


def test_resample_nearest_matches_physical_lookup(labels_and_image):
    labels_uri, image_uri, labels = labels_and_image
    image = NgffImage(image_uri)
    grid = image.transforms[0]
    box = Box.full(image.shape(0))
    out = NgffImage(labels_uri).resample_nearest(0, grid, box)

    # Brute force: every image voxel center -> nearest label voxel.
    idx = np.indices(box.shape).reshape(3, -1).T
    src = np.floor(grid.to_physical(idx) / 8.0 + 0.5).astype(int)
    inside = np.all((src >= 0) & (src < 6), axis=1)
    expected = np.zeros(len(idx), dtype=labels.dtype)
    expected[inside] = labels[tuple(src[inside].T)]
    np.testing.assert_array_equal(out.ravel(), expected)


def test_region_roi_bounds(labels_and_image):
    labels_uri, _, _ = labels_and_image
    roi = RegionRoi.from_annotation(NgffImage(labels_uri), {7})
    # Voxels z 2..3, y 1..4, x 3..4 at 8 um with centers at index*8 -> edges at index*8 - 4.
    np.testing.assert_allclose(roi.lo_um, [12.0, 4.0, 20.0])
    np.testing.assert_allclose(roi.hi_um, [28.0, 36.0, 36.0])
    assert roi.n_voxels == 2 * 4 * 2
    assert roi.volume_mm3 == pytest.approx(16 * 512 * 1e-9)


def test_region_roi_missing_label(labels_and_image):
    labels_uri, _, _ = labels_and_image
    with pytest.raises(ValueError, match="None of labels"):
        RegionRoi.from_annotation(NgffImage(labels_uri), {99})


def test_read_padded_edges(labels_and_image):
    _, image_uri, _ = labels_and_image
    image = NgffImage(image_uri)
    full = image.read(0)
    block = image.read_padded(0, Box((-2, 0, 10), (3, 4, 14)))
    expected = np.pad(full[0:3, 0:4, 10:12], ((2, 0), (0, 0), (0, 2)), mode="edge")
    np.testing.assert_array_equal(block, expected)


def test_roi_mask_queries(labels_and_image):
    labels_uri, image_uri, _ = labels_and_image
    roi = RegionRoi.from_annotation(NgffImage(labels_uri), {7})
    grid = NgffImage(image_uri).transforms[0]
    sampled = roi.sample_on(grid, Box.full((12, 12, 12)))
    expected = NgffImage(labels_uri).resample_nearest(0, grid, Box.full((12, 12, 12))) == 7
    np.testing.assert_array_equal(sampled, expected)
    assert roi.fraction_inside(grid, Box((0, 0, 0), (2, 2, 2))) == 0
    assert roi.fraction_inside(grid, Box((0, 0, 0), (2, 2, 2)), dilate_um=40) > 0
    assert roi.fraction_inside(grid, Box((5, 3, 7), (6, 4, 8))) == 1


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_grown_region_adds_the_requested_volume_as_a_uniform_shell(fraction):
    # shells here are several voxels thick; a one-voxel shell overshoots on ties in distance
    tf = LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0))
    zz, yy, xx = np.indices((60, 80, 80))
    ball = ((zz - 30) * 4.0) ** 2 + ((yy - 40) * 3.0) ** 2 + ((xx - 40) * 3.0) ** 2 < 90.0**2
    roi = RegionRoi(frozenset({1}), ball[5:55, 8:72, 8:72], Box((5, 8, 8), (55, 72, 72)), tf)
    grown = roi.grown(fraction)
    assert grown.growth == fraction
    assert abs(grown.n_voxels / roi.n_voxels - (1 + fraction)) < 0.02
    # the original region is still inside, at the same physical place
    assert grown.sample_on(tf, roi.mask_box)[roi.mask].all()
    # and the shell is uniform: equal growth (in µm) along z and along x despite anisotropy
    extent = [grown.hi_um - grown.lo_um, roi.hi_um - roi.lo_um]
    dz, dx = (extent[0] - extent[1])[[0, 2]]
    assert abs(dz - dx) <= 2 * 4.0
    assert roi.grown(0) is roi
