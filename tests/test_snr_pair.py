import numpy as np

from exaspim_axon_segmentation.snr_pair import _to_a


def test_to_a_maps_b_grid_onto_a_grid():
    # b is a at half the voxel size, shifted by 10 um and scaled 1.0 (A = identity)
    sca, scb = np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0])
    oa, ob = np.zeros(3), np.array([-10.0, -10.0, -10.0])
    b = np.zeros((80, 80, 80), np.float32)
    b[40, 40, 40] = 1.0  # physical (30, 30, 30)
    a = _to_a(b, ob, scb, np.eye(3), np.zeros(3), oa, sca, (30, 30, 30), order=0)
    assert np.argwhere(a > 0).tolist() == [[15, 15, 15]]
