import numpy as np

from exaspim_axon_segmentation.train import read_swc, resample, write_swc


def _write(tmp_path):
    p = tmp_path / "t.swc"
    p.write_text(
        "# ORIGINAL_SOURCE test\n# OFFSET 100.0 200.0 300.0\n"
        "1 0 0 0 0 1 -1\n2 0 10 0 0 1 1\n3 0 10 5 0 1 2\n4 0 0 0 3 1 1\n"
    )
    return p


def test_read_applies_offset(tmp_path):
    swc = read_swc(_write(tmp_path))
    np.testing.assert_allclose(swc.xyz_um[1], [110, 200, 300])
    np.testing.assert_allclose(swc.zyx_um[1], [300, 200, 110])
    assert swc.path_length_um() == 18.0


def test_resample_step_and_topology(tmp_path):
    swc = read_swc(_write(tmp_path))
    fine = resample(swc, step_um=0.748)
    e = fine.edges()
    lengths = np.linalg.norm(fine.xyz_um[e[:, 0]] - fine.xyz_um[e[:, 1]], axis=1)
    assert lengths.max() <= 0.748 + 1e-9
    assert np.isclose(fine.path_length_um(), swc.path_length_um())
    assert (fine.parents == -1).sum() == 1
    position = {int(i): k for k, i in enumerate(fine.ids)}
    assert all(position[int(p)] < k for k, p in enumerate(fine.parents) if p != -1)
    for original in swc.xyz_um:  # every original node survives
        assert np.min(np.linalg.norm(fine.xyz_um - original, axis=1)) < 1e-9


def test_write_round_trip(tmp_path):
    swc = resample(read_swc(_write(tmp_path)), 1.0)
    out = tmp_path / "out.swc"
    write_swc(swc, out, offset_um=np.array([100.0, 200.0, 300.0]))
    back = read_swc(out)
    np.testing.assert_allclose(back.xyz_um, swc.xyz_um, atol=1e-5)
    np.testing.assert_array_equal(back.parents, swc.parents)


def test_empty_swc_round_trips(tmp_path):
    path = tmp_path / "empty.swc"
    path.write_text("# ORIGINAL_SOURCE test\n# OFFSET 0 0 0\n")
    swc = read_swc(path)
    assert len(swc.ids) == 0 and swc.edges().shape == (0, 2)
