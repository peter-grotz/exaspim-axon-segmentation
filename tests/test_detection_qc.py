import numpy as np
from scipy.ndimage import gaussian_filter

from exaspim_axon_segmentation.inference.normalize import IntensityScaling
from exaspim_axon_segmentation.qc.detection_qc import block_qc


def _detector(threshold):
    """A stand-in detector: probability = normalized intensity above a threshold."""
    return lambda v: (gaussian_filter(v, 0.7) > threshold).astype(np.float32)


def _block(rng, tubes=True):
    raw = rng.normal(300, 15, (64, 128, 128)).astype(np.float32)
    if tubes:
        t = np.linspace(0, 1, 800)
        for a, b in (((32, 5, 5), (32, 120, 120)), ((10, 120, 10), (54, 10, 118))):
            p = np.round(np.asarray(a) + t[:, None] * np.subtract(b, a)).astype(int)
            raw[tuple(p.T)] += 900
        raw = gaussian_filter(raw, 0.6)
    return raw


def test_axon_block_passes():
    rng = np.random.default_rng(0)
    raw = _block(rng)
    q = block_qc(raw, _detector(0.3), IntensityScaling(300, 600))
    assert q["verdict"] == "ok", q


def test_noise_block_has_no_signal():
    rng = np.random.default_rng(1)
    raw = _block(rng, tubes=False)
    q = block_qc(raw, _detector(0.3), IntensityScaling(300, 600))
    assert q["verdict"] == "no_signal", q


def test_stretched_noise_is_a_false_detection():
    # a few bright specks pass the signal check, but a per-block scaling stretches the noise
    rng = np.random.default_rng(2)
    raw = _block(rng, tubes=False)
    raw[rng.integers(0, 64, 2500), rng.integers(0, 128, 2500), rng.integers(0, 128, 2500)] += 2000
    stretched = IntensityScaling(290, 40)  # what a dim block's own normalization does
    q = block_qc(raw, _detector(0.3), stretched, shared_scaling=IntensityScaling(300, 600))
    assert q["verdict"] == "false_detection", q
    assert {"contrast", "noise_control", "normalization"} & set(q["failed"])


def test_sparse_dilation_matches_full():
    from scipy.ndimage import binary_dilation

    from exaspim_axon_segmentation.qc.detection_qc import sparse_dilation

    rng = np.random.default_rng(1)
    for shape, density, it in (
        ((70, 90, 130), 1e-4, 2),
        ((64, 64, 64), 3e-3, 6),
        ((50, 129, 65), 0.02, 8),
    ):
        m = rng.random(shape) < density
        m[0, 0, 0] = m[-1, -1, -1] = True  # corners: dilation clipped at the border
        assert np.array_equal(sparse_dilation(m, it, cell=32), binary_dilation(m, iterations=it))


def test_clipped_histogram_matches_values():
    from exaspim_axon_segmentation.qc import detection_qc as q

    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(100, 5, 200_000), rng.normal(900, 50, 3_000)])
    v = np.clip(np.round(v), 0, None).astype(np.float32)
    ref = v.astype(np.float64)
    for _ in range(5):
        m, sd = ref.mean(), ref.std()
        ref = ref[np.abs(ref - m) <= 3 * sd]
    m, sd = q._clipped(v)
    assert np.isclose(m, ref.mean(), rtol=1e-12) and np.isclose(sd, ref.std(), rtol=1e-12)
