import numpy as np
from scipy.ndimage import binary_dilation

from exaspim_axon_segmentation.blockhist import N_RING, classify, current_metrics, histograms, stats
from exaspim_axon_segmentation.snr import SnrConfig, _score, _sub


def _block(seed=0, shape=(40, 60, 60)):
    rng = np.random.default_rng(seed)
    raw = rng.poisson(40, shape).astype(np.float32)
    stored = np.zeros(shape, np.uint8)
    stored[20, 10:50, 30] = 255  # a line of axon
    stored[5:8, 5:8, 5:8] = 231
    stored[30, 30, 10:20] = 229  # just below p >= 0.9
    raw[stored >= 230] += 60
    return raw, stored


def test_rings_match_dilation():
    _, stored = _block()
    ring = classify(stored, 230)
    axon = stored >= 230
    for k in range(1, N_RING + 1):
        assert np.array_equal(ring <= k, binary_dilation(axon, iterations=k))
    assert ring.max() == N_RING + 1


def test_current_metrics_reproduce_score():
    raw, stored = _block()
    inside = np.ones(raw.shape, bool)
    inside[:, :, :10] = False
    h = histograms(raw, classify(stored, 230), inside)
    assert h.sum() == raw.size and h[1].sum() == (~inside).sum()
    want, _ = _score(raw, stored.astype(np.float32) / 255, SnrConfig(axon_p=229.5 / 255))
    got = current_metrics(h, float(np.percentile(_sub(raw), 10.0)))
    for k in ("signal", "background", "noise_sd", "snr"):
        assert np.isclose(got[k], want[k], rtol=1e-6), k


def test_plain_stats():
    raw, stored = _block(1)
    h = histograms(raw, classify(stored, 230), np.ones(raw.shape, bool))
    m, sd, n = stats(h[0, 1:].sum(0))
    bg = raw[stored < 230]
    assert n == bg.size and np.isclose(m, bg.mean()) and np.isclose(sd, bg.std())
