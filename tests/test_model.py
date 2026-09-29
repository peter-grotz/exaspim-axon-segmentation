from pathlib import Path

import numpy as np
import pytest
import torch

from exaspim_axon_segmentation.model import REGISTRY, TrailmapUNet3D, load_keras_weights
from exaspim_axon_segmentation.model.convert import keras_layer_map

REFERENCE = Path(__file__).parent / "data" / "keras_reference_axon.npz"


def test_output_shape_rule():
    assert TrailmapUNet3D.output_shape((64, 64, 64)) == (36, 36, 36)
    assert TrailmapUNet3D.output_shape((192, 128, 72)) == (164, 100, 44)
    for bad in [(60, 64, 64), (65, 64, 64), (56, 64, 64)]:
        with pytest.raises(ValueError):
            TrailmapUNet3D.output_shape(bad)


def test_forward_shape_and_range():
    model = TrailmapUNet3D().eval()
    with torch.no_grad():
        y = model(torch.rand(2, 1, 72, 64, 80))
    assert y.shape == (2, 1, 44, 36, 52)
    assert torch.all((y >= 0) & (y <= 1))


def test_parameter_count_matches_keras():
    # 19,078,337 counted over every array in AXON_model_210409.hdf5 (incl. BN moving stats).
    state = TrailmapUNet3D().state_dict()
    n = sum(v.numel() for k, v in state.items() if not k.endswith("num_batches_tracked"))
    assert n == 19_078_337
    assert len(keras_layer_map()) == 15 + 14 + 3


@pytest.fixture(scope="module")
def axon_model() -> TrailmapUNet3D:
    return load_keras_weights(REGISTRY["trailmap-axon-210409"].fetch())


def _predict(model: TrailmapUNet3D, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))[None, None])[0, 0].numpy()


@pytest.mark.network
def test_parity_with_keras(axon_model):
    ref = np.load(REFERENCE)
    noise = (np.random.default_rng(0).random((128, 128, 128)) ** 8).astype(np.float16)

    cases = {
        "real exaSPIM patch, 64³": (_predict(axon_model, ref["real_input"]), ref["real_output"]),
        "noise, 64³": (_predict(axon_model, noise[:64, :64, :64]), ref["noise64_output"]),
        "noise, 128³ (fully conv)": (
            _predict(axon_model, noise)[::4, ::4, ::4],
            ref["noise128_output_sub4"],
        ),
    }
    for name, (ours, theirs) in cases.items():
        assert ours.shape == theirs.shape, name
        np.testing.assert_allclose(ours, theirs, atol=1e-4, err_msg=name)
