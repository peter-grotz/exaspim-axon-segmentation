import copy

import numpy as np
import pytest
import torch
import zarr

from exaspim_axon_segmentation.inference import IntensityScaling, TilePredictor, plan_tiles
from exaspim_axon_segmentation.inference.predict import predict_tiles
from exaspim_axon_segmentation.io import Box, NgffImage
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.model import TrailmapUNet3D

from .conftest import write_ngff

pytestmark = pytest.mark.heavy


def test_predict_tiles_matches_per_tile_reference(tmp_path):
    device = "cpu"
    rng = np.random.default_rng(0)
    volume = (rng.random((80, 80, 80)) ** 6 * 2000).astype(np.uint16)
    image = NgffImage(
        write_ngff(tmp_path / "img.zarr", [(volume, [4.0, 3.0, 3.0], [1.5, 1.1, 1.1])])
    )
    torch.manual_seed(0)
    model = TrailmapUNet3D().eval()
    for module in model.modules():  # non-trivial BN stats so outputs vary across voxels
        if isinstance(module, torch.nn.BatchNorm3d):
            module.running_var.uniform_(0.5, 2.0)
    scaling = IntensityScaling(offset=0.0, scale=500.0)
    region = Box((5, 10, 0), (70, 80, 50))
    tiles = plan_tiles(region, tile_output=36)

    uri = str(tmp_path / "out.zarr")
    out = create_ngff_array(
        uri, source=image.transforms[0], region=region, chunks=(36, 36, 36), dtype="uint8",
        name="p",
    )  # fmt: skip
    reference = copy.deepcopy(model)  # TilePredictor moves its model to the device
    predictor = TilePredictor(model, torch.device(device), torch.float32)
    stats = predict_tiles(
        image, 0, tiles, region, scaling, predictor, out, zarr.open_group(uri, mode="a"),
        batch_size=3, io_threads=4,
    )  # fmt: skip
    assert stats.n_predicted == len(tiles)

    expected = np.zeros(region.shape, dtype=np.uint8)
    for tile in tiles:
        with torch.no_grad():
            x = torch.from_numpy(scaling(image.read_padded(0, tile.in_box)))[None, None]
            p = reference(x)[0, 0].numpy()
        dst, src = tile.write_box(region)
        expected[dst.slices] = np.round(p[src.slices] * 255).astype(np.uint8)
    got = zarr.open_group(uri, mode="r")["0"][:]
    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 1


def test_auto_device_never_picks_mps():
    from exaspim_axon_segmentation.inference.predict import resolve_device

    assert resolve_device("auto").type in {"cuda", "cpu"}


def test_self_check_rejects_broken_device():
    from exaspim_axon_segmentation.inference.predict import TilePredictor

    class Broken(TilePredictor):
        def __call__(self, batch):
            return np.zeros((batch.shape[0], *(s - 28 for s in batch.shape[1:])), np.float32)

    predictor = Broken(TrailmapUNet3D().eval(), torch.device("cpu"), torch.bfloat16)
    with pytest.raises(RuntimeError, match="disagrees with CPU"):
        predictor.self_check(batch_size=1)
