import json

import numpy as np
import torch

from exaspim_axon_segmentation.inference import IntensityScaling
from exaspim_axon_segmentation.model import TrailmapUNet3D
from exaspim_axon_segmentation.train.labels import Label
from exaspim_axon_segmentation.train.trainer import (
    TrainConfig,
    TrainingVolume,
    augment,
    finetune,
    weighted_bce,
)


def _volume():
    rng = np.random.default_rng(0)
    raw = (rng.random((40, 80, 160)) * 10).astype(np.uint16)
    labels = np.full(raw.shape, Label.BACKGROUND, dtype=np.uint8)
    raw[20, 40, 10:150] = 500
    labels[20, 40, 10:150] = Label.AXON
    labels[19:22, 39:42, 10:150][labels[19:22, 39:42, 10:150] != Label.AXON] = Label.EDGE
    raw[10, 10, 50] = 400
    labels[10, 10, 50] = Label.HARD_NEGATIVE
    labels[:, :, :5] = Label.IGNORE
    return raw, labels


def test_weighted_bce_ignores_zero_weight_voxels():
    logits = torch.tensor([[10.0, -10.0, 10.0]])
    labels = torch.tensor([[Label.AXON, Label.BACKGROUND, Label.IGNORE]], dtype=torch.uint8)
    assert float(weighted_bce(logits, labels, {Label.AXON: 1.0, Label.BACKGROUND: 1.0})) < 1e-3


def test_augment_keeps_labels_aligned():
    raw, labels = _volume()
    r, lab = augment(
        raw[:, :64, :64].astype(np.float32), labels[:, :64, :64], np.random.default_rng(3), 0.0
    )
    assert r.shape == lab.shape
    np.testing.assert_array_equal(r >= 500, lab == Label.AXON)


def test_finetune_smoke(tmp_path):
    raw, labels = _volume()
    config = TrainConfig(
        input_size=64, batch_size=1, steps=2, val_every=1, val_patches=2, max_minutes=5
    )
    scaling = IntensityScaling(offset=0, scale=500)
    volumes = [
        TrainingVolume("a", raw, labels, scaling, (0, 120), (120, 160), "region"),
        TrainingVolume("b", raw, labels, scaling, None, (0, 160), "heldout-b"),
    ]
    summary = finetune(TrailmapUNet3D(), volumes, tmp_path, config, device=torch.device("cpu"))
    lines = [json.loads(row) for row in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in lines] == [0, 1, 2]
    assert {"region/f1", "heldout-b/f1", "mean_f1"} <= set(lines[0])
    assert (tmp_path / "best.safetensors").exists() and (tmp_path / "last.safetensors").exists()
    assert summary["steps_run"] == 2


def test_label_envelope_ignores_far_voxels():
    from exaspim_axon_segmentation.io import Box, LevelTransform
    from exaspim_axon_segmentation.train.labels import LabelConfig, build_labels

    class Image:  # minimal stand-in for NgffImage
        transforms = (LevelTransform((4.0, 3.0, 3.0), (0.0, 0.0, 0.0)),)

        def read(self, level, box):
            image = np.full(box.shape, 5, dtype=np.uint16)
            image[10, 30, :] = 100  # the traced axon
            return image

    points = np.array([[40.0, 30.0 * 3.0, x * 3.0] for x in np.arange(0, 60, 0.3)])
    box = Box.full((20, 60, 60))
    labels, _, _ = build_labels(
        Image(), 0, box, np.ones(box.shape, bool), points, LabelConfig(envelope_um=15.0)
    )
    assert labels[10, 30, 30] == Label.AXON
    assert labels[10, 30 + 3, 30] == Label.BACKGROUND  # 9 um away: inside the envelope
    assert labels[10, 55, 30] == Label.IGNORE  # 75 um away
