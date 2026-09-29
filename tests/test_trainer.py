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
